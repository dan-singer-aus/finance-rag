"""Ingest the corpus under one chunking config, score it, append a result row.

    uv run python -m evals.measure --strategy character-splitting --target-size 800
    uv run python -m evals.measure --strategy by-line --no-captions --k 3 10

Rows go to `results/retrieval_runs.jsonl`, append-only.

Re-ingests every time (~$0.008, a minute or two) so the recorded config is the
one that built the corpus rather than a description of it.

Not a `SUITES` entry in `evals/__main__.py`: a suite reads the system, this
replaces every chunk in it.

⚠️ `by-line` ignores `target_size`, but the row records it anyway — don't read
a delta between two `by-line` rows as a size effect.
"""

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from psycopg import Connection

from db.connection import connection
from embedding import EMBEDDING_MODEL
from evals.recall_fixtures import RECALL_FIXTURES
from evals.score_retrieval import TOP_K
from evals.score_retrieval import main as score_recall
from ingest.chunking import TARGET_SIZE, Chunker, by_characters, by_line
from ingest.pipeline import ingest_corpus

RESULTS_FILE = Path(__file__).parent.parent / "results" / "retrieval_runs.jsonl"

# Name -> a factory binding that strategy's knobs, so `chunk_document` can stay
# a plain `(str) -> list[str]` contract.
CHUNKERS: dict[str, Callable[[int], Chunker]] = {
    "by-line": lambda target_size: by_line,
    "character-splitting": lambda target_size: partial(
        by_characters, target_size=target_size
    ),
}

CORPUS_STATS_SQL = """
    SELECT
        count(*)::int AS chunks,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY length(chunk_text))::int
            AS median_chars,
        max(length(chunk_text))::int AS max_chars,
        (count(*) FILTER (WHERE length(chunk_text) < 120))::int AS under_120
    FROM chunks
"""

# `strpos`, not `LIKE`: excerpts contain `%` and `_`, which LIKE reads as
# wildcards and would over-match.
CONTAINMENT_SQL = """
    SELECT count(*)::int FROM chunks WHERE strpos(chunk_text, %(excerpt)s) > 0
"""


def measure_recall(
    *,
    strategy: str = "character-splitting",
    target_size: int = TARGET_SIZE,
    captions: bool = True,
    ks: Sequence[int] = (TOP_K,),
) -> dict[str, Any]:
    """Re-ingest under this config, score recall at each k, append one row.

    Several `ks` score one ingest, so recall@3 and recall@10 cost the same as
    recall@3 alone. The gap between them is the only view onto a gold chunk
    ranking 8th rather than 30th.
    """
    chunker = CHUNKERS[strategy](target_size)

    with connection() as conn:
        ingest_corpus(conn, captions=captions, chunker=chunker)
        corpus = _corpus_stats(conn)
        containment = _containment(conn)

    recall = [score_recall(k) for k in ks]

    row: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": _git_provenance(),
        "config": {
            "strategy": strategy,
            "target_size": target_size,
            "captions": captions,
            "embedding_model": EMBEDDING_MODEL,
        },
        "corpus": corpus,
        "containment": containment,
        "recall": [
            {
                "k": result.k,
                "hits": result.hits,
                "spans": result.spans,
                "recall": round(result.recall, 3),
            }
            for result in recall
        ],
    }
    _append(row)
    return row


def _containment(conn: Connection) -> dict[str, Any]:
    """How many gold spans survive this chunking intact — the run's ceiling.

    A span straddling a chunk boundary is inside no chunk, so retrieval can
    never find it. Moves with the chunker (3 of 12 were lost at target_size
    400), so recall without it can't be read.
    """
    lost = []
    with conn.cursor() as cursor:
        for fixture in RECALL_FIXTURES:
            for span in fixture.spans:
                cursor.execute(CONTAINMENT_SQL, {"excerpt": span.excerpt})
                (found,) = cursor.fetchone()  # type: ignore[misc]
                if not found:
                    lost.append(f"{fixture.label}/{span.corpus}")

    spans = sum(len(fixture.spans) for fixture in RECALL_FIXTURES)
    return {"spans": spans, "containable": spans - len(lost), "lost": lost}


def _corpus_stats(conn: Connection) -> dict[str, int]:
    """Chunk shape read back from the table, not from `chunk_document`.

    What retrieval searches is what was persisted; a gap between the two should
    show up here.
    """
    with conn.cursor() as cursor:
        cursor.execute(CORPUS_STATS_SQL)
        chunks, median_chars, max_chars, under_120 = cursor.fetchone()  # type: ignore[misc]
    return {
        "chunks": chunks,
        "median_chars": median_chars,
        "max_chars": max_chars,
        "under_120": under_120,
    }


def _git_provenance() -> dict[str, Any]:
    """Which commit produced this number. `dirty` means the sha alone won't
    reproduce it."""
    return {
        "sha": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def _append(row: dict[str, Any]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a") as results:
        results.write(json.dumps(row) + "\n")


def main() -> None:
    args = _parse_args()
    row = measure_recall(
        strategy=args.strategy,
        target_size=args.target_size,
        captions=not args.no_captions,
        ks=args.k,
    )
    _display(row)


def _display(row: dict[str, Any]) -> None:
    config, corpus = row["config"], row["corpus"]
    print(
        f"\n{config['strategy']} | target {config['target_size']} "
        f"| captions {config['captions']}"
    )
    print(
        f"corpus: {corpus['chunks']} chunks | median {corpus['median_chars']} "
        f"| max {corpus['max_chars']} | under-120 {corpus['under_120']}"
    )
    containment = row["containment"]
    ceiling = f"{containment['containable']}/{containment['spans']}"
    lost = f"  lost: {', '.join(containment['lost'])}" if containment["lost"] else ""
    print(f"containable: {ceiling}{lost}")
    for entry in row["recall"]:
        print(f"recall@{entry['k']}: {entry['hits']}/{entry['spans']}")
    print(f"\nappended to {RESULTS_FILE.relative_to(Path.cwd())}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.measure",
        description="Ingest the corpus under one chunking config and score recall.",
    )
    parser.add_argument(
        "--strategy", choices=sorted(CHUNKERS), default="character-splitting"
    )
    parser.add_argument("--target-size", type=int, default=TARGET_SIZE)
    parser.add_argument("--no-captions", action="store_true")
    parser.add_argument(
        "--k", type=int, nargs="+", default=[TOP_K], help="one or more k to score at"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
