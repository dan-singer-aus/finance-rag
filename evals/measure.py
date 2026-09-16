"""Ingest the corpus under one chunking config, score it, append a result row.

    uv run python -m evals.measure fixed --target-size 400 --overlap 40
    uv run python -m evals.measure character-splitting --target-size 800 --k 3 10
    uv run python -m evals.measure by-line --no-captions

Rows go to the `runs` and `run_spans` tables. Observations only — recall@k and
MRR are derived from `run_spans` by query, not stored.

Re-ingests every time (~$0.008, a minute or two) so the recorded config is the
one that built the corpus rather than a description of it.

Not a `SUITES` entry in `evals/__main__.py`: a suite reads the system, this
replaces every chunk in it.

"""

import argparse
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

from psycopg import Connection

from db.connection import connection
from embedding import EMBEDDING_MODEL
from evals.score_retrieval import TOP_K, SpanRank, mrr, recall_at
from evals.score_retrieval import main as score_recall
from evals.store import insert_run
from ingest.chunking import TARGET_SIZE, by_characters, by_line, by_window
from ingest.pipeline import ingest_corpus

CHUNKERS: dict[str, Callable[..., list[str]]] = {
    "by-line": by_line,
    "character-splitting": by_characters,
    "fixed": by_window,
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


def measure_recall(
    *,
    strategy: str,
    knobs: dict[str, int],
    captions: bool = True,
    ks: Sequence[int] = (TOP_K,),
) -> dict[str, Any]:
    chunker = partial(CHUNKERS[strategy], **knobs)

    with connection() as conn:
        ingest_corpus(conn, captions=captions, chunker=chunker)
        corpus = _corpus_stats(conn)

    span_ranks = score_recall()
    merged = [span_rank.merged_rank for span_rank in span_ranks]
    recall = [recall_at(merged, k) for k in ks]

    config: dict[str, Any] = {
        "strategy": strategy,
        **knobs,
        "captions": captions,
        "embedding_model": EMBEDDING_MODEL,
    }
    with connection() as conn:
        run_id = insert_run(conn, config=config, corpus=corpus, span_ranks=span_ranks)
        conn.commit()

    row: dict[str, Any] = {
        "run_id": run_id,
        "config": config,
        "corpus": corpus,
        "containment": _containment(span_ranks),
        "recall": [
            {
                "k": result.k,
                "hits": result.hits,
                "spans": result.spans,
                "recall": round(result.recall, 3),
            }
            for result in recall
        ],
        "mrr": round(mrr(merged), 3),
        "ranks": [
            {
                "label": span_rank.label,
                "corpus": span_rank.span.corpus,
                "merged_rank": span_rank.merged_rank,
                "corpus_rank": span_rank.corpus_rank,
            }
            for span_rank in span_ranks
        ],
    }
    return row


def _containment(span_ranks: list[SpanRank]) -> dict[str, Any]:
    """How many gold spans survive this chunking intact — the run's ceiling.

    A span straddling a chunk boundary is inside no chunk, so retrieval can
    never find it. Moves with the chunker (3 of 12 were lost at target_size
    400), so recall without it can't be read.
    """
    lost = [
        f"{span_rank.label}/{span_rank.span.corpus}"
        for span_rank in span_ranks
        if not span_rank.containable
    ]
    return {
        "spans": len(span_ranks),
        "containable": len(span_ranks) - len(lost),
        "lost": lost,
    }


def _corpus_stats(conn: Connection) -> dict[str, int]:
    with conn.cursor() as cursor:
        cursor.execute(CORPUS_STATS_SQL)
        chunks, median_chars, max_chars, under_120 = cursor.fetchone()  # type: ignore[misc]
    return {
        "chunks": chunks,
        "median_chars": median_chars,
        "max_chars": max_chars,
        "under_120": under_120,
    }


def main() -> None:
    args = vars(_parse_args())
    strategy = args.pop("strategy")
    captions = not args.pop("no_captions")
    ks = args.pop("k")
    row = measure_recall(strategy=strategy, knobs=args, captions=captions, ks=ks)
    _display(row)


def _display(row: dict[str, Any]) -> None:
    config, corpus = row["config"], row["corpus"]
    print("\n" + " | ".join(f"{key} {value}" for key, value in config.items()))

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
    print(f"MRR: {row['mrr']:.3f}")
    print(f"\nrun {row['run_id']}")


def _parse_args() -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-captions", action="store_true")
    common.add_argument(
        "--k", type=int, nargs="+", default=[TOP_K], help="one or more k to score at"
    )

    parser = argparse.ArgumentParser(
        prog="python -m evals.measure",
        description="Ingest the corpus under one chunking config and score recall.",
    )
    strategies = parser.add_subparsers(dest="strategy", required=True)

    strategies.add_parser("by-line", parents=[common])

    characters = strategies.add_parser("character-splitting", parents=[common])
    characters.add_argument("--target-size", type=int, default=TARGET_SIZE)

    fixed = strategies.add_parser("fixed", parents=[common])
    fixed.add_argument("--target-size", type=int, default=TARGET_SIZE)
    fixed.add_argument("--overlap", type=int, help="default: 10%% of --target-size")

    args = parser.parse_args()
    if args.strategy == "fixed" and args.overlap is None:
        args.overlap = args.target_size // 10
    return args


if __name__ == "__main__":
    main()
