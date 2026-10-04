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
from dataclasses import dataclass
from functools import partial
from typing import Any

from psycopg import Connection

from db.connection import connection
from domain.corpus import Corpus
from domain.queries import CorpusQuery
from embedding import EMBEDDING_MODEL
from evals.score_retrieval import (
    DEFAULT_RETRIEVAL,
    RESULTS_WINDOW,
    TOP_K,
    SpanRank,
    graded_only,
    mrr,
    recall_at,
    score_fixtures,
)
from evals.store import insert_run
from ingest.chunking import TARGET_SIZE, by_characters, by_line, by_window
from ingest.pipeline import ingest_corpus
from retrieval.pipeline import RetrievalConfig
from retrieval.reranking import load_reranker
from retrieval.rewriting import REWRITE_MODEL

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


@dataclass(frozen=True)
class MeasurementRun:
    run_id: int
    config: dict[str, Any]
    corpus: dict[str, int]
    span_ranks: list[SpanRank]


@dataclass(frozen=True)
class ScoringConfig:
    retrieval: RetrievalConfig = DEFAULT_RETRIEVAL
    depths: Sequence[int] = (RESULTS_WINDOW,)
    repeats: int = 1


# A module-level singleton rather than a call in the default: the record is
# frozen, so sharing one instance is safe, and B008 bans the call regardless.
DEFAULT_SCORING = ScoringConfig()


def main() -> None:
    args = vars(_parse_args())
    strategy = args.pop("strategy")
    captions = not args.pop("no_captions")
    ks = args.pop("k")
    # Everything still in `args` becomes the chunker's knobs, so a flag that
    # isn't popped is passed to partial() and raises there instead of here.
    requested_depths = args.pop("depths")
    reranker = load_reranker() if args.pop("rerank") else None
    rewrite = args.pop("rewrite")
    repeats = args.pop("repeats")
    depths = (
        tuple(requested_depths) if reranker and requested_depths else (RESULTS_WINDOW,)
    )
    retrieval = RetrievalConfig(reranker=reranker, rewrite=rewrite, k=max(depths))
    runs = measure_recall(
        strategy=strategy,
        knobs=args,
        captions=captions,
        scoring=ScoringConfig(retrieval=retrieval, depths=depths, repeats=repeats),
    )
    for run in runs:
        _display(run, ks)


def measure_recall(
    *,
    strategy: str,
    knobs: dict[str, int],
    captions: bool = True,
    scoring: ScoringConfig = DEFAULT_SCORING,
) -> list[MeasurementRun]:
    """Ingest once, then score `repeats` times, recording a run per depth each."""
    chunker = partial(CHUNKERS[strategy], **knobs)
    rerank = scoring.retrieval.reranker
    # Deliberately OUTSIDE the repeat loop. The corpus is the variable being held
    # constant, so a repeat that re-ingested would vary the chunking as well as
    # the rewriter and no difference between rows could be attributed.
    with connection() as conn:
        ingest_corpus(conn, captions=captions, chunker=chunker)
        corpus = _corpus_stats(conn)

    # Every row shares this, across repeats as well as depths; only
    # candidate_depth differs.
    config: dict[str, Any] = {
        "strategy": strategy,
        **knobs,
        "captions": captions,
        "embedding_model": EMBEDDING_MODEL,
        "reranker": rerank.model_id if rerank else None,
        "rewriter": REWRITE_MODEL if scoring.retrieval.rewrite else None,
    }
    return [
        run
        for _ in range(scoring.repeats)
        for run in _score_and_record(
            config=config,
            corpus=corpus,
            scoring=scoring,
        )
    ]


def _score_and_record(
    *,
    config: dict[str, Any],
    corpus: dict[str, int],
    scoring: ScoringConfig,
) -> list[MeasurementRun]:
    """One scoring pass over every fixture: one run row per candidate depth.

    Re-running this against an unchanged corpus is what makes a repeat: with
    `rewrite` off it is deterministic and every row should agree, so a difference
    is the rewriter and nothing else.
    """
    ranks_by_depth: dict[int, list[SpanRank]] = {depth: [] for depth in scoring.depths}
    # Keyed by label, not appended to: a fixture yields one result per depth
    # carrying the same queries, and `run_queries` is unique per (run, fixture,
    # corpus), so accumulating them would insert the same row once per depth.
    queries_by_label: dict[str, dict[Corpus, CorpusQuery]] = {}
    rerank = scoring.retrieval.reranker
    for result in score_fixtures(scoring.retrieval, depths=scoring.depths):
        ranks_by_depth[result.depth] += result.span_ranks
        queries_by_label[result.fixture.label] = result.issued_queries

    return [
        _record(
            # Both or neither — the 004 CHECK enforces it.
            config=config | {"candidate_depth": depth if rerank else None},
            corpus=corpus,
            span_ranks=span_ranks,
            queries=queries_by_label,
        )
        for depth, span_ranks in ranks_by_depth.items()
    ]


def _record(
    *,
    config: dict[str, Any],
    corpus: dict[str, int],
    span_ranks: list[SpanRank],
    queries: dict[str, dict[Corpus, CorpusQuery]],
) -> MeasurementRun:
    """Write one `runs` row, its spans and its queries, and return what was written."""
    with connection() as conn:
        run_id = insert_run(
            conn,
            config=config,
            corpus=corpus,
            span_ranks=span_ranks,
            queries=queries,
        )
        conn.commit()

    return MeasurementRun(
        run_id=run_id,
        config=config,
        corpus=corpus,
        span_ranks=span_ranks,
    )


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


def _display(run: MeasurementRun, ks: Sequence[int]) -> None:
    config = run.config
    corpus = run.corpus
    print("\n" + " | ".join(f"{key} {value}" for key, value in config.items()))

    print(
        f"corpus: {corpus['chunks']} chunks | median {corpus['median_chars']} "
        f"| max {corpus['max_chars']} | under-120 {corpus['under_120']}"
    )
    containment = _containment(run.span_ranks)
    ranks = [span_rank.corpus_rank for span_rank in graded_only(run.span_ranks)]
    ceiling = f"{containment['containable']}/{containment['spans']}"
    lost = f"  lost: {', '.join(containment['lost'])}" if containment["lost"] else ""
    print(f"containable: {ceiling}{lost}")
    for k in ks:
        result = recall_at(ranks, k)
        print(f"recall@{k}: {result.hits}/{result.spans} ({result.recall:.0%})")
    print(f"MRR: {mrr(ranks):.3f}")
    if len(ranks) != len(run.span_ranks):
        print(f"  (excludes {len(run.span_ranks) - len(ranks)} ungraded span(s))")
    print(f"\nrun {run.run_id}")


def _parse_args() -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-captions", action="store_true")
    common.add_argument(
        "--k", type=int, nargs="+", default=[TOP_K], help="one or more k to score at"
    )
    common.add_argument(
        "--rerank",
        action="store_true",
        help="reorder the candidates with the cross-encoder before scoring",
    )
    common.add_argument(
        "--rewrite",
        action="store_true",
        help="decompose the query per-corpus before retrieval",
    )
    common.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "score the same corpus this many times, one run row per repeat per "
            "depth. Ingests once. Without --rewrite every repeat should agree, "
            "which is the control for the rewriter's own variance."
        ),
    )
    common.add_argument(
        "--depths",
        type=int,
        nargs="+",
        help=(
            f"candidate depths per corpus to score, one run row each "
            f"(default: {RESULTS_WINDOW}). Ignored without --rerank."
        ),
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
