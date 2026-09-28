"""Persist one measurement run: a `runs` row plus a `run_spans` row per gold span.

Lives in `evals/` rather than `db/` because a measurement result is an eval
concept, not a domain one — `SpanRank` is defined here, and `db/` must not
import from a layer above it.
"""

from typing import Any

from psycopg import Connection

from domain.corpus import Corpus
from domain.queries import CorpusQuery
from evals.score_retrieval import SpanRank

# Named parameters: eleven columns is where a positional tuple starts drifting out
# of step with the column list unseen.
INSERT_RUN_SQL = """
    INSERT INTO runs (
        strategy, target_size, overlap, captions, embedding_model,
        reranker, candidate_depth, rewriter,
        chunks, median_chars, max_chars, under_120
    )
    VALUES (
        %(strategy)s, %(target_size)s, %(overlap)s, %(captions)s,
        %(embedding_model)s,
        %(reranker)s, %(candidate_depth)s, %(rewriter)s,
        %(chunks)s, %(median_chars)s, %(max_chars)s, %(under_120)s
    )
    RETURNING id
"""

INSERT_RUN_SPAN_SQL = """
    INSERT INTO run_spans (
        run_id, label, corpus, excerpt, containable, merged_rank, corpus_rank
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

INSERT_RUN_QUERY_SQL = """
    INSERT INTO run_queries (run_id, label, corpus, query, reasoning)
    VALUES (%s, %s, %s, %s, %s)
"""


def insert_run(
    conn: Connection,
    *,
    config: dict[str, Any],
    corpus: dict[str, int],
    span_ranks: list[SpanRank],
    queries: dict[str, dict[Corpus, CorpusQuery]],
) -> int:
    """Insert the run, its spans and its queries, returning the new run's id."""
    params = {
        "strategy": config["strategy"],
        "target_size": config.get("target_size"),
        "overlap": config.get("overlap"),
        "captions": config["captions"],
        "embedding_model": config["embedding_model"],
        "reranker": config.get("reranker"),
        "candidate_depth": config.get("candidate_depth"),
        "rewriter": config.get("rewriter"),
        **corpus,
    }

    with conn.cursor() as cursor:
        cursor.execute(INSERT_RUN_SQL, params)
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Insertion produced no row")
        run_id = int(row[0])

        cursor.executemany(
            INSERT_RUN_SPAN_SQL,
            [
                (
                    run_id,
                    span_rank.label,
                    span_rank.span.corpus,
                    span_rank.span.excerpt,
                    span_rank.containable,
                    span_rank.merged_rank,
                    span_rank.corpus_rank,
                )
                for span_rank in span_ranks
            ],
        )

        cursor.executemany(
            INSERT_RUN_QUERY_SQL,
            [
                (
                    run_id,
                    label,
                    corpus_name,
                    corpus_query.query,
                    corpus_query.reasoning,
                )
                for label, by_corpus in queries.items()
                for corpus_name, corpus_query in by_corpus.items()
            ],
        )

    return run_id
