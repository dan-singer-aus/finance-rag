"""Persist one measurement run: a `runs` row plus a `run_spans` row per gold span.

Lives in `evals/` rather than `db/` because a measurement result is an eval
concept, not a domain one — `SpanRank` is defined here, and `db/` must not
import from a layer above it.
"""

from typing import Any

from psycopg import Connection

from evals.score_retrieval import SpanRank

# Named parameters: eleven columns is where a positional tuple starts drifting out
# of step with the column list unseen.
INSERT_RUN_SQL = """
    INSERT INTO runs (
        strategy, target_size, overlap, captions, embedding_model,
        reranker, candidate_depth,
        chunks, median_chars, max_chars, under_120
    )
    VALUES (
        %(strategy)s, %(target_size)s, %(overlap)s, %(captions)s,
        %(embedding_model)s,
        %(reranker)s, %(candidate_depth)s,
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


def insert_run(
    conn: Connection,
    *,
    config: dict[str, Any],
    corpus: dict[str, int],
    span_ranks: list[SpanRank],
) -> int:
    """Insert the run and its spans, returning the new run's id."""
    params = {
        "strategy": config["strategy"],
        "target_size": config.get("target_size"),
        "overlap": config.get("overlap"),
        "captions": config["captions"],
        "embedding_model": config["embedding_model"],
        "reranker": config.get("reranker"),
        "candidate_depth": config.get("candidate_depth"),
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

    return run_id
