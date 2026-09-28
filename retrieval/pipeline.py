from psycopg import Connection

from db.search import search
from domain.chunks import RetrievedChunk
from domain.corpus import CORPORA, Corpus
from domain.queries import RewrittenQueries
from embedding import embed

DEFAULT_K = 5
RRF_K = 60


def retrieve(
    conn: Connection,
    query: str,
    k: int = DEFAULT_K,
    rewritten: RewrittenQueries | None = None,
) -> list[RetrievedChunk]:
    # Takes the rewritten result, not a callable that computes it -- a caller
    # that also reranks needs this same result to build per-chunk queries, and
    # calling the rewriter a second time there would both waste the call and
    # risk a different decomposition on each of two non-deterministic calls.
    # Computed once, up front, only in the no-rewriter case -- so the baseline
    # arm keeps embedding the query exactly once, byte-identical to before.
    # Embedding this same text twice is not free: text-embedding-3-small is
    # not bitwise deterministic between calls, so calling it once per corpus
    # here would inject noise into an arm that previously had none.
    shared_embedding = embed([query])[0] if rewritten is None else None
    scored = []

    for corpus in CORPORA:
        corpus_query = _query_for(corpus, query, rewritten)
        if corpus_query is None:
            continue  # this corpus was ruled out (or a full refusal)

        embedding = (
            shared_embedding
            if shared_embedding is not None
            else embed([corpus_query])[0]
        )
        corpus_results = search(conn, embedding, corpus, k)
        for rank, chunk in enumerate(corpus_results, start=1):
            scored.append((chunk, _rrf_score(rank)))

    return [chunk for chunk, _ in sorted(scored, key=lambda x: x[1], reverse=True)]


def _query_for(
    corpus: Corpus, query: str, rewritten: RewrittenQueries | None
) -> str | None:
    """Text to embed for one corpus, or None if the corpus was ruled out."""
    if rewritten is None:
        return query
    return rewritten.for_corpus(corpus).query


def _rrf_score(rank: int) -> float:
    return 1 / (rank + RRF_K)
