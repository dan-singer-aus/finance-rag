from dataclasses import dataclass

from psycopg import Connection

from domain.chunks import RankedChunk, RetrievedChunk
from domain.corpus import Corpus
from domain.queries import CorpusQuery, issued
from domain.retrieval import RetrievalResult
from retrieval.candidates import DEFAULT_K, find_candidates
from retrieval.reranking import Reranker, rerank
from retrieval.rewriting import rewrite_query


@dataclass(frozen=True)
class RetrievalConfig:
    reranker: Reranker | None = None
    rewrite: bool = False
    k: int = DEFAULT_K


def retrieve(
    conn: Connection, question: str, config: RetrievalConfig
) -> RetrievalResult:
    # Computed once and reused for both retrieval and reranking -- rewrite_query
    # is a non-deterministic model call, so calling it twice could decompose
    # the same question two different ways and misalign the two steps.
    rewritten = rewrite_query(conn, question) if config.rewrite else None
    # Resolved once, here: what was actually sent per corpus. Reranking reads
    # this rather than re-deriving it from `rewritten`, which is how rerank()
    # once came to score against a query the search never used.
    issued_queries = issued(question, rewritten)
    candidates = find_candidates(conn, question, k=config.k, rewritten=rewritten)
    ranked = (
        rerank(
            _queries_for(candidates, issued_queries),
            candidates,
            config.reranker.score_pairs,
        )
        if config.reranker is not None
        else _by_cosine(candidates)
    )
    return RetrievalResult(
        issued_queries=issued_queries, candidates=candidates, ranked=ranked
    )


def _queries_for(
    chunks: list[RetrievedChunk], issued_queries: dict[Corpus, CorpusQuery]
) -> list[str]:
    """Each chunk's own corpus-specific query."""
    return [_corpus_query(issued_queries, chunk.corpus) for chunk in chunks]


def _corpus_query(issued_queries: dict[Corpus, CorpusQuery], corpus: Corpus) -> str:
    corpus_query = issued_queries[corpus].query
    if corpus_query is None:
        raise ValueError(
            f"find_candidates() returned a chunk from {corpus!r}, "
            "which the rewriter ruled out"
        )
    return corpus_query


def _by_cosine(chunks: list[RetrievedChunk]) -> list[RankedChunk]:
    """Pair each chunk with its cosine score, which is what ordered this list."""
    return [RankedChunk(chunk=chunk, score=chunk.score) for chunk in chunks]
