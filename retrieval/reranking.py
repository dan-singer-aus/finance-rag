from collections.abc import Callable
from dataclasses import dataclass

from domain.chunks import RankedChunk, RetrievedChunk

# One query per pair, not one shared query for the whole batch: a chunk
# retrieved via a rewritten per-corpus query must be scored against THAT
# query, not the original composite question it was never meant to fully
# answer on its own -- see the needs-both regression this fixes.
type PairScorer = Callable[[list[tuple[str, str]]], list[float]]


@dataclass(frozen=True)
class Reranker:
    score_pairs: PairScorer
    model_id: str


def load_reranker() -> Reranker:
    """Build the cross-encoder reranker, importing torch only when asked for."""
    # sentence-transformers lives in the non-default `rerank` group, so a
    # top-level import would cost every unreranked run a multi-second torch
    # import and break CI, which is deliberately denied the package.
    from retrieval.cross_encoder import reranker  # noqa: PLC0415

    return reranker()


def rerank(
    queries: list[str], chunks: list[RetrievedChunk], scorer: PairScorer
) -> list[RankedChunk]:
    """queries[i] is scored against chunks[i] -- same length, positionally paired."""
    pairs = list(zip(queries, (chunk.chunk_text for chunk in chunks), strict=True))
    scores = scorer(pairs)
    ranked = sorted(
        zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    return [RankedChunk(chunk=chunk, score=score) for chunk, score in ranked]
