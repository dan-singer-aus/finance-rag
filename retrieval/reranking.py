from collections.abc import Callable
from dataclasses import dataclass

from domain.chunks import RankedChunk, RetrievedChunk

type PairScorer = Callable[[str, list[str]], list[tuple[int, float]]]


@dataclass(frozen=True)
class Reranker:
    score_pairs: PairScorer
    model_id: str


def rerank(
    query: str, chunks: list[RetrievedChunk], scorer: PairScorer
) -> list[RankedChunk]:
    texts = [chunk.chunk_text for chunk in chunks]
    scores = scorer(query, texts)
    ranked = sorted(scores, key=lambda pair: pair[1], reverse=True)
    return [RankedChunk(chunk=chunks[index], score=score) for index, score in ranked]
