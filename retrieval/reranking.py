from collections.abc import Callable

from domain.chunks import RerankedChunk, RetrievedChunk

type PairScorer = Callable[[str, list[str]], list[tuple[int, float]]]


def rerank(
    query: str, chunks: list[RetrievedChunk], scorer: PairScorer
) -> list[RerankedChunk]:
    texts = [chunk.chunk_text for chunk in chunks]
    scores = scorer(query, texts)
    ranked = sorted(scores, key=lambda pair: pair[1], reverse=True)
    return [RerankedChunk(chunk=chunks[index], score=score) for index, score in ranked]
