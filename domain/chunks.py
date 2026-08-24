from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str

@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]


