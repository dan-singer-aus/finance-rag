from dataclasses import dataclass
from datetime import date

from domain.corpus import Corpus


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str

@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_text: str
    chunk_index: int
    source_id: int
    title: str
    doc_type: str
    fiscal_year: int | None
    source_url: str | None
    corpus: Corpus
    company: str | None
    ticker: str | None
    section: str | None
    period_end: date | None
    score: float

