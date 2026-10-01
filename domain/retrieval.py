from dataclasses import dataclass

from domain.chunks import RankedChunk, RetrievedChunk
from domain.corpus import Corpus
from domain.queries import CorpusQuery


@dataclass(frozen=True)
class RetrievalResult:
    issued_queries: dict[Corpus, CorpusQuery]
    candidates: list[RetrievedChunk]
    ranked: list[RankedChunk]
