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

    @property
    def provenance(self) -> str:
        """Where this chunk came from, in one line.

        Derived entirely from the record's own fields — no I/O, no policy — and
        every consumer wants the same string, so it lives here rather than being
        re-forked at each call site. It was written out three times (the
        retrieval CLI, the grounding scoreboard, the generator's evidence
        block) before that stopped being a coincidence.

        Note the generator's prompt documents this exact format, so changing it
        changes what the model reads: `prompts/answer.yml` has to move with it.
        """
        if self.corpus == "letters":
            return self.title
        return f"{self.company} FY{self.fiscal_year} {self.doc_type} {self.section}"

