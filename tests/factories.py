"""Builders for the record types the pure functions take.

`RetrievedChunk` has thirteen fields and every test needs one or two of them.
Naming the defaults here keeps each test showing only what it is actually
about — a test that says `chunk(score=0.9, corpus="filings")` reads as the case
it covers, where a full constructor call reads as noise.
"""

from datetime import date

from domain.chunks import RetrievedChunk
from domain.corpus import Corpus


def chunk(
    *,
    score: float = 0.5,
    corpus: Corpus = "filings",
    chunk_text: str = "some text",
    chunk_index: int = 0,
    source_id: int = 1,
) -> RetrievedChunk:
    """A RetrievedChunk with only the fields a test cares about."""
    return RetrievedChunk(
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        source_id=source_id,
        title="A Source",
        doc_type="10-K",
        fiscal_year=2025,
        source_url=None,
        corpus=corpus,
        company="VISA INC.",
        ticker="V",
        section="item-7-mda",
        period_end=date(2025, 9, 30),
        score=score,
    )
