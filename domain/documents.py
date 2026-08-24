from dataclasses import dataclass
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class BaseDocument:
    """Fields every corpus document carries."""

    title: str
    doc_type: str
    fiscal_year: int
    source_url: str
    content: str

@dataclass(frozen=True)
class LetterDocument(BaseDocument):
    corpus: Literal["letters"]
    author: str

@dataclass(frozen=True)
class FilingDocument(BaseDocument):
    corpus: Literal["filings"]
    company: str
    ticker: str
    cik: int
    section: str
    period_end: date
    accession: str

type Document = LetterDocument | FilingDocument


