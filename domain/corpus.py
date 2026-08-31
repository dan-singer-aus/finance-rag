from typing import Literal

type Corpus = Literal["letters", "filings"]

CORPORA: tuple[Corpus, ...] = ("filings", "letters")
