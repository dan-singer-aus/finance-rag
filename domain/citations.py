from dataclasses import dataclass
from typing import Literal

from domain.chunks import RetrievedChunk

type Entailment = Literal["entailed", "contradicted", "not_stated", "subjective"]

@dataclass(frozen=True)
class Claim:
    text: str
    citations: list[int]

@dataclass(frozen=True)
class LocatedClaim:
    claim: Claim
    cited_chunks: list[RetrievedChunk]
    unresolved_citations: list[int]

@dataclass(frozen=True)
class ClaimVerdict:
    located: LocatedClaim
    entailment: Entailment
    reason: str
