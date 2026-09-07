from dataclasses import dataclass
from typing import Literal

from domain.chunks import RetrievedChunk

type Entailment = Literal["entailed", "contradicted", "not_stated", "unverifiable"]

@dataclass(frozen=True)
class Claim:
    text: str
    citations: list[int]

@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    cited_chunks: list[RetrievedChunk]
    entailment: Entailment
    reason: str

@dataclass(frozen=True)
class LocatedClaim:
    claim: Claim
    cited_chunks: list[RetrievedChunk]
    unresolved_citations: list[int]
    