from dataclasses import dataclass
from typing import Literal

from domain.chunks import RetrievedChunk

type Entailment = Literal["entailed", "contradicted", "not_stated", "unverifiable"]


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    cited_chunks: list[RetrievedChunk]
    entailment: Entailment
    reason: str