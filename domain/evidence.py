from dataclasses import dataclass
from typing import Literal

from domain.chunks import RetrievedChunk

type SupportStatus = Literal["supported", "weak", "unsupported"]

@dataclass(frozen=True)
class ClaimSupport:
    claim: str
    status: SupportStatus
    chunks: list[RetrievedChunk]
