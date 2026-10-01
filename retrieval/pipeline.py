from dataclasses import dataclass

from retrieval.candidates import DEFAULT_K
from retrieval.reranking import Reranker


@dataclass(frozen=True)
class RetrievalConfig:
    reranker: Reranker | None = None
    rewrite: bool = False
    k: int = DEFAULT_K
