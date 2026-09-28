from functools import cache

from sentence_transformers import CrossEncoder

from retrieval.reranking import Reranker

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@cache
def _model() -> CrossEncoder:
    model: CrossEncoder = CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION)
    return model


def score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """Score each (query, text) pair independently, order preserved."""
    scores = _model().predict(pairs)
    return [float(score) for score in scores]


def reranker() -> Reranker:
    return Reranker(
        model_id=f"{RERANK_MODEL}@{RERANK_REVISION}", score_pairs=score_pairs
    )
