from functools import cache

from sentence_transformers import CrossEncoder

from retrieval.reranking import Reranker

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@cache
def _model() -> CrossEncoder:
    model: CrossEncoder = CrossEncoder(RERANK_MODEL, revision=RERANK_REVISION)
    return model


def score_pairs(query: str, texts: list[str]) -> list[tuple[int, float]]:
    ranks = _model().rank(query, texts)
    return [(int(rank["corpus_id"]), float(rank["score"])) for rank in ranks]


def reranker() -> Reranker:
    return Reranker(
        model_id=f"{RERANK_MODEL}@{RERANK_REVISION}", score_pairs=score_pairs
    )
