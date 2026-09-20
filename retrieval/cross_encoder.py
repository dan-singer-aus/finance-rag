from functools import cache

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


@cache
def _model() -> CrossEncoder:
    model: CrossEncoder = CrossEncoder(RERANK_MODEL)
    return model


def score_pairs(query: str, texts: list[str]) -> list[tuple[int, float]]:
    ranks = _model().rank(query, texts)
    return [(int(rank["corpus_id"]), float(rank["score"])) for rank in ranks]
