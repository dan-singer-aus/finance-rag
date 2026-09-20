from retrieval.reranking import PairScorer, rerank
from tests.factories import chunk


def scorer_returning(pairs: list[tuple[int, float]]) -> PairScorer:
    """A scorer that ignores its input and returns fixed (index, score) pairs."""

    def scorer(query: str, texts: list[str]) -> list[tuple[int, float]]:
        return pairs

    return scorer


def test_each_chunk_gets_the_score_for_its_own_index() -> None:
    chunks = [chunk(chunk_text="a"), chunk(chunk_text="b"), chunk(chunk_text="c")]
    scorer = scorer_returning([(0, 0.1), (2, 0.9), (1, 0.5)])

    result = rerank("q", chunks, scorer)

    assert [item.chunk.chunk_text for item in result] == ["c", "b", "a"]
    assert [item.score for item in result] == [0.9, 0.5, 0.1]


def test_the_scorer_receives_the_chunk_texts_in_order() -> None:
    seen: dict[str, object] = {}

    def scorer(query: str, texts: list[str]) -> list[tuple[int, float]]:
        seen["query"], seen["texts"] = query, texts
        return [(0, 1.0), (1, 0.5)]

    rerank(
        "what drove revenue?", [chunk(chunk_text="a"), chunk(chunk_text="b")], scorer
    )

    assert seen["texts"] == ["a", "b"]
    assert seen["query"] == "what drove revenue?"


def test_no_chunks_is_no_results() -> None:
    assert rerank("q", [], scorer_returning([])) == []
