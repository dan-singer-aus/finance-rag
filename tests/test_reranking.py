from retrieval.reranking import PairScorer, rerank
from tests.factories import chunk


def scorer_returning(scores: list[float]) -> PairScorer:
    """A scorer that ignores its input and returns fixed scores, in order."""

    def scorer(pairs: list[tuple[str, str]]) -> list[float]:
        return scores

    return scorer


def test_each_chunk_gets_the_score_for_its_own_position() -> None:
    chunks = [chunk(chunk_text="a"), chunk(chunk_text="b"), chunk(chunk_text="c")]
    scorer = scorer_returning([0.1, 0.9, 0.5])

    result = rerank(["q", "q", "q"], chunks, scorer)

    assert [item.chunk.chunk_text for item in result] == ["b", "c", "a"]
    assert [item.score for item in result] == [0.9, 0.5, 0.1]


def test_the_scorer_receives_each_chunk_paired_with_its_own_query() -> None:
    seen: dict[str, object] = {}

    def scorer(pairs: list[tuple[str, str]]) -> list[float]:
        seen["pairs"] = pairs
        return [1.0, 0.5]

    rerank(
        ["filings question", "letters question"],
        [chunk(chunk_text="a"), chunk(chunk_text="b")],
        scorer,
    )

    assert seen["pairs"] == [
        ("filings question", "a"),
        ("letters question", "b"),
    ]


def test_no_chunks_is_no_results() -> None:
    assert rerank([], [], scorer_returning([])) == []
