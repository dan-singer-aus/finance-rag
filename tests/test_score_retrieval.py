from evals.recall_fixtures import GoldSpan
from evals.score_retrieval import SpanRank, graded_only, is_reachable, mrr, recall_at


def _gold_span(*, graded: bool) -> GoldSpan:
    return GoldSpan(
        corpus="filings",
        company=None,
        fiscal_year=2025,
        section=None,
        excerpt="x",
        why="test",
        graded=graded,
    )


def test_a_span_at_exactly_k_is_a_hit() -> None:
    assert recall_at([3], k=3).hits == 1


def test_a_span_past_k_is_not() -> None:
    assert recall_at([4], k=3).hits == 0


def test_an_unranked_span_never_counts_as_a_hit() -> None:
    assert recall_at([None], k=100).hits == 0


def test_misses_stay_in_the_denominator() -> None:
    """Recall is hits over EVERY span, not over the ones that ranked."""
    result = recall_at([1, None, 5], k=3)

    assert (result.hits, result.spans) == (1, 3)
    assert result.recall == 1 / 3


def test_raising_k_never_lowers_recall_and_cannot_pass_the_ranked_spans() -> None:
    ranks: list[int | None] = [1, None, 5, 12]
    hits = [recall_at(ranks, k).hits for k in range(1, 20)]

    assert hits == sorted(hits)
    assert max(hits) == 3


def test_mrr_is_one_when_every_span_ranks_first() -> None:
    assert mrr([1, 1, 1]) == 1.0


def test_mrr_is_the_reciprocal_of_the_rank() -> None:
    assert mrr([4]) == 0.25


def test_an_unranked_span_contributes_zero_but_still_counts() -> None:
    assert mrr([1, None]) == 0.5
    assert mrr([None, None]) == 0.0


def test_mrr_does_not_depend_on_the_order_of_the_spans() -> None:
    assert mrr([1, None, 4]) == mrr([4, 1, None])


def test_graded_only_excludes_ungraded_spans() -> None:
    graded = SpanRank(
        label="F2",
        span=_gold_span(graded=True),
        containable=True,
        merged_rank=1,
        corpus_rank=1,
    )
    ungraded = SpanRank(
        label="F1",
        span=_gold_span(graded=False),
        containable=True,
        merged_rank=1,
        corpus_rank=1,
    )

    assert graded_only([graded, ungraded]) == [graded]


def test_is_reachable_requires_both_containable_and_a_rank() -> None:
    span = _gold_span(graded=False)

    reachable = SpanRank(
        label="F1", span=span, containable=True, merged_rank=5, corpus_rank=5
    )
    not_containable = SpanRank(
        label="F1", span=span, containable=False, merged_rank=5, corpus_rank=5
    )
    not_ranked = SpanRank(
        label="F1", span=span, containable=True, merged_rank=None, corpus_rank=None
    )

    assert is_reachable(reachable)
    assert not is_reachable(not_containable)
    assert not is_reachable(not_ranked)
