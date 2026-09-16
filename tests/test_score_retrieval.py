from evals.score_retrieval import mrr, recall_at


def test_a_span_at_exactly_k_is_a_hit() -> None:
    assert recall_at([3], k=3).hits == 1


def test_a_span_past_k_is_not() -> None:
    assert recall_at([4], k=3).hits == 0


def test_an_unranked_span_never_counts_as_a_hit() -> None:
    assert recall_at([None], k=100).hits == 0


def test_misses_stay_in_the_denominator() -> None:
    """recall is hits over EVERY span, not over the ones that ranked."""
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
