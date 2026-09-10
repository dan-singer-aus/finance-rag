"""`locate_citations` — the deterministic half of the citation checker.

Only the pure layer is tested here. `split_claims` and `judge_claims` are model
calls: non-deterministic, and they cost money. They are measured by
`evals/score_citations.py`, which reports `match` / `mismatch` rather than
asserting, because a disagreement there might mean the fixture is wrong.
"""

from domain.citations import Claim
from grounding.citations import locate_citations
from tests.factories import chunk


def test_resolves_a_marker_to_the_chunk_at_that_position() -> None:
    """`[n]` is a 1-based position in the context, so [2] is the second chunk.

    The bug this guards: the first implementation matched on `chunk_index`,
    which is a chunk's ordinal WITHIN ITS SOURCE DOCUMENT. Both are small ints,
    so the wrong one returns confident, wrong chunks rather than failing.
    """
    context = [chunk(chunk_text="first"), chunk(chunk_text="second")]

    located = locate_citations([Claim(text="a claim", citations=[2])], context)

    assert [c.chunk_text for c in located[0].cited_chunks] == ["second"]
    assert located[0].unresolved_citations == []


def test_a_marker_past_the_end_is_unresolved_rather_than_an_error() -> None:
    """An invalid marker is the finding, not a bug — so it is recorded, not raised."""
    context = [chunk(), chunk()]

    located = locate_citations([Claim(text="a claim", citations=[14])], context)

    assert located[0].cited_chunks == []
    assert located[0].unresolved_citations == [14]


def test_marker_zero_is_unresolved_and_does_not_wrap_to_the_last_chunk() -> None:
    """The lower bound is load-bearing: `chunks[0 - 1]` is `chunks[-1]` in Python.

    Without `1 <= n`, a `[0]` marker silently returns the LAST chunk — a wrong
    answer rather than an error, which is the worse failure.
    """
    context = [chunk(chunk_text="first"), chunk(chunk_text="last")]

    located = locate_citations([Claim(text="a claim", citations=[0])], context)

    assert located[0].cited_chunks == []
    assert located[0].unresolved_citations == [0]


def test_valid_and_invalid_markers_on_one_claim_are_split() -> None:
    context = [chunk(chunk_text="first"), chunk(chunk_text="second")]

    located = locate_citations([Claim(text="a claim", citations=[2, 14, 0])], context)

    assert [c.chunk_text for c in located[0].cited_chunks] == ["second"]
    assert located[0].unresolved_citations == [14, 0]


def test_an_uncited_claim_resolves_to_nothing_without_complaint() -> None:
    """Uncited is a legitimate state, distinct from citing a source that does not exist."""
    located = locate_citations([Claim(text="a claim", citations=[])], [chunk()])

    assert located[0].cited_chunks == []
    assert located[0].unresolved_citations == []


def test_markers_keep_the_order_the_answer_wrote_them_in() -> None:
    """Order is a fact about the generator's output, so it is preserved, not sorted."""
    context = [
        chunk(chunk_text="first"),
        chunk(chunk_text="second"),
        chunk(chunk_text="third"),
    ]

    located = locate_citations([Claim(text="a claim", citations=[3, 1])], context)

    assert [c.chunk_text for c in located[0].cited_chunks] == ["third", "first"]


def test_every_claim_gets_a_result_in_order() -> None:
    """The scoreboard pairs claims to expectations by position, so nothing may be dropped."""
    context = [chunk(chunk_text="first"), chunk(chunk_text="second")]
    claims = [
        Claim(text="cites one", citations=[1]),
        Claim(text="cites nothing", citations=[]),
        Claim(text="cites two", citations=[2]),
    ]

    located = locate_citations(claims, context)

    assert [lc.claim.text for lc in located] == [
        "cites one",
        "cites nothing",
        "cites two",
    ]
