"""`_classify` — the evidence linker's support rule.

It is pure by design (chunks in, status out, no connection and no embedding
call) precisely so it can be exercised with hand-written chunk lists. These
tests pin the RULES, not the threshold values — those are provisional, and a
test asserting a specific number would have to be edited every time they are
calibrated, which turns the test into an echo of the code.
"""

from grounding.evidence import SUPPORT_THRESHOLDS, _classify
from tests.factories import chunk

FILINGS_PASS = SUPPORT_THRESHOLDS["filings"] + 0.05
FILINGS_FAIL = SUPPORT_THRESHOLDS["filings"] - 0.05
LETTERS_PASS = SUPPORT_THRESHOLDS["letters"] + 0.05
LETTERS_FAIL = SUPPORT_THRESHOLDS["letters"] - 0.05


def test_nothing_clearing_its_threshold_is_unsupported() -> None:
    chunks = [
        chunk(corpus="filings", score=FILINGS_FAIL),
        chunk(corpus="letters", score=LETTERS_FAIL),
    ]

    assert _classify(chunks) == "unsupported"


def test_a_qualifying_filing_is_supported() -> None:
    assert _classify([chunk(corpus="filings", score=FILINGS_PASS)]) == "supported"


def test_letters_alone_cap_at_weak() -> None:
    """A framework says what to look for; it cannot testify about a company.

    "Visa is capital-light" grounded solely in Buffett is a statement about
    Buffett, so letters-only support is capped rather than promoted.
    """
    assert _classify([chunk(corpus="letters", score=LETTERS_PASS)]) == "weak"


def test_a_qualifying_filing_alongside_letters_is_supported() -> None:
    chunks = [
        chunk(corpus="letters", score=LETTERS_PASS),
        chunk(corpus="filings", score=FILINGS_PASS),
    ]

    assert _classify(chunks) == "supported"


def test_a_failing_filing_does_not_promote_letters_only_support() -> None:
    """Only chunks that CLEAR their threshold count toward the corpus rule."""
    chunks = [
        chunk(corpus="filings", score=FILINGS_FAIL),
        chunk(corpus="letters", score=LETTERS_PASS),
    ]

    assert _classify(chunks) == "weak"


def test_the_thresholds_are_per_corpus_not_global() -> None:
    """A letters score that clears its own bar would fail the filings bar.

    This is the measured finding the two thresholds exist for: the filings
    scores are nearly flat because every filing chunk is about one of three
    companies, so a single global cut point reads the flatter column.
    """
    assert LETTERS_PASS < SUPPORT_THRESHOLDS["filings"]
    assert _classify([chunk(corpus="letters", score=LETTERS_PASS)]) == "weak"
    assert _classify([chunk(corpus="filings", score=LETTERS_PASS)]) == "unsupported"


def test_input_order_does_not_change_the_verdict() -> None:
    """`_classify` must not assume its input is sorted — it filters, it does not index."""
    qualifying = chunk(corpus="filings", score=FILINGS_PASS)
    failing = chunk(corpus="filings", score=FILINGS_FAIL)

    assert _classify([failing, qualifying]) == _classify([qualifying, failing]) == "supported"


def test_no_chunks_at_all_is_unsupported() -> None:
    assert _classify([]) == "unsupported"
