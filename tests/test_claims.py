"""`_citations_in` — marker extraction, the code half of the splitter.

The model reports each claim's source sentence verbatim and deliberately does
NOT attribute markers; code pulls them out with a regex. That split exists
because two runs once disagreed on whether a comma-joined sentence held one
claim or two, which moved a specific figure between cited and uncited. Judgement
to the model, mechanics to code — and mechanics is the part that can be pinned
down here.
"""

from grounding.claims import _citations_in


def test_extracts_a_single_marker() -> None:
    assert _citations_in("Net revenue increased 11% [2].") == [2]


def test_extracts_adjacent_markers_in_order() -> None:
    """A claim can legitimately rest on more than one chunk: `[2][5]`."""
    assert _citations_in("Growth was broad based [2][5].") == [2, 5]


def test_extracts_markers_separated_by_prose() -> None:
    assert _citations_in("Revenue rose [2] while costs fell [7].") == [2, 7]


def test_a_sentence_with_no_markers_yields_nothing() -> None:
    assert _citations_in("Management expects growth to continue.") == []


def test_multi_digit_markers_are_read_whole() -> None:
    """`[14]` must not read as 1 and 4 — it is the fabricated-source case."""
    assert _citations_in("A claim [14].") == [14]


def test_currency_and_brackets_in_financial_prose_are_not_markers() -> None:
    """Filing prose is full of `$29.0 billion` and bracketed asides.

    Only `[digits]` counts, which is why this is a regex over the sentence and
    not a looser scan.
    """
    assert _citations_in("Capex was $29.0 billion (up 12%) [3].") == [3]
    assert _citations_in("See Item 1A [Risk Factors] for detail.") == []
