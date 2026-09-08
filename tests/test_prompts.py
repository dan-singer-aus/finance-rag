"""The prompt loader — substitution and its two guards.

Both behaviours here are load-bearing and neither is obvious from reading the
call site, which is why they are pinned: a prompt that renders wrongly reaches
the model as slightly odd text rather than as an error, and produces a plausible
answer built on the wrong input.
"""

import pytest

from prompts import Prompt, load


def _prompt(user: str) -> Prompt:
    return Prompt(name="test", system="irrelevant", user=user)


def test_substitutes_a_placeholder() -> None:
    assert _prompt("Q: <%question%>").render(question="why?") == "Q: why?"


def test_a_placeholder_used_twice_is_filled_twice() -> None:
    rendered = _prompt("<%name%> and <%name%>").render(name="Visa")

    assert rendered == "Visa and Visa"


def test_dollar_signs_in_the_value_survive_intact() -> None:
    """`string.Template` would treat `$29` as a substitution and mangle it.

    Financial prose is full of these, which is why substitution is plain
    `str.replace` — it assigns meaning to no character at all.
    """
    rendered = _prompt("<%evidence%>").render(evidence="Capex was $29.0 billion.")

    assert rendered == "Capex was $29.0 billion."


def test_braces_in_the_value_survive_intact() -> None:
    """`str.format` would raise or misread on `{...}` appearing in the corpus."""
    rendered = _prompt("<%evidence%>").render(evidence="A set {a, b} and a brace }.")

    assert rendered == "A set {a, b} and a brace }."


def test_an_unfilled_placeholder_raises_rather_than_reaching_the_model() -> None:
    """Otherwise a literal `<%letters%>` reaches the model and it answers anyway."""
    with pytest.raises(ValueError, match="letters"):
        _prompt("<%filings%> <%letters%>").render(filings="…")


def test_the_error_names_what_was_missing_and_what_was_given() -> None:
    with pytest.raises(ValueError) as caught:
        _prompt("<%a%> <%b%>").render(a="…")

    message = str(caught.value)
    assert "b" in message
    assert "a" in message


def test_an_unknown_field_does_not_silently_pass() -> None:
    """Passing `contxet=` leaves `<%context%>` unfilled, so it raises."""
    with pytest.raises(ValueError, match="context"):
        _prompt("<%context%>").render(contxet="…")


@pytest.mark.parametrize("name", ["answer", "answer-naive", "split", "judge"])
def test_every_shipped_prompt_loads(name: str) -> None:
    """A YAML typo is otherwise found at the moment of a paid call."""
    prompt = load(name)

    assert prompt.system.strip()
    assert prompt.user.strip()


def test_a_missing_prompt_names_the_ones_that_exist() -> None:
    with pytest.raises(FileNotFoundError, match="judge"):
        load("judgement")
