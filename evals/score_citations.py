"""Scores the citation checker against the fixture in `evals/answer_fixtures.py`.

    uv run python -m evals citations

Runs offline — no database, no embedding call — because `CONTEXT` is frozen.
The model calls are the splitter and one judge call per claim, per run.

⚠️ **Runs the pipeline RUNS times, because one green pass can be luck.** The
splitter is a model call and has already been seen to return a different number
of claims on identical input. A cell that flips between runs is a different
finding from a cell that is consistently wrong, and a single pass cannot tell
them apart.

⚠️ **A scoreboard, not a test suite** — `match` / `mismatch`, never
`PASS` / `FAIL`. A disagreement might mean the checker is wrong or might mean
the fixture is, and pass/fail language asserts the first reading. That is what
invites tuning the prompt until the board goes green.

Claims are paired with expectations **by position**, which holds only because
they happen to come out in order. A run returning the wrong number of claims is
therefore unusable rather than merely short — it is excluded from the
aggregation and reported as a count, since how often that happens is itself a
result.
"""

import textwrap
from collections import Counter

from domain.citations import ClaimVerdict
from evals.answer_fixtures import ANSWER, CONTEXT, EXPECTED_VERDICTS, ExpectedVerdict
from grounding.citations import judge_claims, locate_citations
from grounding.claims import split_claims

RUNS = 3


def main() -> None:
    runs = [_run_once() for _ in range(RUNS)]
    usable = [run for run in runs if len(run) == len(EXPECTED_VERDICTS)]

    _display_run_counts(runs, usable)
    if not usable:
        print("no usable runs — nothing to score")
        return

    matches = 0
    for verdicts, expected in zip(
        zip(*usable, strict=True), EXPECTED_VERDICTS, strict=True
    ):
        entailments = Counter(verdict.entailment for verdict in verdicts)
        citations = Counter(
            tuple(verdict.located.claim.citations) for verdict in verdicts
        )

        entailment_agreed = entailments[expected.entailment]
        citations_agreed = citations[tuple(expected.cites)]
        matched = entailment_agreed == len(usable) and citations_agreed == len(usable)

        _display_result(
            verdicts[0], expected, entailment_agreed, citations_agreed, len(usable)
        )
        if matched:
            matches += 1
        else:
            _display_disagreement(entailments, citations, expected)

    print(f"\n{matches}/{len(EXPECTED_VERDICTS)} matched across {len(usable)} run(s)")


def _run_once() -> list[ClaimVerdict]:
    claims = split_claims(ANSWER)
    located = locate_citations(claims, CONTEXT)
    return judge_claims(located, CONTEXT)


def _display_run_counts(
    runs: list[list[ClaimVerdict]], usable: list[list[ClaimVerdict]]
) -> None:
    """How many claims each run produced, and how many runs are scoreable.

    The splitter's denominator is itself a measurement — a checker that sees six
    claims twice out of three is not the same instrument as one that sees six
    every time — so this prints even when nothing is wrong.
    """
    counts = ", ".join(str(len(run)) for run in runs)
    print(f"claims per run: {counts}  (expected {len(EXPECTED_VERDICTS)})")
    if len(usable) != len(runs):
        print(
            f"⚠️  {len(runs) - len(usable)} run(s) excluded — wrong claim count, "
            f"so positions don't align"
        )
    print()


def _display_result(
    verdict: ClaimVerdict,
    expected: ExpectedVerdict,
    entailment_agreed: int,
    citations_agreed: int,
    total: int,
) -> None:
    """One row per expected verdict: the claim, then each axis as expected + agreement.

    Each axis carries its own marker, so a failure says WHICH half disagreed —
    a row can be wrong on the entailment, on the citations, or on both, and
    those are different findings. The claim text is taken from the first usable
    run purely as a label; the splitter may word it differently in the others.
    """
    claim = textwrap.shorten(verdict.located.claim.text, width=58, placeholder="…")
    overall = _mark(entailment_agreed == total and citations_agreed == total)
    entailment_mark = _mark(entailment_agreed == total)
    citations_mark = _mark(citations_agreed == total)
    print(
        f"{overall}  {claim:<58}  "
        f"{entailment_mark} {expected.entailment:<12} {entailment_agreed}/{total}  "
        f"{citations_mark} cites {_cites(expected.cites):>5} {citations_agreed}/{total}"
    )


def _display_disagreement(
    entailments: Counter[str],
    citations: Counter[tuple[int, ...]],
    expected: ExpectedVerdict,
) -> None:
    """What actually came back, and the reasoning the expectation rests on.

    `why` is printed because on a disagreement the two candidates are "the
    checker is wrong" and "the fixture is wrong" — without the recorded
    reasoning, the tempting fix is to edit the expectation until it matches.
    """
    citation_tally = {_cites(list(cites)): n for cites, n in citations.items()}
    print(f"      entailment: {_tally(entailments)}")
    print(f"      citations:  {_tally(citation_tally)}")
    print(
        textwrap.fill(
            expected.why,
            width=88,
            initial_indent="      why:  ",
            subsequent_indent="            ",
        )
    )
    print()


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"


def _cites(citations: list[int]) -> str:
    return ", ".join(str(n) for n in citations) if citations else "—"


def _tally(counts: dict[str, int]) -> str:
    """Distribution across runs, commonest first: `2× entailed, 1× not_stated`.

    Count first because a citation value is itself a number — `2 2` is
    unreadable in a way `2× 2` is not.
    """
    ordered = sorted(counts.items(), key=lambda item: -item[1])
    return ", ".join(f"{count}× {value}" for value, count in ordered)
