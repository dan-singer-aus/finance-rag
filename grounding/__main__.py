"""Runs the evidence linker over every claim in `evals/claim_fixtures.py`.

The instrument for Lab 3, the way `retrieval/__main__.py` is for Lab 2 — five
hand-checked claims on one screen, so a change to the threshold or the corpus
rule is visible immediately rather than one query at a time.

**This is a scoreboard, not a test suite, and the wording matters.** Two of the
five fixtures encode judgement calls rather than facts: the letters-only claim
is explicitly a placeholder for a decision being made now, and the pricing-power
one is deliberately borderline. So a disagreement might mean the classifier is
wrong — or it might mean the fixture is. Print `match` / `mismatch`, not
`PASS` / `FAIL`, because pass/fail language quietly asserts the first reading,
and the failure mode that invites is real: see red, nudge the threshold until it
goes green, call it progress. Nothing here can settle that question; L5's metric
can.

On a mismatch, print the fixture's `why` alongside the top chunk's score and
provenance. That pair is the whole diagnostic — why the answer was expected, and
what retrieval actually handed the classifier.

Run with: uv run python -m grounding
"""


import textwrap

from db.connection import connection
from domain.chunks import RetrievedChunk
from domain.corpus import CORPORA, Corpus
from domain.evidence import ClaimSupport
from evals.claim_fixtures import CLAIM_FIXTURES, ClaimFixture
from grounding.evidence import link_evidence


def main() -> None:
    matches = 0
    with connection() as conn:
        for fixture in CLAIM_FIXTURES:
            support = link_evidence(conn, fixture.claim)
            matched = support.status == fixture.expected_status
            _display_result(fixture, support, matched)

            if matched:
                matches += 1
            else:
                _display_why(fixture)

    _display_summary(matches, len(CLAIM_FIXTURES))


def _display_result(fixture: ClaimFixture, support: ClaimSupport, matched: bool) -> None:
    marker = "✓" if matched else "✗"
    claim = textwrap.shorten(fixture.claim, width=60, placeholder="…")
    print(f"{marker}  {claim:<60}  {fixture.expected_status:<12} | {support.status:<12}")
    for corpus in CORPORA:
        print(f"      {corpus}: {_format_best_chunk_in_corpus(support.chunks, corpus)}")

def _display_why(fixture: ClaimFixture) -> None:
    print(
        textwrap.fill(
            fixture.why,
            width=88,
            initial_indent="      why:  ",
            subsequent_indent="            ",
        )
    )
    print()


def _display_summary(matches: int, total: int) -> None:
    print(f"\n{matches}/{total} matched")


def _format_best_chunk_in_corpus(chunks: list[RetrievedChunk], corpus: Corpus) -> str:
    """The highest-scoring chunk retrieval returned, qualifying or not.

    On an `unsupported` verdict nothing cleared the threshold, so there is no
    qualifying chunk to show — and "how close did the best one get" is the
    number that says whether the threshold is wrong or the retrieval is.
    """

    in_corpus = [chunk for chunk in chunks if chunk.corpus == corpus]
    if not in_corpus:
        return f"no chunks for {corpus} returned"

    top = in_corpus[0]  # search() returns them score-ordered
    return f"{top.score:.3f}  {top.provenance}  #{top.chunk_index}"


if __name__ == "__main__":
    main()