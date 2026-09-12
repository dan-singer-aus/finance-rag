"""Scores recall@3 against the fixtures in `evals/recall_fixtures.py`.

    uv run python -m evals retrieval

**Retrieves live**, like `score_evidence.py` and unlike `score_citations.py`:
there is nothing to freeze, because the retrieval pipeline itself is what's
being measured.

A `GoldSpan` is checked by plain text containment (`excerpt in chunk_text`),
not chunk id or offset — so the check is agnostic to how the corpus happens to
be chunked, and survives a re-chunk unchanged.

`retrieve()`'s `k` is a per-corpus quota, not a result count — it returns up
to `TOP_K` chunks *per corpus*, merged and score-sorted. The slice to the true
top `TOP_K` happens here, in the caller, not inside `retrieve()`: `retrieve()`
also serves generation, which needs the quota's guarantee (never silently
drop a corpus), so changing what `k` means there would break that. Same
principle `generate()` already uses for its own filings-then-letters
regrouping — a caller-specific view of the merged result belongs with the
caller.

Scored per **span**, not per fixture — a needs-both fixture (B1, B3)
contributes two independent checks to the denominator, not one collapsed
verdict. Whether "success" means both spans hit or either one is then a
reporting question, answered from the same two numbers, not baked into the
check.
"""

import textwrap
from dataclasses import dataclass

from db.connection import connection
from domain.chunks import RetrievedChunk
from evals.recall_fixtures import RECALL_FIXTURES, GoldSpan, RecallFixture
from retrieval.pipeline import retrieve

TOP_K = 3


@dataclass(frozen=True)
class RecallResult:
    """The three numbers a recall run produces. `recall` is a property, not a
    field — a stored copy could disagree with `hits / spans`."""

    k: int
    hits: int
    spans: int

    @property
    def recall(self) -> float:
        return self.hits / self.spans


def main(k: int = TOP_K) -> RecallResult:
    """Score recall@k over every fixture, print the report, return the numbers.

    `k` is a parameter so one ingested corpus can be scored at several windows
    without re-ingesting.
    """
    hits = 0
    spans = 0
    with connection() as conn:
        for fixture in RECALL_FIXTURES:
            results = retrieve(conn, fixture.query, k=k)[:k]
            fixture_hits = sum(_is_hit(span, results) for span in fixture.spans)

            _display_result(fixture, fixture_hits, results)

            hits += fixture_hits
            spans += len(fixture.spans)

    result = RecallResult(k=k, hits=hits, spans=spans)
    print(f"\nrecall@{k}: {hits}/{spans} ({result.recall:.0%})")
    return result


def _is_hit(span: GoldSpan, chunks: list[RetrievedChunk]) -> bool:
    return any(span.excerpt in chunk.chunk_text for chunk in chunks)


def _display_result(
    fixture: RecallFixture, fixture_hits: int, results: list[RetrievedChunk]
) -> None:
    """One row per fixture: label, query, and the fraction of its spans hit.

    A fixture with all spans hit stays a single line. Anything less prints
    what each span expected and what the top-k actually held, so a miss is
    diagnosable from the output rather than sending you back to a shell
    command.
    """
    total = len(fixture.spans)
    query = textwrap.shorten(fixture.query, width=64, placeholder="…")
    print(
        f"{_mark(fixture_hits == total)}  {fixture.label:<4} {query:<64}  {fixture_hits}/{total}"
    )

    if fixture_hits < total:
        _display_miss(fixture, results)


def _display_miss(fixture: RecallFixture, results: list[RetrievedChunk]) -> None:
    for span in fixture.spans:
        hit = _is_hit(span, results)
        print(f"      {_mark(hit)} {span.corpus}")
        if not hit:
            print(
                textwrap.fill(
                    span.why,
                    width=88,
                    initial_indent="        why:  ",
                    subsequent_indent="              ",
                )
            )
    print("      top-k returned:")
    for chunk in results:
        print(f"        {round(chunk.score, 3)}  {chunk.provenance}")
    print()


def _mark(ok: bool) -> str:
    return "✓" if ok else "✗"
