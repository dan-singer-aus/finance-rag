import textwrap
from dataclasses import dataclass

from psycopg import Connection

from db.connection import connection
from domain.chunks import RetrievedChunk
from evals.recall_fixtures import RECALL_FIXTURES, GoldSpan, RecallFixture
from retrieval.pipeline import retrieve

CONTAINMENT_SQL = """
    SELECT count(*)::int FROM chunks WHERE strpos(chunk_text, %(excerpt)s) > 0
"""


TOP_K = 3
RESULTS_WINDOW = 100
DISPLAY_CHUNKS = 5


@dataclass(frozen=True)
class SpanRank:
    label: str
    span: GoldSpan
    containable: bool
    merged_rank: int | None
    corpus_rank: int | None


@dataclass(frozen=True)
class RecallResult:
    k: int
    hits: int
    spans: int

    @property
    def recall(self) -> float:
        return self.hits / self.spans


def main(k: int = TOP_K) -> list[SpanRank]:
    """Score recall@k over every fixture, print the report, return the numbers."""
    span_ranks: list[SpanRank] = []
    with connection() as conn:
        for fixture in RECALL_FIXTURES:
            results = retrieve(conn, fixture.query, k=RESULTS_WINDOW)
            fixture_ranks = [
                _rank_span(conn, fixture, span, results) for span in fixture.spans
            ]
            _display_result(fixture, fixture_ranks, results)
            span_ranks += fixture_ranks

    merged = [span_rank.merged_rank for span_rank in span_ranks]
    recall = recall_at(merged, k)
    print(f"\nrecall@{k}: {recall.hits}/{recall.spans} ({recall.recall:.0%})")
    print(f"MRR: {mrr(merged):.3f}")
    return span_ranks


def _rank_of(span: GoldSpan, chunks: list[RetrievedChunk]) -> int | None:
    """1-based position of the first chunk containing the span's excerpt."""
    return next(
        (
            rank
            for rank, chunk in enumerate(chunks, start=1)
            if span.excerpt in chunk.chunk_text
        ),
        None,
    )


def _rank_span(
    conn: Connection,
    fixture: RecallFixture,
    span: GoldSpan,
    results: list[RetrievedChunk],
) -> SpanRank:
    same_corpus = [chunk for chunk in results if chunk.corpus == span.corpus]
    return SpanRank(
        label=fixture.label,
        span=span,
        containable=_is_containable(conn, span),
        merged_rank=_rank_of(span, results),
        corpus_rank=_rank_of(span, same_corpus),
    )


def _is_containable(conn: Connection, span: GoldSpan) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(CONTAINMENT_SQL, {"excerpt": span.excerpt})
        (found,) = cursor.fetchone()  # type: ignore[misc]
    return bool(found)


def recall_at(ranks: list[int | None], k: int) -> RecallResult:
    """How many spans landed in the top k."""
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return RecallResult(k=k, hits=hits, spans=len(ranks))


def mrr(ranks: list[int | None]) -> float:
    """Mean reciprocal rank — a span that never appeared contributes 0."""
    return sum(1 / rank for rank in ranks if rank is not None) / len(ranks)


def _display_result(
    fixture: RecallFixture,
    fixture_ranks: list[SpanRank],
    results: list[RetrievedChunk],
) -> None:
    query = textwrap.shorten(fixture.query, width=72, placeholder="…")
    print(f"{fixture.label:<4} {query}")

    for span_rank in fixture_ranks:
        print(
            f"     {span_rank.span.corpus:<8}"
            f"  merged {_rank(span_rank.merged_rank):>4}"
            f"  corpus {_rank(span_rank.corpus_rank):>4}"
        )
        if span_rank.merged_rank is None:
            _display_miss(span_rank, results)


def _display_miss(span_rank: SpanRank, results: list[RetrievedChunk]) -> None:
    if not span_rank.containable:
        print("        not containable — no chunk holds this excerpt whole")
        return

    print(
        textwrap.fill(
            span_rank.span.why,
            width=88,
            initial_indent="        why:  ",
            subsequent_indent="              ",
        )
    )
    print(f"        top {DISPLAY_CHUNKS} returned:")
    for chunk in results[:DISPLAY_CHUNKS]:
        print(f"          {round(chunk.score, 3)}  {chunk.provenance}")
    print()


def _rank(rank: int | None) -> str:
    return "–" if rank is None else str(rank)
