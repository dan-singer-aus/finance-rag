import textwrap
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from psycopg import Connection

from db.connection import connection
from domain.chunks import RankedChunk, RetrievedChunk
from domain.corpus import CORPORA
from evals.recall_fixtures import RECALL_FIXTURES, GoldSpan, RecallFixture
from retrieval.pipeline import retrieve
from retrieval.reranking import PairScorer, rerank

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
class FixtureResult:
    fixture: RecallFixture
    depth: int
    span_ranks: list[SpanRank]
    ranked: list[RankedChunk]


@dataclass(frozen=True)
class RecallResult:
    k: int
    hits: int
    spans: int

    @property
    def recall(self) -> float:
        return self.hits / self.spans


def main(k: int = TOP_K, scorer: PairScorer | None = None) -> list[SpanRank]:
    """Score recall@k over every fixture, print the report, return the numbers."""
    span_ranks: list[SpanRank] = []
    for result in score_fixtures(scorer):
        _display_result(result)
        span_ranks += result.span_ranks

    merged = [span_rank.merged_rank for span_rank in span_ranks]
    recall = recall_at(merged, k)
    print(f"\nrecall@{k}: {recall.hits}/{recall.spans} ({recall.recall:.0%})")
    print(f"MRR: {mrr(merged):.3f}")
    return span_ranks


def score_fixtures(
    scorer: PairScorer | None = None,
    depths: Sequence[int] = (RESULTS_WINDOW,),
) -> Iterator[FixtureResult]:
    """Retrieve, rank and score every fixture, once per candidate depth."""
    # One retrieve and one rerank per fixture serve every depth: the scorer is
    # pointwise, so a shallower depth is a filter over the same scored list.
    with connection() as conn:
        for fixture in RECALL_FIXTURES:
            chunks = retrieve(conn, fixture.query, k=max(depths))
            cosine_ranks = _cosine_ranks(chunks)
            ranked = (
                rerank(fixture.query, chunks, scorer)
                if scorer is not None
                else _by_cosine(chunks)
            )
            for depth in depths:
                at_depth = _within_depth(ranked, cosine_ranks, depth)
                yield FixtureResult(
                    fixture=fixture,
                    depth=depth,
                    span_ranks=[
                        _rank_span(conn, fixture, span, at_depth)
                        for span in fixture.spans
                    ],
                    ranked=at_depth,
                )


def _by_cosine(chunks: list[RetrievedChunk]) -> list[RankedChunk]:
    """Pair each chunk with its cosine score, which is what ordered this list."""
    return [RankedChunk(chunk=chunk, score=chunk.score) for chunk in chunks]


def _cosine_ranks(chunks: list[RetrievedChunk]) -> dict[tuple[int, int], int]:
    """Each chunk's 1-based position within its own corpus, before reranking."""
    ranks: dict[tuple[int, int], int] = {}
    for corpus in CORPORA:
        in_corpus = [chunk for chunk in chunks if chunk.corpus == corpus]
        for rank, chunk in enumerate(in_corpus, start=1):
            ranks[(chunk.source_id, chunk.chunk_index)] = rank
    return ranks


def _within_depth(
    ranked: list[RankedChunk], cosine_ranks: dict[tuple[int, int], int], depth: int
) -> list[RankedChunk]:
    """The reranked list, keeping only candidates cosine ranked within depth."""
    return [
        item
        for item in ranked
        if cosine_ranks[(item.chunk.source_id, item.chunk.chunk_index)] <= depth
    ]


def _rank_of(span: GoldSpan, ranked: list[RankedChunk]) -> int | None:
    """1-based position of the first chunk containing the span's excerpt."""
    return next(
        (
            rank
            for rank, item in enumerate(ranked, start=1)
            if span.excerpt in item.chunk.chunk_text
        ),
        None,
    )


def _rank_span(
    conn: Connection,
    fixture: RecallFixture,
    span: GoldSpan,
    ranked: list[RankedChunk],
) -> SpanRank:
    same_corpus = [item for item in ranked if item.chunk.corpus == span.corpus]
    return SpanRank(
        label=fixture.label,
        span=span,
        containable=_is_containable(conn, span),
        merged_rank=_rank_of(span, ranked),
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


def _display_result(result: FixtureResult) -> None:
    fixture = result.fixture
    ranked = result.ranked
    fixture_ranks = result.span_ranks
    query = textwrap.shorten(fixture.query, width=72, placeholder="…")
    print(f"{fixture.label:<4} {query}")

    for span_rank in fixture_ranks:
        print(
            f"     {span_rank.span.corpus:<8}"
            f"  merged {_rank(span_rank.merged_rank):>4}"
            f"  corpus {_rank(span_rank.corpus_rank):>4}"
        )
        if span_rank.merged_rank is None:
            _display_miss(span_rank, ranked)


def _display_miss(span_rank: SpanRank, ranked: list[RankedChunk]) -> None:
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
    for item in ranked[:DISPLAY_CHUNKS]:
        print(f"          {round(item.score, 3)}  {item.chunk.provenance}")

    print()


def _rank(rank: int | None) -> str:
    return "–" if rank is None else str(rank)
