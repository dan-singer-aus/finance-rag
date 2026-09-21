import argparse
import textwrap

from db.connection import connection
from domain.chunks import RetrievedChunk
from retrieval.pipeline import retrieve
from retrieval.reranking import rerank


def main() -> None:
    args = _parse_args()

    with connection() as conn:
        results = retrieve(conn, args.query)

    ranked = _reranked(args.query, results) if args.rerank else _as_scored(results)

    for rank, (chunk, score) in enumerate(ranked, start=1):
        print(_format_result(rank, chunk, score))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m retrieval")
    parser.add_argument("query", type=str, help="the question to search for")
    parser.add_argument(
        "--rerank", action="store_true", help="reorder with the cross-encoder"
    )
    return parser.parse_args()


def _as_scored(chunks: list[RetrievedChunk]) -> list[tuple[RetrievedChunk, float]]:
    """Cosine ordered this list, so cosine is the score that explains it."""
    return [(chunk, chunk.score) for chunk in chunks]


def _reranked(
    query: str, chunks: list[RetrievedChunk]
) -> list[tuple[RetrievedChunk, float]]:
    # sentence-transformers is in the non-default `rerank` group, so importing
    # it at module level breaks the plain query path where that isn't installed.
    from retrieval.cross_encoder import reranker  # noqa: PLC0415

    scorer = reranker().score_pairs
    return [(item.chunk, item.score) for item in rerank(query, chunks, scorer)]


def _format_result(rank: int, chunk: RetrievedChunk, score: float) -> str:
    wrapped_text = textwrap.fill(chunk.chunk_text, width=100)
    return f"{rank} {round(score, 3)}\n{chunk.provenance}\n{wrapped_text}"


if __name__ == "__main__":
    main()
