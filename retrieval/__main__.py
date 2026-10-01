import argparse
import textwrap

from db.connection import connection
from domain.chunks import RetrievedChunk
from retrieval.pipeline import RetrievalConfig, retrieve
from retrieval.reranking import load_reranker


def main() -> None:
    args = _parse_args()
    config = RetrievalConfig(
        reranker=load_reranker() if args.rerank else None, rewrite=args.rewrite
    )

    with connection() as conn:
        result = retrieve(conn, args.query, config)

    for rank, item in enumerate(result.ranked, start=1):
        print(_format_result(rank, item.chunk, item.score))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m retrieval")
    parser.add_argument("query", type=str, help="the question to search for")
    parser.add_argument(
        "--rerank", action="store_true", help="reorder with the cross-encoder"
    )
    parser.add_argument(
        "--rewrite", action="store_true", help="decompose the query per corpus"
    )
    return parser.parse_args()


def _format_result(rank: int, chunk: RetrievedChunk, score: float) -> str:
    wrapped_text = textwrap.fill(chunk.chunk_text, width=100)
    return f"{rank} {round(score, 3)}\n{chunk.provenance}\n{wrapped_text}"


if __name__ == "__main__":
    main()
