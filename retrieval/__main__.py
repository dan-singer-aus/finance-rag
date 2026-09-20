import sys
import textwrap

from db.connection import connection
from domain.chunks import RetrievedChunk
from retrieval.cross_encoder import score_pairs
from retrieval.pipeline import retrieve
from retrieval.reranking import rerank


def main() -> None:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if not args:
        sys.exit("Usage: python -m retrieval <query> [--rerank]")
    query = args[0]

    with connection() as conn:
        results = retrieve(conn, query)

    if "--rerank" in sys.argv:
        for rank, scored in enumerate(rerank(query, results, score_pairs), start=1):
            print(_format_result(rank, scored.chunk, scored.score))
    else:
        for rank, chunk in enumerate(results, start=1):
            print(_format_result(rank, chunk, chunk.score))


def _format_result(rank: int, chunk: RetrievedChunk, score: float) -> str:
    wrapped_text = textwrap.fill(chunk.chunk_text, width=100)
    return f"{rank} {round(score, 3)}\n{chunk.provenance}\n{wrapped_text}"


if __name__ == "__main__":
    main()
