import sys
import textwrap

from db.connection import connection
from domain.chunks import RetrievedChunk
from retrieval.pipeline import retrieve


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Usage: python -m retrieval <query>")

    query = sys.argv[1]
    with connection() as conn:
        results = retrieve(conn, query)

        for rank, chunk in enumerate(results, start=1):
            print(_format_result(rank, chunk))

def _format_result(rank: int, chunk: RetrievedChunk) -> str:
    wrapped_text = textwrap.fill(chunk.chunk_text, width=100)
    return f"{rank} {round(chunk.score, 3)}\n{chunk.provenance}\n{wrapped_text}"




if __name__ == "__main__":
    main()


