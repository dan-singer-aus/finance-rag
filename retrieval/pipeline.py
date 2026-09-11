from psycopg import Connection

from db.search import search
from domain.chunks import RetrievedChunk
from domain.corpus import CORPORA
from embedding import embed

DEFAULT_K = 5
RRF_K = 60


def retrieve(conn: Connection, query: str, k: int = DEFAULT_K) -> list[RetrievedChunk]:
    query_embedding = embed([query])[0]
    scored = []

    for corpus in CORPORA:
        corpus_results = search(conn, query_embedding, corpus, k)
        for rank, chunk in enumerate(corpus_results, start=1):
            scored.append((chunk, _rrf_score(rank)))

    return [chunk for chunk, _ in sorted(scored, key=lambda x: x[1], reverse=True)]


def _rrf_score(rank: int) -> float:
    return 1 / (rank + RRF_K)
