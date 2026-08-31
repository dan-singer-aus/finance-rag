

from psycopg import Connection

from db.search import search
from domain.chunks import RetrievedChunk
from domain.corpus import CORPORA
from embedding import embed

DEFAULT_K = 5

def retrieve(conn: Connection, query: str, k: int = DEFAULT_K) -> list[RetrievedChunk]:
    query_embedding = embed([query])[0]
    results =[]

    for corpus in CORPORA:
        results += search(conn, query_embedding, corpus, k)
    return sorted(results, key=lambda chunk: chunk.score, reverse=True)
