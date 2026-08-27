

from psycopg import Connection

from db.search import search
from domain.chunks import RetrievedChunk
from embedding import embed

DEFAULT_K = 5

def retrieve(conn: Connection, query: str, k: int = DEFAULT_K) -> list[RetrievedChunk]:
    query_embedding = embed([query])[0]
    return search(conn=conn, embedding=query_embedding, k=k)
