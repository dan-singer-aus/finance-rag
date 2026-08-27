from psycopg import Connection
from psycopg.rows import class_row

from domain.chunks import RetrievedChunk

SEARCH_SQL = """
    SELECT
        chunks.chunk_text,
        chunks.chunk_index,
        chunks.source_id,
        sources.title,
        sources.doc_type,
        sources.fiscal_year,
        sources.source_url,
        sources.corpus,
        sources.company,
        sources.ticker,
        sources.section,
        sources.period_end,
        1 - (chunks.embedding <=> %(embedding)s::vector) AS score
    FROM chunks
    JOIN sources ON chunks.source_id = sources.id
    ORDER BY chunks.embedding <=> %(embedding)s::vector
    LIMIT %(k)s
"""

def search(conn: Connection, embedding: list[float], k: int) -> list[RetrievedChunk]:
    with conn.cursor(row_factory=class_row(RetrievedChunk)) as cursor:
        cursor.execute(SEARCH_SQL, {"embedding": embedding, "k": k})
        return cursor.fetchall()






