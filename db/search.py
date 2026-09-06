from psycopg import Connection
from psycopg.rows import class_row

from domain.chunks import RetrievedChunk
from domain.corpus import Corpus

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
    WHERE sources.corpus = %(corpus)s
    ORDER BY chunks.embedding <=> %(embedding)s::vector
    LIMIT %(k)s
"""

def search(conn: Connection, embedding: list[float], corpus: Corpus, k: int) -> list[RetrievedChunk]:
    with conn.cursor(row_factory=class_row(RetrievedChunk)) as cursor:
        cursor.execute(SEARCH_SQL, {"embedding": embedding, "k": k, "corpus": corpus})
        return cursor.fetchall()






