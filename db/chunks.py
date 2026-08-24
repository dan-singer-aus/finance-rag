from psycopg import Connection

from domain.chunks import EmbeddedChunk

DELETE_CHUNKS_SQL = "DELETE FROM chunks WHERE source_id = %s"
INSERT_CHUNKS_SQL = """
    INSERT INTO chunks (source_id, chunk_index, chunk_text, embedding)
    VALUES (%s, %s, %s, %s)
"""

def delete_for_source(conn: Connection, source_id: int) -> None:
    with conn.cursor() as cursor:
        cursor.execute(DELETE_CHUNKS_SQL, (source_id,))

def insert_many(conn: Connection, source_id: int, chunks: list[EmbeddedChunk]) -> None:
    rows = [
        (source_id, embedded.chunk.index, embedded.chunk.text, embedded.embedding)
        for embedded in chunks
    ]
    with conn.cursor() as cursor:
        cursor.executemany(INSERT_CHUNKS_SQL, rows)

