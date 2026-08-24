from pathlib import Path

import frontmatter
from psycopg import Connection

from db.chunks import delete_for_source, insert_many
from db.sources import insert_source
from domain.chunks import EmbeddedChunk
from domain.documents import Document
from embedding import embed
from ingest.chunking import chunk_document
from ingest.parsing import parse_document

CORPUS_FOLDER = Path(__file__).parent.parent / "corpus"

def ingest_document(conn: Connection, document: Document) -> None:
    chunks = chunk_document(document)
    vectors = embed([chunk.text for chunk in chunks])
    embedded_chunks = [EmbeddedChunk(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)]
    source_id = insert_source(conn, document)
    delete_for_source(conn, source_id)
    insert_many(conn, source_id, embedded_chunks)
    print(f"{document.title}: {len(embedded_chunks)} chunks")

def get_files(folder_name: str) -> list[Path]:
    folder = CORPUS_FOLDER / folder_name
    return sorted(folder.glob("*.md"))

def read_document(file: Path) -> Document:
    post = frontmatter.load(file)
    return parse_document(post)

