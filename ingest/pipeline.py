import logging
from pathlib import Path

import frontmatter
from psycopg import Connection

from db.chunks import delete_for_source, insert_many
from db.sources import insert_source
from domain.chunks import EmbeddedChunk
from domain.documents import Document
from embedding import embed
from ingest.chunking import Chunker, by_characters, chunk_document
from ingest.parsing import parse_document

logger = logging.getLogger(__name__)

CORPUS_FOLDER = Path(__file__).parent.parent / "corpus"


def ingest_corpus(
    conn: Connection,
    *,
    captions: bool = True,
    chunker: Chunker = by_characters,
) -> None:
    """Re-chunk, re-embed and replace every source under one configuration.

    Commits per document, not per run — an idempotent upsert plus a per-document
    commit is what makes a crashed run restartable.
    """
    for file in get_files("letters") + get_files("filings"):
        ingest_document(conn, read_document(file), captions=captions, chunker=chunker)
        conn.commit()


def ingest_document(
    conn: Connection,
    document: Document,
    *,
    captions: bool = True,
    chunker: Chunker = by_characters,
) -> None:
    chunks = chunk_document(document, captions=captions, chunker=chunker)
    vectors = embed([chunk.text for chunk in chunks])
    embedded_chunks = [
        EmbeddedChunk(chunk, vector)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    source_id = insert_source(conn, document)
    delete_for_source(conn, source_id)
    insert_many(conn, source_id, embedded_chunks)
    logger.info("%s: %d chunks", document.title, len(embedded_chunks))


def get_files(folder_name: str) -> list[Path]:
    folder = CORPUS_FOLDER / folder_name
    return sorted(folder.glob("*.md"))


def read_document(file: Path) -> Document:
    post = frontmatter.load(file)
    return parse_document(post)
