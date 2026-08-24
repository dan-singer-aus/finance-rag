from domain.chunks import Chunk
from domain.documents import Document

CHARACTER_LIMIT = 4000


def _split_oversized(text: str) -> list[str]:
    """Cut text into successive slices of at most CHARACTER_LIMIT characters."""
    return [text[i:i + CHARACTER_LIMIT] for i in range(0, len(text), CHARACTER_LIMIT)]


def chunk_document(document: Document) -> list[Chunk]:
    """Split a document into chunks of text, each within the character limit."""
    lines = document.content.split("\n")
    blocks = [stripped for line in lines if (stripped := line.strip())]
    pieces = []
    for block in blocks:
        pieces.extend(_split_oversized(block))
    return [Chunk(index=i, text=piece) for i, piece in enumerate(pieces)]


