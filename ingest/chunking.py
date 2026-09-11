from domain.chunks import Chunk
from domain.documents import Document
from ingest.segmentation import Block, segment

CHARACTER_LIMIT = 4000


def _split_oversized(text: str) -> list[str]:
    """Cut text into successive slices of at most CHARACTER_LIMIT characters."""
    return [text[i : i + CHARACTER_LIMIT] for i in range(0, len(text), CHARACTER_LIMIT)]


def by_line(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def chunk_document(
    document: Document,
    *,
    captions: bool = True,
) -> list[Chunk]:
    """Split a document into chunks of text, each within the character limit."""
    pieces = [
        piece
        for block in segment(document.content, captions=captions)
        for piece in _block_pieces(block)
    ]
    return [Chunk(index=i, text=piece) for i, piece in enumerate(pieces)]


def _block_pieces(block: Block) -> list[str]:
    """The texts one block contributes, before anything numbers them."""
    text = "\n".join(block.lines)
    pieces = [text] if block.kind == "table" else by_line(text)
    return [capped for piece in pieces for capped in _split_oversized(piece)]
