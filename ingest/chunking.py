from collections.abc import Callable

from domain.chunks import Chunk
from domain.documents import Document
from ingest.segmentation import Block, segment

CHARACTER_LIMIT = 4000
TARGET_SIZE = 2000
SEPARATOR = ["\n\n", "\n", ". ", " "]

# One block's prose into pieces. Knobs are bound by the caller, so the contract
# is the same for every strategy.
type Chunker = Callable[[str], list[str]]


def _split_oversized(text: str) -> list[str]:
    """Cut text into successive slices of at most CHARACTER_LIMIT characters."""
    return [text[i : i + CHARACTER_LIMIT] for i in range(0, len(text), CHARACTER_LIMIT)]


def by_line(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _split_recursive(text: str, separators: list[str], target_size: int) -> list[str]:
    """Split text into pieces by separators, then split those into pieces."""
    if len(text) <= target_size:
        return [text]
    if len(separators) == 0:
        return [text]
    separator = separators[0]
    pieces = text.split(separator)
    parts = [piece + separator for piece in pieces[:-1]] + pieces[-1:]

    return [
        capped
        for part in parts
        for capped in _split_recursive(part, separators[1:], target_size)
    ]


def _pack(pieces: list[str], target_size: int) -> list[str]:
    """Pack pieces into chunks of at most target_size characters."""
    chunks = []
    current_chunk = ""
    for piece in pieces:
        if len(current_chunk) + len(piece) <= target_size:
            current_chunk += piece
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = piece
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def by_characters(text: str, target_size: int = TARGET_SIZE) -> list[str]:
    """Split text into pieces of at most target_size characters."""
    pieces = _split_recursive(text, SEPARATOR, target_size)
    packed_pieces = _pack(pieces, target_size)
    return [piece.strip() for piece in packed_pieces if piece.strip()]


def chunk_document(
    document: Document,
    *,
    captions: bool = True,
    chunker: Chunker = by_characters,
) -> list[Chunk]:
    """Split a document into chunks of text, each within the character limit.

    `chunker` arrives already bound to its own knobs — `partial(by_characters,
    target_size=800)` — so a strategy with different knobs needs no change here.
    """
    pieces = [
        piece
        for block in segment(document.content, captions=captions)
        for piece in _block_pieces(block, chunker)
    ]
    return [Chunk(index=i, text=piece) for i, piece in enumerate(pieces)]


def _block_pieces(block: Block, chunker: Chunker) -> list[str]:
    """The texts one block contributes, before anything numbers them."""
    text = "\n".join(block.lines)
    pieces = [text] if block.kind == "table" else chunker(text)
    return [capped for piece in pieces for capped in _split_oversized(piece)]
