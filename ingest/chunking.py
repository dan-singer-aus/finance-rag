from collections.abc import Callable

from domain.chunks import Chunk
from domain.documents import Document
from ingest.segmentation import Block, segment

CHARACTER_LIMIT = 4000
TARGET_SIZE = 2000
SEPARATORS = ["\n\n", "\n", ". ", " "]

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
    pieces = _split_recursive(text, SEPARATORS, target_size)
    packed_pieces = _pack(pieces, target_size)
    return [piece.strip() for piece in packed_pieces if piece.strip()]


def by_window(text: str, target_size: int = TARGET_SIZE, overlap: int = 0) -> list[str]:
    """Fixed-size windows (with overlaps), blind to sentence and paragraph breaks."""
    if not 0 <= overlap < target_size:
        raise ValueError(
            f"need 0 <= overlap < target_size, got {overlap=}, {target_size=}"
        )

    chunks = []
    start = 0
    while start < len(text):
        end = _window_end(text, start, target_size)
        if chunk := text[start:end].strip():
            chunks.append(chunk)
        if end == len(text):
            break
        start = _next_start(text, start, end, overlap)
    return chunks


def _window_end(text: str, start: int, target_size: int) -> int:
    """The last whitespace that fits, or a hard cut at the limit if there is none."""
    limit = start + target_size
    if limit >= len(text):
        return len(text)
    for i in range(limit, start, -1):
        if text[i].isspace():
            return i
    return limit


def _next_start(text: str, start: int, end: int, overlap: int) -> int:
    next_start = max(end - overlap, start)
    while next_start > start and _inside_word(text, next_start):
        next_start -= 1
    if next_start == start:
        next_start = end
    while next_start < len(text) and text[next_start].isspace():
        next_start += 1
    return next_start


def _inside_word(text: str, i: int) -> bool:
    """Both neighbours are non-whitespace, so a cut at i would split a word."""
    return not text[i - 1].isspace() and not text[i].isspace()


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
