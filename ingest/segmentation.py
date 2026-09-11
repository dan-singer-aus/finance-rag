"""Split a document into typed blocks before any chunking strategy sees it.

Partitioning (what kind of thing is this run of lines?) and splitting (how big
should a chunk be?) are separate jobs. Every chunking strategy shares this pass,
so a strategy comparison varies only the prose chunking and table handling can't
confound the delta.

Operates on RAW lines. The whitespace-column signal that identifies a table in
the letters lives entirely in the unstripped line, so nothing here may strip or
drop blanks — that stays a prose-path concern, downstream.
"""

import re
from dataclasses import dataclass
from typing import Literal

type BlockKind = Literal["prose", "table"]

# Two or more runs of two-plus whitespace is the column-gap signature of an
# ASCII table. Measured over both corpora: 1,314 of 5,887 non-blank lines, with
# no prose false positives found by sampling. Leading indentation deliberately
# counts as a gap (hence rstrip, not strip) — it recovers 64 further real table
# lines, all of them header rows, rule rows, or indented two-column rows.
_COLUMN_GAP = re.compile(r"\s{2,}")
_MIN_COLUMN_GAPS = 2

# How far above a table run to look for its caption. Bounded because a longer
# walk starts inventing relationships the layout doesn't support.
_CAPTION_LOOKBACK = 3


@dataclass(frozen=True)
class Block:
    """A run of consecutive lines of one kind, verbatim and unstripped."""

    kind: BlockKind
    lines: list[str]


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("|"):
        return True
    return len(_COLUMN_GAP.findall(line.rstrip())) >= _MIN_COLUMN_GAPS


def _is_image_only(stripped: str) -> bool:
    """A bare `![alt](src)` line — real in the filings, useless as a caption."""
    return stripped.startswith("![") and stripped.endswith(")")


def _caption(lines: list[str], start: int) -> list[str]:
    """The nearest line above a table run that could label it, or nothing.

    Returned as a list so the caller can prepend unconditionally. The line is
    COPIED, not moved: `The following table sets forth our consolidated
    statements of income` both labels the table and belongs to the narrative
    above it, and moving it leaves a hole in the prose. A duplicated line across
    a structural boundary is just overlap.
    """
    for index in range(start - 1, max(start - 1 - _CAPTION_LOOKBACK, -1), -1):
        line = lines[index]
        if _is_table_line(line):
            break
        stripped = line.strip()
        if not stripped or _is_image_only(stripped):
            continue
        return [line]
    return []


def segment(content: str, *, captions: bool = True) -> list[Block]:
    """Partition document text into alternating prose and table blocks.

    Takes text rather than a Document: segmentation reads no metadata, so this
    stays a pure function of the content and imports nothing from `domain`.

    `captions` exists to measure one thing at a time. Grouping table rows and
    labelling the result are two separate changes, and shipping them together
    produced a single recall delta attributable to neither.
    """
    lines = content.split("\n")
    blocks: list[Block] = []
    prose: list[str] = []
    index = 0

    while index < len(lines):
        if not _is_table_line(lines[index]):
            prose.append(lines[index])
            index += 1
            continue

        start = index
        while index < len(lines) and _is_table_line(lines[index]):
            index += 1

        if prose:
            blocks.append(Block("prose", prose))
            prose = []
        caption = _caption(lines, start) if captions else []
        blocks.append(Block("table", caption + lines[start:index]))

    if prose:
        blocks.append(Block("prose", prose))
    return blocks
