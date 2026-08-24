"""Fetch Berkshire Hathaway shareholder letters and convert them to markdown.

Ancillary corpus-prep tooling — not learning-target code. Kept in the repo rather
than run once and deleted, so the corpus is reproducible and can grow later.

Source: https://www.berkshirehathaway.com/letters/ — freely published by Berkshire.
Each output file records its source URL, so any chunk stays traceable to it.

Two wrinkles this handles, both of which would otherwise corrupt the corpus:

1. ENCODING. The older letters are cp1252, not UTF-8. Read as UTF-8 they turn
   every curly quote and apostrophe into a replacement character — which would
   then be embedded, and would break any later quote-matching check.

2. HARD WRAPPING vs TABLES. The letters are almost entirely one big <PRE> block:
   prose is hard-wrapped at ~65 chars, and financial tables live in the same
   block, aligned with spaces. Unwrapping everything would destroy the tables;
   unwrapping nothing leaves prose with a line break every 65 characters, which
   makes chunk boundaries and quote extraction unnecessarily awkward. So blocks
   are classified and treated differently — see `_looks_tabular`.

Usage:
    uv run python scripts/fetch_letters.py 1983 1987 1991 1996 1999 2007
"""

import argparse
import html
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "corpus" / "letters"
BASE = "https://www.berkshirehathaway.com/letters"

USER_AGENT = "Mozilla/5.0 (compatible; personal-research-corpus/1.0)"

# Berkshire's URL scheme is not consistent across years; these are the patterns
# that actually resolve. Probed rather than assumed.
HTML_PATTERNS = ["{base}/{year}.html", "{base}/{year}htm.html", "{base}/{year}ltr.html"]
PDF_PATTERNS = ["{base}/{year}ltr.pdf"]


def _get(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 200:
                return None
            # See fetch_filings._get — .read() is Any off a union return type.
            body: bytes = response.read()
            return body
    except (urllib.error.URLError, TimeoutError):
        # A URL that doesn't resolve is an EXPECTED outcome here — this function
        # probes candidate patterns, so misses are normal control flow, not faults.
        # Anything else (a decode fault, a bug) is genuinely unexpected: let it raise.
        return None


SEPARATOR_RE = re.compile(r"^[\s*_\-]*(?:\*[\s*]*){3,}$")


def _is_separator(block: str) -> bool:
    """Buffett's `* * * * *` section dividers — a real boundary, not content."""
    return bool(SEPARATOR_RE.fullmatch(block.strip().replace("**", "*")))


def _looks_tabular(block: str) -> bool:
    """Heuristic: is this block a financial table rather than prose?

    Tables align columns with whitespace, so most lines have an interior gap.
    Prose, being hard-wrapped, normally doesn't. Three corrections learned from
    the actual corpus, each of which silently corrupted a letter before:

    - Columns are sometimes separated by TABS, not spaces (1996's holdings
      table). `expandtabs` first, or the gap test can't see them.
    - A table can be a SINGLE line (1983). The old `len(lines) < 2` guard sent
      those to `_unwrap`, which is harmless for one line but mislabels it.
    - PDF-extracted PROSE contains irregular interior gaps (2007), so the gap
      test alone produces false positives. Sentence structure is the counter-
      signal: real prose has sentence boundaries and few digits, and that
      outranks the gap evidence.
    """
    lines = [line.expandtabs(8) for line in block.split("\n") if line.strip()]
    if not lines:
        return False

    digit_heavy = sum(1 for line in lines if len(re.findall(r"\d", line)) >= 8)

    # Prose veto, checked first: sentence boundaries + low digit density.
    sentence_breaks = len(re.findall(r"[.!?][\"')”]?\s+[A-Z]", block))
    if (
        sentence_breaks >= 2
        and len(block.split()) > 40
        and digit_heavy < len(lines) * 0.3
    ):
        return False

    # Dot leaders ("General Re ....... $555") are an unambiguous table marker in
    # these documents, and the only reliable one in the PDF years, where extraction
    # collapses column whitespace and the gap test goes blind.
    if sum(1 for line in lines if re.search(r"\.{4,}", line)) >= max(1, len(lines) * 0.4):
        return True

    gapped = sum(1 for line in lines if re.search(r"\S {3,}\S", line))
    if len(lines) == 1:
        return bool(re.search(r"\S {3,}\S", lines[0]) and re.search(r"\d", lines[0]))
    return gapped >= len(lines) * 0.5 or digit_heavy >= len(lines) * 0.6


def _unwrap(block: str) -> str:
    """Join hard-wrapped lines of a prose block back into one paragraph.

    Rejoins compounds the original split at a line break ("Per-\\nshare"). Naive
    joining gives "Per- share", which reads fine but is not what the document
    says — and this corpus gets searched for verbatim quotes later, so a stray
    space inside a word is a real defect, not a cosmetic one.

    The hyphen is KEPT, not swallowed. These letters are hard-wrapped plain text,
    and hard wrapping breaks at spaces — it never inserts a soft hyphen to split a
    word. So a hyphen at end of line is always a real one ("Per-share",
    "first-class"), and dropping it produces "Pershare".
    """
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and not out.endswith("--") and line[:1].islower():
            out += line
        else:
            out += " " + line
    return re.sub(r"[ \t]{2,}", " ", out)


def html_to_markdown(raw: bytes) -> str:
    # cp1252 covers the smart quotes/dashes; the letters contain nothing outside it.
    text = raw.decode("cp1252", errors="replace")

    text = re.sub(r"(?is)<script.*?</script>", "", text)
    text = re.sub(r"(?is)<style.*?</style>", "", text)
    text = re.sub(r"(?is)<head.*?</head>", "", text)

    # <B> around a short line is a heading in these documents; keep the emphasis
    # as bold and let the chunker decide what a boundary is.
    text = re.sub(r"(?is)<b>(.*?)</b>", r"**\1**", text)
    # <I> is NOT converted. These documents open it before a table and close it
    # several lines inside, so a DOTALL match sprays stray '*' through columnar
    # data. Italics carry no retrieval signal here — drop the markup, keep the text.
    text = re.sub(r"(?i)</?i>", "", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Blank lines separate blocks; each block is prose or a table.
    blocks = re.split(r"\n\s*\n", text)
    out: list[str] = []
    for block in blocks:
        if not block.strip():
            continue
        # Page furniture: a block that is nothing but a page number. Harmless to
        # read past, but it would become its own scrap of retrievable text.
        if re.fullmatch(r"\s*\d{1,3}\s*", block):
            continue
        # Normalised to a horizontal rule. Faithful — it's the same section break
        # the original draws with asterisks — and it also removes the stray '**'
        # the bold-stripping leaves behind in those lines.
        if _is_separator(block):
            out.append("---")
            continue
        if _looks_tabular(block):
            # Fenced so the alignment survives and a chunker won't reflow it.
            # Emphasis is stripped inside: it won't render in a code fence anyway,
            # and stray markers would end up in the embedded text as noise.
            out.append("```\n" + block.strip("\n").rstrip().replace("**", "") + "\n```")
        else:
            unwrapped = _unwrap(block)
            if unwrapped:
                out.append(unwrapped)

    body = "\n\n".join(out)
    # Empty bold left by stripped tags. Must NOT span a blank line, or it welds
    # two adjacent headings into one (`**A.**\n\n**B**` -> `**A.B**`).
    body = re.sub(r"\*\*[ \t]*\*\*", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def pdf_to_markdown(raw: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    blocks = re.split(r"\n\s*\n", text)
    out = []
    for block in blocks:
        if not block.strip() or re.fullmatch(r"\s*\d{1,3}\s*", block):
            continue
        if _is_separator(block):
            out.append("---")
            continue
        out.append(
            "```\n" + block.strip("\n").rstrip() + "\n```"
            if _looks_tabular(block)
            else _unwrap(block)
        )
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(out)).strip()


def fetch_year(year: int) -> tuple[str, str] | None:
    """Return (markdown, source_url) for a year, or None if nothing resolved."""
    for pattern in HTML_PATTERNS:
        url = pattern.format(base=BASE, year=year)
        raw = _get(url)
        # The bare {year}.html for some years is a tiny frameset stub, not the
        # letter. Size is a crude but reliable way to tell them apart.
        if raw and len(raw) > 10_000:
            return html_to_markdown(raw), url

    for pattern in PDF_PATTERNS:
        url = pattern.format(base=BASE, year=year)
        raw = _get(url)
        if raw and len(raw) > 10_000:
            return pdf_to_markdown(raw), url

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("years", nargs="+", type=int)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []

    for year in args.years:
        print(f"{year} ...", end=" ", flush=True)
        result = fetch_year(year)
        if result is None:
            print("NOT FOUND")
            failed.append(year)
            continue

        body, url = result
        front_matter = (
            "---\n"
            f"title: Berkshire Hathaway Shareholder Letter {year}\n"
            "author: Warren Buffett\n"
            "corpus: letters\n"
            "doc_type: shareholder-letter\n"
            f"fiscal_year: {year}\n"
            f"source_url: {url}\n"
            "---\n\n"
        )
        path = OUT_DIR / f"{year}.md"
        path.write_text(front_matter + body + "\n", encoding="utf-8")
        print(f"ok  ({len(body.split()):,} words → {path.name})")

    if failed:
        sys.exit(f"Failed: {failed}")


if __name__ == "__main__":
    main()
