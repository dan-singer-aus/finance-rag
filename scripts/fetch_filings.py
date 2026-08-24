r"""Fetch 10-K narrative sections from SEC EDGAR and convert them to markdown.

Ancillary corpus-prep tooling — not learning-target code. Kept in the repo rather
than run once and deleted, so the corpus is reproducible and can grow later
(same rationale as `fetch_letters.py`).

Source: https://www.sec.gov/edgar — filings are public domain. Each output file
records its accession number and source URL, so any chunk stays traceable.

Three sections per filing, chosen because they are the NARRATIVE ones:

    Item 1   Business            — competitive position, products, markets
    Item 1A  Risk Factors        — discrete titled risks, dozens per filing
    Item 7   MD&A                — management's analysis of results

Four wrinkles this handles, each of which would otherwise corrupt the corpus:

1. EMPHASIS IS CSS, NOT TAGS. Modern inline-XBRL filings carry *all* emphasis as
   `style="font-weight:700"` on a <span>. Checked across Visa, Exxon and Meta:
   ZERO <b>/<strong> tags between them, ~5,200 font-weight declarations. A plain
   HTML→markdown pass therefore drops every risk-factor title into
   undifferentiated prose — which matters, because in Item 1A those titles ARE
   the section's structure. `_promote_styled_emphasis` restores them first.

   Emphasis is preserved as **bold**/*italic*, deliberately NOT as markdown
   headings. Bold is what the document actually says; a heading level would be an
   interpretation, and where the chunk boundaries fall is a decision for the
   chunker, not for corpus prep.

2. THE TABLE OF CONTENTS LOOKS EXACTLY LIKE THE SECTIONS. "Item 1A Risk Factors"
   appears in the TOC before it appears as a heading, and both render as pipe
   table rows — Exxon lays its real headings out in single-cell tables. So "is a
   table row" cannot be the test; the page number is. A TOC row carries a bare
   integer cell after the title, a heading row does not.

3. CROSS-REFERENCES LOOK LIKE HEADINGS TOO. Bodies are full of "see Item
   1A—Risk Factors of this report". Two guards: the match must start its own line
   (a real heading does; prose references don't), and the expected section title
   must follow the item number.

4. NON-BREAKING SPACES. Filers separate the item number from the title with
   `&#160;`, so `ITEM\xa01A.` never matches a `\s`-based pattern in some regex
   modes and reads as a broken word downstream. Normalised to plain spaces.

5. AN ITEM MAY BE A SIGNPOST, NOT THE SECTION. Exxon's Item 7 is one sentence —
   "Reference is made to the section entitled ... in the Financial Section of this
   report" — with the actual MD&A tens of thousands of words later under its own
   heading. Extraction "succeeds", writes a 105-word file, and the corpus quietly
   loses its most important section for that company. Hence MIN_SECTION_WORDS and
   the per-section fallback patterns; short sections are reported loudly.

6. RUNNING HEADERS ARE NOT CONTENT. "Table of Contents" and the section title
   repeat at every page break — 15-40 times inside a single section. Each one
   would be chunked and embedded as if it were text. Stripped by repetition.

Two more per-filer variations, both of which produced a SILENT miss rather than
an error — worth knowing before trusting a new ticker:

- Emphasis may be split mid-heading. Visa emits one bold run ("**ITEM 1A.  Risk
  Factors**"); Meta emits two ("**Item 1A.** **Risk Factors**"). The markers land
  in the middle of the heading, so the patterns must tolerate '*' as separator
  whitespace, not just \s.

- A ticker does not reliably identify the filing entity. After ExxonMobil's 2026
  holding-company reorganisation, ticker XOM maps to the NEW entity (CIK
  2115436), which has never filed a 10-K; every 10-K is under the OLD entity (CIK
  34088), which still lists XOM as its ticker too. Pass `XOM:34088` to override.

Usage:
    uv run python scripts/fetch_filings.py V XOM:34088 META --years 2
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from bs4 import BeautifulSoup
from markdownify import markdownify

OUT_DIR = Path(__file__).resolve().parent.parent / "corpus" / "filings"

# SEC requires a descriptive UA with contact details, and rate-limits to 10 req/s.
USER_AGENT = "dan-singer-learning-project daniel@charidy.com"
REQUEST_SPACING_SECONDS = 0.2

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# (slug, human title, start pattern, end pattern). The end of each section is the
# start of the next item — the ITEM number alone is too weak a signal, so the
# expected title is required on both ends.
#
# GAP matches separator whitespace that may have emphasis markers embedded in it,
# because filers close and reopen bold mid-heading (see the docstring). Using \s
# here silently misses every Meta section.
GAP = r"[\s*]*"
ITEM = r"ITEM" + GAP + r"{number}" + GAP + r"[.\-—–:]?" + GAP
MDA_TITLE = r"MANAGEMENT.S" + GAP + r"DISCUSSION" + GAP + r"AND" + GAP + r"ANALYSIS\b"


class Section(NamedTuple):
    slug: str
    title: str
    start: str
    end: str
    # Some filers satisfy an item by pointing at another part of the document
    # instead of writing it there ("Reference is made to..."). When the primary
    # patterns yield almost nothing, these locate the prose itself. See wrinkle 5.
    fallback_start: str | None = None
    fallback_end: str | None = None


SECTIONS = [
    Section(
        "item-1-business",
        "Item 1. Business",
        ITEM.format(number="1") + r"BUSINESS\b",
        ITEM.format(number="1A") + r"RISK" + GAP + r"FACTORS\b",
    ),
    Section(
        "item-1a-risk-factors",
        "Item 1A. Risk Factors",
        ITEM.format(number="1A") + r"RISK" + GAP + r"FACTORS\b",
        ITEM.format(number="1B") + r"UNRESOLVED" + GAP + r"STAFF" + GAP + r"COMMENTS\b",
    ),
    Section(
        "item-7-mda",
        "Item 7. Management's Discussion and Analysis",
        ITEM.format(number="7") + MDA_TITLE,
        ITEM.format(number="7A")
        + r"QUANTITATIVE"
        + GAP
        + r"AND"
        + GAP
        + r"QUALITATIVE\b",
        fallback_start=MDA_TITLE,
        fallback_end=(
            r"MANAGEMENT.S" + GAP + r"REPORT" + GAP + r"ON" + GAP + r"INTERNAL\b"
        ),
    ),
]

# A heading occupies its own line, optionally wrapped in the emphasis markers that
# step 1 restored. Anchoring here is what separates a heading from a cross-reference.
LINE_START = r"(?im)^[#*\s|]{0,8}"

# Narrative sections run to thousands of words; a few hundred means the filer
# pointed elsewhere rather than wrote it here.
MIN_SECTION_WORDS = 500


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    time.sleep(REQUEST_SPACING_SECONDS)
    with urllib.request.urlopen(request, timeout=60) as response:
        # urlopen's return type is a union, so .read() is Any. Annotating the
        # local pins it without copying the payload, which bytes() would.
        body: bytes = response.read()
        return body


def _promote_styled_emphasis(raw: str) -> str:
    """Rewrite CSS-styled emphasis into real tags so markdownify can see it.

    Without this the Item 1A risk titles — the only thing marking where one risk
    ends and the next begins — arrive indistinguishable from body prose. See
    wrinkle 1 in the module docstring.
    """
    soup = BeautifulSoup(raw, "html.parser")
    for span in soup.find_all("span"):
        # BeautifulSoup returns a LIST for multi-valued attributes, so .get() is
        # `str | AttributeValueList`. `style` isn't multi-valued by default, which
        # is why this never fired — but the list form is legal, and .replace()
        # doesn't exist on it. Normalise rather than assume.
        raw_style = span.get("style") or ""
        if not isinstance(raw_style, str):
            raw_style = ";".join(raw_style)
        style = raw_style.replace(" ", "").lower()
        # 600+ covers the semibold weights filers occasionally use; `bold` is the
        # keyword form. Neither appeared in the sampled filings, but both are legal
        # CSS and cost nothing to accept.
        bold = re.search(r"font-weight:(bold|[6-9]00)", style)
        italic = "font-style:italic" in style
        if not (bold or italic):
            continue
        # Wrap rather than replace: the span may carry other content or nesting.
        inner = soup.new_tag("em" if italic and not bold else "strong")
        span.wrap(inner)
        if bold and italic:
            inner.wrap(soup.new_tag("em"))
    return str(soup)


def _to_markdown(raw: bytes) -> str:
    html_text = raw.decode("utf-8", errors="replace")
    html_text = _promote_styled_emphasis(html_text)
    # `strip=['a']` drops the pervasive "Table of Contents" anchors; their link
    # targets are internal ids that mean nothing outside the original document.
    text = markdownify(html_text, heading_style="ATX", strip=["a"])
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Adjacent bold runs, which happen when a filer closes and reopens the styled
    # span mid-heading ("**Item 1.****Business**"). Left alone the words weld
    # together in the text that gets embedded.
    #
    # NOTE: do NOT try to tidy emphasis with a general `\*\*...\*\*` rewrite. A
    # regex cannot tell an opening marker from a closing one, so it happily pairs
    # the close of one heading with the open of the next and moves both markers
    # onto the plain paragraph between them — silently unbolding every heading in
    # the document. That cost an audit to find; markdownify's own output is right.
    text = text.replace("****", "** **")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_contents_row(line: str) -> bool:
    """Does this table row carry a page number, i.e. is it a contents entry?

    Both the table of contents and (for some filers) the real headings are pipe
    table rows, so the row itself proves nothing. What separates them is that a
    contents entry points at a page and a heading does not.
    """
    if not line.lstrip().startswith("|"):
        return False
    return any(cell.strip().isdigit() for cell in line.split("|"))


def _find_heading(text: str, pattern: str, search_from: int = 0) -> int | None:
    """First line-anchored, non-contents occurrence of a section heading."""
    for match in re.finditer(LINE_START + pattern, text[search_from:]):
        start = search_from + match.start()
        line_start = text.rfind("\n", 0, start) + 1
        line_end = text.find("\n", start)
        line = text[line_start : line_end if line_end != -1 else len(text)]
        if _is_contents_row(line):
            continue
        return start
    return None


def _looks_like_prose(text: str, start: int, window: int = 600) -> bool:
    """Is this heading followed by prose, or by more table rows?

    Distinguishes a real heading from an entry in a sub-table of contents, which
    carries no page number (so `_is_contents_row` clears it) but is surrounded by
    its sibling rows.
    """
    following = text[text.find("\n", start) + 1 : start + window]
    lines = [line for line in following.split("\n") if line.strip()]
    if not lines:
        return False
    return sum(1 for line in lines if line.lstrip().startswith("|")) < len(lines) / 2


def _slice(text: str, start_pattern: str, end_pattern: str, prose: bool) -> str | None:
    search_from = 0
    while (start := _find_heading(text, start_pattern, search_from)) is not None:
        if prose and not _looks_like_prose(text, start):
            search_from = start + 1
            continue
        end = _find_heading(text, end_pattern, search_from=start + 1)
        return text[start : end if end is not None else len(text)].strip()
    return None


def _strip_page_furniture(section: str) -> str:
    """Drop repeated running headers, page rules, and empty table rows.

    Filings repeat "Table of Contents" and the section title at every page break —
    15-40 times in a single section here. Left in, each becomes text that gets
    chunked and embedded, and a running header is the emptiest possible chunk.
    Repetition plus brevity is the signal; a real heading does not recur.
    """
    lines = section.split("\n")
    counts = Counter(line.strip() for line in lines if len(line.split()) < 15)
    furniture = {line for line, n in counts.items() if n >= 3 and line}
    kept = [
        line
        for line in lines
        if line.strip() not in furniture
        and not re.fullmatch(r"[\s|:\-]*", line)  # page rules, empty table rows
        and not re.fullmatch(r"\s*\d{1,4}\s*", line)  # bare page numbers
        # Navigation rows survive the repetition test when a section happens to
        # contain only one or two of them, as at the tail of Exxon's MD&A.
        and not re.fullmatch(
            r"[\s|]*(?:\w+ )?Table of Contents[\s|\w]*", line, re.IGNORECASE
        )
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def extract_sections(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for section in SECTIONS:
        body = _slice(text, section.start, section.end, prose=False)
        # A section can resolve and still be a signpost rather than the thing.
        if (
            (body is None or len(body.split()) < MIN_SECTION_WORDS)
            and section.fallback_start
            and section.fallback_end
        ):
            body = (
                _slice(text, section.fallback_start, section.fallback_end, prose=True)
                or body
            )
        if body is not None:
            found[section.slug] = _strip_page_furniture(body)
    return found


def resolve_cik(argument: str, table: dict) -> tuple[str, int]:
    """Accepts `TICKER` or `TICKER:CIK`, the latter overriding the lookup.

    The override exists because a ticker identifies a *listing*, not a filer. See
    the XOM case in the module docstring — the lookup is right and still useless.
    """
    ticker, _, override = argument.partition(":")
    if override:
        return ticker.upper(), int(override)
    for entry in table.values():
        if entry["ticker"].upper() == ticker.upper():
            return ticker.upper(), int(entry["cik_str"])
    sys.exit(f"Unknown ticker: {ticker}")


def recent_10ks(cik: int, count: int) -> tuple[str, list[dict]]:
    """Return the filer's own name plus its most recent 10-Ks.

    The name comes from the submissions feed rather than the ticker table on
    purpose: with a CIK override those two disagree, and the filer's own name is
    the one that belongs in a citation.
    """
    data = json.loads(_get(SUBMISSIONS_URL.format(cik=cik)))
    recent = data["filings"]["recent"]
    filings = [
        {
            "accession": recent["accessionNumber"][i],
            "document": recent["primaryDocument"][i],
            "period_end": recent["reportDate"][i],
        }
        for i in range(len(recent["form"]))
        if recent["form"][i] == "10-K"
    ]
    return data["name"], filings[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--years", type=int, default=2, help="most recent N 10-Ks")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ticker_table = json.loads(_get(TICKER_MAP_URL))
    missing: list[str] = []

    for argument in args.tickers:
        ticker, cik = resolve_cik(argument, ticker_table)
        company, filings = recent_10ks(cik, args.years)
        if not filings:
            # Loud, because the ticker resolved fine and the fetch succeeded —
            # nothing here looks like a failure except the absence of output.
            print(f"{ticker} ... NO 10-K FILINGS under CIK {cik} ({company})")
            missing.append(f"{ticker} (no 10-K under CIK {cik})")
            continue
        for filing in filings:
            accession = filing["accession"].replace("-", "")
            url = ARCHIVE_URL.format(
                cik=cik, accession=accession, document=filing["document"]
            )
            # The fiscal-year LABEL is the year the period ends in. That is right
            # for these companies but not universal — a January year-end (Nvidia)
            # labels its year ahead of the end date. `period_end` is the
            # unambiguous field; prefer it for any year-over-year comparison.
            fiscal_year = filing["period_end"][:4]
            print(f"{ticker} FY{fiscal_year} ...", end=" ", flush=True)

            sections = extract_sections(_to_markdown(_get(url)))
            if not sections:
                print("NO SECTIONS FOUND")
                missing.append(f"{ticker} FY{fiscal_year} (all)")
                continue

            written = []
            for slug, title, *_ in SECTIONS:
                body = sections.get(slug)
                if body is None:
                    missing.append(f"{ticker} FY{fiscal_year} {slug}")
                    continue
                front_matter = (
                    "---\n"
                    f"title: {company} FY{fiscal_year} Form 10-K — {title}\n"
                    f"company: {company}\n"
                    f"ticker: {ticker.upper()}\n"
                    f"cik: {cik}\n"
                    "corpus: filings\n"
                    "doc_type: 10-K\n"
                    f"section: {slug}\n"
                    f"fiscal_year: {fiscal_year}\n"
                    f"period_end: {filing['period_end']}\n"
                    f"accession: {filing['accession']}\n"
                    f"source_url: {url}\n"
                    "---\n\n"
                )
                path = OUT_DIR / f"{ticker.upper()}-FY{fiscal_year}-{slug}.md"
                path.write_text(front_matter + body + "\n", encoding="utf-8")
                words = len(body.split())
                # A section that resolved but came back tiny is the failure mode
                # that hides: the file exists, the run says ok, and the corpus is
                # quietly missing a document. Usually means the filer incorporated
                # the section by reference instead of writing it inline.
                flag = "  <-- SUSPICIOUSLY SHORT" if words < MIN_SECTION_WORDS else ""
                written.append(f"{slug}={words:,}w{flag}")
            print("ok  " + "  ".join(written))

    if missing:
        print("\nNot found:", *missing, sep="\n  ")


if __name__ == "__main__":
    main()
