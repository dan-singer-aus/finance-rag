from datetime import date

from psycopg import Connection

from domain.documents import Document, FilingDocument, LetterDocument

SOURCE_INSERTION_SQL = """
    INSERT INTO sources (corpus, company, doc_type, title, section, fiscal_year, source_url, ticker, cik, period_end)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (corpus, ticker, fiscal_year, section) DO UPDATE SET
        doc_type = EXCLUDED.doc_type,
        title = EXCLUDED.title,
        source_url = EXCLUDED.source_url
    RETURNING id
"""

# The INSERT's column list, in order. Named so the helpers stay readable.
type SourceValues = tuple[
    str,          # corpus
    str | None,   # company
    str,          # doc_type
    str,          # title
    str | None,   # section
    int,          # fiscal_year
    str,          # source_url
    str | None,   # ticker
    int | None,   # cik
    date | None,  # period_end
]


def insert_source(conn: Connection, source: Document) -> int:
    if source.corpus == "filings":
        values = _filing_values(source)
    elif source.corpus == "letters":
        values = _letter_values(source)
    else:
        raise ValueError(f"Unknown corpus: {source.corpus}")

    with conn.cursor() as cursor:
        cursor.execute(SOURCE_INSERTION_SQL, values)
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError("Insertion produced no row")
        # `row[0]` is Any — the promise that RETURNING id yields an int lives in
        # the SQL string, which no checker can read. int() is free (psycopg
        # already hands back an int) and turns the promise into a real check.
        return int(row[0])


def _filing_values(source: FilingDocument) -> SourceValues:
    return (
        source.corpus,
        source.company,
        source.doc_type,
        source.title,
        source.section,
        source.fiscal_year,
        source.source_url,
        source.ticker,
        source.cik,
        source.period_end,
    )


def _letter_values(source: LetterDocument) -> SourceValues:
    return (
        source.corpus,
        None,  # company — Berkshire is implicit
        source.doc_type,
        source.title,
        None,  # section — letters are continuous prose
        source.fiscal_year,
        source.source_url,
        None,  # ticker
        None,  # cik
        None,  # period_end
    )
