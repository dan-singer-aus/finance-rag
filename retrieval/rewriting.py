from psycopg import Connection
from pydantic import BaseModel

from domain.queries import CorpusQuery, RewrittenQueries
from llm import parse_call
from prompts import load as load_prompt

REWRITE_MODEL = "gpt-5.5-2026-04-23"
REWRITE_PROMPT = "rewrite"

COMPANIES_SQL = "SELECT DISTINCT company FROM sources WHERE corpus = 'filings'"


class _CorpusQuery(BaseModel):
    reasoning: str
    query: str | None


class _CorpusQueries(BaseModel):
    filings: _CorpusQuery
    letters: _CorpusQuery


def rewrite_query(conn: Connection, question: str) -> RewrittenQueries:
    """Decompose a question into up to two corpus-specific retrieval queries."""
    prompt = load_prompt(REWRITE_PROMPT)
    result = parse_call(
        system=prompt.system,
        user=prompt.render(companies=_companies(conn), question=question),
        model=REWRITE_MODEL,
        schema=_CorpusQueries,
    )
    return RewrittenQueries(
        filings=_as_domain(result.filings),
        letters=_as_domain(result.letters),
    )


def _as_domain(query: _CorpusQuery) -> CorpusQuery:
    return CorpusQuery(query=query.query, reasoning=query.reasoning)


def _companies(conn: Connection) -> str:
    """The covered filings companies, comma-separated -- fills <%companies%>."""
    with conn.cursor() as cursor:
        cursor.execute(COMPANIES_SQL)
        return ", ".join(row[0] for row in cursor.fetchall())
