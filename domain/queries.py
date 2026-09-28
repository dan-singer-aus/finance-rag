from dataclasses import dataclass

from domain.corpus import CORPORA, Corpus


@dataclass(frozen=True)
class CorpusQuery:
    query: str | None
    reasoning: str | None


@dataclass(frozen=True)
class RewrittenQueries:
    filings: CorpusQuery
    letters: CorpusQuery

    def for_corpus(self, corpus: Corpus) -> CorpusQuery:
        return self.filings if corpus == "filings" else self.letters


def issued(query: str, rewritten: RewrittenQueries | None) -> dict[Corpus, CorpusQuery]:
    """What was actually sent per corpus -- the rewrite, or the original question."""
    if rewritten is None:
        return {corpus: CorpusQuery(query=query, reasoning=None) for corpus in CORPORA}
    return {corpus: rewritten.for_corpus(corpus) for corpus in CORPORA}
