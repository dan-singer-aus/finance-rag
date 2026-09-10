"""The evidence linker: does the corpus support this claim, and how strongly?

⚠️ **`supported` means "the corpus discusses this at close range", not "this is
true".** Similarity measures whether a chunk is *about* a claim, never whether
it *agrees* — "margins rose" and "margins fell" retrieve the same chunks at the
same scores. A claim can be confidently supported by a passage that contradicts
it, and **no threshold value fixes that**; it needs entailment checking, which
is `grounding/citations.py`.

⚠️ **The thresholds are PROVISIONAL — a hypothesis for L5, not a measurement.**
Five data points, and the gap the letters threshold exploits (0.413 → 0.467) is
0.054 wide. They are per corpus because a single global cut point could not
discriminate at all: the filings scores are nearly flat (0.022 across four
unrelated claims, versus 0.193 for letters) because every filing chunk is about
Visa, Exxon or Meta, so any claim naming one has a near neighbour whether or not
it is relevant. When tuning, prefer a false `unsupported`: declining when you
had evidence is honest, asserting when you didn't is the failure this stage
exists to prevent. Thresholds are query-time, so a change costs only a re-run.

**Letters-only support caps at `weak`.** A framework says what to look for but
cannot testify about a specific company — "Visa is capital-light" grounded
solely in Buffett is a statement about Buffett. `_classify` can't tell a claim
about a company from one about the framework, so capping is the honest
resolution: *there is evidence, but not the kind that settles this*.

`_classify` is pure — chunks in, status out, no connection, no embedding call —
so it can be exercised offline with hand-written chunk lists. It must not assume
its input is sorted; filter, don't index.
"""

from psycopg import Connection

from domain.chunks import RetrievedChunk
from domain.corpus import Corpus
from domain.evidence import ClaimSupport, SupportStatus
from retrieval.pipeline import retrieve

SUPPORT_THRESHOLDS: dict[Corpus, float] = {"filings": 0.60, "letters": 0.45}


def link_evidence(conn: Connection, claim: str) -> ClaimSupport:
    results = retrieve(conn, claim)
    status = _classify(results)
    return ClaimSupport(claim, status, results)


def _classify(chunks: list[RetrievedChunk]) -> SupportStatus:
    supported_chunks = [chunk for chunk in chunks if _clears_threshold(chunk)]
    if not supported_chunks:
        return "unsupported"
    if any(chunk.corpus == "filings" for chunk in supported_chunks):
        return "supported"
    return "weak"


def _clears_threshold(chunk: RetrievedChunk) -> bool:
    return chunk.score > SUPPORT_THRESHOLDS[chunk.corpus]
