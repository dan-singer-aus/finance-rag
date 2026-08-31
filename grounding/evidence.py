"""Lab 3 — the evidence linker: does the corpus actually support this claim?

Grounding, not retrieval. `retrieval/` finds chunks; this decides whether what
came back is enough to stand behind. The two fail differently, which is why they
are separate packages: a retrieval failure means the right chunk didn't rank, a
grounding failure means fabrication — and grounding failures survive perfect
retrieval.

Two things drive the verdict.

**The score threshold, which is PER CORPUS and not by preference.** A single
global threshold was tried first and could not discriminate at all. Measured
across the five claim fixtures:

                            filings   letters
    XOM capex (factual)      0.756     0.413
    Visa capital-light       0.553     0.467
    Visa supply-chain        0.536     0.300   <- no answer exists anywhere
    XOM capital-hungry       0.558     0.493
    Visa pricing power       0.541     0.472

The filings column is **flat** — excluding the verbatim-figure claim, four
unrelated claims span 0.022. The letters column spans 0.193, nearly ten times
wider. The cause is corpus saturation: every filing chunk is about Visa, Exxon
or Meta, so any claim naming one of them has a nearest neighbour *about that
company* whether or not it says anything relevant. Nearest-neighbour always
returns something, and here everything is equally close.

So the two corpora carry different amounts of information at the same score, and
a single cut point reads whichever column happens to be flatter. Hence a
threshold each. Note the letters column does the real work: the claim with no
answer anywhere scores 0.300, by far the lowest number in the table, because
Buffett genuinely has nothing to say about supply chains.

**These numbers are PROVISIONAL — five data points, and the gap the letters
threshold exploits (0.413 to 0.467) is 0.054 wide.** They are a hypothesis for
L5 to test, not a measurement. Where a judgement call was needed, a false
`unsupported` is the safer error: declining when you had evidence is honest;
asserting when you didn't is the failure this whole stage exists to prevent.
Thresholds are query-time, so changing them costs nothing but a re-run.

**The corpus composition.** Letters-only support caps at `weak`, never
`supported`. A framework tells you what to look for but cannot testify about a
specific company — "Visa is capital-light" grounded solely in Buffett is a
statement about Buffett, not about Visa. The linker can't distinguish a claim
about a company from a claim about the framework, so it can't branch on that;
capping at `weak` is the honest resolution, saying *there is evidence, but not
the kind that settles this*.

`_classify` is deliberately pure — chunks in, status out, no connection and no
embedding call. It holds the logic L5 will spend its time calibrating, and a
pure function can be exercised with hand-written chunk lists offline, instantly,
with no API spend. It also must not assume its input is sorted: filtering rather
than indexing keeps it correct whatever order the caller passes.

**Known limit, and it is the real one.** Similarity measures whether a chunk is
*about* a claim, never whether it *agrees* with it. "Visa's margins rose" and
"Visa's margins fell" retrieve the same chunks at the same scores. So
`supported` here means "the corpus discusses this at close range", which is
weaker than it sounds — a claim can be confidently supported by a passage that
contradicts it. Closing that gap needs entailment checking, which means reading
the chunk against the claim with a model rather than comparing vectors. That is
L4's job; no threshold value can substitute for it.
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


