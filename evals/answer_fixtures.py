"""A forged generated answer, the context it was written against, and the
verdict each of its claims should receive.

⚠️ **CONTEXT IS FROZEN. Do not make this retrieve live.** It is a verbatim
capture of `retrieve(conn, QUESTION)` from 2026-08-31. The faults below are
defined *relative to these ten chunks* — "nothing in the context says this" is
only true of this context. Retrieve live and the next chunking change swaps the
context underneath the fixture, turning a planted fabrication into a supported
claim while the scoreboard stays green. Freezing also means the checker runs
with no database and no embedding call.

**ANSWER is hand-written, not generated**, so that exactly one claim lands in
each cell of the grid:

                      | cited                | uncited
    ------------------+----------------------+---------------------------
    entailed          | claim 1              | claim 4  (hygiene failure)
    contradicted      | claim 2  (the worst) | --
    not_stated        | claim 3              | claim 5  (fabrication)
    subjective        | --                   | claim 6  (vague)

The two empty cells are reachable but uninteresting: an uncited contradiction is
a fabrication with worse luck, and a cited vague claim tests nothing the uncited
one doesn't.

**A third citation state is deliberately absent**: a marker pointing at nothing
("[14]" when ten were retrieved). It has a home — `LocatedClaim.unresolved_citations`
— but adding it makes the grid three columns wide. Left until two are scored
reliably.

⚠️ **OPEN — the scoreboard's problem.** Expectations are keyed by `gist`, a human
label, because the splitter chooses the wording and the fixture cannot know it
in advance. How an actual claim pairs with its expected verdict is undecided;
today it works because they come out in order.
"""

from dataclasses import dataclass
from datetime import date

from domain.chunks import RetrievedChunk
from domain.citations import Entailment

QUESTION = "What drove Visa's net revenue growth in fiscal 2025?"

# Forged. Citation markers are 1-based positions into CONTEXT, which is the
# convention a generator prompt would specify. Sentence order is the order the
# faults appear in the grid above.
ANSWER = (
    "Visa's net revenue increased 11% in fiscal 2025, primarily on growth in "
    "processed transactions, cross-border volume and payments volume [2]. "
    "Exchange rate movements were a meaningful tailwind to that growth [2]. "
    "Visa's global headcount expanded over the year to support this growth [1]. "
    "Of the 329 billion transactions carried under Visa's brand during fiscal "
    "2025, 258 billion were processed by Visa itself. Management expects net "
    "revenue growth to remain in the low double digits through fiscal 2026. "
    "Visa remains well positioned in a rapidly evolving payments landscape."
)


@dataclass(frozen=True)
class ExpectedVerdict:
    """What one claim should come back as, and the check that produced it.

    `why` exists because when a verdict disagrees, the candidates are "the
    checker is wrong" and "the fixture is wrong" — and without the recorded
    reasoning the tempting fix is to adjust the expectation until it matches,
    which deletes the test while leaving it green.
    """

    gist: str
    cites: list[
        int
    ]  # 1-based positions into CONTEXT; empty if the claim carries no marker
    entailment: Entailment
    why: str


EXPECTED_VERDICTS = [
    ExpectedVerdict(
        gist="net revenue increased 11%, on processed transactions and volume",
        cites=[2],
        entailment="entailed",
        why=(
            "The control. Chunk [2] states it almost verbatim: 'Net revenue "
            "increased 11% over the prior year, primarily due to the growth in "
            "processed transactions, nominal cross-border volume, and nominal "
            "payments volume.' Right citation, right content. If the checker "
            "cannot pass this one, nothing else it reports means anything."
        ),
    ),
    ExpectedVerdict(
        gist="exchange rate movements were a meaningful tailwind",
        cites=[2],
        entailment="contradicted",
        why=(
            "THE ONE THE LAB EXISTS FOR. The cited chunk says the opposite in "
            "plain words: 'Exchange rate movements did not have a material "
            "impact on net revenue growth.' Note what makes it dangerous -- "
            "this is the CORRECT chunk to cite for a claim about exchange rates "
            "and net revenue. It is top-of-context, on-topic, and maximally "
            "similar to the claim. Every retrieval signal says supported. L3's "
            "threshold labels it supported at high confidence, and would do so "
            "at any threshold value, because similarity measures aboutness and "
            "not agreement. Only reading the two texts together separates them."
        ),
    ),
    ExpectedVerdict(
        gist="global headcount expanded over the year",
        cites=[1],
        entailment="not_stated",
        why=(
            "Cited, plausible, and simply absent. Chunk [1] is Visa's strategy "
            "sentence -- 'accelerate our revenue growth in consumer payments, "
            "new flows and value-added services' -- which says nothing about "
            "headcount either way. Distinct from claim 2 on purpose: the "
            "document does not disagree, it is silent, and 'silent' and "
            "'contradicted' are different facts about the world that a single "
            "supported/unsupported axis collapses. Headcount may well have "
            "risen; nothing retrieved establishes it."
        ),
    ),
    ExpectedVerdict(
        gist="258 of 329 billion transactions processed by Visa itself",
        cites=[],
        entailment="entailed",
        why=(
            "A hygiene failure, not a lie. Chunk [3] states both figures: 'During "
            "fiscal 2025, 329 billion payments and cash transactions with Visa's "
            "brand were processed by Visa or other networks... Of the 329 billion "
            "total transactions, 258 billion were processed by Visa.' True, in "
            "the context, no marker attached. This is the case that proves why "
            "the checker needs the chunks and not just the answer text: on the "
            "string alone this is indistinguishable from claim 5, and the two "
            "want completely different responses -- fix the generator's prompt "
            "versus do not ship the sentence."
        ),
    ),
    ExpectedVerdict(
        gist="management expects low-double-digit growth through fiscal 2026",
        cites=[],
        entailment="not_stated",
        why=(
            "The fabrication, and forward-looking on a public company, which is "
            "the shape that actually causes harm. No chunk contains a fiscal 2026 "
            "outlook; no chunk attributes any expectation to management. It reads "
            "as a reasonable extrapolation from the 11% in chunk [2], which is "
            "exactly how generators produce these -- not by inventing wildly, but "
            "by continuing a sentence one step past what the document licences."
        ),
    ),
    ExpectedVerdict(
        gist="Visa remains well positioned in a rapidly evolving landscape",
        cites=[],
        entailment="subjective",
        why=(
            "Not false -- unfalsifiable. There is no state of the world, and no "
            "chunk, that could settle 'well positioned'. It is included because "
            "answers are full of these and a checker with nowhere to put them "
            "must force one: score it entailed (chunk [3] is broadly flattering "
            "about Visa's position) and the metric flatters itself, score it "
            "not_stated and the report cries fabrication over boilerplate. "
            "Neither error is acceptable, which is why 'subjective' is a value "
            "on the entailment axis rather than a judgement call left to the "
            "model's mood."
        ),
    ),
]


# Verbatim capture: retrieve(conn, QUESTION) on 2026-08-31, top-5 per corpus,
# in returned (score-descending) order. Do not hand-edit -- regenerate.
CONTEXT = [
    # [1]
    RetrievedChunk(
        chunk_text="Visa’s strategy is to accelerate our revenue growth in consumer payments, new flows and value-added services, and fortify the key foundations of our business model.",
        chunk_index=28,
        source_id=15,
        title="VISA INC. FY2024 Form 10-K — Item 1. Business",
        doc_type="10-K",
        fiscal_year=2024,
        source_url="https://www.sec.gov/Archives/edgar/data/1403161/000140316124000058/v-20240930.htm",
        corpus="filings",
        company="VISA INC.",
        ticker="V",
        section="item-1-business",
        period_end=date(2024, 9, 30),
        score=0.7215,
    ),
    # [2]
    RetrievedChunk(
        chunk_text="*Highlights for fiscal 2025*. Net revenue increased 11% over the prior year, primarily due to the growth in processed transactions, nominal cross-border volume, and nominal payments volume, partially offset by higher client incentives. See *Results of Operations—Net Revenue* below for further discussion. Exchange rate movements did not have a material impact on net revenue growth.",
        chunk_index=18,
        source_id=20,
        title="VISA INC. FY2025 Form 10-K — Item 7. Management's Discussion and Analysis",
        doc_type="10-K",
        fiscal_year=2025,
        source_url="https://www.sec.gov/Archives/edgar/data/1403161/000140316125000089/v-20250930.htm",
        corpus="filings",
        company="VISA INC.",
        ticker="V",
        section="item-7-mda",
        period_end=date(2025, 9, 30),
        score=0.688,
    ),
    # [3]
    RetrievedChunk(
        chunk_text="Visa is one of the world’s leaders in digital payments. Our purpose is to uplift everyone, everywhere by being the best way to pay and be paid. Since Visa’s early days in 1958, we have been in the business of facilitating secure, reliable and efficient global commerce and money movement. We provide transaction processing services (primarily authorization, clearing and settlement) among consumers, issuing and acquiring financial institutions and sellers in a structure we call the “four-party” model. Please see *Our Core Business* discussion below. As the payments ecosystem continues to evolve, we have broadened this model to include digital banks, digital wallets, a range of financial technology companies (fintechs), governments and non-governmental organizations (NGOs). We are focused on extending, enhancing and investing in our proprietary advanced transaction processing network, VisaNet, to offer a single connection point for facilitating money movement to multiple endpoints through various form factors and innovative technologies across more than 200 countries and territories. Visa is committed to advancing innovation within the payment technology sector. Building upon our track record of industry leadership, including early adoption and integration of artificial intelligence (AI) models in payment systems, Visa continues to invest in the development and deployment of next-generation technologies, such as generative AI (GenAI), stablecoins and agentic commerce. During fiscal 2025, 329 billion payments and cash transactions with Visa’s brand were processed by Visa or other networks, equating to an average of 901 million transactions per day. Of the 329 billion total transactions, 258 billion were processed by Visa.",
        chunk_index=2,
        source_id=18,
        title="VISA INC. FY2025 Form 10-K — Item 1. Business",
        doc_type="10-K",
        fiscal_year=2025,
        source_url="https://www.sec.gov/Archives/edgar/data/1403161/000140316125000089/v-20250930.htm",
        corpus="filings",
        company="VISA INC.",
        ticker="V",
        section="item-1-business",
        period_end=date(2025, 9, 30),
        score=0.6777,
    ),
    # [4]
    RetrievedChunk(
        chunk_text="Visa’s growth has been driven by the strength of our core products — credit, debit and prepaid.",
        chunk_index=37,
        source_id=15,
        title="VISA INC. FY2024 Form 10-K — Item 1. Business",
        doc_type="10-K",
        fiscal_year=2024,
        source_url="https://www.sec.gov/Archives/edgar/data/1403161/000140316124000058/v-20240930.htm",
        corpus="filings",
        company="VISA INC.",
        ticker="V",
        section="item-1-business",
        period_end=date(2024, 9, 30),
        score=0.6736,
    ),
    # [5] Note: identical text to [4], different fiscal year. Not a bug in the
    # capture -- the sentence is repeated verbatim across the FY2024 and FY2025
    # filings, and both chunks are genuinely in the context.
    RetrievedChunk(
        chunk_text="Visa’s growth has been driven by the strength of our core products — credit, debit and prepaid.",
        chunk_index=37,
        source_id=18,
        title="VISA INC. FY2025 Form 10-K — Item 1. Business",
        doc_type="10-K",
        fiscal_year=2025,
        source_url="https://www.sec.gov/Archives/edgar/data/1403161/000140316125000089/v-20250930.htm",
        corpus="filings",
        company="VISA INC.",
        ticker="V",
        section="item-1-business",
        period_end=date(2025, 9, 30),
        score=0.6736,
    ),
    # [6] The letters half of the context is noise for this question, and is kept
    # rather than trimmed: a real generator sees it, and a checker that grounds a
    # Visa claim in a Berkshire table has a bug worth catching.
    RetrievedChunk(
        chunk_text="Our gain in net worth during 1995 was $5.3 billion, or 45.0%. Per-share book value grew by a little less, 43.1%, because we paid stock for two acquisitions, increasing our shares outstanding by 1.3%. Over the last 31 years (that is, since present management took over) per-share book value has grown from $19 to $14,426, or at a rate of 23.6% compounded annually.",
        chunk_index=2,
        source_id=6,
        title="Berkshire Hathaway Shareholder Letter 1995",
        doc_type="shareholder-letter",
        fiscal_year=1995,
        source_url="https://www.berkshirehathaway.com/letters/1995.html",
        corpus="letters",
        company=None,
        ticker=None,
        section=None,
        period_end=None,
        score=0.3813,
    ),
    # [7]
    RetrievedChunk(
        chunk_text="American Express ..............        300,000              263,265(1)(2)",
        chunk_index=238,
        source_id=4,
        title="Berkshire Hathaway Shareholder Letter 1991",
        doc_type="shareholder-letter",
        fiscal_year=1991,
        source_url="https://www.berkshirehathaway.com/letters/1991.html",
        corpus="letters",
        company=None,
        ticker=None,
        section=None,
        period_end=None,
        score=0.3809,
    ),
    # [8]
    RetrievedChunk(
        chunk_text="I've told you that over time look-through earnings must increase at about 15% annually if our intrinsic business value is to grow at that rate. Our look-through earnings in 1992 were $604 million, and they will need to grow to more than $1.8 billion by the year 2000 if we are to meet that 15% goal. For us to get there, our operating subsidiaries and investees must deliver excellent performances, and we must exercise some skill in capital allocation as well.",
        chunk_index=72,
        source_id=5,
        title="Berkshire Hathaway Shareholder Letter 1992",
        doc_type="shareholder-letter",
        fiscal_year=1992,
        source_url="https://www.berkshirehathaway.com/letters/1992.html",
        corpus="letters",
        company=None,
        ticker=None,
        section=None,
        period_end=None,
        score=0.3735,
    ),
    # [9]
    RetrievedChunk(
        chunk_text='The strong revenue gains of 1985-87 almost guaranteed the industry an excellent underwriting performance in 1987 and, indeed, it was a banner year. But the news soured as the quarters rolled by: Best\'s estimates that year-over-year volume increases were 12.9%, 11.1%, 5.7%, and 5.6%. In 1988, the revenue gain is certain to be far below our 10% "equilibrium" figure. Clearly, the party is over.',
        chunk_index=109,
        source_id=2,
        title="Berkshire Hathaway Shareholder Letter 1987",
        doc_type="shareholder-letter",
        fiscal_year=1987,
        source_url="https://www.berkshirehathaway.com/letters/1987.html",
        corpus="letters",
        company=None,
        ticker=None,
        section=None,
        period_end=None,
        score=0.3713,
    ),
    # [10]
    RetrievedChunk(
        chunk_text="49,456,900  American Express Company .............   $1,392.7   $2,046.3",
        chunk_index=184,
        source_id=6,
        title="Berkshire Hathaway Shareholder Letter 1995",
        doc_type="shareholder-letter",
        fiscal_year=1995,
        source_url="https://www.berkshirehathaway.com/letters/1995.html",
        corpus="letters",
        company=None,
        ticker=None,
        section=None,
        period_end=None,
        score=0.3711,
    ),
]
