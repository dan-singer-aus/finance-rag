from dataclasses import dataclass

from domain.corpus import Corpus


@dataclass(frozen=True)
class GoldSpan:
    corpus: Corpus
    company: str | None
    fiscal_year: int
    section: str | None
    excerpt: str
    why: str


@dataclass(frozen=True)
class RecallFixture:
    query: str
    spans: list[GoldSpan]
    label: str


recall_fixture_1 = RecallFixture(
    query="What drove Visa's net revenue growth in fiscal 2025?",
    spans=[
        GoldSpan(
            corpus="filings",
            company="VISA INC.",
            fiscal_year=2025,
            section="item-7-mda",
            excerpt="Net revenue increased 11% over the prior year, primarily due to the growth in processed transactions, nominal cross-border volume, and nominal payments volume, partially offset by higher client incentives.",
            why="Expect every configuration to hit it. If it ever misses, the harness is broken, not retrieval",
        ),
    ],
    label="F1",
)

recall_fixture_2 = RecallFixture(
    query="What did Exxon's capital investment actually come to in 2025, including acquisitions?",
    spans=[
        GoldSpan(
            corpus="filings",
            company="EXXON MOBIL CORP",
            fiscal_year=2025,
            section="item-7-mda",
            excerpt="Cash Capex in 2025 was $29.0 billion, including $2.6 billion of acquisitions, reflecting the Corporation’s continued active investment program.",
            why="Year-discrimination test. Verified at line 357. Missed live -- top 10 don't return it; rank 1 is a different figure for a similar line item.",
        ),
    ],
    label="F2",
)

recall_fixture_3 = RecallFixture(
    query="Does Visa lend money to cardholders or take on the risk that they don't pay?",
    spans=[
        GoldSpan(
            corpus="filings",
            company="VISA INC.",
            fiscal_year=2025,
            section="item-1-business",
            excerpt="Visa is not a financial institution. We do not issue cards, extend credit or set rates and fees for account holders of Visa products nor do we earn revenue from or bear credit risk with respect to any of these activities.",
            why="Vocabulary-mismatch test. Confirmed absent from top 20 filings-only, by text not just provenance -- likely confounded with corpus saturation, not proven to isolate vocab mismatch alone.",
        ),
    ],
    label="F4",
)

recall_fixture_4 = RecallFixture(
    query="How much is Meta planning to spend on infrastructure next year to support AI?",
    spans=[
        GoldSpan(
            corpus="filings",
            company="Meta Platforms, Inc.",
            fiscal_year=2025,
            section="item-7-mda",
            excerpt="We anticipate making capital expenditures of approximately $115 billion to $135 billion in 2026 to support our AI efforts and core business.",
            why="Within-source discrimination test. Clean hit, rank 1 -- though the decoy that actually surfaced was a different document (FY2024's own guidance), not the same-file 2025-actual-vs-2026-guidance decoy F5 was designed around.",
        ),
    ],
    label="F5",
)

recall_fixture_5 = RecallFixture(
    query="What does Buffett say makes an economic franchise?",
    spans=[
        GoldSpan(
            corpus="letters",
            company=None,
            fiscal_year=1991,
            section=None,
            excerpt="An economic franchise arises from a product or service that: (1) is needed or desired; (2) is thought by its customers to have no close substitute and; (3) is not subject to price regulation.",
            why="Ceiling case, letters side. Clean hit, rank 1. Previously proven live 2026-09-08.",
        ),
    ],
    label="L1",
)

recall_fixture_6 = RecallFixture(
    query="Why does Buffett like competing against businesses that can't afford to keep reinvesting?",
    spans=[
        GoldSpan(
            corpus="letters",
            company=None,
            fiscal_year=1989,
            section=None,
            excerpt="Capital outlays at a business can be skipped, of course, in any given month, just as a human can skip a day or even a week of eating. But if the skipping becomes routine and is not made up, the body weakens and eventually dies.",
            why="Vocabulary-mismatch test, letters side. Confirmed absent from top 10 -- previously ranked 4th under different phrasing, now a full miss. Deliberately the hardest letters case.",
        ),
    ],
    label="L2",
)

recall_fixture_7 = RecallFixture(
    query="What kind of business does Buffett say is the worst to own, and what industry does he use as the example?",
    spans=[
        GoldSpan(
            corpus="letters",
            company=None,
            fiscal_year=2007,
            section=None,
            excerpt="The worst sort of business is one that grows rapidly, requires significant capital to engender the growth",
            why="Passage-discrimination test. Miss -- right letter (2007) surfaces at rank 2 but with an unrelated passage (Dexter), not this one; rank 1 is a cross-letter near-paraphrase from 1992.",
        ),
    ],
    label="L3",
)

recall_fixture_8 = RecallFixture(
    query="How many companies does Buffett think an investor needs to understand?",
    spans=[
        GoldSpan(
            corpus="letters",
            company=None,
            fiscal_year=1996,
            section=None,
            excerpt="You don't have to be an expert on every company, or even many. You only have to be able to evaluate companies within your circle of competence. The size of that circle is not very important; knowing its boundaries, however, is vital.",
            why="Vocabulary-mismatch test, unnamed concept. Miss -- none of top 10 from 1996; rank 1 is (again) a 1992 near-paraphrase without the 'circle of competence' name. Second time 1992 decoys a different target letter.",
        ),
    ],
    label="L5",
)

recall_fixture_9 = RecallFixture(
    query="Is Visa the capital-light kind of business Buffett favours?",
    spans=[
        GoldSpan(
            corpus="letters",
            company=None,
            fiscal_year=1987,
            section=None,
            excerpt="because so little capital is required to run these businesses, they can grow while concurrently making almost all of their earnings available for deployment in new opportunities",
            why="Framework half of a needs-both question. States the capital-light principle directly, unlike the mechanism-only 2007 excerpt first considered. Miss at top-3; found at rank 19/30 within its own corpus -- real but buried.",
        ),
        GoldSpan(
            corpus="filings",
            company="VISA INC.",
            fiscal_year=2025,
            section="item-7-mda",
            excerpt="Most U.S. dollar settlements are settled within the same day and do not result in a receivable or payable balance, while settlements in currencies other than the U.S. dollar generally remain outstanding for one to two business days, which is consistent with industry practice for such transactions. In general, during fiscal 2025, we were not required to fund settlement-related working capital.",
            why="Disclosure half. No Visa filing ever uses capital-light vocabulary (checked); this is the most direct statement of the underlying mechanism -- no working-capital funding need -- available in the corpus. Miss even at k=30 within its own corpus.",
        ),
    ],
    label="B1",
)

recall_fixture_10 = RecallFixture(
    query="Does Visa satisfy Buffett's three conditions for an economic franchise, given how it is regulated?",
    spans=[
        GoldSpan(
            corpus="letters",
            company=None,
            fiscal_year=1991,
            section=None,
            excerpt="An economic franchise arises from a product or service that: (1) is needed or desired; (2) is thought by its customers to have no close substitute and; (3) is not subject to price regulation.",
            why="Framework half, same excerpt as L1. Hit at rank 3, top-3 edge.",
        ),
        GoldSpan(
            corpus="filings",
            company="VISA INC.",
            fiscal_year=2025,
            section="item-1a-risk-factors",
            excerpt="In several jurisdictions, we have been designated as a “systemically important payment system.”",
            why="Disclosure half -- the one place the two corpora disagree. Miss at top-3; found at rank 18/30 within its own corpus.",
        ),
    ],
    label="B3",
)

RECALL_FIXTURES = [
    recall_fixture_1,
    recall_fixture_2,
    recall_fixture_3,
    recall_fixture_4,
    recall_fixture_5,
    recall_fixture_6,
    recall_fixture_7,
    recall_fixture_8,
    recall_fixture_9,
    recall_fixture_10,
]
