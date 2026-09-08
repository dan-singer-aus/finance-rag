# Finance RAG

A source-grounded research system over SEC filings **and** Berkshire Hathaway
shareholder letters. Ask it a question, get an answer where every factual claim
carries a citation to a specific chunk of a specific document — and where the
system declines rather than guessing when the evidence doesn't support an
answer.

The premise is that the two corpora play **different roles**. The filings are
the *subject* — what a company says about itself, and the only thing that can
establish a fact about that company. The letters are the *lens* — a framework
for judging what it said. A question like *"what supply-chain risks did Visa
flag?"* needs only the first. *"Is Visa the capital-light kind of business
Buffett favours?"* needs both, and needs them kept apart: a letter states a
principle, but it cannot testify about a company.

> Not investment advice. This is a research and retrieval system over public
> documents.

---

## Status: in progress

The ingestion, retrieval and generation path works end to end, and so does the
citation checker that grades what it produces — an answer is decomposed into
claims, each claim's `[n]` markers are resolved to the chunks they name, and
each claim is judged against the retrieved context. That checker is itself
scored, against a fixture of six hand-written faults, over repeated runs.

What the evaluation layer still lacks is the other half: **whether retrieval
found the right thing in the first place** is unmeasured, and the retrieval
quality work it gates hasn't started. See [Roadmap](#roadmap) for what's
deliberately not done yet.

**Working today:**

```bash
# Ingest both corpora: parse → chunk → embed → store
uv run python -m ingest

# Semantic search over the chunk store
uv run python -m retrieval "What drove Visa's net revenue growth in fiscal 2025?"

# Full retrieval + grounded generation, with citations
uv run python -m generation "Is Visa the capital-light kind of business Buffett favours?"

# Eval suites — retrieval-stage (evidence linker) and generation-stage
# (citation coverage + entailment). Name the ones you want; there is no
# run-everything default, because unlike a test suite these cost money.
uv run python -m evals evidence citations
```

**Corpus as ingested:** 5,891 chunks across 26 source documents, reproducible
and idempotent — three consecutive runs produce an identical store.

---

## What's interesting here

Most of the value in this project has come from things that *didn't* work, and
from measuring them rather than assuming.

### A similarity threshold cannot tell you whether a claim is supported

The system grades claims against retrieved evidence. The obvious first
implementation is a cosine-similarity threshold: if the best chunk scores above
*x*, the claim is supported. Two separate measurements say that can't work.

**The filings scores are flat, and it's a property of the corpus.** Across five
unrelated test claims, the best-scoring filing chunk spanned a range of **0.022**
while the letters spanned **0.193** — nearly ten times wider. The cause is
corpus saturation: every filing chunk is about one of three companies, so any
claim naming one of them has a nearest neighbour *about that company* whether or
not it's relevant. Nearest-neighbour search always returns something, and here
everything is equally close. A single global threshold reads whichever
distribution is flatter and looks broken for no visible reason. The fix is
per-corpus thresholds — the concrete form of "similarity scores aren't
comparable across heterogeneous sources".

**And underneath that: similarity measures aboutness, not agreement.** *"Margins
rose"* and *"margins fell"* retrieve the same chunks at the same scores. So a
claim can be confidently "supported" by a passage that flatly contradicts it,
and **no threshold value recovers the difference** — the information isn't in the
number. That's the evidenced case for a model in the loop rather than a bigger
constant, and it's why the citation checker judges entailment separately from
whether a citation exists at all.

### Vector search has no "no match"

The first live query was *"What supply-chain risks did Visa flag?"* It returned
five chunks — all Visa, all from Item 1A, with no metadata filter at all, which
is real evidence the embeddings are doing their job across 26 documents.

None of them was about supply chain. Checking the source: Visa's FY2025 risk
factors mention "supply chain" exactly twice, both incidental. The honest answer
is *"Visa doesn't flag it substantively"* — and nothing in the retrieval output
says so, because nearest-neighbour search returns exactly *k* rows whatever
exists, and the scores of a good and a bad match aren't visually separable.

The consequence shapes the whole design: a generator **cannot distinguish
"absent from the corpus" from "absent from the top-k"**, so a retrieval miss
produces the same fluent, correctly-cited, false sentence as a true absence. And
the likelier failure isn't fabrication but *over-interpretation* — the retrieved
chunks mention "third-party service providers", which a helpful model will
happily frame as supply-chain risk while every word still traces to a real
source. A citation checker passes that. **"I don't know" therefore has to be
measured at the retrieval layer, not prompted for at the generation layer.**

### Don't chase determinism in a stochastic step — make the check robust to it

The citation checker decomposes an answer into individual claims, because
citations attach to claims and not to paragraphs. Sentence boundaries aren't
claim boundaries in either direction — one sentence can carry two claims, two
sentences can carry one — so decomposition is a model call.

The first version also had the model decide which claim each `[n]` marker
belonged to. Running it twice over an identical input produced **opposite
verdicts**: one run split a comma-joined sentence in two and reported a figure as
uncited, the next kept it whole and reported the same figure as cited. A
stochastic step was moving a deterministic result.

The fix wasn't a tighter rule or a lower temperature — that reduces the variance
without removing the dependency on it. Instead the model now returns the *source
sentence* verbatim and attributes nothing, and marker extraction happens in code
with a regex. Every claim from a sentence carries that sentence's markers, so
the decomposition can wobble freely without moving an answer.

### Prompt specification bought auditability, not groundedness

Two prompt arms exist so prompt sensitivity is measurable rather than assumed: a
fully specified one (closed-world evidence policy, citation rules, figure and
period handling, cross-corpus reasoning, explicit abstention) and a spare one
carrying only the three requirements the system's design actually needs.

Early result, on a small number of runs: **both arms abstained** on a question
the model certainly knows from pretraining, because abstention is in both. The
extra ~2,000 tokens of specification bought citation discipline and attribution —
not fewer fabrications. The opposite of the intuition, and only visible because
the spare arm exists.

Once the citation checker existed, that reading could be replaced with a
measurement. Same question, same retrieved context, both arms:

| | claims | cited | entailed |
|---|---|---|---|
| specified arm | 3 | **3/3** | 3/3 |
| spare arm | 5 | **3/5** | 5/5 |

**Groundedness identical; citation coverage not.** The spare arm put one `[2]`
at the end of two sentences, leaving the headline figure and the entire driver
list uncited while being perfectly true — a claim that traces to a real chunk
but doesn't say so. That is the failure the checker is for, and the effect had
been predicted in writing before the metric existed to test it.

### The instrument gets calibrated too — on frozen input

The citation checker is what grades live answers. But a grader whose own
accuracy is unknown just relocates the trust problem, so it is itself scored
against a fixture: a hand-written answer with one deliberate fault per cell of
the grid it can express — cited-and-contradicted, uncited-but-true, cited-but-
not-stated, and so on — paired with the retrieved context that answer was
written against, **frozen as literals**.

The freezing is the point. Those faults are defined *relative to those ten
chunks*: "nothing in the context says this" is only true of that context.
Retrieve live and the next chunking change swaps the evidence underneath the
fixture, quietly turning a planted fabrication into a supported claim while the
scoreboard stays green.

So the two halves of the eval program are deliberately asymmetric — **freeze the
input when the ground truth is defined relative to it; keep it live when
retrieval is the thing being measured.** The retrieval-stage suite retrieves
live for exactly the reason this one doesn't. A consequence worth stating: a
chunking change *should* move the retrieval numbers and *should not* move this
one. If it moved both, you could no longer tell the system getting worse from
the instrument drifting.

It also runs three times rather than once. The claim splitter is a model call,
and it has been observed returning a different number of claims from identical
input — so a cell that flips between runs is reported as a different finding
from a cell that is consistently wrong.

---

## Architecture

```
corpus/  ──►  ingest/  ──►  ┌──────────────┐  ◄──  retrieval/  ──►  generation/
                            │  Postgres 17 │                             │
                            │  + pgvector  │                        grounding/
                            └──────────────┘
```

| Path | Owns |
|---|---|
| `domain/` | shared record types — stdlib only, depends on nothing |
| `db/` | storage: connection, one module per table, `search.py`, SQL migrations |
| `ingest/` | parse → chunk → embed → store, one transaction per document |
| `retrieval/` | query embedding + vector search + result shaping |
| `generation/` | grounded answer generation from a supplied context |
| `grounding/` | evidence linking, claim decomposition, citation checking |
| `prompts/` | prompt library (YAML) + typed loader |
| `evals/` | fixtures, ground truth, and the scoreboards over them |
| `llm.py`, `embedding.py` | the two vendor adapters, owned by no layer |
| `web/` | Next.js app — scaffolded, not built |

Dependencies point inward. `domain/` imports nothing and everything imports it.

**A few decisions worth calling out:**

- **Plain SQL migrations, no ORM.** Two languages hit this database, so the
  neutral format wins — and pgvector DDL (operator classes, HNSW opclass
  pairing) is what ORMs model worst.
- **Generation never retrieves.** `generate(question, context, prompt_name)`
  takes its evidence as a parameter and opens no connection. Beyond the obvious
  separation-of-concerns argument, it's what makes an agentic retrieval loop
  possible later: the loop retrieves, grades, rewrites, retrieves again, and
  then generates *from the set it chose*.
- **Chunks are derived data and get replaced wholesale.** Updating chunk text
  without re-embedding leaves a row whose vector contradicts its own content,
  and nothing downstream can detect that.
- **Model snapshots are pinned to dated versions**, never floating aliases. An
  evaluation whose score moves when a vendor rotates an alias can't attribute a
  change to your own code.
- **The eval fixture freezes its retrieved context verbatim.** The planted faults
  are defined relative to *those* chunks, so retrieving live would let a chunker
  change silently swap the context and turn a green scoreboard into a measurement
  of a different question.

---

## Corpus

Curated markdown, committed to the repo, re-fetchable via `scripts/`.

- **`corpus/filings/`** — 18 sections (~200,000 words): Visa, ExxonMobil and Meta
  × FY2024 and FY2025 × Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A).
  The three companies were picked for **contrast on capital intensity**, not
  index rank, so a question judged against Buffett's framework has three
  genuinely different correct answers. Two fiscal years so that stale-source
  handling has a real year-over-year delta to work with.
- **`corpus/letters/`** — 8 Berkshire Hathaway shareholder letters (~94,000
  words), chosen by which ideas they argue — economic goodwill, owner earnings,
  economic franchise, businesses that eat capital — rather than by recency.
  Argument-dense prose gives retrieval something to discriminate on; annual
  performance recaps don't.

**Everything in this repository is public.** SEC filings are US government
works and not subject to copyright; the shareholder letters are published
freely by Berkshire Hathaway and are used here with attribution and a link to
source. That's a deliberate constraint rather than a convenience: it's what
makes the corpus safe to commit, safe to deploy publicly, and safe to show. It
also ruled things out — earnings-call transcripts would have added a genuinely
useful third document shape, but the accessible sources either bar scraping or
gate redistribution behind a separate agreement, so they're excluded on
licensing grounds rather than on capability.

The corpus is small on purpose. Chunking means corpus size doesn't compete for
context — only the top-*k* reaches the model — so the binding constraint isn't
size but **evaluability**. Ground truth requires knowing which source genuinely
holds the best answer, and you can't write that for material you skimmed. A
corpus you haven't read produces mis-grounded ground truth by construction.

---

## Setup

Requires Docker, [uv](https://docs.astral.sh/uv/), and an OpenAI API key.

```bash
cp .env.example .env

docker compose up -d                    # Postgres 17 + pgvector on localhost:5434
uv sync
uv run python scripts/migrate.py
uv run python -m ingest                 # ~$0.01 in embedding calls
```

Then any of the commands under [Status](#status-in-progress). Postgres is on
port **5434** to avoid colliding with other local containers; a pgweb console
comes up alongside it on **localhost:8081**.

Embeddings use `text-embedding-3-small` (1536 dimensions); generation uses a
pinned GPT-5.5 snapshot. Re-embedding the entire corpus costs well under a cent,
which is deliberate — it means chunking strategy can be changed and re-measured
freely.

---

## Roadmap

Listed in the order they're being built, because each one needs the measurement
the previous one provides.

**Evaluation — half built.** The generation-stage half is done: the citation
coverage checker (decompose an answer into claims → resolve each `[n]` to a
chunk → judge each claim against the retrieved context), scored against a frozen
fixture over repeated runs. The retrieval-stage half — a question set with
hand-written ground truth, reporting recall@k — is next. Nothing
below this line is worth doing before that number exists — every retrieval
technique is a claimed improvement, and a claimed improvement without a baseline
is a vibe.

**Retrieval quality.** Each of these is a known technique with a known failure
direction, to be adopted only if it moves the number:

- **Better chunking.** The current chunker is deliberately naive — a fixed
  budget with no packing — and its weaknesses are already visible in real
  results: a four-word section heading outranks a substantive paragraph, because
  cosine similarity is length-normalised and a heading that is 100% on-topic
  points nearer the query than a paragraph that is also about three other
  things. Section-aware chunking and packing are the obvious next passes.
- **Hybrid search (semantic + BM25).** Pure vector search misses exact terms —
  tickers, product names, figures. Worth noting the failure direction: on the
  one query where the *correct* answer ranked 4th, keyword search would have made
  it worse, because Buffett states the concept in metaphor and has none of the
  query's vocabulary.
- **Reranking** with a cross-encoder that reads query and chunk together —
  probably the largest single quality lever after chunking, and the right fix
  for the metaphor case above.
- **Metadata filters and corpus routing.** Filtering by company, fiscal year,
  section and corpus. A filings-only question currently still spends part of its
  budget on letters chunks.
- **Query rewriting and HyDE.** Rewriting a vague question into one that matches
  the vocabulary of the section that answers it; generating a hypothetical answer
  and retrieving against *that* embedding rather than against the question.

**Agentic retrieval.** Upgrading the single-pass pipeline into a loop: grade the
retrieved chunks for relevance, rewrite the query and retry when they fail,
run a grounding check before returning, cap the iterations, and surface
"couldn't ground an answer" as a legible outcome rather than a confident wrong
one. The pipeline is built first on purpose — you can't show that the loop beats
the pipeline without the pipeline's numbers.

**Web app.** A Next.js front end over the retrieval and generation path, showing
the retrieved evidence and the loop's steps alongside the answer. Deployed
bring-your-own-key.
