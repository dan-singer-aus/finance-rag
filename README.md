# 02 — Finance RAG

Stage 2 of the AI Engineering track. A source-grounded research system over
company filings **and** Buffett shareholder letters that answers with citations,
distinguishes *what a company disclosed* from *the framework judging it*, and
declines when the evidence is weak.

Study guide: `vault-k/AI/study-guides/stage-2-rag.md`

> Not investment advice. A research/retrieval system over public documents.

## Shape

Two languages, one database, split along the learning target:

| Path | Owns |
|---|---|
| `ingest/`, `evals/` (Python) | ingestion, chunking, embedding, retrieval evals |
| `web/` (Next.js, TS) | the API route, the read path, the UI |
| `db/migrations/*.sql` | the schema — **the single source of truth for both** |

Plain SQL migrations rather than an ORM: two languages hit this database, so the
neutral format wins, and pgvector DDL (operator classes, HNSW opclass pairing) is
what ORMs model worst. The read path uses raw `pg` with hand-written row types in
`web/src/lib/types.ts` — the trade is that nothing checks those types still match
the tables, so migrations and `types.ts` change in the same commit.

## Setup

```bash
cp .env.example .env          # Python side
cp .env.example web/.env.local  # Next side

docker compose up -d          # Postgres 17 + pgvector on localhost:5434
uv sync                       # Python deps
uv run python scripts/migrate.py

cd web && npm install && npm run dev
```

Port **5434**, not 5432/5433 — both are taken by other containers on this machine.

## Corpus

Curated markdown, committed to the repo. Not built by a scraper — see the study
guide on why a small corpus you have *read* beats a large one you haven't
(eval ground truth you didn't read is mis-grounded by construction).

- `corpus/filings/` — 2–3 companies × Item 1A (Risk Factors) + MD&A + one
  earnings-call transcript each.
- `corpus/letters/` — six Buffett shareholder letters, chosen by which metrics
  they ground (moat / capital intensity / pricing power), not by recency.

## Status

Scaffolding only. No lab code written yet — see `docs/todo.md`.
