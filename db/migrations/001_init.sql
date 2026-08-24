-- 001_init.sql — pgvector extension + the corpus tables.
--
-- Scaffolding only (per the Stage 2 guide, Lab 1: "Claude scaffolds the Postgres
-- table + pgvector extension"). The chunking STRATEGY is yours; expect to evolve
-- this schema as that strategy firms up.

CREATE EXTENSION IF NOT EXISTS vector;

-- A source document: one filing section, one transcript, or one letter.
CREATE TABLE sources (
    id          bigserial PRIMARY KEY,
    corpus      text NOT NULL CHECK (corpus IN ('filings', 'letters')),
    company     text,                 -- NULL for letters (Berkshire is implicit)
    doc_type    text NOT NULL,       
    title       text NOT NULL,
    section     text,                 
    fiscal_year int,
    source_url  text,
    ticker      text,
    cik         int,
    period_end  date,
    created_at  timestamptz NOT NULL DEFAULT now(),

    UNIQUE NULLS NOT DISTINCT (corpus, ticker, fiscal_year, section)
);

CREATE TABLE chunks (
    id          bigserial PRIMARY KEY,
    source_id   bigint NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,         -- position within the source; keeps order recoverable
    chunk_text  text NOT NULL,
    embedding   vector(1536),         -- ⚠️ dimension is model-coupled: 1536 = OpenAI
                                      -- text-embedding-3-small. Changing model = new migration.
    created_at  timestamptz NOT NULL DEFAULT now(),

    UNIQUE (source_id, chunk_index)
);

CREATE INDEX chunks_source_id_idx ON chunks (source_id);
CREATE INDEX sources_corpus_idx   ON sources (corpus);

-- ---------------------------------------------------------------------------
-- Vector index — DELIBERATELY NOT CREATED YET. Two reasons, both worth knowing:
--
-- 1. At ~600–800 chunks a sequential scan is fast AND exact. HNSW is an
--    APPROXIMATE index — turning it on here would trade correctness for speed
--    you don't need, and could quietly change your recall@3 numbers.
--
-- 2. The operator class must MATCH the distance operator you query with:
--       <=>  cosine distance          → vector_cosine_ops
--       <->  L2 / Euclidean distance  → vector_l2_ops
--       <#>  negative inner product   → vector_ip_ops
--    Mismatch them and Postgres silently ignores the index and seq-scans.
--    That pairing is yours to make in Lab 2, once you've picked the operator.
--
-- When the corpus grows enough to need it, uncomment the matching line:
--
-- CREATE INDEX chunks_embedding_hnsw_idx ON chunks
--     USING hnsw (embedding vector_cosine_ops);
-- ---------------------------------------------------------------------------
