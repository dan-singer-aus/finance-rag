-- 003_runs.sql — measurement results: one row per chunking run, one row per
-- gold span in that run.
--
-- Observations only. recall@k, MRR and the containment ceiling are all derivable
-- from run_spans, so they are queries rather than columns — a stored copy can
-- disagree with the ranks it was computed from.

CREATE TABLE runs (
    id              bigserial PRIMARY KEY,
    ran_at          timestamptz NOT NULL DEFAULT now(),
    strategy        text NOT NULL,
    target_size     int,                  -- NULL for by-line, which has no size target
    overlap         int,                  -- NULL unless the strategy takes one
    captions        boolean NOT NULL,
    embedding_model text NOT NULL,
    chunks          int NOT NULL,
    median_chars    int NOT NULL,
    max_chars       int NOT NULL,
    under_120       int NOT NULL
);

-- Where one gold span landed under one run.
--
-- merged_rank / corpus_rank are NULL when no chunk in the window contained the
-- excerpt. `containable` separates the two reasons for that: false means the
-- chunker split the excerpt so no chunk could ever hold it (the run's ceiling),
-- true means it survived chunking and retrieval ranked it too low.
--
-- `excerpt` is stored rather than looked up by label, so a row stays readable
-- after the fixture set changes — otherwise an edited F2 silently makes every
-- older F2 row refer to a different span.
CREATE TABLE run_spans (
    id          bigserial PRIMARY KEY,
    run_id      bigint NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    label       text NOT NULL,
    corpus      text NOT NULL CHECK (corpus IN ('filings', 'letters')),
    excerpt     text NOT NULL,
    containable boolean NOT NULL,
    merged_rank int,
    corpus_rank int
);

CREATE INDEX run_spans_run_id_idx ON run_spans (run_id);
