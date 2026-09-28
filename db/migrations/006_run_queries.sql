-- 006_run_queries.sql — the query actually sent, per fixture and corpus.
--
-- Grain is (run, fixture, corpus), NOT (run, span): a single-corpus fixture
-- still issues a query to both corpora, so this table has rows run_spans has
-- no counterpart for.
--
-- query NULL = the rewriter ruled this corpus out. That is a result, not a
-- missing value.
--
-- reasoning is the rewriter's own account of that judgement, and is NULL on a
-- run with no rewriting -- those rows have no reasoning, and '' would be a lie.
-- A log, not a metric: read it when diagnosing, don't branch on its contents.

CREATE TABLE run_queries (
    run_id    bigint NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    label     text   NOT NULL,
    corpus    text   NOT NULL CHECK (corpus IN ('filings', 'letters')),
    query     text,
    reasoning text,
    PRIMARY KEY (run_id, label, corpus)
);
