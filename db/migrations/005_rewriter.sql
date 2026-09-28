-- 005_rewriter.sql -- a run row records whether query rewriting was applied.
--
-- Independent of reranker/candidate_depth: rewriting changes what gets
-- embedded and searched, reranking reorders what came back, and either can
-- run without the other. No CHECK pairing them.

ALTER TABLE runs
    ADD COLUMN rewriter text;  -- NULL = no rewriting, else the pinned model id
