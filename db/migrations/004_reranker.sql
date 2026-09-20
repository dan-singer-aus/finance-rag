-- 004_reranker.sql — a run row records whether a cross-encoder reordered the
-- results, and how many candidates it was given.


ALTER TABLE runs
    -- NULL means no reranking. Otherwise the pinned model id INCLUDING its
    ADD COLUMN reranker text,

    -- NULL means NOT APPLICABLE, not "off": truncating a cosine-ordered list
    -- does not reorder it, so on an unreranked run the depth changes no rank
    ADD COLUMN candidate_depth int,
    ADD CONSTRAINT reranker_depth_together
        CHECK ((reranker IS NULL) = (candidate_depth IS NULL));

-- One ingest can produce SEVERAL run rows — one per candidate depth. The
-- cross-encoder scores each (query, chunk) pair independently, so a shallower
-- depth is a filter over the same scored list rather than a second run.