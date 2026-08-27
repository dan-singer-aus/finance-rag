-- 002_embedding_not_null.sql — embedding becomes mandatory.
--
-- Similarity search only finds chunks that have an embedding, so a chunk
-- without one is dead weight in the table.
--
-- It also breaks the search result type: `1 - (embedding <=> $1)` is NULL when
-- the embedding is NULL, and that arrives as None in RetrievedChunk.score,
-- which is typed `float`.
--
-- Applying this scans the table to check no existing rows violate it.
ALTER TABLE chunks
ALTER COLUMN embedding SET NOT NULL;