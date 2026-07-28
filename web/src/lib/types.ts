/**
 * Row types for db/migrations/001_init.sql — hand-written, and hand-maintained.
 *
 * These are snake_case because that is what Postgres actually returns; the node
 * driver does no key mapping. Convert to camelCase at the boundary (in the
 * mapper below), so snake_case never leaks past the data layer into React.
 *
 * ⚠️ When you change a migration, change this file in the same commit.
 */

export type Corpus = "filings" | "letters";

/** A row from `sources`, as Postgres returns it. */
export interface SourceRow {
  id: string; // bigserial — node-postgres returns int8 as string to avoid precision loss
  corpus: Corpus;
  company: string | null;
  doc_type: string;
  title: string;
  fiscal_year: number | null;
  source_url: string | null;
  created_at: Date;
}

/** A row from `chunks`, as Postgres returns it. */
export interface ChunkRow {
  id: string;
  source_id: string;
  chunk_index: number;
  chunk_text: string;
  section: string | null;
  created_at: Date;
  // `embedding` is deliberately absent: never SELECT it into the app. 1536 floats
  // per row is a lot of bytes to ship for something the UI cannot use. Compute
  // distance in SQL and select the score, not the vector.
}

/** The app-facing shape. Add the retrieval score here when you write Lab 2. */
export interface Chunk {
  id: string;
  sourceId: string;
  chunkIndex: number;
  chunkText: string;
  section: string | null;
  createdAt: Date;
}

export function toChunk(row: ChunkRow): Chunk {
  return {
    id: row.id,
    sourceId: row.source_id,
    chunkIndex: row.chunk_index,
    chunkText: row.chunk_text,
    section: row.section,
    createdAt: row.created_at,
  };
}
