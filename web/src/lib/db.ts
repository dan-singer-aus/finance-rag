import { Pool } from "pg";

/**
 * Postgres pool for the read path.
 *
 * Ancillary scaffolding — not learning-target code. The retrieval query itself
 * (Lab 2) is yours to write.
 *
 * Cached on `globalThis` because Next's dev server hot-reloads modules on every
 * edit; without this you'd leak a new pool per reload until Postgres refuses
 * connections. Standard Next + node-driver idiom.
 */
const globalForDb = globalThis as unknown as { pool?: Pool };

export const pool =
  globalForDb.pool ??
  new Pool({
    connectionString: process.env.DATABASE_URL,
    max: 10,
  });

if (process.env.NODE_ENV !== "production") globalForDb.pool = pool;

/**
 * Typed query helper. The type parameter is an *assertion*, not a guarantee —
 * nothing checks that `T` still matches the table. That's the trade you took by
 * choosing hand-written types over an ORM: if a migration changes a column,
 * this keeps compiling and fails at runtime. Keep `types.ts` next to the
 * migrations in your head.
 */
export async function query<T>(
  text: string,
  params?: readonly unknown[],
): Promise<T[]> {
  const result = await pool.query(text, params as unknown[]);
  return result.rows as T[];
}
