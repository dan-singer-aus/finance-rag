"""Apply pending .sql migrations from db/migrations, in filename order.

Ancillary tooling — not learning-target code. Run with: uv run migrate
"""

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "db" / "migrations"


def main() -> None:
    load_dotenv()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        sys.exit("DATABASE_URL is not set (copy .env.example to .env).")

    migrations = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not migrations:
        sys.exit(f"No migrations found in {MIGRATIONS_DIR}")

    with psycopg.connect(database_url) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        conn.commit()

        applied = {
            row[0]
            for row in conn.execute("SELECT filename FROM schema_migrations").fetchall()
        }

        pending = [m for m in migrations if m.name not in applied]
        if not pending:
            print("Nothing to apply — schema is up to date.")
            return

        for migration in pending:
            print(f"Applying {migration.name} ...", end=" ", flush=True)
            # Each migration runs in its own transaction: a failure rolls back
            # that file only, so a partial schema never gets recorded as applied.
            conn.execute(migration.read_text())
            conn.execute(
                "INSERT INTO schema_migrations (filename) VALUES (%s)",
                (migration.name,),
            )
            conn.commit()
            print("ok")


if __name__ == "__main__":
    main()
