"""Lab 1 — corpus ingestor. STUB: this is yours to write.

The pipeline, roughly:

    read markdown from corpus/  →  chunk  →  embed  →  insert into sources + chunks

Open decisions the lab is actually about (don't let the stub imply answers):

  - Do filings and letters share one chunker, or do they need two? Filings have
    hard section boundaries worth preserving; the letters are continuous
    discursive prose where the boundary call is yours. Deciding this IS the lab.
  - Chunk size and overlap, and how you'd know if you picked badly.
  - What goes in `sources` vs `chunks` — the schema in 001_init.sql is a starting
    point, not a constraint. Change it if your chunking strategy wants something
    different.

Run with: uv run python -m ingest.ingest
"""


def main() -> None:
    raise NotImplementedError("Lab 1 — write the ingestion pipeline here.")


if __name__ == "__main__":
    main()
