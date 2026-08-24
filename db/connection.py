"""Postgres connection helper.

Ancillary scaffolding — not learning-target code.

The one non-obvious bit is `register_vector`. Without it, psycopg has no idea what
a `vector` column is: you'd have to format embeddings into the string literal
'[0.1,0.2,...]' yourself on the way in, and parse that string back into floats on
the way out. Registering the adapter makes the type mostly invisible.

Two things it does NOT do, both verified against this stack (pgvector 0.5.0,
psycopg 3.3.4) — you'll meet both in Lab 2:

1. READS come back as a `pgvector.Vector` object, not a list or numpy array.
   It has no `len()`. Use `.to_list()`, `.to_numpy()`, or `.dimensions`.

2. A plain Python list PARAMETER is adapted as `double precision[]`, not
   `vector`. On an INSERT that's fine — Postgres coerces it to the column type.
   But in an expression there is no column to infer from, so:

       SELECT embedding <=> %s        -- ERROR: operator does not exist:
                                      -- vector <=> double precision[]

   Fix it either way:
       SELECT embedding <=> %s::vector      , (query_embedding,)
       SELECT embedding <=> %s              , (Vector(query_embedding),)

   This bites precisely on the similarity query, and the error names the
   operator rather than the parameter — so it reads like the operator is
   missing when the real problem is the argument's type.

Sanity check on the operator itself: `[1,1,0...] <=> [1,0,0...]` is 0.29289,
i.e. 1 - cos(45°). If your distances aren't in [0, 2], something is wrong.
A zero vector yields `nan` — cosine is undefined at zero magnitude.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (copy .env.example to .env).")
    return url


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Open a connection with the pgvector type adapter registered."""
    with psycopg.connect(_database_url()) as conn:
        register_vector(conn)
        yield conn
