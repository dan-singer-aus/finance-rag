import argparse

from db.connection import connection
from ingest.pipeline import ingest_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-captions", action="store_true")
    args = parser.parse_args()

    with connection() as conn:
        ingest_corpus(conn, captions=not args.no_captions)


if __name__ == "__main__":
    main()
