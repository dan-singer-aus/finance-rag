import argparse
import logging

from db.connection import connection
from ingest.pipeline import ingest_corpus


def main() -> None:
    # Only the entry point configures logging, and only inside main() — at
    # module scope this would be an import-time side effect, and in a library
    # module it would hijack the configuration of whatever imported it.
    # `%(message)s` reproduces the bare lines the previous print() emitted.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-captions", action="store_true")
    args = parser.parse_args()

    with connection() as conn:
        ingest_corpus(conn, captions=not args.no_captions)


if __name__ == "__main__":
    main()
