import argparse

from db.connection import connection
from ingest.pipeline import get_files, ingest_document, read_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-captions", action="store_true")
    args = parser.parse_args()

    files = get_files("letters") + get_files("filings")

    with connection() as conn:
        for file in files:
            document = read_document(file)
            ingest_document(conn, document, captions=not args.no_captions)
            conn.commit()


if __name__ == "__main__":
    main()
