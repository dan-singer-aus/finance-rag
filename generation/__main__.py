"""One-shot CLI for the naive answer path: retrieve, generate, print.

    uv run python -m generation "What drove Visa's net revenue growth in fiscal 2025?"
    uv run python -m generation "..." --prompt answer-naive --evidence

Uses argparse rather than hand-rolled sys.argv like `retrieval/` and
`grounding/` do — those take a single positional, this takes a positional plus
an option plus a flag, which is where hand-rolling stops paying.

Status goes to stderr, the answer to stdout, so the output composes:
`python -m generation "..." > answer.txt` writes the answer and nothing else.
That separation is the reason this is one-shot rather than a REPL.
"""

import argparse
import sys
import textwrap

from db.connection import connection
from domain.answers import GeneratedAnswer
from generation.answer import DEFAULT_PROMPT, generate
from retrieval.pipeline import retrieve


def main() -> None:
    args = _parse_args()

    # Only retrieval needs the connection. Generation is a slow model call, and
    # holding a Postgres connection open across it buys nothing.
    with connection() as conn:
        print(f"Retrieving evidence for: {args.query}", file=sys.stderr)
        context = retrieve(conn, args.query)

    print(f"Generating with prompt {args.prompt!r}...", file=sys.stderr)
    answer = generate(question=args.query, context=context, prompt_name=args.prompt)

    if args.evidence:
        _display_evidence(answer)

    print(answer.text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m generation")
    parser.add_argument("query", type=str, help="the question to answer")
    parser.add_argument(
        "--prompt",
        type=str,
        # From answer.py rather than retyped, so changing the default arm in one
        # place cannot leave the CLI on the old one.
        default=DEFAULT_PROMPT,
        help=f"prompt arm in prompts/ (default: {DEFAULT_PROMPT})",
    )
    parser.add_argument(
        "--evidence",
        action="store_true",
        help="print the numbered context the answer was generated from",
    )
    return parser.parse_args()


def _display_evidence(answer: GeneratedAnswer) -> None:
    """Print the context in the order its citation numbers were assigned.

    `enumerate(answer.context, 1)` IS the numbering — context is stored in the
    order the generator numbered it — so this cannot drift from what the model
    saw, and [3] in the answer is this list's third entry.
    """
    print(f"\nEVIDENCE ({len(answer.context)} chunks)", file=sys.stderr)
    for number, chunk in enumerate(answer.context, start=1):
        print(f"\n[{number}] {chunk.score:.3f}  {chunk.provenance}", file=sys.stderr)
        print(textwrap.fill(chunk.chunk_text, width=100), file=sys.stderr)
    print(file=sys.stderr)


if __name__ == "__main__":
    main()

