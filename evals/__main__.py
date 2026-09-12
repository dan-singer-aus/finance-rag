"""One entry point for every eval suite.

    uv run python -m evals evidence
    uv run python -m evals citations
    uv run python -m evals retrieval
    uv run python -m evals evidence citations retrieval

Ancillary scaffolding — dispatch only. What each suite measures, and what counts
as a match, lives in the suite.

**Three suites because an eval program needs more than one stage.** `evidence`
and `retrieval` are both retrieval-stage measurements (did the corpus support
this claim at all; did the right source land in the top-k); `citations` is the
generation-stage one (did the answer cite what it used, and does the evidence
bear it out). Tracking only generation hides retrieval regressions; tracking
only retrieval misses fabrication.

**There is deliberately no run-everything default**, which departs from test-
runner convention. That convention assumes running the suite is free. `evidence`
and `citations` cost a dollar or two and a couple of minutes per invocation,
almost all of it in model calls — so a bare `python -m evals` prints the
choices and spends nothing. `nargs="+"` gives that for free and lets any
subset be named, which a `--all` flag could not. (`retrieval` alone is free —
no model calls, just embeddings and a live DB — but it stays behind the same
opt-in for consistency, not because of cost.)
"""

import argparse
from collections.abc import Callable

from evals import score_citations, score_evidence, score_retrieval

# `object`, not `None`: a suite may return its numbers for another caller.
# Dispatch ignores them — printing the report is the suite's job.
SUITES: dict[str, Callable[[], object]] = {
    "evidence": score_evidence.main,
    "citations": score_citations.main,
    "retrieval": score_retrieval.main,
}


def main() -> None:
    args = _parse_args()

    for name in args.suites:
        print(f"\n=== {name} ===\n")
        SUITES[name]()


def _parse_args() -> argparse.Namespace:
    choices = sorted(SUITES)
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run one or more eval suites. Naming none prints this.",
    )
    parser.add_argument(
        "suites",
        nargs="+",
        choices=choices,
        metavar="SUITE",
        help=f"one or more of: {', '.join(choices)}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
