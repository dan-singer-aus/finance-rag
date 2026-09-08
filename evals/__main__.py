"""One entry point for every eval suite.

    uv run python -m evals evidence
    uv run python -m evals citations
    uv run python -m evals evidence citations

Ancillary scaffolding — dispatch only. What each suite measures, and what counts
as a match, lives in the suite.

**Two suites because an eval program needs both stages.** `evidence` is the
retrieval-stage measurement (did the corpus support this claim at all);
`citations` is the generation-stage one (did the answer cite what it used, and
does the evidence bear it out). Tracking only the second hides retrieval
regressions; tracking only the first misses fabrication.

**There is deliberately no run-everything default**, which departs from test-
runner convention. That convention assumes running the suite is free. This one
costs a dollar or two and a couple of minutes per invocation, almost all of it
in `citations`' judge calls — so a bare `python -m evals` prints the choices and
spends nothing. `nargs="+"` gives that for free and lets any subset be named,
which a `--all` flag could not.
"""

import argparse
from collections.abc import Callable

from evals import score_citations, score_evidence

SUITES: dict[str, Callable[[], None]] = {
    "evidence": score_evidence.main,
    "citations": score_citations.main,
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
