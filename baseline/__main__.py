from __future__ import annotations

import sys
from collections.abc import Sequence

from .branch_search import main as branch_search_main
from .cpsat.__main__ import main as cpsat_main
from .genetic import main as genetic_main


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a baseline CLI while preserving the historical genetic CLI."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "branch-search":
        return branch_search_main(arguments[1:])
    if arguments and arguments[0] == "cpsat":
        return cpsat_main(arguments[1:])
    if arguments and arguments[0] == "genetic":
        return genetic_main(arguments[1:])
    return genetic_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
