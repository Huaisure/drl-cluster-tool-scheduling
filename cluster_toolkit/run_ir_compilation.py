"""Convert a supported ClusterProblem JSON to canonical Constraint IR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .constraint_ir import SemanticError, TimeDomain, compile_problem
from .problem import load_problem


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("--output", type=Path, help="new output file; existing files are never overwritten")
    parser.add_argument("--ticks-per-second", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        ir = compile_problem(load_problem(args.problem), TimeDomain(
            unit="second", ticks_per_unit=args.ticks_per_second,
        ))
        payload = ir.canonical_json() + "\n"
        if args.output is None:
            print(payload, end="")
        else:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(payload)
        return 0
    except SemanticError as error:
        print(json.dumps({"code": error.code.value, "path": error.path, "message": str(error)}), file=sys.stderr)
    except (OSError, ValueError) as error:
        print(f"Input/output error: {error}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
