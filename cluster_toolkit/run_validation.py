"""Load a problem and validate one action sequence with the local Toolkit."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from .problem import load_problem
from .validator import ValidationReport, ValidatorSuite


RawAction = Mapping[str, Any]


def load_actions(path: str | Path) -> list[RawAction]:
    decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(decoded, list):
        raise ValueError("Actions JSON root must be a list")
    if not all(isinstance(item, dict) for item in decoded):
        raise ValueError("Every action must be a JSON object")
    return decoded


def validate_action_sequence(
    problem_path: str | Path,
    actions_path: str | Path,
) -> ValidationReport:
    return ValidatorSuite(load_problem(problem_path)).validate(load_actions(actions_path))


def print_report(report: ValidationReport, stream: TextIO | None = None) -> None:
    output = sys.stdout if stream is None else stream
    checked = ", ".join(
        f"{kind}={count}" for kind, count in sorted(report.checked_subjects.items())
    )
    print(f"Checked subjects: {checked or 'none'}", file=output)
    if report.ok:
        print("Validation passed: no constraint violations found.", file=output)
        return
    print(f"Validation failed: {len(report.issues)} issue(s).", file=output)
    for number, issue in enumerate(report.issues, start=1):
        action = "-" if issue.action_index is None else str(issue.action_index)
        print(
            f"{number}. [{issue.constraint_id}] "
            f"{issue.subject_kind}={issue.subject_id!r}, action={action}",
            file=output,
        )
        print(f"   {issue.message}", file=output)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a cluster-tool action sequence against a problem"
    )
    parser.add_argument("problem", type=Path)
    parser.add_argument("actions", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        report = validate_action_sequence(args.problem, args.actions)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
