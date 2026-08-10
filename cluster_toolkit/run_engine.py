"""Inspect the initial semantic actions of the local Cluster Engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .cluster_engine import AdvanceAction, ClusterEngine, PickAction, PlaceAction
from .problem import load_problem


def _write_json(value: Any, path: Path | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    if path is None:
        print(text)
    else:
        path.write_text(text + "\n", encoding="utf-8")


def _action_dict(action: object) -> dict[str, object]:
    if isinstance(action, PickAction):
        return {
            "action_type": "pick",
            "robot_id": action.robot_id,
            "wafer_key": list(action.wafer_key),
        }
    if isinstance(action, PlaceAction):
        return {
            "action_type": "place",
            "wafer_key": list(action.wafer_key),
            "target_module_id": action.target_module_id,
        }
    if isinstance(action, AdvanceAction):
        return {"action_type": "advance"}
    raise TypeError(f"unsupported action: {action!r}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the minimal Cluster Engine")
    parser.add_argument("problem", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        engine = ClusterEngine(load_problem(args.problem))
        engine.reset()
        payload = {
            "time": engine.state.time,
            "complete": engine.is_complete(),
            "deadlocked": engine.is_deadlocked(),
            "actions": [_action_dict(action) for action in engine.available_actions()],
            "next_event_time": engine.next_event_time(),
        }
        _write_json(payload, args.output)
        return 0
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
