"""CLI for procedural PPO Cluster Tool curriculum datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .cluster_generator import RLDatasetGenerator, RLGenerationConfig


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate PPO curriculum problems")
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard", "edge"))
    parser.add_argument("--count", type=int, dest="instance_count")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--without-reference-actions", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def load_rl_generation_config(args: argparse.Namespace) -> RLGenerationConfig:
    raw: dict[str, Any] = {}
    if args.config is not None:
        decoded = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(decoded, Mapping):
            raise ValueError("RL generation config JSON root must be an object")
        raw.update(decoded)
    for field in ("split", "difficulty", "instance_count", "seed"):
        value = getattr(args, field)
        if value is not None:
            raw[field] = value
    if args.seed_only:
        raw["materialize_problems"] = False
        raw["include_reference_actions"] = False
    elif args.without_reference_actions:
        raw["include_reference_actions"] = False
    return RLGenerationConfig.model_validate(raw)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        config = load_rl_generation_config(args)
        manifest = RLDatasetGenerator(config).generate(
            args.output,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "manifest": str(args.output / "manifest.json"),
                    "instance_count": len(manifest["instances"]),
                    "split": config.split,
                    "seed": config.seed,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"RL generation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
