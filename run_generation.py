"""Backward-compatible CLI for the legacy fixed-topology dataset generator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from cluster_toolkit.cluster_generator import DatasetGenerator, GenerationConfig


def load_generation_config(
    config_path: str | Path | None = None,
    **overrides: object,
) -> GenerationConfig:
    values: dict[str, object] = {}
    if config_path is not None:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("generation config must contain a JSON object")
        values.update(raw)
    values.update({key: value for key, value in overrides.items() if value is not None})
    return GenerationConfig.model_validate(values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fixed-topology instances")
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--profile", choices=("small", "medium", "large"))
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
        config = load_generation_config(
            args.config,
            profile=args.profile,
            instance_count=args.count,
            seed=args.seed,
        )
        generator = DatasetGenerator.from_template(args.template, config)
        manifest = generator.generate(args.output, overwrite=args.overwrite)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "instance_count": len(manifest.instances),
                "manifest": str(args.output / "manifest.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

