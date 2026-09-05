"""Migrate graph2 IR checkpoints to graph3 with behavior-preserving zeros."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .graph import FEATURE_VERSION, NUMERIC_WIDTH


def migrate_graph2_checkpoint(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    saved = torch.load(source, map_location="cpu", weights_only=True)
    if saved.get("feature_version") != "ir-graph-2":
        raise ValueError("source checkpoint is not ir-graph-2")
    if "model" not in saved:
        raise ValueError("source checkpoint has no model state")
    state = saved["model"]
    weight = state.get("numeric.weight")
    if weight is None or weight.ndim != 2 or weight.shape[1] != 1:
        raise ValueError("source numeric projection is not graph2 width")
    expanded = weight.new_zeros((weight.shape[0], NUMERIC_WIDTH))
    expanded[:, 0] = weight[:, 0]
    state["numeric.weight"] = expanded
    saved["feature_version"] = FEATURE_VERSION
    saved["migrated_from"] = {
        "path": str(source),
        "feature_version": "ir-graph-2",
        "new_numeric_columns_initialized_to_zero": list(range(1, NUMERIC_WIDTH)),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(saved, temporary)
    temporary.replace(destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    migrate_graph2_checkpoint(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
