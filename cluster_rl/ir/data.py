"""Materialized, disjoint small PM datasets and strict raw/IR input loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from cluster_toolkit.constraint_ir import ConstraintIRV1, TimeDomain, compile_problem
from cluster_toolkit.problem import load_problem, parse_problem


def load_ir(path: Path) -> ConstraintIRV1:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw.get("schema_version"), str) and "reference" in raw["schema_version"]:
        return ConstraintIRV1.model_validate(raw)
    return compile_problem(load_problem(path), TimeDomain(unit="second", ticks_per_unit=1000))


def load_cases(paths: list[Path], *, expected_split: str | None = None) -> list[tuple[Path, ConstraintIRV1]]:
    cases = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "instances" not in raw:
            cases.append((path.resolve(), load_ir(path)))
            continue
        split = raw.get("config", {}).get("split")
        if expected_split is not None and split is not None and split != expected_split:
            raise ValueError(f"expected {expected_split} manifest, got {split}: {path}")
        for entry in raw["instances"]:
            name = entry.get("ir_file", entry.get("problem_file"))
            if not isinstance(name, str):
                raise ValueError(f"unmaterialized manifest entry in {path}")
            target = (path.parent / name).resolve()
            if target.parent != path.parent.resolve():
                raise ValueError(f"manifest files must be in its directory: {name}")
            ir = load_ir(target)
            if entry.get("problem_hash", ir.problem_hash) != ir.problem_hash:
                raise ValueError(f"manifest hash mismatch: {target}")
            cases.append((target, ir))
    if not cases:
        raise ValueError("a non-empty dataset is required")
    if len({ir.problem_hash for _, ir in cases}) != len(cases):
        raise ValueError("dataset contains duplicate IR problems")
    return cases


def generate_dataset(directory: Path, *, seed: int = 17, train_count: int = 8,
                     validation_count: int = 4, test_count: int = 4) -> dict[str, Path]:
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("all dataset splits must be non-empty")
    directory.mkdir(parents=True, exist_ok=False)
    rng, seen, manifests = random.Random(seed), set(), {}
    for split, count in (("train", train_count), ("validation", validation_count), ("test", test_count)):
        destination = directory / split
        destination.mkdir()
        entries = []
        for i in range(count):
            for _ in range(100):
                wafers = 1 + (i % 2)
                pm_ids = [f"m{j}" for j in range(2 + i % 2)]
                raw = {
                    "Modules": {"io": {"type": "IO", "capacity": wafers},
                                **{mid: {"type": "PM"} for mid in pm_ids}},
                    "ClusterTool": {"r": {"module_ids": ["io", *pm_ids],
                                          "arm_type": "dual_arm" if i % 3 == 2 else "single_arm",
                                          "pick_time": rng.randint(1, 3), "place_time": rng.randint(1, 3),
                                          "travel_times": rng.randint(0, 2)}},
                    "routes": {"route": [
                        {"module_ids": rng.sample(pm_ids, rng.randint(1, len(pm_ids))),
                         "process_time": rng.randint(2, 15)} for _ in range(1 + i % 3)
                    ]},
                    "initial_state": {"robots": {"r": {"position_module_id": "io"}}, "wafers": [
                        {"route_id": "route", "wafer_index": f"0-{wafers-1}", "priority": 0,
                         "location": {"kind": "module", "module_id": "io"}},
                    ]},
                }
                ir = compile_problem(parse_problem(raw), TimeDomain(unit="second", ticks_per_unit=1000))
                if ir.problem_hash not in seen:
                    seen.add(ir.problem_hash)
                    break
            else:
                raise RuntimeError("could not generate another unique problem")
            source, compiled = f"{i:04d}.json", f"{i:04d}.ir.json"
            (destination / source).write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
            (destination / compiled).write_text(ir.canonical_json() + "\n", encoding="utf-8")
            entries.append({"problem_file": source, "ir_file": compiled, "problem_hash": ir.problem_hash})
        path = destination / "manifest.json"
        path.write_text(json.dumps({"format": "ir-training-1", "config": {"split": split, "seed": seed},
                                    "instances": entries}, indent=2) + "\n", encoding="utf-8")
        manifests[split] = path
    return manifests


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory; existing datasets are never overwritten")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=4)
    args = parser.parse_args(argv)
    generate_dataset(args.output, seed=args.seed, train_count=args.train_count,
                     validation_count=args.validation_count, test_count=args.test_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
