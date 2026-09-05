"""Prepare solver-labelled pipeline runs and replay them as IR supervision."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Callable, Iterable
from urllib.parse import quote, unquote

from cluster_toolkit.cluster_generator.pipeline_models import SchedulingInstance
from cluster_toolkit.cluster_generator.problem_adapter import to_cluster_problem
from cluster_toolkit.constraint_ir import ConstraintIRV1, TimeDomain, compile_problem

from .env import IRSchedulingEnv
from .graph import IRGraph


DATASET_FORMAT = "ir-sft-1"


@dataclass(frozen=True)
class SFTCase:
    path: Path
    problem: ConstraintIRV1
    actions_path: Path
    actions: tuple[dict[str, object], ...]
    metadata: dict[str, object]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_actions(path: Path, expected_sha256: str | None = None) -> tuple[dict[str, object], ...]:
    raw = gzip.decompress(path.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"expert action hash mismatch: {path}")
    actions = json.loads(raw)
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"expert action file is empty or invalid: {path}")
    return tuple(actions)


def _candidate_key(candidate) -> tuple[str, int, str, str, str, str]:
    parts = candidate.operator_template_id.split("/")
    if len(parts) != 8 or parts[0] != "operator":
        raise ValueError(
            f"unsupported selectable operator id: {candidate.operator_template_id}"
        )
    _, kind, phase, module_id, robot_id, _, _, _ = map(unquote, parts)
    bindings = {item.parameter: item.value for item in candidate.bindings}
    return kind, int(phase), module_id, robot_id, bindings["wafer"], bindings["holder"]


def _action_key(action: dict[str, object]) -> tuple[str, int, str, str, str, str]:
    kind = str(action["action_type"]).lower()
    step_index = int(action["step_index"])
    phase = 2 * step_index if kind == "pick" else 2 * step_index - 1
    module_id = str(action["module_id"])
    robot_id = str(action["tm_id"])
    owner = "/".join((
        "wafer",
        quote(str(action["route_id"]), safe=""),
        quote(str(action["wafer_index"]), safe=""),
    ))
    holder = "/".join(("module", quote(module_id, safe="")))
    return kind, phase, module_id, robot_id, owner, holder


def replay_expert(
    problem: ConstraintIRV1,
    actions: Iterable[dict[str, object]],
    *,
    on_choice: Callable[[IRGraph, int], None] | None = None,
    on_choice_set: Callable[[IRGraph, tuple[int, ...]], None] | None = None,
    on_choice_context: Callable[
        [IRGraph, tuple[int, ...], IRSchedulingEnv], None
    ] | None = None,
    max_decisions: int = 10000,
) -> dict[str, object]:
    """Replay module/robot/wafer expert choices through the generic IR session.

    Independent operations need not have the same serialization as the source
    solver. At every IR frame we therefore commit the earliest still-unconsumed
    expert action that is currently legal. Explicit Wait labels are inserted
    only when no remaining expert operation is legal before the next event.
    """

    remaining = sorted([
        (_action_key(action), index, float(action.get("start", index)))
        for index, action in enumerate(actions)
    ], key=lambda item: (item[2], item[1]))
    env = IRSchedulingEnv(
        problem,
        max_decisions=max_decisions,
        encode_observations=(
            on_choice is not None
            or on_choice_set is not None
            or on_choice_context is not None
        ),
    )
    observation, _ = env.reset(seed=0)
    choices = waits = committed = 0
    while env.reason is None:
        candidate_keys = [_candidate_key(candidate) for candidate in env.frame.intents]
        selected: tuple[int, int] | None = None
        acceptable: list[int] = []
        if remaining:
            earliest_start = remaining[0][2]
            for remaining_index, (key, _, start) in enumerate(remaining):
                if start != earliest_start:
                    break
                try:
                    action_index = candidate_keys.index(key)
                except ValueError:
                    continue
                acceptable.append(action_index)
                if selected is None:
                    selected = remaining_index, action_index
        if selected is None:
            if env.wait_tick is None:
                preview = [item[0] for item in remaining[:3]]
                raise ValueError(
                    "expert replay has no matching legal action and cannot wait; "
                    f"tick={env.snapshot.tick}, remaining={preview}"
                )
            action_index = len(env.frame.intents)
            acceptable = [action_index]
            waits += 1
        else:
            remaining_index, action_index = selected
            remaining.pop(remaining_index)
            committed += 1
        action_count = len(env.frame.intents) + (env.wait_tick is not None)
        if action_count > 1:
            choices += 1
            if on_choice is not None:
                assert observation is not None
                on_choice(observation, action_index)
            if on_choice_set is not None:
                assert observation is not None
                on_choice_set(observation, tuple(sorted(set(acceptable))))
            if on_choice_context is not None:
                assert observation is not None
                on_choice_context(
                    observation,
                    tuple(sorted(set(acceptable))),
                    env,
                )
        observation, _, _, _, _ = env.step(action_index)
    if env.reason != "success" or remaining:
        raise ValueError(
            f"expert replay ended as {env.reason} with {len(remaining)} actions remaining"
        )
    audit = env.audit()
    if not audit.ok:
        raise ValueError(f"expert IR replay audit failed: {audit.issues}")
    return {
        "success": True,
        "termination_reason": env.reason,
        "ir_makespan": env.snapshot.tick / problem.time_domain.ticks_per_unit,
        "expert_action_count": committed,
        "inserted_wait_count": waits,
        "supervised_choice_count": choices,
    }


def verify_expert_coverage(
    problem: ConstraintIRV1,
    actions: Iterable[dict[str, object]],
) -> None:
    """Check that every source action has a corresponding compiled IR binding."""

    domains = {item.id: item for item in problem.binding_domains}
    available: set[tuple[str, int, str, str, str, str]] = set()
    for rule in problem.dynamic_intents:
        domain = domains[rule.binding_domain_id]
        for row in domain.rows:
            bindings = dict(zip((item.name for item in domain.parameters), row.values))
            parts = rule.operator_template_id.split("/")
            if len(parts) != 8 or parts[0] != "operator":
                continue
            _, kind, phase, module_id, robot_id, _, _, _ = map(unquote, parts)
            available.add((
                kind,
                int(phase),
                module_id,
                robot_id,
                bindings["wafer"],
                bindings["holder"],
            ))
    missing = [key for key in map(_action_key, actions) if key not in available]
    if missing:
        raise ValueError(f"{len(missing)} expert actions are not representable in IR: {missing[:3]}")


def load_sft_cases(
    paths: list[Path],
    *,
    expected_split: str | None = None,
    limit: int | None = None,
    max_wafer_count: int | None = None,
) -> list[SFTCase]:
    pending: list[tuple[Path, dict[str, object]]] = []
    for manifest_path in paths:
        manifest = _read_json(manifest_path)
        if manifest.get("format") != DATASET_FORMAT:
            raise ValueError(f"not an {DATASET_FORMAT} manifest: {manifest_path}")
        split = manifest.get("config", {}).get("split")
        if expected_split is not None and split != expected_split:
            raise ValueError(
                f"expected {expected_split} manifest, got {split}: {manifest_path}"
            )
        for entry in manifest["instances"]:
            if max_wafer_count is not None and int(entry["wafer_count"]) > max_wafer_count:
                continue
            pending.append((manifest_path.parent.resolve(), dict(entry)))
    pending.sort(key=lambda item: (
        int(item[1]["wafer_count"]),
        int(item[1]["topology_cell_count"]),
        str(item[1]["instance_id"]),
    ))
    if limit is not None:
        groups: dict[tuple[object, object], list[tuple[Path, dict[str, object]]]] = defaultdict(list)
        for item in pending:
            groups[(item[1]["expert_solver"], item[1]["topology_cell_count"])].append(item)
        balanced: list[tuple[Path, dict[str, object]]] = []
        ordered_groups = [groups[key] for key in sorted(groups, key=str)]
        while len(balanced) < limit and any(ordered_groups):
            for group in ordered_groups:
                if group and len(balanced) < limit:
                    balanced.append(group.pop(0))
        pending = balanced
    cases: list[SFTCase] = []
    seen: set[str] = set()
    for base, entry in pending:
        problem_path = (base / entry["problem_file"]).resolve()
        actions_path = (base / entry["expert_actions_file"]).resolve()
        if problem_path.parent != base or actions_path.parent != base:
            raise ValueError("SFT manifest files must be in the manifest directory")
        raw = _read_json(problem_path)
        instance = SchedulingInstance.model_validate(raw)
        problem = compile_problem(
            to_cluster_problem(instance),
            TimeDomain(unit="second", ticks_per_unit=1),
        )
        if problem.problem_hash != entry["problem_hash"]:
            raise ValueError(f"problem hash mismatch: {problem_path}")
        if problem.problem_hash in seen:
            raise ValueError("SFT dataset contains duplicate IR problems")
        seen.add(problem.problem_hash)
        actions = _read_actions(actions_path, entry.get("actions_sha256"))
        cases.append(SFTCase(problem_path, problem, actions_path, actions, entry))
    if not cases:
        raise ValueError("a non-empty SFT dataset is required")
    return cases


def _valid_solver_makespans(instance_dir: Path) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for path in (instance_dir / "solutions").glob("*/*.solution.json"):
        record = _read_json(path)
        if (
            record.get("validation_status") == "VALID"
            and record.get("solution_status") in {"FEASIBLE", "OPTIMAL"}
            and record.get("makespan") is not None
        ):
            values[str(record["solver_name"])].append(float(record["makespan"]))
    return {name: min(items) for name, items in values.items()}


def _assign_splits(entries: list[dict[str, object]], seed: int) -> dict[str, list[dict[str, object]]]:
    strata: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        strata[(entry["wafer_scale"], entry["topology_cell_count"], entry["expert_solver"])].append(entry)
    splits = {"train": [], "validation": [], "test": []}
    for key in sorted(strata, key=str):
        rows = strata[key]
        rows.sort(key=lambda item: hashlib.sha256(
            f"{seed}:{item['instance_id']}".encode("utf-8")
        ).hexdigest())
        count = len(rows)
        validation_count = max(1, round(count * 0.1)) if count >= 3 else 0
        test_count = max(1, round(count * 0.1)) if count >= 3 else 0
        if validation_count + test_count >= count:
            validation_count = test_count = 0
        splits["validation"].extend(rows[:validation_count])
        splits["test"].extend(rows[validation_count:validation_count + test_count])
        splits["train"].extend(rows[validation_count + test_count:])
    for name in splits:
        splits[name].sort(key=lambda item: str(item["instance_id"]))
    if not all(splits.values()):
        raise ValueError("not enough compatible instances for three non-empty splits")
    return splits


def prepare_dataset(run_root: Path, output: Path, *, seed: int = 1701) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rejected: list[dict[str, str]] = []
    accepted: list[dict[str, object]] = []
    instance_root = run_root / "instances"
    for index_path in sorted(instance_root.glob("*/solution_index.json")):
        index = _read_json(index_path)
        if not index.get("usable"):
            continue
        instance_dir = index_path.parent
        try:
            metadata = _read_json(instance_dir / "metadata.json")
            problem_path = instance_dir / "problem.json"
            instance = SchedulingInstance.model_validate(_read_json(problem_path))
            ir = compile_problem(
                to_cluster_problem(instance),
                TimeDomain(unit="second", ticks_per_unit=1),
            )
            solution_path = instance_dir / str(index["best_solution_file"])
            solution = _read_json(solution_path)
            actions_path = instance_dir / str(solution["actions_file"])
            actions = _read_actions(actions_path, solution.get("actions_sha256"))
            verify_expert_coverage(ir, actions)
            solver_makespans = _valid_solver_makespans(instance_dir)
            accepted.append({
                "instance_id": instance.instance_id,
                "problem_hash": ir.problem_hash,
                "source_problem": problem_path,
                "source_actions": actions_path,
                "actions_sha256": solution["actions_sha256"],
                "expert_solver": solution["solver_name"],
                "expert_solution_id": solution["solution_id"],
                "expert_makespan": float(solution["makespan"]),
                "branch_search_makespan": solver_makespans.get("branch_search"),
                "genetic_makespan": solver_makespans.get("genetic"),
                "wafer_scale": metadata["wafer_scale"],
                "wafer_count": metadata["wafer_count"],
                "topology_cell_count": metadata["topology_cell_count"],
                "topology_archetype_id": metadata["topology_archetype_id"],
                "expert_action_count": len(actions),
                "static_ir_coverage": True,
            })
        except Exception as error:  # Preserve every rejection for dataset audit.
            rejected.append({
                "instance_id": str(index.get("instance_id", instance_dir.name)),
                "error": f"{type(error).__name__}: {error}",
            })
    splits = _assign_splits(accepted, seed)
    manifests = {}
    for split, rows in splits.items():
        destination = output / split
        destination.mkdir()
        materialized = []
        for row in rows:
            instance_id = str(row["instance_id"])
            problem_name = f"{instance_id}.problem.json"
            actions_name = f"{instance_id}.expert.actions.json.gz"
            shutil.copyfile(row.pop("source_problem"), destination / problem_name)
            shutil.copyfile(row.pop("source_actions"), destination / actions_name)
            materialized.append({
                **row,
                "problem_file": problem_name,
                "expert_actions_file": actions_name,
            })
        manifest = {
            "format": DATASET_FORMAT,
            "config": {"split": split, "seed": seed},
            "instances": materialized,
        }
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifests[split] = str(manifest_path)
    summary = {
        "format": DATASET_FORMAT,
        "source_run": str(run_root.resolve()),
        "seed": seed,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "best_expert_counts": {
            solver: sum(row["expert_solver"] == solver for row in accepted)
            for solver in ("genetic", "branch_search")
        },
        "manifests": manifests,
        "rejections": rejected,
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# IR SFT dataset\n\n"
        "This immutable dataset was selected from validated GA/Branch Search "
        "incumbents. Splits are disjoint by generated instance and stratified "
        "by scale, cell count, and winning expert. `dataset_summary.json` "
        "records every compatibility rejection. Expert actions retain their "
        "original SHA-256. Inclusion checks that every action is representable "
        "by the compiled IR; training replays the complete trajectory through "
        "the IR kernel and independently audits its terminal state.\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_dataset(args.run_root, args.output, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
