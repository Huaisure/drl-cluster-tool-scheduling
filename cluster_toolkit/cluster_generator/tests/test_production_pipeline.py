from __future__ import annotations

from collections import Counter
import json
import random
from pathlib import Path

import pytest
from pydantic import ValidationError

import cluster_toolkit.cluster_generator.heuristic as heuristic_module
import cluster_toolkit.cluster_generator.production as production_module
from cluster_toolkit.cluster_generator.labeling import (
    SolverTask,
    _write_terminal_failure,
    reduce_run,
    run_labeling,
    run_status,
)
from cluster_toolkit.cluster_generator.pipeline_catalog import PipelineCatalog
from cluster_toolkit.cluster_generator.pipeline_models import IntInterval
from cluster_toolkit.cluster_generator.production import (
    _assign_instance_topologies,
    _selected_catalog_topologies,
    load_run,
    materialize_plan,
)
from cluster_toolkit.cluster_generator.production_models import (
    ProductionRunSpec,
    SolverBudgets,
)
from cluster_toolkit.cluster_generator.solutions import (
    SolutionRecord,
    TerminationReason,
)


_REPOSITORY_ROOT = Path(__file__).parents[3]
_ARCHETYPE_IDS = {
    "single_compact",
    "single_parallel",
    "dual_balanced",
    "dual_front_bottleneck",
    "dual_rear_bottleneck",
    "dual_parallel_handoff",
    "triple_balanced",
    "triple_middle_bottleneck",
    "triple_asymmetric",
}


def _source_catalog() -> PipelineCatalog:
    return PipelineCatalog.load(
        _REPOSITORY_ROOT / "topologies",
        _REPOSITORY_ROOT / "recipe_generation_profiles",
    )


def _tiny_spec() -> ProductionRunSpec:
    return ProductionRunSpec(
        run_id="tiny-production-run",
        master_seed=9,
        instance_count=1,
        topology_count=1,
        cell_count_weights={1: 1.0, 2: 0.0, 3: 0.0},
        recipe_counts=(1,),
        wafer_scales=("small",),
        wafer_ranges={"small": IntInterval(minimum=2, maximum=2)},
        periodic_fraction=0.0,
        genetic_seeds=(0,),
        max_parallel_tasks=2,
        budgets=SolverBudgets(
            direct_short_seconds=1,
            direct_long_seconds=1,
            periodic_cycle_short_seconds=1,
            periodic_cycle_long_seconds=1,
            periodic_transition_short_seconds=1,
            periodic_transition_long_seconds=1,
            genetic_seconds=1,
            branch_search_seconds=1,
            hard_kill_grace_seconds=2,
        ),
    )


def test_source_catalog_recursively_loads_file_backed_archetypes() -> None:
    topologies = [
        topology
        for topology in _source_catalog().topologies
        if topology.archetype_id is not None
    ]

    assert len(topologies) == 32
    assert {topology.archetype_id for topology in topologies} == _ARCHETYPE_IDS
    assert len(
        {
            (topology.archetype_id, topology.arm_profile_id)
            for topology in topologies
        }
    ) == 32


def test_run_spec_rejects_wafer_ranges_that_cannot_fit_periodic_ratios() -> None:
    raw = _tiny_spec().model_dump(mode="json")
    raw.update(
        {
            "recipe_counts": [3],
            "periodic_fraction": 0.5,
            "wafer_ranges": {"small": {"minimum": 2, "maximum": 3}},
        }
    )

    with pytest.raises(ValidationError, match="cannot fit periodic ratio"):
        ProductionRunSpec.model_validate(raw)


def test_catalog_configuration_fails_before_creating_run_directory(
    tmp_path: Path,
) -> None:
    raw = _tiny_spec().model_dump(mode="json")
    raw["topology_count"] = 33
    spec = ProductionRunSpec.model_validate(raw)
    run_root = tmp_path / "invalid-run"

    with pytest.raises(ValueError, match="topology_count exceeds"):
        materialize_plan(run_root, spec)

    assert not run_root.exists()


def test_supervisor_failure_records_concrete_solver_version(tmp_path: Path) -> None:
    plan = materialize_plan(tmp_path, _tiny_spec())
    task = SolverTask(
        instance_id=plan.entries[0].instance_id,
        solver_name="cpsat_direct",
        attempt="short",
        seed=0,
        time_limit_seconds=1,
    )

    _write_terminal_failure(
        tmp_path,
        task,
        termination_reason=TerminationReason.INTERRUPTED,
        runtime_seconds=1,
        error="test hard deadline",
    )

    path = next(
        (tmp_path / "instances" / task.instance_id / "solutions").rglob(
            "*.solution.json"
        )
    )
    record = SolutionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    assert record.solver_version == "0.2.0"


def test_topology_selection_uses_catalog_contents_without_code_fallback() -> None:
    source = _source_catalog()
    retained = {
        topology.topology_id: topology
        for topology in source.topologies
        if topology.archetype_id != "triple_asymmetric"
    }
    catalog = PipelineCatalog(
        topologies=retained,
        profiles={
            "atmospheric_linear_default": source.profile(
                "atmospheric_linear_default"
            )
        },
    )
    spec = ProductionRunSpec(
        run_id="catalog-selection",
        master_seed=3,
        instance_count=100,
        topology_count=28,
    )

    selected = _selected_catalog_topologies(spec, catalog)

    assert len(selected) == 28
    assert all(
        topology.archetype_id != "triple_asymmetric" for topology in selected
    )


def test_plan_is_reproducible_strictly_valid_and_saves_no_witness(
    tmp_path: Path,
) -> None:
    spec = _tiny_spec()
    first = materialize_plan(tmp_path, spec)
    repeated = materialize_plan(tmp_path, spec)

    assert first == repeated
    entry = first.entries[0]
    instance_dir = tmp_path / "instances" / entry.instance_id
    metadata = json.loads((instance_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["generation_validation"] == "VALID"
    assert metadata["serial_witness_saved"] is False
    assert metadata["run_id"] == spec.run_id
    assert metadata["master_seed"] == spec.master_seed
    assert metadata["topology_seed"] == entry.topology_seed
    assert metadata["topology_archetype_id"] == "single_compact"
    assert metadata["robot_arm_profile_id"] == entry.robot_arm_profile_id
    assert entry.topology_archetype_id == "single_compact"
    assert entry.robot_arm_profile_id is not None
    assert metadata["instance_seed"] == entry.request.seed
    assert not list(instance_dir.glob("*witness*"))
    source_topology = _source_catalog().topology(entry.request.topology_id)
    run_topology = json.loads(
        (tmp_path / "topologies" / f"{source_topology.topology_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_topology == source_topology.model_dump(mode="json")


def test_plan_does_not_depend_on_json_mapping_order(tmp_path: Path) -> None:
    spec = _tiny_spec()
    first = materialize_plan(tmp_path / "first", spec)
    raw = spec.model_dump(mode="json")
    raw["cell_count_weights"] = dict(
        reversed(list(raw["cell_count_weights"].items()))
    )
    raw["route_pattern_weights"] = dict(
        reversed(list(raw["route_pattern_weights"].items()))
    )
    reloaded = ProductionRunSpec.model_validate(raw)
    second = materialize_plan(tmp_path / "second", reloaded)

    assert first == second


def test_plan_uses_bounded_process_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _tiny_spec().model_copy(
        update={"instance_count": 4, "max_parallel_tasks": 4}
    )
    observed: dict[str, int] = {}

    class RecordingExecutor:
        def __init__(self, *, max_workers, mp_context):
            observed["max_workers"] = max_workers
            observed["start_method_is_spawn"] = int(
                mp_context.get_start_method() == "spawn"
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def map(self, function, tasks):
            materialized_tasks = tuple(tasks)
            observed["task_count"] = len(materialized_tasks)
            return map(function, materialized_tasks)

    monkeypatch.setattr(production_module, "ProcessPoolExecutor", RecordingExecutor)

    plan = materialize_plan(tmp_path, spec)

    assert len(plan.entries) == 4
    assert observed == {
        "max_workers": 4,
        "start_method_is_spawn": 1,
        "task_count": 4,
    }


def test_parallel_plan_matches_single_worker_output(tmp_path: Path) -> None:
    raw = _tiny_spec().model_dump(mode="python")
    raw.update({"instance_count": 4, "max_parallel_tasks": 1})
    serial_spec = ProductionRunSpec.model_validate(raw)
    raw["max_parallel_tasks"] = 4
    parallel_spec = ProductionRunSpec.model_validate(raw)

    serial_plan = materialize_plan(tmp_path / "serial", serial_spec)
    parallel_plan = materialize_plan(tmp_path / "parallel", parallel_spec)

    assert serial_plan == parallel_plan
    for entry in serial_plan.entries:
        serial_instance = tmp_path / "serial" / "instances" / entry.instance_id
        parallel_instance = tmp_path / "parallel" / "instances" / entry.instance_id
        assert (serial_instance / "problem.json").read_bytes() == (
            parallel_instance / "problem.json"
        ).read_bytes()
        assert (serial_instance / "metadata.json").read_bytes() == (
            parallel_instance / "metadata.json"
        ).read_bytes()


def test_plan_validates_each_serial_witness_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_suite = heuristic_module.ValidatorSuite
    validation_count = 0

    class CountingValidatorSuite(original_suite):
        def validate(self, *args, **kwargs):
            nonlocal validation_count
            validation_count += 1
            return super().validate(*args, **kwargs)

    monkeypatch.setattr(heuristic_module, "ValidatorSuite", CountingValidatorSuite)

    materialize_plan(tmp_path, _tiny_spec())

    assert validation_count == 1


def test_run_resume_reduce_and_status_are_idempotent(tmp_path: Path) -> None:
    materialize_plan(tmp_path, _tiny_spec())

    first = run_labeling(tmp_path)
    status = run_status(tmp_path)
    second = run_labeling(tmp_path)

    assert first["short_tasks_completed"] == 5
    assert status["complete"] is True
    assert status["usable_instance_count"] == 1
    assert status["quarantined_instance_count"] == 0
    assert second["short_tasks_completed"] == 0
    assert second["long_tasks_completed"] == 0
    assert reduce_run(tmp_path) == 1


def test_load_run_accepts_schema_v1_plan_without_topology_seed(
    tmp_path: Path,
) -> None:
    plan = materialize_plan(tmp_path, _tiny_spec())
    plan_path = tmp_path / "plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    for entry in payload["entries"]:
        entry.pop("topology_seed")
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    _, reloaded = load_run(tmp_path)

    assert len(reloaded.entries) == len(plan.entries)
    assert all(entry.topology_seed is None for entry in reloaded.entries)


def test_plan_materializes_every_layout_archetype_before_arm_variants(
    tmp_path: Path,
) -> None:
    spec = _tiny_spec().model_copy(
        update={
            "run_id": "all-layout-archetypes",
            "instance_count": 9,
            "topology_count": 9,
            "cell_count_weights": {1: 1.0, 2: 1.0, 3: 1.0},
            "wafer_ranges": {"small": IntInterval(minimum=1, maximum=1)},
        }
    )
    materialize_plan(tmp_path, ProductionRunSpec.model_validate(spec.model_dump()))

    snapshots = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "topologies").glob("*.json")
    ]

    assert len(snapshots) == 9
    assert {snapshot["archetype_id"] for snapshot in snapshots} == _ARCHETYPE_IDS
    assert all(snapshot["arm_profile_id"] for snapshot in snapshots)


def test_large_plan_stratifies_cells_archetypes_and_arm_profiles() -> None:
    spec = ProductionRunSpec(
        run_id="distribution-audit",
        master_seed=23,
        instance_count=10_000,
    )
    topologies = list(_selected_catalog_topologies(spec, _source_catalog()))

    assigned = _assign_instance_topologies(
        spec,
        topologies,
        random.Random(spec.master_seed),
    )

    assert Counter(len(topology.cell_order) for topology in assigned) == {
        1: 5_000,
        2: 4_000,
        3: 1_000,
    }
    archetype_counts = Counter(topology.archetype_id for topology in assigned)
    assert archetype_counts["single_compact"] == 2_500
    assert archetype_counts["single_parallel"] == 2_500
    assert {
        archetype_counts[archetype_id]
        for archetype_id in (
            "dual_balanced",
            "dual_front_bottleneck",
            "dual_rear_bottleneck",
            "dual_parallel_handoff",
        )
    } == {1_000}
    assert max(
        archetype_counts[archetype_id]
        for archetype_id in (
            "triple_balanced",
            "triple_middle_bottleneck",
            "triple_asymmetric",
        )
    ) - min(
        archetype_counts[archetype_id]
        for archetype_id in (
            "triple_balanced",
            "triple_middle_bottleneck",
            "triple_asymmetric",
        )
    ) <= 1
    for archetype_id in _ARCHETYPE_IDS:
        arm_counts = Counter(
            topology.arm_profile_id
            for topology in assigned
            if topology.archetype_id == archetype_id
        )
        assert max(arm_counts.values()) - min(arm_counts.values()) <= 1
