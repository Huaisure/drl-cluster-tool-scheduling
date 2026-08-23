from __future__ import annotations

from collections import Counter
import json
import random
from pathlib import Path

from cluster_toolkit.cluster_generator.labeling import (
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
