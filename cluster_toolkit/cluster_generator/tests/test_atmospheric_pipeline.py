from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from baseline.cpsat.direct import solve_instance as solve_direct_instance
from baseline.cpsat.periodic import solve_periodic_instance
from cluster_toolkit.cluster_generator.pipeline import InstanceGenerator
from cluster_toolkit.cluster_generator.pipeline_catalog import PipelineCatalog
from cluster_toolkit.cluster_generator.heuristic import build_safe_reference_schedule
from cluster_toolkit.cluster_generator.pipeline_models import (
    EquipmentTiming,
    GenerationProvenance,
    InstanceGenerationRequest,
    ModuleKind,
    ModuleTag,
    Recipe,
    RecipeStep,
    RecipeGenerationProfile,
    RobotArmKind,
    RobotTiming,
    SchedulingInstance,
    TopologyModule,
    TopologyRobot,
    TopologyTemplate,
    WorkloadItem,
)
from cluster_toolkit.cluster_generator.problem_adapter import to_cluster_problem
from cluster_toolkit.cluster_generator.topology_family import (
    AtmosphericTopologyRequest,
    generate_atmospheric_topology,
)
from cluster_toolkit.validator import ValidatorSuite


def _topology() -> TopologyTemplate:
    return TopologyTemplate(
        schema_version=2,
        topology_id="atmospheric-linear-2cell",
        topology_version="2.0.0",
        cell_order=("C0", "C1"),
        modules={
            "IO": TopologyModule(kind=ModuleKind.IO, cell_id="C0"),
            "P0": TopologyModule(
                kind=ModuleKind.CHAMBER,
                cell_id="C0",
                tags=(ModuleTag.PROCESS,),
            ),
            "P1": TopologyModule(
                kind=ModuleKind.CHAMBER,
                cell_id="C1",
                tags=(ModuleTag.PROCESS,),
            ),
            "B01": TopologyModule(
                kind=ModuleKind.CHAMBER,
                connected_cell_ids=("C0", "C1"),
                tags=(ModuleTag.BUFFER,),
            ),
        },
        robots={
            "TM0": TopologyRobot(
                cell_id="C0",
                module_ids=("IO", "P0", "B01"),
                arm_kind=RobotArmKind.DUAL,
            ),
            "TM1": TopologyRobot(
                cell_id="C1",
                module_ids=("P1", "B01"),
                arm_kind=RobotArmKind.SINGLE,
            ),
        },
    )


_REPOSITORY_ROOT = Path(__file__).parents[3]
_EXPECTED_LAYOUTS = {
    "single_compact": ((2,), ()),
    "single_parallel": ((6,), ()),
    "dual_balanced": ((3, 3), (1,)),
    "dual_front_bottleneck": ((2, 6), (1,)),
    "dual_rear_bottleneck": ((6, 2), (1,)),
    "dual_parallel_handoff": ((4, 4), (2,)),
    "triple_balanced": ((3, 3, 3), (1, 1)),
    "triple_middle_bottleneck": ((5, 2, 5), (1, 1)),
    "triple_asymmetric": ((2, 4, 6), (1, 2)),
}


def _catalog_archetype_topologies() -> tuple[TopologyTemplate, ...]:
    return tuple(
        TopologyTemplate.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(
            (_REPOSITORY_ROOT / "topologies" / "atmospheric_archetypes").glob(
                "*.json"
            )
        )
    )


def test_fixed_atmospheric_archetypes_match_declared_hardware_layouts() -> None:
    for topology in _catalog_archetype_topologies():
        assert topology.archetype_id is not None
        process_counts, buffer_counts = _EXPECTED_LAYOUTS[topology.archetype_id]

        assert topology.arm_profile_id is not None
        assert len(topology.cell_order) == len(process_counts)
        assert sum(
            ModuleTag.ALIGN in module.effective_tags
            for module in topology.modules.values()
        ) == 1
        for cell_index, expected_count in enumerate(
            process_counts
        ):
            assert sum(
                module.cell_id == f"C{cell_index}"
                and ModuleTag.PROCESS in module.effective_tags
                for module in topology.modules.values()
            ) == expected_count
        for boundary_index, expected_count in enumerate(
            buffer_counts
        ):
            assert sum(
                module.connected_cell_ids
                == (f"C{boundary_index}", f"C{boundary_index + 1}")
                and ModuleTag.BUFFER in module.effective_tags
                for module in topology.modules.values()
            ) == expected_count


def test_all_archetype_arm_variants_are_unique_and_schema_valid() -> None:
    topologies = _catalog_archetype_topologies()
    variants = {
        (topology.archetype_id, topology.arm_profile_id)
        for topology in topologies
    }

    assert len(topologies) == 32
    assert len({topology.topology_id for topology in topologies}) == 32
    assert len(variants) == 32
    assert {topology.archetype_id for topology in topologies} == set(
        _EXPECTED_LAYOUTS
    )


def _instance() -> SchedulingInstance:
    topology = _topology()
    return SchedulingInstance(
        schema_version=2,
        instance_id="atmospheric-instance",
        topology=topology,
        timing=EquipmentTiming(
            robots={
                robot_id: RobotTiming(pick_time=1, place_time=1, travel_time=1)
                for robot_id in topology.robots
            }
        ),
        recipes=(
            Recipe(
                recipe_id="R0",
                steps=(
                    RecipeStep(
                        step_id="S0",
                        candidate_module_ids=("P0",),
                        process_time=3,
                    ),
                    RecipeStep(
                        step_id="T0",
                        candidate_module_ids=("B01",),
                        process_time=0,
                    ),
                    RecipeStep(
                        step_id="S1",
                        candidate_module_ids=("P1",),
                        process_time=4,
                    ),
                    RecipeStep(
                        step_id="T1",
                        candidate_module_ids=("B01",),
                        process_time=0,
                    ),
                    RecipeStep(
                        step_id="S2",
                        candidate_module_ids=("P0",),
                        process_time=2,
                    ),
                ),
            ),
        ),
        workload=(WorkloadItem(recipe_id="R0", wafer_count=2),),
        source_module_id="IO",
        sink_module_id="IO",
        provenance=GenerationProvenance(
            generator_name="test",
            generator_version="2.0.0",
            seed=1,
            profile_id="test",
            profile_version="2.0.0",
            wafer_scale="small",
        ),
    )


def test_v2_atmospheric_route_runs_through_engine_and_strict_validator() -> None:
    problem = to_cluster_problem(_instance())

    result = build_safe_reference_schedule(problem)
    report = ValidatorSuite(problem).validate(
        result.actions,
        require_complete=True,
        exact_action_durations=True,
    )

    assert report.ok
    assert {action["tm_id"] for action in result.actions} == {"TM0", "TM1"}


def test_v2_rejects_candidate_modules_that_cross_cells() -> None:
    instance = _instance()
    recipe = instance.recipes[0].model_copy(
        update={
            "steps": (
                RecipeStep(
                    step_id="S0",
                    candidate_module_ids=("P0", "P1"),
                    process_time=3,
                ),
            )
        }
    )

    with pytest.raises(ValidationError, match="candidates must stay in one Cell"):
        instance.model_copy(update={"recipes": (recipe,)}, deep=True).__class__(
            **instance.model_copy(update={"recipes": (recipe,)}).model_dump()
        )


def test_robot_handoff_is_rejected_outside_buffer() -> None:
    problem = to_cluster_problem(_instance())
    result = build_safe_reference_schedule(problem)
    actions = [dict(action) for action in result.actions]
    pick_from_p0 = next(
        action
        for action in actions
        if action["action_type"] == "pick"
        and action["module_id"] == "P0"
        and action["step_index"] == 1
    )
    pick_from_p0["tm_id"] = "TM1"

    report = ValidatorSuite(problem).validate(actions, require_complete=True)

    assert any(
        issue.constraint_id in {"robot.reachability", "wafer.process_order"}
        for issue in report.issues
    )


def test_non_pm_visit_duration_blocks_early_pick() -> None:
    instance = _instance()
    steps = list(instance.recipes[0].steps)
    steps[1] = steps[1].model_copy(update={"process_time": 5})
    recipe = instance.recipes[0].model_copy(update={"steps": tuple(steps)})
    updated = SchedulingInstance.model_validate(
        instance.model_copy(update={"recipes": (recipe,)}).model_dump()
    )
    problem = to_cluster_problem(updated)
    actions = [dict(action) for action in build_safe_reference_schedule(problem).actions]
    pick = next(
        action
        for action in actions
        if action["action_type"] == "pick"
        and action["module_id"] == "B01"
        and action["step_index"] == 2
    )
    pick["start"] -= 1

    report = ValidatorSuite(problem).validate(actions)

    assert any(
        issue.constraint_id == "wafer.interval_overlap"
        for issue in report.issues
    )


def test_strict_validator_rejects_incomplete_schedule() -> None:
    problem = to_cluster_problem(_instance())
    actions = [
        dict(action)
        for action in build_safe_reference_schedule(problem).actions
        if action["wafer_index"] == 0
    ]

    assert ValidatorSuite(problem).validate(actions).ok
    strict = ValidatorSuite(problem).validate(actions, require_complete=True)
    assert any(issue.constraint_id == "wafer.completeness" for issue in strict.issues)


def _atmospheric_profile() -> RecipeGenerationProfile:
    return RecipeGenerationProfile(
        schema_version=2,
        profile_id="atmospheric-test",
        profile_version="2.0.0",
        applies_to=(),
        applies_to_families=("atmospheric_linear",),
        compiler="atmospheric_linear",
        pm_step_count_weights={4: 1.0},
        candidate_pm_count_weights={1: 0.5, 2: 0.5},
        reentry_probability=0.0,
        process_time_anchor_weights={10: 1.0},
        process_time_jitter=0.0,
        robot_time={"minimum": 1, "maximum": 1},
        alignment_probability=1.0,
        alignment_time={"minimum": 2, "maximum": 2},
        buffer_hold_time={"minimum": 0, "maximum": 1},
        route_pattern_weights={"multi_transition": 1.0},
    )


@pytest.mark.parametrize("cell_count", [1, 2, 3])
def test_atmospheric_family_compiles_full_executable_routes(cell_count: int) -> None:
    topology = generate_atmospheric_topology(
        AtmosphericTopologyRequest(seed=cell_count, cell_count=cell_count)
    )
    profile = _atmospheric_profile()
    generated = InstanceGenerator(
        PipelineCatalog(
            topologies={topology.topology_id: topology},
            profiles={profile.profile_id: profile},
        )
    ).generate(
        InstanceGenerationRequest(
            topology_id=topology.topology_id,
            profile_id=profile.profile_id,
            recipe_count=2,
            wafer_scale="small",
            seed=100 + cell_count,
            periodic_ratio=(1, 1),
        )
    )

    assert generated.instance.schema_version == 2
    assert generated.metadata["periodic_eligible"] is True
    domains = [
        frozenset(step.candidate_module_ids)
        for recipe in generated.instance.recipes
        for step in recipe.steps
    ]
    assert all(
        not left.intersection(right) or left == right
        for index, left in enumerate(domains)
        for right in domains[index + 1 :]
    )
    if cell_count > 1:
        assert any(
            ModuleTag.BUFFER
            in topology.modules[step.candidate_module_ids[0]].effective_tags
            for recipe in generated.instance.recipes
            for step in recipe.steps
        )

    problem = to_cluster_problem(generated.instance)
    reference = build_safe_reference_schedule(problem)
    assert ValidatorSuite(problem).validate(
        reference.actions,
        require_complete=True,
        exact_action_durations=True,
    ).ok


@pytest.mark.parametrize(
    "topology",
    _catalog_archetype_topologies(),
    ids=lambda topology: f"{topology.archetype_id}-{topology.arm_profile_id}",
)
def test_every_archetype_generates_a_strictly_valid_instance(
    topology: TopologyTemplate,
) -> None:
    profile = _atmospheric_profile()
    generated = InstanceGenerator(
        PipelineCatalog(
            topologies={topology.topology_id: topology},
            profiles={profile.profile_id: profile},
        ),
        wafer_ranges={"small": (1, 1)},
    ).generate(
        InstanceGenerationRequest(
            topology_id=topology.topology_id,
            profile_id=profile.profile_id,
            recipe_count=1,
            wafer_scale="small",
            seed=41,
            route_pattern=(
                "local" if len(topology.cell_order) == 1 else "multi_transition"
            ),
        )
    )

    problem = to_cluster_problem(generated.instance)
    reference = build_safe_reference_schedule(problem)

    assert ValidatorSuite(problem).validate(
        reference.actions,
        require_complete=True,
        exact_action_durations=True,
    ).ok


def test_atmospheric_topology_generation_is_reproducible_and_bounded() -> None:
    request = AtmosphericTopologyRequest(seed=7, cell_count=3)
    first = generate_atmospheric_topology(request)
    repeated = generate_atmospheric_topology(request)

    assert first == repeated
    assert len(first.cell_order) == 3
    assert len(first.robots) == 3
    for cell_id in first.cell_order:
        process_count = sum(
            module.cell_id == cell_id
            and ModuleTag.PROCESS in module.effective_tags
            for module in first.modules.values()
        )
        assert 2 <= process_count <= 6


def test_direct_cpsat_uses_buffer_for_multi_robot_handoff() -> None:
    result = solve_direct_instance(
        _instance(),
        time_limit_seconds=3,
        random_seed=1,
        num_search_workers=1,
    )

    assert result.status in {"FEASIBLE", "OPTIMAL"}
    assert result.validation_ok is True
    assert {action["tm_id"] for action in result.actions} == {"TM0", "TM1"}


def test_periodic_cpsat_uses_buffer_for_multi_robot_handoff() -> None:
    instance = _instance()
    instance = instance.model_copy(
        update={
            "provenance": instance.provenance.model_copy(
                update={"periodic_ratio": (1,)}
            )
        }
    )
    result = solve_periodic_instance(
        instance,
        time_limit_seconds=5,
        random_seed=1,
        num_search_workers=1,
    )

    assert result.status == "FEASIBLE"
    assert result.validation_ok is True
    assert result.cycle is not None
    assert {action["tm_id"] for action in result.actions} == {"TM0", "TM1"}


def test_direct_cpsat_supports_multiple_capacity_one_buffer_candidates() -> None:
    instance = _instance()
    topology = instance.topology
    modules = {
        **topology.modules,
        "B02": TopologyModule(
            kind=ModuleKind.CHAMBER,
            connected_cell_ids=("C0", "C1"),
            tags=(ModuleTag.BUFFER,),
        ),
    }
    robots = {
        robot_id: robot.model_copy(
            update={"module_ids": (*robot.module_ids, "B02")}
        )
        for robot_id, robot in topology.robots.items()
    }
    topology = TopologyTemplate.model_validate(
        topology.model_copy(update={"modules": modules, "robots": robots}).model_dump()
    )
    steps = tuple(
        step.model_copy(
            update={"candidate_module_ids": ("B01", "B02")}
        )
        if step.candidate_module_ids == ("B01",)
        else step
        for step in instance.recipes[0].steps
    )
    recipe = instance.recipes[0].model_copy(update={"steps": steps})
    instance = SchedulingInstance.model_validate(
        instance.model_copy(
            update={"topology": topology, "recipes": (recipe,)}
        ).model_dump()
    )

    result = solve_direct_instance(
        instance,
        time_limit_seconds=3,
        random_seed=2,
        num_search_workers=1,
    )

    assert result.status in {"FEASIBLE", "OPTIMAL"}
    assert result.validation_ok is True
    assert all(
        module.capacity == 1
        for module in to_cluster_problem(instance).Modules.values()
        if module.type.value != "IO"
    )
