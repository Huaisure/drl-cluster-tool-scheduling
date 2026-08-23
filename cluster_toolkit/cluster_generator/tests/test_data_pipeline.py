from __future__ import annotations

import json
import gzip
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from cluster_toolkit.cluster_generator import (
    GlobalOptimalityStatus,
    InstanceCorpus,
    InstanceGenerationRequest,
    InstanceGenerator,
    InstanceSolutions,
    PipelineCatalog,
    SolutionRecord,
    SolverStatus,
    TerminationReason,
    ValidationStatus,
    WorkflowStatus,
    build_safe_reference_schedule,
    to_cluster_problem,
)
from cluster_toolkit.problem import ModuleType
from cluster_toolkit.validator import ValidatorSuite
from cluster_toolkit.cluster_generator.solutions import ComponentResult


REPOSITORY_ROOT = Path(__file__).parents[3]


@pytest.fixture
def generator() -> InstanceGenerator:
    catalog = PipelineCatalog.load(
        REPOSITORY_ROOT / "topologies",
        REPOSITORY_ROOT / "recipe_generation_profiles",
    )
    return InstanceGenerator(catalog)


def _request(**overrides) -> InstanceGenerationRequest:
    values = {
        "topology_id": "direct_single_cell_4pm_dual_arm",
        "profile_id": "direct_single_cell_default",
        "recipe_count": 2,
        "wafer_scale": "small",
        "seed": 17,
        "periodic_ratio": None,
    }
    values.update(overrides)
    return InstanceGenerationRequest(**values)


def test_new_schema_is_reproducible_and_has_no_capability_matrix(
    generator: InstanceGenerator,
) -> None:
    first = generator.generate(_request())
    repeated = generator.generate(_request())
    different = generator.generate(_request(seed=18))

    assert first.instance.model_dump_json() == repeated.instance.model_dump_json()
    assert first.instance_id == repeated.instance_id
    assert first.instance_id != different.instance_id

    raw = first.instance.model_dump(mode="json")
    encoded = json.dumps(raw)
    assert "process_ids" not in encoded
    assert "capability" not in encoded
    assert raw["objective"] == "makespan"
    assert all(item["priority"] == 0 for item in raw["workload"])
    assert all(item["release_time"] == 0 for item in raw["workload"])
    assert all(
        isinstance(step["process_time"], int)
        for recipe in raw["recipes"]
        for step in recipe["steps"]
    )


@pytest.mark.parametrize(
    "ratio",
    [
        (1, 1, 1),
        (1, 2, 1),
        (2, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 2, 1),
        (2, 1, 2),
    ],
)
def test_three_recipe_periodic_workload_is_an_exact_multiple(
    generator: InstanceGenerator,
    ratio: tuple[int, int, int],
) -> None:
    generated = generator.generate(
        _request(recipe_count=3, periodic_ratio=ratio, seed=sum(ratio))
    )
    counts = tuple(item.wafer_count for item in generated.instance.workload)
    factors = {count // part for count, part in zip(counts, ratio, strict=True)}

    assert len(factors) == 1
    assert counts == tuple(next(iter(factors)) * part for part in ratio)
    assert generated.metadata["periodic_eligible"] is True


def test_unsupported_periodic_ratio_is_rejected(generator: InstanceGenerator) -> None:
    with pytest.raises(ValueError, match="unsupported periodic ratio"):
        generator.generate(_request(periodic_ratio=(1, 3)))


def test_adapter_preserves_instance_execution_facts(
    generator: InstanceGenerator,
) -> None:
    instance = generator.generate(_request(recipe_count=3, periodic_ratio=(1, 2, 1))).instance

    problem = to_cluster_problem(instance)

    assert problem.schema_version == 1
    assert problem.meta["adapter"]["source_instance_id"] == instance.instance_id
    assert all(not module.process_ids for module in problem.Modules.values())
    assert problem.Modules[instance.source_module_id].capacity == sum(
        item.wafer_count for item in instance.workload
    )
    assert all(
        module.capacity == 1
        for module_id, module in problem.Modules.items()
        if module_id != instance.source_module_id
    )
    assert {
        module_id: module.type
        for module_id, module in problem.Modules.items()
    } == {
        module_id: ModuleType(module.kind.value)
        for module_id, module in instance.topology.modules.items()
    }

    for robot_id, robot in instance.topology.robots.items():
        adapted = problem.ClusterTool[robot_id]
        timing = instance.timing.robots[robot_id]
        assert adapted.module_ids == robot.module_ids
        assert adapted.arm_type.value == robot.arm_kind.value
        assert adapted.travel_times == timing.travel_time
        assert adapted.pick_time == timing.pick_time
        assert adapted.place_time == timing.place_time
        assert problem.initial_state.robots[robot_id].position_module_id is None

    for recipe in instance.recipes:
        visits = problem.routes[recipe.recipe_id].visits
        assert len(visits) == len(recipe.steps)
        for visit, step in zip(visits, recipe.steps, strict=True):
            assert visit.module_ids == step.candidate_module_ids
            assert visit.process_time == step.process_time
            assert visit.process_id is None

    workload = {item.recipe_id: item.wafer_count for item in instance.workload}
    adapted_wafer_keys = {wafer.wafer_key for wafer in problem.initial_state.wafers}
    expected_wafer_keys = {
        (recipe_id, wafer_index)
        for recipe_id, count in workload.items()
        for wafer_index in range(count)
    }
    assert adapted_wafer_keys == expected_wafer_keys
    assert all(
        wafer.location.module_id == instance.source_module_id
        for wafer in problem.initial_state.wafers
    )


def test_adapted_problem_runs_through_existing_engine_and_validator(
    generator: InstanceGenerator,
) -> None:
    instance = generator.generate(_request(recipe_count=1, seed=5)).instance
    problem = to_cluster_problem(instance)

    reference = build_safe_reference_schedule(problem)

    assert reference.actions
    assert reference.makespan > 0
    assert reference.makespan.is_integer()
    assert ValidatorSuite(problem).validate(reference.actions).ok


def test_corpus_uses_one_immutable_directory_per_problem(
    generator: InstanceGenerator,
    tmp_path: Path,
) -> None:
    generated = generator.generate(_request())
    corpus = InstanceCorpus(tmp_path)

    instance_dir = corpus.materialize(generated)
    repeated = corpus.materialize(generated)

    assert repeated == instance_dir
    assert (instance_dir / "problem.json").is_file()
    assert (instance_dir / "metadata.json").is_file()
    for solver_name in InstanceCorpus.SOLVER_DIRECTORIES:
        assert (instance_dir / "solutions" / solver_name).is_dir()


def test_solution_reducer_ignores_invalid_schedules_and_preserves_proof_status(
    generator: InstanceGenerator,
    tmp_path: Path,
) -> None:
    generated = generator.generate(_request())
    instance_dir = InstanceCorpus(tmp_path).materialize(generated)
    solutions = InstanceSolutions(instance_dir)

    solutions.write(
        SolutionRecord(
            instance_id=generated.instance_id,
            solution_id="periodic-0",
            solver_name="cpsat_periodic",
            solver_version="0.1",
            solver_config_hash="periodic-config",
            seed=0,
            status=SolverStatus.FEASIBLE,
            validation_status=ValidationStatus.VALID,
            validator_name="validator_suite",
            validator_version="0.1",
            makespan=120,
            runtime_seconds=10,
            time_limit_seconds=1800,
            components={
                "cycle": ComponentResult(
                    status=SolverStatus.OPTIMAL,
                    objective=20,
                    best_bound=20,
                    runtime_seconds=2,
                )
            },
        )
    )
    best_path = solutions.write(
        SolutionRecord(
            instance_id=generated.instance_id,
            solution_id="direct-0",
            solver_name="cpsat_direct",
            solver_version="0.1",
            solver_config_hash="direct-config",
            seed=0,
            status=SolverStatus.OPTIMAL,
            validation_status=ValidationStatus.VALID,
            validator_name="validator_suite",
            validator_version="0.1",
            global_optimality_status=GlobalOptimalityStatus.PROVEN_OPTIMAL,
            makespan=110,
            best_bound=110,
            best_bound_scope="full_problem",
            runtime_seconds=20,
            time_limit_seconds=1800,
        )
    )
    solutions.write(
        SolutionRecord(
            instance_id=generated.instance_id,
            solution_id="invalid-0",
            solver_name="genetic",
            solver_version="0.1",
            solver_config_hash="genetic-config",
            seed=0,
            status=SolverStatus.FEASIBLE,
            validation_status=ValidationStatus.INVALID,
            validator_name="validator_suite",
            validator_version="0.1",
            makespan=90,
            runtime_seconds=5,
        )
    )

    index = solutions.reduce()

    assert index.record_count == 3
    assert index.valid_solution_count == 2
    assert index.best_makespan == 110
    assert index.best_solution_file == best_path.relative_to(instance_dir).as_posix()
    assert index.best_bound == 110
    assert index.certified_gap == 0
    assert index.global_optimality_status is GlobalOptimalityStatus.PROVEN_OPTIMAL


def test_component_optimality_cannot_be_declared_global() -> None:
    with pytest.raises(ValidationError, match="full-problem status OPTIMAL"):
        SolutionRecord(
            instance_id="instance-1",
            solution_id="periodic-0",
            solver_name="cpsat_periodic",
            solver_version="0.1",
            solver_config_hash="config",
            seed=0,
            status=SolverStatus.FEASIBLE,
            validation_status=ValidationStatus.VALID,
            validator_name="validator_suite",
            validator_version="0.1",
            global_optimality_status=GlobalOptimalityStatus.PROVEN_OPTIMAL,
            makespan=100,
            best_bound=100,
            best_bound_scope="full_problem",
            runtime_seconds=1,
            components={
                "cycle": ComponentResult(
                    status=SolverStatus.OPTIMAL,
                    objective=10,
                    best_bound=10,
                    runtime_seconds=1,
                )
            },
        )


def test_solution_actions_are_deterministically_compressed_and_hashed(
    generator: InstanceGenerator,
    tmp_path: Path,
) -> None:
    generated = generator.generate(_request())
    instance_dir = InstanceCorpus(tmp_path).materialize(generated)
    solutions = InstanceSolutions(instance_dir)
    actions = [
        {
            "action_type": "pick",
            "tm_id": "TM1",
            "module_id": "IO",
            "route_id": "R0",
            "wafer_index": 0,
            "step_index": 0,
            "start": 0,
            "end": 1,
        }
    ]
    path = solutions.write(
        SolutionRecord(
            instance_id=generated.instance_id,
            solution_id="compressed",
            solver_name="genetic",
            solver_version="test",
            solver_config_hash="hash",
            seed=0,
            status=SolverStatus.FEASIBLE,
            termination_reason=TerminationReason.NORMAL,
            validation_status=ValidationStatus.VALID,
            workflow_status=WorkflowStatus.TERMINAL,
            validator_name="validator_suite",
            validator_version="2",
            makespan=1,
            runtime_seconds=0.1,
        ),
        actions,
    )

    record = SolutionRecord.model_validate_json(path.read_text(encoding="utf-8"))
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["solution_status"] == "FEASIBLE"
    assert "status" not in persisted
    assert record.actions_file is not None
    compressed = instance_dir / record.actions_file
    raw = gzip.decompress(compressed.read_bytes())
    assert json.loads(raw) == actions
    assert record.action_count == 1
    assert record.actions_sha256 == hashlib.sha256(raw).hexdigest()


def test_not_eligible_is_an_orthogonal_terminal_state() -> None:
    record = SolutionRecord(
        instance_id="instance",
        solution_id="not-eligible",
        solver_name="cpsat_periodic",
        solver_version="test",
        solver_config_hash="hash",
        seed=0,
        status=SolverStatus.UNKNOWN,
        termination_reason=TerminationReason.NOT_ELIGIBLE,
        validation_status=ValidationStatus.NOT_RUN,
        workflow_status=WorkflowStatus.TERMINAL,
        runtime_seconds=0,
    )

    assert record.termination_reason is TerminationReason.NOT_ELIGIBLE
