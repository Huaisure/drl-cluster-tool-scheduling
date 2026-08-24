from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import baseline.cpsat.periodic as periodic_module
from baseline.cpsat import (
    periodic_ratio,
    solve_cpsat_instance,
    solve_periodic_instance,
)
from baseline.cpsat.__main__ import main
from baseline.__main__ import main as baseline_main
from cluster_toolkit.cluster_generator import (
    InstanceGenerationRequest,
    InstanceGenerator,
    PipelineCatalog,
    to_cluster_problem,
)
from cluster_toolkit.cluster_generator.pipeline_models import WorkloadItem
from cluster_toolkit.validator import ValidatorSuite


REPOSITORY_ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def generator() -> InstanceGenerator:
    return InstanceGenerator(
        PipelineCatalog.load(
            REPOSITORY_ROOT / "topologies",
            REPOSITORY_ROOT / "recipe_generation_profiles",
        )
    )


def _periodic_instance(
    generator: InstanceGenerator,
    *,
    recipe_count: int,
    ratio: tuple[int, ...],
    seed: int,
):
    return generator.generate(
        InstanceGenerationRequest(
            topology_id="direct_single_cell_4pm_dual_arm",
            profile_id="direct_single_cell_default",
            recipe_count=recipe_count,
            wafer_scale="small",
            seed=seed,
            periodic_ratio=ratio,
        )
    ).instance


@pytest.mark.parametrize(
    ("ratio", "seed"),
    [((1,), 5), ((1, 2), 17), ((1, 2, 1), 23)],
)
def test_periodic_pipeline_materializes_valid_startup_steady_and_closedown(
    generator: InstanceGenerator,
    ratio: tuple[int, ...],
    seed: int,
) -> None:
    instance = _periodic_instance(
        generator,
        recipe_count=len(ratio),
        ratio=ratio,
        seed=seed,
    )

    result = solve_periodic_instance(
        instance,
        time_limit_seconds=15,
        random_seed=seed,
        num_search_workers=1,
    )

    assert result.status == "FEASIBLE"
    assert result.ratio == ratio
    assert result.period is not None and result.period > 0
    assert result.cycle is not None
    assert result.cycle.status in {"FEASIBLE", "OPTIMAL"}
    if result.cycle.status == "OPTIMAL":
        assert result.cycle.best_bound == result.period
    assert result.pipeline_depth_periods is not None
    assert result.pipeline_depth_periods > 0
    assert result.steady_cycle_count is not None
    assert result.steady_cycle_count > 0
    assert result.startup is not None and result.startup.action_count > 0
    assert result.closedown is not None and result.closedown.action_count > 0
    assert result.validation_ok is True
    assert {action["periodic_phase"] for action in result.actions} == {
        "startup",
        "steady",
        "closedown",
    }
    problem = to_cluster_problem(instance)
    assert ValidatorSuite(problem).validate(result.actions).ok
    assert len(result.actions) == 2 * sum(
        item.wafer_count * (len(problem.routes[item.recipe_id].visits) + 1)
        for item in instance.workload
    )


def test_boundary_selection_rejects_empty_transition_phases(
    generator: InstanceGenerator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _periodic_instance(
        generator,
        recipe_count=1,
        ratio=(1,),
        seed=5,
    )
    solved = periodic_module._SolvedCycle(
        status="FEASIBLE",
        period=10,
        best_bound=5,
        runtime_seconds=0.1,
        actions=(
            {"start": 0, "end": 1},
            {"start": 2, "end": 3},
        ),
        transfer_wraps={},
        process_wraps={},
    )

    def fake_rotate(instance, ratio, solved, shift):
        del instance, ratio
        return replace(
            solved,
            actions=({"start": shift, "end": shift + 1, "shift": shift},),
        )

    def fake_materialize(instance, ratio, repeat_count, solved):
        del instance, ratio, repeat_count
        shift = int(solved.actions[0]["shift"])
        if shift == 0:
            return (
                ({"start": 0, "end": 1},),
                {"startup": 0, "steady": 1, "closedown": 0},
                0,
            )
        return (
            ({"start": 0, "end": 10 + shift},),
            {"startup": 1, "steady": 1, "closedown": 1},
            1,
        )

    class AlwaysValid:
        def __init__(self, problem):
            del problem

        def validate(self, actions, **kwargs):
            del actions, kwargs
            return SimpleNamespace(ok=True)

    monkeypatch.setattr(periodic_module, "_rotate_cycle_boundary", fake_rotate)
    monkeypatch.setattr(
        periodic_module,
        "_materialize_finite_schedule",
        fake_materialize,
    )
    monkeypatch.setattr(periodic_module, "ValidatorSuite", AlwaysValid)

    candidates, _ = periodic_module._select_cycle_boundaries(
        instance,
        (1,),
        3,
        solved,
    )

    assert candidates
    assert all(candidate.pipeline_depth > 0 for candidate in candidates)
    assert all(candidate.phase_counts["startup"] > 0 for candidate in candidates)
    assert all(candidate.phase_counts["closedown"] > 0 for candidate in candidates)


def test_unsupported_ratio_is_skipped_and_portfolio_uses_direct(
    generator: InstanceGenerator,
) -> None:
    instance = _periodic_instance(
        generator,
        recipe_count=2,
        ratio=(1, 1),
        seed=31,
    ).model_copy(
        update={
            "workload": (
                WorkloadItem(recipe_id="R0", wafer_count=1),
                WorkloadItem(recipe_id="R1", wafer_count=3),
            )
        }
    )

    assert periodic_ratio(instance) is None
    skipped = solve_periodic_instance(instance, time_limit_seconds=1)
    assert skipped.status == "NOT_ELIGIBLE"
    assert not skipped.actions

    routed = solve_cpsat_instance(instance, time_limit_seconds=2)
    assert routed.method == "direct"
    assert routed.result.status in {"FEASIBLE", "OPTIMAL"}
    assert routed.result.validation_ok is True


def test_portfolio_routes_supported_ratio_to_periodic(
    generator: InstanceGenerator,
) -> None:
    instance = _periodic_instance(
        generator,
        recipe_count=1,
        ratio=(1,),
        seed=5,
    )

    routed = solve_cpsat_instance(instance, time_limit_seconds=5, random_seed=5)

    assert routed.method == "periodic"
    assert routed.result.status == "FEASIBLE"
    assert routed.result.validation_ok is True


def test_cpsat_cli_auto_mode_writes_periodic_actions(
    generator: InstanceGenerator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    instance = _periodic_instance(
        generator,
        recipe_count=1,
        ratio=(1,),
        seed=5,
    )
    problem_path = tmp_path / "problem.json"
    actions_path = tmp_path / "actions.json"
    problem_path.write_text(instance.model_dump_json(indent=2), encoding="utf-8")

    exit_code = main(
        [
            str(problem_path),
            "--output",
            str(actions_path),
            "--mode",
            "auto",
            "--time-limit",
            "5",
            "--seed",
            "5",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    actions = json.loads(actions_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["method"] == "periodic"
    assert summary["cycle_status"] in {"FEASIBLE", "OPTIMAL"}
    assert summary["action_count"] == len(actions)
    assert ValidatorSuite(to_cluster_problem(instance)).validate(actions).ok


def test_top_level_baseline_cli_dispatches_cpsat(
    generator: InstanceGenerator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    instance = _periodic_instance(
        generator,
        recipe_count=1,
        ratio=(1,),
        seed=5,
    )
    problem_path = tmp_path / "problem.json"
    actions_path = tmp_path / "actions.json"
    problem_path.write_text(instance.model_dump_json(indent=2), encoding="utf-8")

    exit_code = baseline_main(
        [
            "cpsat",
            str(problem_path),
            "--output",
            str(actions_path),
            "--time-limit",
            "5",
            "--seed",
            "5",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["method"] == "periodic"
    assert actions_path.is_file()
