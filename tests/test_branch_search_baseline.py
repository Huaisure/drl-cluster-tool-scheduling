from __future__ import annotations

import json
from pathlib import Path

import pytest

from baseline import solve_branch_search_instance
from baseline.__main__ import main as baseline_main
from cluster_toolkit.cluster_generator import (
    InstanceGenerationRequest,
    InstanceGenerator,
    PipelineCatalog,
    to_cluster_problem,
)
from cluster_toolkit.cluster_generator.pipeline_models import (
    SchedulingInstance,
    WorkloadItem,
)
from cluster_toolkit.validator import ValidatorSuite


REPOSITORY_ROOT = Path(__file__).parents[1]


def _generated_instance() -> SchedulingInstance:
    catalog = PipelineCatalog.load(
        REPOSITORY_ROOT / "topologies",
        REPOSITORY_ROOT / "recipe_generation_profiles",
    )
    instance = InstanceGenerator(catalog).generate(
        InstanceGenerationRequest(
            topology_id="direct_single_cell_4pm_dual_arm",
            profile_id="direct_single_cell_default",
            recipe_count=2,
            wafer_scale="small",
            seed=17,
        )
    ).instance
    return instance.model_copy(
        update={
            "workload": tuple(
                WorkloadItem(recipe_id=recipe.recipe_id, wafer_count=1)
                for recipe in instance.recipes
            )
        }
    )


def test_branch_search_solves_generated_instance_reproducibly() -> None:
    instance = _generated_instance()

    first = solve_branch_search_instance(instance, planning_horizon=3)
    second = solve_branch_search_instance(instance, planning_horizon=3)

    assert first.validation_ok is True
    assert first.makespan == second.makespan
    assert [dict(action) for action in first.actions] == [
        dict(action) for action in second.actions
    ]
    assert len(first.actions) == 2 * sum(
        len(recipe.steps) + 1 for recipe in instance.recipes
    )
    assert ValidatorSuite(to_cluster_problem(instance)).validate(first.actions).ok
    with pytest.raises(TypeError):
        first.actions[0]["start"] = 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"planning_horizon": 0}, "planning_horizon"),
        ({"planning_horizon": True}, "planning_horizon"),
        ({"safety_lookahead_depth": -1}, "safety_lookahead_depth"),
        ({"time_limit_seconds": 0}, "time_limit_seconds"),
    ],
)
def test_branch_search_rejects_invalid_configuration(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_branch_search_instance(_generated_instance(), **kwargs)


def test_branch_search_cli_reads_generated_problem_and_writes_actions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    instance = _generated_instance()
    problem_path = tmp_path / "problem.json"
    output_path = tmp_path / "actions.json"
    problem_path.write_text(
        instance.model_dump_json(indent=2),
        encoding="utf-8",
    )

    exit_code = baseline_main(
        [
            "branch-search",
            str(problem_path),
            "--output",
            str(output_path),
            "--planning-horizon",
            "2",
        ]
    )

    actions = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["solver"] == "branch_search"
    assert summary["validation_ok"] is True
    assert summary["action_count"] == len(actions)
    assert ValidatorSuite(to_cluster_problem(instance)).validate(actions).ok
