from __future__ import annotations

from pathlib import Path

import pytest

from baseline.cpsat import solve_instance
from cluster_toolkit.cluster_generator import (
    InstanceGenerationRequest,
    InstanceGenerator,
    PipelineCatalog,
    to_cluster_problem,
)
from cluster_toolkit.cluster_generator.pipeline_models import RobotArmKind, WorkloadItem
from cluster_toolkit.validator import ValidatorSuite


REPOSITORY_ROOT = Path(__file__).parents[1]


def _instance(*, arm: str, recipe_count: int, seed: int):
    catalog = PipelineCatalog.load(
        REPOSITORY_ROOT / "topologies",
        REPOSITORY_ROOT / "recipe_generation_profiles",
    )
    generator = InstanceGenerator(catalog)
    instance = generator.generate(
        InstanceGenerationRequest(
            topology_id="direct_single_cell_4pm_dual_arm",
            profile_id="direct_single_cell_default",
            recipe_count=recipe_count,
            wafer_scale="small",
            seed=seed,
        )
    ).instance
    if arm == "single":
        robots = {
            robot_id: robot.model_copy(update={"arm_kind": RobotArmKind.SINGLE})
            for robot_id, robot in instance.topology.robots.items()
        }
        instance = instance.model_copy(
            update={"topology": instance.topology.model_copy(update={"robots": robots})}
        )
    return instance


@pytest.mark.parametrize(
    ("arm", "recipe_count", "seed"),
    [("single", 1, 5), ("dual", 2, 17), ("dual", 3, 23)],
)
def test_direct_cpsat_solves_and_validates_current_instances(
    arm: str,
    recipe_count: int,
    seed: int,
) -> None:
    instance = _instance(arm=arm, recipe_count=recipe_count, seed=seed)
    result = solve_instance(
        instance,
        time_limit_seconds=2,
        random_seed=seed,
        num_search_workers=1,
    )

    assert result.status in {"OPTIMAL", "FEASIBLE"}
    assert result.validation_ok is True
    assert result.makespan is not None
    assert result.best_bound is not None
    assert len(result.actions) == 2 * sum(
        item.wafer_count
        * (
            len(
                next(
                    recipe
                    for recipe in instance.recipes
                    if recipe.recipe_id == item.recipe_id
                ).steps
            )
            + 1
        )
        for item in instance.workload
    )
    assert ValidatorSuite(to_cluster_problem(instance)).validate(result.actions).ok
    assert all(
        action["end"] - action["start"]
        == (
            instance.timing.robots[action["tm_id"]].pick_time
            if action["action_type"] == "pick"
            else instance.timing.robots[action["tm_id"]].place_time
        )
        for action in result.actions
    )


def test_direct_cpsat_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="time_limit_seconds"):
        solve_instance(_instance(arm="dual", recipe_count=1, seed=1), time_limit_seconds=0)


def test_direct_cpsat_reports_full_problem_optimality_only_after_full_solve() -> None:
    instance = _instance(arm="dual", recipe_count=1, seed=1).model_copy(
        update={"workload": (WorkloadItem(recipe_id="R0", wafer_count=1),)}
    )

    result = solve_instance(instance, time_limit_seconds=5, num_search_workers=1)

    assert result.status == "OPTIMAL"
    assert result.validation_ok is True
    assert result.makespan == result.best_bound
