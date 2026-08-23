from __future__ import annotations

import warnings
from typing import Any

from cluster_toolkit.problem import ClusterProblem, parse_problem

from .pipeline_models import ModuleKind, ModuleTag, SchedulingInstance, TopologyModule


ADAPTER_NAME = "scheduling_instance_to_cluster_problem"
ADAPTER_VERSION = "0.1.0"


def to_cluster_problem(instance: SchedulingInstance) -> ClusterProblem:
    """Convert one canonical instance into the legacy in-memory execution model.

    The adapter intentionally targets execution schema version 1 so canonical
    Route candidates remain the source of truth and no process capability
    matrix is introduced.  Generic CHAMBER tags are mapped to the legacy
    PM/AL/BUFFER vocabulary while preserving every Robot's reachability.
    """

    if any(
        module.physical_kind is ModuleKind.LOAD_LOCK
        for module in instance.topology.modules.values()
    ):
        raise NotImplementedError(
            "the first atmospheric execution adapter does not support LOAD_LOCK"
        )

    total_wafers = sum(item.wafer_count for item in instance.workload)
    workload_by_recipe = {item.recipe_id: item for item in instance.workload}

    raw_problem: dict[str, Any] = {
        "schema_version": 1,
        "_meta": {
            "adapter": {
                "name": ADAPTER_NAME,
                "version": ADAPTER_VERSION,
                "source_instance_id": instance.instance_id,
            }
        },
        "Modules": {
            module_id: {
                "type": _legacy_module_type(module),
                "capacity": total_wafers if module.physical_kind is ModuleKind.IO else 1,
            }
            for module_id, module in instance.topology.modules.items()
        },
        "ClusterTool": {
            robot_id: {
                "module_ids": list(robot.module_ids),
                "arm_type": robot.arm_kind.value,
                "travel_times": instance.timing.robots[robot_id].travel_time,
                "pick_time": instance.timing.robots[robot_id].pick_time,
                "place_time": instance.timing.robots[robot_id].place_time,
            }
            for robot_id, robot in instance.topology.robots.items()
        },
        "routes": {
            recipe.recipe_id: [
                {
                    "module_ids": list(step.candidate_module_ids),
                    "process_time": step.process_time,
                }
                for step in recipe.steps
            ]
            for recipe in instance.recipes
        },
        "initial_state": {
            "robots": {
                robot_id: {"position_module_id": None}
                for robot_id in instance.topology.robots
            },
            "wafers": [
                {
                    "route_id": recipe.recipe_id,
                    "wafer_index": str(wafer_index),
                    "priority": workload_by_recipe[recipe.recipe_id].priority,
                    "step_index": 0,
                    "location": {
                        "kind": "module",
                        "module_id": instance.source_module_id,
                    },
                    "process_end_time": None,
                }
                for recipe in instance.recipes
                for wafer_index in range(
                    workload_by_recipe[recipe.recipe_id].wafer_count
                )
            ],
        },
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Explicit Module.capacity overrides the type-based default",
            category=UserWarning,
        )
        return parse_problem(raw_problem)


def _legacy_module_type(module: TopologyModule) -> str:
    """Map generic v2 CHAMBER roles onto the execution model vocabulary."""

    if module.physical_kind is ModuleKind.IO:
        return "IO"
    if module.physical_kind is ModuleKind.LOAD_LOCK:
        return "LL"
    tags = module.effective_tags
    if ModuleTag.BUFFER in tags:
        return "BUFFER"
    if ModuleTag.ALIGN in tags:
        return "AL"
    return "PM"
