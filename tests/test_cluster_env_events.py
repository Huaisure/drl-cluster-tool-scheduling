from __future__ import annotations

import numpy as np
import pytest

from cluster_rl.cluster_env import ClusterEnv, RobotPhase
from problem import parse_problem


def _problem(*, robot_count: int = 1, wafer_count: int = 1):
    return parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO", "capacity": wafer_count},
                "PM1": {"type": "PM"},
                "PM2": {"type": "PM"},
            },
            "ClusterTool": {
                f"TM{index + 1}": {
                    "module_ids": ["IO1", "PM1", "PM2"],
                    "arm_type": "single_arm",
                    "travel_times": 2,
                    "pick_time": 1,
                    "place_time": 1.5,
                }
                for index in range(robot_count)
            },
            "routes": {"A": [{"module_id": "PM1", "process_time": 5}]},
            "initial_state": {
                "robots": {
                    f"TM{index + 1}": {"position_module_id": "PM2"}
                    for index in range(robot_count)
                },
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": str(index),
                        "priority": index,
                        "location": {"kind": "module", "module_id": "IO1"},
                    }
                    for index in range(wafer_count)
                ],
            },
        }
    )


def _deadlock_problem():
    return parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO", "capacity": 2},
                "PM1": {"type": "PM"},
                "PM2": {"type": "PM"},
            },
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["IO1", "PM1", "PM2"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                }
            },
            "routes": {
                "A": [
                    {"module_id": "PM1", "process_time": 0},
                    {"module_id": "PM2", "process_time": 0},
                ],
                "B": [
                    {"module_id": "PM2", "process_time": 0},
                    {"module_id": "PM1", "process_time": 0},
                ],
            },
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "PM1"},
                    },
                    {
                        "route_id": "B",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "PM2"},
                    },
                ]
            },
        }
    )


def test_observation_projects_engine_pending_operation() -> None:
    env = ClusterEnv(_problem())
    observation, _ = env.reset()
    wafer_sentinel = len(env.wafer_keys)
    module_sentinel = len(env.module_ids)

    assert observation["robot_holding"].tolist() == [[wafer_sentinel]]
    assert observation["robot_phase"].tolist() == [RobotPhase.IDLE]
    assert observation["robot_operation_module"].tolist() == [module_sentinel]

    observation, *_ = env.step(0)
    assert observation["robot_phase"].tolist() == [RobotPhase.TRAVEL_TO_PICK]
    assert observation["time_to_operation_start"].tolist() == [2.0]
    assert observation["time_to_operation_end"].tolist() == [3.0]

    observation, *_ = env.step(int(env.action_space.n) - 1)
    assert observation["robot_phase"].tolist() == [RobotPhase.PICKING]
    assert observation["robot_holding"].tolist() == [[0]]


def test_pending_pick_reserves_one_wafer_across_robots() -> None:
    env = ClusterEnv(_problem(robot_count=2))
    observation, _ = env.reset()

    assert observation["action_mask"][:2].tolist() == [1, 1]
    observation, *_ = env.step(0)
    assert not observation["action_mask"][1]


def test_safety_mask_blocks_pick_that_forces_a_deadlock() -> None:
    env = ClusterEnv(_deadlock_problem())
    observation, _ = env.reset()

    assert observation["legal_action_mask"][:2].tolist() == [1, 1]
    assert not observation["action_mask"].any()
    with pytest.raises(ValueError, match="not allowed"):
        env.step(0)


def test_zero_depth_preserves_raw_deadlock_behavior() -> None:
    env = ClusterEnv(_deadlock_problem(), safety_lookahead_depth=0)
    env.reset()

    observation, *_ = env.step(0)
    _, _, terminated, truncated, info = env.step(int(env.action_space.n) - 1)

    assert not terminated and truncated
    assert info["reason"] == "deadlock"
    assert not info["action_mask"].any()


def test_safety_mask_keeps_only_robot_that_can_reach_next_target() -> None:
    problem = parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO"},
                "BUFFER1": {"type": "BUFFER"},
                "PM2": {"type": "PM"},
            },
            "ClusterTool": {
                "VTM1": {
                    "module_ids": ["BUFFER1"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                },
                "VTM2": {
                    "module_ids": ["IO1", "BUFFER1", "PM2"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                },
            },
            "routes": {
                "A": [
                    {"module_id": "BUFFER1", "process_time": 0},
                    {"module_id": "PM2", "process_time": 0},
                ]
            },
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {
                            "kind": "module",
                            "module_id": "BUFFER1",
                        },
                    }
                ]
            },
        }
    )
    env = ClusterEnv(problem)
    observation, _ = env.reset()

    assert observation["legal_action_mask"][:2].tolist() == [1, 1]
    assert observation["action_mask"][:2].tolist() == [0, 1]
