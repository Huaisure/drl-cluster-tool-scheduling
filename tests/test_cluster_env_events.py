from __future__ import annotations

import numpy as np
import pytest

from cluster_rl.cluster_env import ClusterEnv, RobotPhase
from cluster_toolkit.problem import parse_problem


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


def test_zero_depth_still_applies_static_deadlock_rules() -> None:
    env = ClusterEnv(_deadlock_problem(), safety_lookahead_depth=0)
    observation, _ = env.reset()

    assert observation["legal_action_mask"][:2].tolist() == [1, 1]
    assert not observation["action_mask"].any()
    with pytest.raises(ValueError, match="not allowed"):
        env.step(0)


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


def _full_single_arm_cycle_problem(*, include_helper_robot: bool = False):
    robots = {
        "ATM1": {
            "module_ids": ["IO1", "AL1", "LL1"],
            "arm_type": "single_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        }
    }
    if include_helper_robot:
        robots["ATM2"] = {
            "module_ids": ["AL1", "LL1"],
            "arm_type": "single_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        }
    return parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO"},
                "AL1": {"type": "AL"},
                "LL1": {"type": "BUFFER"},
            },
            "ClusterTool": robots,
            "routes": {
                "A": [{"module_id": "AL1", "process_time": 0}],
                "B": [
                    {"module_id": "AL1", "process_time": 0},
                    {"module_id": "LL1", "process_time": 0},
                ],
            },
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "IO1"},
                    },
                    {
                        "route_id": "B",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "AL1"},
                    },
                ]
            },
        }
    )


def test_static_mask_blocks_pick_that_fills_only_unblocking_robot() -> None:
    env = ClusterEnv(
        _full_single_arm_cycle_problem(),
        safety_lookahead_depth=0,
    )
    observation, _ = env.reset()

    # Picking A0 fills ATM1 while A0 waits for AL1.  AL1 contains B0, which
    # also needs ATM1 to move onward, so the source Pick closes a wait cycle.
    assert observation["legal_action_mask"][0]
    assert not observation["action_mask"][0]
    assert observation["action_mask"][1]


def test_static_mask_keeps_pick_when_another_robot_can_release_target() -> None:
    env = ClusterEnv(
        _full_single_arm_cycle_problem(include_helper_robot=True),
        safety_lookahead_depth=0,
    )
    observation, _ = env.reset()

    # ATM2 can move B0 from AL1 to LL1, so ATM1 picking A0 is not guaranteed
    # to deadlock and remains available to the policy.
    assert observation["legal_action_mask"][0]
    assert observation["action_mask"][0]


def test_static_mask_keeps_dual_arm_exchange_pick() -> None:
    problem = parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO"},
                "PM1": {"type": "PM"},
                "PM2": {"type": "PM"},
            },
            "ClusterTool": {
                "VTM1": {
                    "module_ids": ["IO1", "PM1", "PM2"],
                    "arm_type": "dual_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                }
            },
            "routes": {
                "A": [{"module_id": "PM1", "process_time": 0}],
                "B": [
                    {"module_id": "PM1", "process_time": 0},
                    {"module_id": "PM2", "process_time": 0},
                ],
            },
            "initial_state": {
                "robots": {"VTM1": {"position_module_id": "PM1"}},
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {
                            "kind": "robot",
                            "robot_id": "VTM1",
                            "arm_id": "arm0",
                        },
                    },
                    {
                        "route_id": "B",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "PM1"},
                    },
                ],
            },
        }
    )
    env = ClusterEnv(problem, safety_lookahead_depth=0)
    observation, _ = env.reset()

    # Picking B0 fills VTM1 but simultaneously releases PM1 at Pick.end, so
    # the already held A0 can be placed and the exchange remains recoverable.
    assert observation["legal_action_mask"][1]
    assert observation["action_mask"][1]


def test_static_mask_handles_target_occupied_at_place_start() -> None:
    problem = parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO"},
                "AL1": {"type": "AL"},
                "LL1": {"type": "BUFFER"},
            },
            "ClusterTool": {
                "ATM1": {
                    "module_ids": ["IO1", "AL1"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                },
                "ATM2": {
                    "module_ids": ["AL1", "LL1"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                },
            },
            "routes": {
                "A": [{"module_id": "AL1", "process_time": 0}],
                "B": [
                    {"module_id": "AL1", "process_time": 0},
                    {"module_id": "LL1", "process_time": 0},
                ],
            },
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "IO1"},
                    },
                    {
                        "route_id": "B",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {
                            "kind": "robot",
                            "robot_id": "ATM2",
                            "arm_id": "arm0",
                        },
                    },
                ]
            },
        }
    )
    env = ClusterEnv(problem, safety_lookahead_depth=0)
    env.reset()

    place_b_in_al = env._pick_action_count + len(env.module_ids) + env.module_ids.index(
        "AL1"
    )
    observation, *_ = env.step(place_b_in_al)

    # Place.start reserves AL1 before B0.module_id changes at Place.end.  The
    # mask must use AL1 as B0's future Pick source during this interval.
    assert env.engine.state.wafers[("B", 0)].module_id is None
    assert ("B", 0) in env.engine.state.module_occupants["AL1"]
    assert observation["action_mask"].shape == (env.action_space.n,)
