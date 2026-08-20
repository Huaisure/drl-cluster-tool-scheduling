from __future__ import annotations

import numpy as np
import pytest

from cluster_rl.cluster_env import ClusterEnv, RobotPhase
from cluster_toolkit.cluster_engine import PickAction, PlaceAction
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
    assert observation["legal_action_mask"][0]
    assert observation["action_mask"][0]


def _pending_ll_place_cycle_problem(*, include_helper_robot: bool = False):
    robots = {
        "ATM1": {
            "module_ids": ["IO1", "AL1", "LL1"],
            "arm_type": "single_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        },
        "VTM1": {
            "module_ids": ["PM1", "LL1"],
            "arm_type": "dual_arm",
            "travel_times": 5,
            "pick_time": 1,
            "place_time": 1,
        },
    }
    required_states = {"ATM1": "atmosphere", "VTM1": "vacuum"}
    if include_helper_robot:
        robots["ATM2"] = {
            "module_ids": ["IO1", "LL1"],
            "arm_type": "single_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        }
        required_states["ATM2"] = "atmosphere"
    return parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO", "capacity": 2},
                "AL1": {"type": "AL"},
                "PM1": {"type": "PM"},
                "LL1": {
                    "type": "LL",
                    "load_lock": {
                        "initial_state": "vacuum",
                        "atmosphere_to_vacuum_time": 5,
                        "vacuum_to_atmosphere_time": 7,
                        "tm_required_states": required_states,
                    },
                },
            },
            "ClusterTool": robots,
            "routes": {
                "A": [
                    {"module_id": "AL1", "process_time": 0},
                    {"module_id": "LL1", "process_time": 0},
                ],
                "B": [{"module_id": "LL1", "process_time": 0}],
            },
            "initial_state": {
                "robots": {"VTM1": {"position_module_id": "PM1"}},
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "AL1"},
                    },
                    {
                        "route_id": "B",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {
                            "kind": "robot",
                            "robot_id": "VTM1",
                            "arm_id": "arm0",
                        },
                    },
                ],
            },
        }
    )


def test_static_mask_counts_pending_place_as_future_ll_blocker() -> None:
    env = ClusterEnv(_pending_ll_place_cycle_problem())
    env.reset()

    place_b_in_ll = env._encode_engine_action(PlaceAction(("B", 0), "LL1"))
    observation, *_ = env.step(place_b_in_ll)
    pick_a_with_atm1 = env._encode_engine_action(PickAction("ATM1", ("A", 0)))

    # LL1 is empty now but reserved for B0.  Once B0 enters, only ATM1 can
    # move it to IO1, so filling ATM1 with A0 closes the future wait cycle.
    assert not env.engine.state.module_occupants["LL1"]
    assert any(
        operation.action_type == "place" and operation.module_id == "LL1"
        for operation in env.engine.state.pending_operations
    )
    assert observation["legal_action_mask"][pick_a_with_atm1]
    assert not observation["action_mask"][pick_a_with_atm1]


def test_pending_place_cycle_keeps_external_same_side_robot_exit() -> None:
    env = ClusterEnv(_pending_ll_place_cycle_problem(include_helper_robot=True))
    env.reset()

    place_b_in_ll = env._encode_engine_action(PlaceAction(("B", 0), "LL1"))
    observation, *_ = env.step(place_b_in_ll)
    pick_a_with_atm1 = env._encode_engine_action(PickAction("ATM1", ("A", 0)))

    # ATM2 can remove B0 after the pending Place completes, so ATM1 picking A0
    # is not an inevitable deadlock and must remain available.
    assert observation["legal_action_mask"][pick_a_with_atm1]
    assert observation["action_mask"][pick_a_with_atm1]


def test_place_is_blocked_when_it_closes_another_robots_wait_cycle() -> None:
    env = ClusterEnv(_pending_ll_place_cycle_problem())
    observation, _ = env.reset()

    pick_a_with_atm1 = env._encode_engine_action(PickAction("ATM1", ("A", 0)))
    observation, *_ = env.step(pick_a_with_atm1)
    place_b_in_ll = env._encode_engine_action(PlaceAction(("B", 0), "LL1"))

    # A0 was safe while LL1 was free.  Reserving LL1 for outbound B0 would
    # make ATM1 wait for B0 while B0 can leave only through ATM1.
    assert observation["legal_action_mask"][place_b_in_ll]
    assert not observation["action_mask"][place_b_in_ll]


def test_place_is_blocked_when_it_consumes_last_exchange_slot() -> None:
    problem = parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO"},
                "M1": {"type": "BUFFER"},
                "M2": {"type": "BUFFER"},
                "M3": {"type": "BUFFER"},
            },
            "ClusterTool": {
                "R1": {
                    "module_ids": ["IO1", "M1", "M2", "M3"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                }
            },
            "routes": {
                "A": [
                    {"module_id": "M1", "process_time": 0},
                    {"module_id": "M2", "process_time": 0},
                ],
                "B": [
                    {"module_id": "M2", "process_time": 0},
                    {"module_id": "M1", "process_time": 0},
                ],
                "C": [
                    {"module_id": "M3", "process_time": 0},
                    {"module_id": "M1", "process_time": 0},
                ],
            },
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "M1"},
                    },
                    {
                        "route_id": "B",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "M2"},
                    },
                    {
                        "route_id": "C",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {
                            "kind": "robot",
                            "robot_id": "R1",
                            "arm_id": "arm0",
                        },
                    },
                ]
            },
        }
    )
    env = ClusterEnv(problem)
    observation, _ = env.reset()
    place_c_in_m3 = env._encode_engine_action(PlaceAction(("C", 0), "M3"))

    # After this Place every module is full.  Any next Pick fills the single
    # arm with a wafer whose destination is another full module.
    assert observation["legal_action_mask"][place_c_in_m3]
    assert not observation["action_mask"][place_c_in_m3]


def test_candidate_action_does_not_consume_focused_robot_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = ClusterEnv(_problem(), safety_lookahead_depth=2)
    env.reset()
    action = PickAction("TM1", ("A", 0))
    observed: list[tuple[int, ...]] = []

    def record_depth(
        engine,
        remaining_by_robot: tuple[int, ...],
        focus_robot_id,
        watched_module_ids,
        memo,
    ) -> bool:
        observed.append(remaining_by_robot)
        return True

    monkeypatch.setattr(
        env.safety_filter,
        "_state_has_safe_continuation",
        record_depth,
    )

    assert env.safety_filter.action_has_safe_continuation(env.engine, action, 2)
    assert observed == [(2,)]


def _multi_robot_wait_cycle_problem(*, include_escape_robot: bool = False):
    robots = {
        "R1": {
            "module_ids": ["IO1", "M1", "M2", "OUT1"],
            "arm_type": "dual_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        },
        "R2": {
            "module_ids": ["IO1", "M1", "M2", "OUT2"],
            "arm_type": "single_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        },
    }
    if include_escape_robot:
        robots["R3"] = {
            "module_ids": ["M2", "OUT2"],
            "arm_type": "single_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        }
    return parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO"},
                "M1": {"type": "BUFFER"},
                "M2": {"type": "BUFFER"},
                "OUT1": {"type": "PM"},
                "OUT2": {"type": "PM"},
            },
            "ClusterTool": robots,
            "routes": {
                "A": [{"module_id": "M2", "process_time": 0}],
                "B": [
                    {"module_id": "M2", "process_time": 0},
                    {"module_id": "OUT2", "process_time": 0},
                ],
                "C": [{"module_id": "M1", "process_time": 0}],
                "D": [
                    {"module_id": "M1", "process_time": 0},
                    {"module_id": "OUT1", "process_time": 0},
                ],
                "E": [{"module_id": "M2", "process_time": 0}],
            },
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {
                            "kind": "robot",
                            "robot_id": "R1",
                            "arm_id": "arm0",
                        },
                    },
                    {
                        "route_id": "B",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "M2"},
                    },
                    {
                        "route_id": "C",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {
                            "kind": "robot",
                            "robot_id": "R2",
                            "arm_id": "arm0",
                        },
                    },
                    {
                        "route_id": "D",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "M1"},
                    },
                    {
                        "route_id": "E",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "IO1"},
                    },
                ]
            },
        }
    )


def _pick_index(env: ClusterEnv, wafer_key, robot_id: str) -> int:
    return (
        env._wafer_index[wafer_key] * len(env._robot_ids)
        + env._robot_index[robot_id]
    )


def test_static_mask_blocks_closed_multi_robot_wait_cycle() -> None:
    env = ClusterEnv(
        _multi_robot_wait_cycle_problem(),
        safety_lookahead_depth=0,
    )
    observation, _ = env.reset()
    pick_e_with_r1 = _pick_index(env, ("E", 0), "R1")

    # R1 -> M2 -> R2 -> M1 -> R1 has no free arm, target, pending
    # operation, or external Robot that can open the cycle.
    assert observation["legal_action_mask"][pick_e_with_r1]
    assert not observation["action_mask"][pick_e_with_r1]


def test_static_mask_keeps_cycle_candidate_with_external_robot_exit() -> None:
    env = ClusterEnv(
        _multi_robot_wait_cycle_problem(include_escape_robot=True),
        safety_lookahead_depth=0,
    )
    observation, _ = env.reset()
    pick_e_with_r1 = _pick_index(env, ("E", 0), "R1")

    # R3 can transfer B0 from M2 to OUT2, so the apparent R1/R2 cycle has an
    # external exit and cannot be declared an inevitable deadlock.
    assert observation["legal_action_mask"][pick_e_with_r1]
    assert observation["action_mask"][pick_e_with_r1]


def _ll_exit_side_cycle_problem(*, include_same_side_robot: bool = False):
    robots = {
        "ATM1": {
            "module_ids": ["LL1", "BUFFER1"],
            "arm_type": "single_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        },
        "VTM1": {
            "module_ids": ["IO1", "LL1", "BUFFER1"],
            "arm_type": "dual_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        },
    }
    required_states = {
        "ATM1": "atmosphere",
        "VTM1": "vacuum",
    }
    if include_same_side_robot:
        robots["VTM2"] = {
            "module_ids": ["LL1", "BUFFER1"],
            "arm_type": "single_arm",
            "travel_times": 0,
            "pick_time": 1,
            "place_time": 1,
        }
        required_states["VTM2"] = "vacuum"
    return parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO"},
                "LL1": {
                    "type": "LL",
                    "load_lock": {
                        "initial_state": "vacuum",
                        "atmosphere_to_vacuum_time": 5,
                        "vacuum_to_atmosphere_time": 7,
                        "tm_required_states": required_states,
                    },
                },
                "BUFFER1": {"type": "BUFFER"},
            },
            "ClusterTool": robots,
            "routes": {
                "A": [{"module_id": "LL1", "process_time": 0}],
                "B": [
                    {"module_id": "LL1", "process_time": 0},
                    {"module_id": "BUFFER1", "process_time": 0},
                ],
                "E": [{"module_id": "LL1", "process_time": 0}],
            },
            "initial_state": {
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
                        "location": {"kind": "module", "module_id": "LL1"},
                    },
                    {
                        "route_id": "E",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "IO1"},
                    },
                ]
            },
        }
    )


def test_static_mask_does_not_treat_opposite_ll_side_as_cycle_exit() -> None:
    env = ClusterEnv(
        _ll_exit_side_cycle_problem(),
        safety_lookahead_depth=0,
    )
    observation, _ = env.reset()
    pick_e_with_vtm1 = _pick_index(env, ("E", 0), "VTM1")

    # ATM1 reaches both LL1 and BUFFER1, but B0 can currently leave LL1 only
    # from the vacuum side.  Filling VTM1 therefore closes the wait cycle.
    assert observation["legal_action_mask"][pick_e_with_vtm1]
    assert not observation["action_mask"][pick_e_with_vtm1]


def test_static_mask_keeps_same_ll_side_external_robot_exit() -> None:
    env = ClusterEnv(
        _ll_exit_side_cycle_problem(include_same_side_robot=True),
        safety_lookahead_depth=0,
    )
    observation, _ = env.reset()
    pick_e_with_vtm1 = _pick_index(env, ("E", 0), "VTM1")

    # VTM2 is an empty external Robot on the required vacuum side, so it can
    # remove B0 and the Pick is not an inevitable deadlock.
    assert observation["legal_action_mask"][pick_e_with_vtm1]
    assert observation["action_mask"][pick_e_with_vtm1]
