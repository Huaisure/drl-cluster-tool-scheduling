from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cluster_rl.cluster_env import ClusterEnv
from problem import parse_problem
from validator import ValidatorSuite


def _raw_problem(
    *,
    routes: dict[str, list[dict[str, object]]] | None = None,
    wafer_routes: tuple[str, ...] = ("A",),
    robot_position: str | None = None,
    travel_time: float = 2,
    pick_time: float = 1,
    place_time: float = 1.5,
) -> dict[str, object]:
    routes = routes or {
        "A": [
            {
                "module_ids": ["PM1", "PM2"],
                "process_time": 5,
            }
        ]
    }
    indexes: Counter[str] = Counter()
    wafers = []
    for route_id in wafer_routes:
        wafers.append(
            {
                "route_id": route_id,
                "wafer_index": str(indexes[route_id]),
                "step_index": 0,
                "location": {"kind": "module", "module_id": "LP"},
            }
        )
        indexes[route_id] += 1

    initial_state: dict[str, object] = {"wafers": wafers}
    if robot_position is not None:
        initial_state["robots"] = {
            "TM1": {"position_module_id": robot_position}
        }
    return {
        "Modules": {
            "LP": {"type": "LP"},
            "PM1": {"type": "PM"},
            "PM2": {"type": "PM"},
        },
        "ClusterTool": {
            "TM1": {
                "module_ids": ["LP", "PM1", "PM2"],
                "arm_type": "single_arm",
                "travel_times": travel_time,
                "pick_time": pick_time,
                "place_time": place_time,
            }
        },
        "routes": routes,
        "initial_state": initial_state,
    }


def _problem(**kwargs):
    return parse_problem(_raw_problem(**kwargs))


def _place_action(env: ClusterEnv, module_id: str) -> int:
    entity_index = len(env.wafer_keys) + env.module_ids.index(module_id)
    return entity_index * len(env._robot_ids)


def _advance_action(env: ClusterEnv) -> int:
    return int(env.action_space.n) - 1


def test_reset_exposes_stable_spaces_and_only_pick_actions() -> None:
    env = ClusterEnv(_problem(wafer_routes=("A", "A")))

    observation, info = env.reset(seed=7)

    assert env.wafer_keys == (("A", 0), ("A", 1))
    assert env.module_ids == ("LP", "PM1", "PM2")
    assert env.action_space.n == len(env.wafer_keys) + len(env.module_ids) + 1
    assert env.observation_space.contains(observation)
    assert observation["action_mask"].shape == (env.action_space.n,)
    assert observation["action_mask"].tolist() == [1, 0, 0, 0, 0, 0]
    assert observation["robot_loc"].tolist() == [len(env.module_ids)]
    assert info == {"time": 0.0}
    assert env.actions == ()


def test_action_decode_and_explicit_advance_boundaries() -> None:
    problem = _problem(robot_position="PM2")
    env = ClusterEnv(problem)
    observation, _ = env.reset()

    assert env._decode_action(0) == ("pick", 0, 0)
    assert env._decode_action(_place_action(env, "PM1")) == (
        "place",
        env.module_ids.index("PM1"),
        0,
    )
    assert env._decode_action(_advance_action(env)) == (
        "advance",
        None,
        None,
    )

    observation, reward, terminated, truncated, info = env.step(0)
    assert not terminated and not truncated
    assert reward == 0.0
    assert info["time"] == 0.0
    assert np.flatnonzero(observation["action_mask"]).tolist() == [
        _advance_action(env)
    ]
    assert env.actions[0]["action_type"] == "pick"
    assert (env.actions[0]["start"], env.actions[0]["end"]) == (2.0, 3.0)

    observation, *_ = env.step(_advance_action(env))
    assert info["time"] == 0.0
    assert env._time == 2.0
    assert env._robots[0].holding == [0]
    assert np.flatnonzero(observation["action_mask"]).tolist() == [
        _advance_action(env)
    ]

    observation, *_ = env.step(_advance_action(env))
    assert env._time == 3.0
    assert observation["action_mask"][_place_action(env, "PM1")]


def test_invalid_action_raises_without_changing_state() -> None:
    env = ClusterEnv(_problem())
    before, _ = env.reset()

    with pytest.raises(ValueError, match="not allowed"):
        env.step(_place_action(env, "LP"))

    after, _, _, _, info = env.step(0)
    assert env.actions[0]["start"] == 0.0
    assert info["time"] == 0.0
    assert before["wafer_step"].tolist() == after["wafer_step"].tolist()
    assert len(env.actions) == 1


def test_repeated_pm_route_can_release_then_reoccupy_the_same_module() -> None:
    problem = _problem(
        routes={
            "A": [
                {"module_id": "PM1", "process_time": 0},
                {"module_id": "PM1", "process_time": 0},
            ]
        }
    )
    env = ClusterEnv(problem)
    observation, _ = env.reset()

    transport_actions = iter(
        (
            0,
            _place_action(env, "PM1"),
            0,
            _place_action(env, "PM1"),
            0,
            _place_action(env, "LP"),
        )
    )
    next_transport = next(transport_actions)
    terminated = False
    for _ in range(30):
        action = (
            next_transport
            if next_transport >= 0
            and observation["action_mask"][next_transport]
            else _advance_action(env)
        )
        assert observation["action_mask"][action]
        observation, _, terminated, _, _ = env.step(action)
        if action == next_transport:
            next_transport = next(transport_actions, -1)
        if terminated:
            break

    assert terminated
    assert ValidatorSuite(problem).validate(env.actions).ok


def test_masked_random_episode_finishes_with_a_valid_schedule() -> None:
    problem = _problem(wafer_routes=("A", "A", "A"))
    env = ClusterEnv(problem)
    observation, _ = env.reset(seed=11)
    rng = np.random.default_rng(11)

    for _ in range(100):
        legal_actions = np.flatnonzero(observation["action_mask"])
        action = int(rng.choice(legal_actions))
        observation, reward, terminated, truncated, info = env.step(action)
        assert reward == 0.0
        if terminated or truncated:
            break

    assert terminated and not truncated
    assert info["is_success"]
    assert ValidatorSuite(problem).validate(env.actions).ok


def test_ignored_constraints_do_not_enter_environment_state() -> None:
    raw = _raw_problem()
    raw["just_in_time"] = {"residency_time": 1}
    raw["cleaning"] = {
        "module_ids": ["PM1"],
        "process_switch": {"clean_time": 3},
    }
    raw["routes"]["A"][0]["residency_time"] = 1

    env = ClusterEnv(parse_problem(raw))

    observation, _ = env.reset()
    assert set(observation) == {
        "wafer_loc",
        "wafer_step",
        "process_remaining",
        "robot_loc",
        "robot_holding",
        "robot_phase",
        "robot_operation_wafer",
        "robot_operation_module",
        "time_to_operation_start",
        "time_to_operation_end",
        "action_mask",
    }
