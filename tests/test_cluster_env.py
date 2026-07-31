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
    return len(env.wafer_keys) + env.module_ids.index(module_id)


def test_reset_exposes_stable_spaces_and_only_pick_actions() -> None:
    env = ClusterEnv(_problem(wafer_routes=("A", "A")))

    observation, info = env.reset(seed=7)

    assert env.wafer_keys == (("A", 0), ("A", 1))
    assert env.module_ids == ("LP", "PM1", "PM2")
    assert env.action_space.n == len(env.wafer_keys) + len(env.module_ids)
    assert env.observation_space.contains(observation)
    assert observation["action_mask"].shape == (env.action_space.n,)
    assert observation["action_mask"].tolist() == [1, 1, 0, 0, 0]
    assert observation["robot_module"] == len(env.module_ids)
    assert info == {"time": 0.0}
    assert env.actions == ()


def test_pick_place_timing_wait_and_success_reward_match_makespan() -> None:
    problem = _problem(robot_position="PM2")
    env = ClusterEnv(problem)
    observation, _ = env.reset()
    total_reward = 0.0

    observation, reward, terminated, truncated, info = env.step(0)
    total_reward += reward
    assert not terminated and not truncated
    assert info["time"] == 3.0
    assert observation["action_mask"].tolist() == [0, 0, 1, 1]
    assert env.actions[0]["action_type"] == "pick"
    assert (env.actions[0]["start"], env.actions[0]["end"]) == (2.0, 3.0)

    observation, reward, terminated, _, info = env.step(
        _place_action(env, "PM1")
    )
    total_reward += reward
    assert not terminated
    assert info["time"] == 11.5
    assert observation["action_mask"].tolist() == [1, 0, 0, 0]
    assert env.actions[1]["action_type"] == "place"
    assert (env.actions[1]["start"], env.actions[1]["end"]) == (5.0, 6.5)

    _, reward, _, _, _ = env.step(0)
    total_reward += reward
    observation, reward, terminated, truncated, info = env.step(
        _place_action(env, "LP")
    )
    total_reward += reward

    assert terminated and not truncated
    assert info == {"time": 16.0, "is_success": True, "reason": "completed"}
    assert total_reward == -16.0
    assert not observation["action_mask"].any()
    assert ValidatorSuite(problem).validate(env.actions).ok
    with pytest.raises(TypeError):
        env.actions[0]["start"] = 0


def test_invalid_action_raises_without_changing_state() -> None:
    env = ClusterEnv(_problem())
    before, _ = env.reset()

    with pytest.raises(ValueError, match="not allowed"):
        env.step(_place_action(env, "LP"))

    after, _, _, _, info = env.step(0)
    assert env.actions[0]["start"] == 0.0
    assert info["time"] == 1.0
    assert before["wafer_step"].tolist() == after["wafer_step"].tolist()


def test_capacity_and_remaining_process_time_are_derived() -> None:
    env = ClusterEnv(_problem(wafer_routes=("A", "A")))
    observation, _ = env.reset()

    observation, *_ = env.step(0)
    observation, *_ = env.step(_place_action(env, "PM1"))

    assert observation["process_remaining"].tolist() == [5.0, 0.0]
    assert observation["action_mask"].tolist() == [0, 1, 0, 0, 0]


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

    for action in (
        0,
        _place_action(env, "PM1"),
        0,
        _place_action(env, "PM1"),
        0,
        _place_action(env, "LP"),
    ):
        assert observation["action_mask"][action]
        observation, _, terminated, _, _ = env.step(action)

    assert terminated
    assert ValidatorSuite(problem).validate(env.actions).ok


def test_crossing_routes_end_as_a_penalized_deadlock() -> None:
    problem = _problem(
        routes={
            "A": [
                {"module_id": "PM1", "process_time": 0},
                {"module_id": "PM2", "process_time": 0},
            ],
            "B": [
                {"module_id": "PM2", "process_time": 0},
                {"module_id": "PM1", "process_time": 0},
            ],
        },
        wafer_routes=("A", "B"),
    )
    env = ClusterEnv(problem)
    observation, _ = env.reset()
    total_reward = 0.0

    for action in (
        0,
        _place_action(env, "PM1"),
        1,
        _place_action(env, "PM2"),
    ):
        assert observation["action_mask"][action]
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    assert terminated and not truncated
    assert not info["is_success"]
    assert info["reason"] == "deadlock"
    assert total_reward == -info["failure_cost"]
    assert total_reward < -info["time"]
    assert not observation["action_mask"].any()


def test_masked_random_episode_finishes_with_a_valid_schedule() -> None:
    problem = _problem(wafer_routes=("A", "A", "A"))
    env = ClusterEnv(problem)
    observation, _ = env.reset(seed=11)
    rng = np.random.default_rng(11)
    total_reward = 0.0

    for _ in range(20):
        legal_actions = np.flatnonzero(observation["action_mask"])
        action = int(rng.choice(legal_actions))
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    assert terminated and not truncated
    assert info["is_success"]
    assert total_reward == -info["time"]
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
        "wafer_module",
        "wafer_step",
        "process_remaining",
        "robot_module",
        "action_mask",
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda raw: raw["ClusterTool"].update(
                {
                    "TM2": {
                        "module_ids": ["LP"],
                        "arm_type": "single_arm",
                        "pick_time": 1,
                        "place_time": 1,
                    }
                }
            ),
            "exactly one TM",
        ),
        (
            lambda raw: raw["ClusterTool"]["TM1"].update(
                {"arm_type": "dual_arm"}
            ),
            "single-arm",
        ),
        (
            lambda raw: raw["Modules"].update({"LL1": {"type": "LL"}}),
            "only LP and PM",
        ),
        (
            lambda raw: raw["Modules"].update({"LP2": {"type": "LP"}}),
            "exactly one LP",
        ),
        (
            lambda raw: raw["ClusterTool"]["TM1"].update(
                {"module_ids": ["LP", "PM1"]}
            ),
            "cannot reach modules",
        ),
        (
            lambda raw: raw["initial_state"].update({"wafers": []}),
            "at least one wafer",
        ),
    ],
)
def test_unsupported_problem_shapes_fail_fast(change, message: str) -> None:
    raw = _raw_problem()
    change(raw)

    with pytest.raises(ValueError, match=message):
        ClusterEnv(parse_problem(raw))
