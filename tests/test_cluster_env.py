from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cluster_rl.cluster_env import ClusterEnv, LoadLockSide
from cluster_toolkit.problem import parse_problem
from tests.problem_fixtures import load_lock_problem
from cluster_toolkit.validator import ValidatorSuite


def _raw_problem(
    *,
    wafer_routes: tuple[str, ...] = ("A",),
    priorities: tuple[int, ...] | None = None,
    arm_type: str = "single_arm",
    robot_position: str | None = None,
    routes: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    priorities = priorities or (0,) * len(wafer_routes)
    indexes: Counter[str] = Counter()
    wafers = []
    for route_id, priority in zip(wafer_routes, priorities, strict=True):
        wafers.append(
            {
                "route_id": route_id,
                "wafer_index": str(indexes[route_id]),
                "priority": priority,
                "location": {"kind": "module", "module_id": "IO1"},
            }
        )
        indexes[route_id] += 1
    return {
        "Modules": {
            "IO1": {"type": "IO", "capacity": len(wafers)},
            "PM1": {"type": "PM"},
            "PM2": {"type": "PM"},
        },
        "ClusterTool": {
            "TM1": {
                "module_ids": ["IO1", "PM1", "PM2"],
                "arm_type": arm_type,
                "travel_times": 2,
                "pick_time": 1,
                "place_time": 1.5,
            }
        },
        "routes": routes or {
            route_id: [
                {"module_ids": ["PM1", "PM2"], "process_time": 5}
            ]
            for route_id in set(wafer_routes)
        },
        "initial_state": {
            "robots": {"TM1": {"position_module_id": robot_position}},
            "wafers": wafers,
        },
    }


def _problem(**kwargs):
    return parse_problem(_raw_problem(**kwargs))


def _pick_action(env: ClusterEnv, wafer_index: int, robot_index: int = 0) -> int:
    return wafer_index * len(env._robot_ids) + robot_index


def _place_action(env: ClusterEnv, wafer_index: int, module_id: str) -> int:
    return (
        len(env.wafer_keys) * len(env._robot_ids)
        + wafer_index * len(env.module_ids)
        + env.module_ids.index(module_id)
    )


def _advance_action(env: ClusterEnv) -> int:
    return int(env.action_space.n) - 1


def test_safety_lookahead_depth_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ClusterEnv(_problem(), safety_lookahead_depth=-1)


def test_reset_uses_explicit_pick_place_action_layout_and_static_features() -> None:
    env = ClusterEnv(_problem(wafer_routes=("A", "A"), priorities=(0, 0)))

    observation, info = env.reset(seed=7)

    assert env.action_space.n == 2 * 1 + 2 * 3 + 1
    assert env.observation_space.contains(observation)
    assert np.flatnonzero(observation["action_mask"]).tolist() == [0]
    assert observation["wafer_priority"].tolist() == [0.0, 0.0]
    assert observation["wafer_index"].tolist() == [0.0, 1.0]
    assert info == {"time": 0.0}


def test_engine_priority_then_env_recipe_index_tie_break() -> None:
    env = ClusterEnv(
        _problem(
            wafer_routes=("A", "A", "B", "B"),
            priorities=(0, 0, 0, 1),
        )
    )
    observation, _ = env.reset()

    # Engine keeps global priority 0. Env then keeps the smallest index in
    # each Recipe, so A0 and B0 remain while A1 and B1 are masked.
    assert np.flatnonzero(observation["action_mask"]).tolist() == [0, 2]


def test_action_decode_and_explicit_place_selects_the_wafer() -> None:
    raw = _problem(wafer_routes=("A", "A"), arm_type="dual_arm").model_dump(
        mode="json", by_alias=True
    )
    raw["initial_state"] = {
        "wafers": [
            {
                "route_id": "A",
                "wafer_index": str(index),
                "priority": 0,
                "location": {
                    "kind": "robot",
                    "robot_id": "TM1",
                    "arm_id": f"arm{index}",
                },
            }
            for index in range(2)
        ]
    }
    env = ClusterEnv(parse_problem(raw))
    observation, _ = env.reset()
    place_0 = _place_action(env, 0, "PM1")
    place_1 = _place_action(env, 1, "PM1")

    assert env._decode_action(place_0) == ("place", 0, env.module_ids.index("PM1"))
    assert env._decode_action(place_1) == ("place", 1, env.module_ids.index("PM1"))
    assert observation["action_mask"][place_0]
    assert observation["action_mask"][place_1]

    env.step(place_1)
    assert env.actions[-1]["wafer_index"] == 1


def test_pick_and_advance_preserve_engine_event_boundaries() -> None:
    env = ClusterEnv(_problem(robot_position="PM2"))
    observation, _ = env.reset()

    observation, reward, terminated, truncated, info = env.step(0)
    assert reward == 0 and not terminated and not truncated
    assert (env.actions[0]["start"], env.actions[0]["end"]) == (2.0, 3.0)
    assert np.flatnonzero(observation["action_mask"]).tolist() == [_advance_action(env)]

    observation, *_ = env.step(_advance_action(env))
    assert env.time == 2.0
    assert observation["robot_phase"].tolist() == [2]
    observation, *_ = env.step(_advance_action(env))
    assert env.time == 3.0
    assert observation["action_mask"][_place_action(env, 0, "PM1")]


def test_greedy_masked_episode_finishes_with_a_valid_schedule() -> None:
    problem = _problem(wafer_routes=("A", "A", "A"))
    env = ClusterEnv(problem)
    observation, _ = env.reset()
    total_reward = 0.0

    for _ in range(200):
        legal = np.flatnonzero(observation["action_mask"])
        place = legal[
            (legal >= env._pick_action_count)
            & (legal < env.action_space.n - 1)
        ]
        action = int(place[0] if len(place) else legal[0])
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    assert terminated and not truncated
    assert total_reward == -info["time"]
    assert ValidatorSuite(problem).validate(env.actions).ok


def test_invalid_action_raises_without_dispatching() -> None:
    env = ClusterEnv(_problem())
    env.reset()

    with pytest.raises(ValueError, match="not allowed"):
        env.step(_place_action(env, 0, "IO1"))
    assert env.actions == ()


def test_load_lock_observation_distinguishes_empty_and_occupied_timers() -> None:
    env = ClusterEnv(load_lock_problem())
    observation, _ = env.reset()
    ll_index = env.module_ids.index("LL1")
    io_index = env.module_ids.index("IO1")

    assert env.observation_space.contains(observation)
    assert observation["ll_pump_time"][ll_index] == 5
    assert observation["ll_vent_time"][ll_index] == 7
    assert observation["ll_last_pick_side"][ll_index] == LoadLockSide.ATMOSPHERE
    assert observation["ll_empty_transition_progress"][ll_index] == 0
    assert observation["ll_occupied_exit_side"][ll_index] == LoadLockSide.NONE
    assert observation["ll_occupied_transition_progress"][ll_index] == 0
    assert observation["ll_pump_time"][io_index] == 0

    runtime = env.engine.state.load_locks["LL1"]
    env.engine.state.time = 2.5
    observation = env._observation()
    assert observation["ll_empty_transition_progress"][ll_index] == 0.5

    runtime.occupied_exit_side = "vacuum"
    runtime.occupied_transition_start = env.time
    runtime.occupied_transition_duration = 5
    runtime.occupied_ready_at = env.time + 5
    observation = env._observation()
    assert observation["ll_empty_transition_progress"][ll_index] == 0
    assert observation["ll_occupied_exit_side"][ll_index] == LoadLockSide.VACUUM
    assert observation["ll_occupied_transition_progress"][ll_index] == 0

    env.engine.state.time += 2.5
    observation = env._observation()
    assert observation["ll_occupied_transition_progress"][ll_index] == 0.5
