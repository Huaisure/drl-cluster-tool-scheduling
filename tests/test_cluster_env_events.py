from __future__ import annotations

from cluster_rl.cluster_env import ClusterEnv, RobotPhase
from problem import parse_problem


def _problem(
    *,
    robot_count: int = 1,
    wafer_count: int = 1,
    arm_type: str = "single_arm",
    robot_modules: dict[str, list[str]] | None = None,
):
    robots = {
        f"TM{index + 1}": {
            "module_ids": (robot_modules or {}).get(
                f"TM{index + 1}", ["LP", "PM1", "PM2"]
            ),
            "arm_type": arm_type,
            "travel_times": 2,
            "pick_time": 1,
            "place_time": 1.5,
        }
        for index in range(robot_count)
    }
    return parse_problem(
        {
            "Modules": {
                "LP": {"type": "LP"},
                "PM1": {"type": "PM"},
                "PM2": {"type": "PM"},
            },
            "ClusterTool": robots,
            "routes": {
                "A": [{"module_id": "PM1", "process_time": 5}],
            },
            "initial_state": {
                "robots": {
                    robot_id: {
                        "position_module_id": (
                            "PM2"
                            if "PM2" in robot["module_ids"]
                            else robot["module_ids"][0]
                        )
                    }
                    for robot_id, robot in robots.items()
                },
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": str(index),
                        "step_index": 0,
                        "location": {"kind": "module", "module_id": "LP"},
                    }
                    for index in range(wafer_count)
                ],
            },
        }
    )


def test_advance_applies_pick_and_place_boundaries() -> None:
    env = ClusterEnv(_problem())
    env.reset()

    env._pick(0, 0)
    assert env._time == 0.0
    assert env._robots[0].holding == []
    assert env._wafers[0].module_id == "LP"

    assert env._advance()
    assert env._time == 2.0
    assert env._robots[0].holding == [0]
    assert env._wafers[0].module_id == "LP"

    assert env._advance()
    assert env._time == 3.0
    assert env._wafers[0].module_id is None
    assert env._wafers[0].robot_id == "TM1"

    pm_index = env.module_ids.index("PM1")
    env._place(pm_index, 0)
    assert env._wafers[0].step_index == 0
    assert env._module_occupancy(pm_index) == 0
    assert not env._has_capacity(pm_index)

    assert env._advance()
    assert env._time == 5.0
    assert env._module_occupancy(pm_index) == 1
    assert not env._has_capacity(pm_index)
    assert env._robots[0].holding == [0]

    assert env._advance()
    assert env._time == 6.5
    assert env._robots[0].holding == []
    assert env._wafers[0].module_id == "PM1"
    assert env._wafers[0].step_index == 1
    assert env._wafers[0].ready_at == 11.5

    assert env._advance()
    assert env._time == 11.5
    assert env._wafers[0].ready_at <= env._time
    assert not env._advance()


def test_observation_exposes_robot_holding_and_pending_operation() -> None:
    env = ClusterEnv(_problem())
    observation, _ = env.reset()
    wafer_sentinel = len(env.wafer_keys)
    module_sentinel = len(env.module_ids)
    lp_index = env.module_ids.index("LP")

    assert env.observation_space.contains(observation)
    assert observation["robot_holding"].tolist() == [[wafer_sentinel]]
    assert observation["robot_phase"].tolist() == [RobotPhase.IDLE]
    assert observation["robot_operation_wafer"].tolist() == [wafer_sentinel]
    assert observation["robot_operation_module"].tolist() == [module_sentinel]

    observation, *_ = env.step(0)
    assert observation["robot_phase"].tolist() == [
        RobotPhase.TRAVEL_TO_PICK
    ]
    assert observation["robot_operation_wafer"].tolist() == [0]
    assert observation["robot_operation_module"].tolist() == [lp_index]
    assert observation["time_to_operation_start"].tolist() == [2.0]
    assert observation["time_to_operation_end"].tolist() == [3.0]
    assert observation["robot_holding"].tolist() == [[wafer_sentinel]]

    observation, *_ = env.step(int(env.action_space.n) - 1)
    assert observation["robot_phase"].tolist() == [RobotPhase.PICKING]
    assert observation["time_to_operation_start"].tolist() == [0.0]
    assert observation["time_to_operation_end"].tolist() == [1.0]
    assert observation["robot_holding"].tolist() == [[0]]

    observation, *_ = env.step(int(env.action_space.n) - 1)
    assert observation["robot_phase"].tolist() == [RobotPhase.IDLE]
    assert observation["robot_operation_wafer"].tolist() == [wafer_sentinel]
    assert observation["robot_operation_module"].tolist() == [module_sentinel]
    assert observation["time_to_operation_end"].tolist() == [0.0]
    assert observation["robot_holding"].tolist() == [[0]]


def test_advance_applies_all_starts_at_the_same_timestamp() -> None:
    env = ClusterEnv(_problem(robot_count=2, wafer_count=2))
    env.reset()
    env._wafers[1].module_id = "PM2"
    env._robots[1].module_id = "LP"

    env._pick(0, 0)
    env._pick(1, 1)

    assert env._advance()
    assert env._time == 2.0
    assert env._robots[0].holding == [0]
    assert env._robots[1].holding == [1]
    assert [wafer.module_id for wafer in env._wafers] == ["LP", "PM2"]

    assert env._advance()
    assert env._time == 3.0
    assert all(wafer.module_id is None for wafer in env._wafers)


def test_pending_pick_reserves_the_wafer_across_robots() -> None:
    env = ClusterEnv(_problem(robot_count=2))
    observation, _ = env.reset()

    assert observation["action_mask"][:2].tolist() == [1, 1]
    env.step(0)

    assert env._is_pick_reserved(0)
    assert env._action_mask()[1] == 0
    try:
        env._pick(0, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("the second robot must not reserve the same wafer")


def test_entity_major_action_decode_with_multiple_robots() -> None:
    env = ClusterEnv(_problem(robot_count=2))
    env.reset()
    pm_index = env.module_ids.index("PM1")
    place_entity = len(env.wafer_keys) + pm_index

    assert env._decode_action(0) == ("pick", 0, 0)
    assert env._decode_action(1) == ("pick", 0, 1)
    assert env._decode_action(place_entity * 2) == ("place", pm_index, 0)
    assert env._decode_action(place_entity * 2 + 1) == (
        "place",
        pm_index,
        1,
    )
    assert env._decode_action(int(env.action_space.n) - 1) == (
        "advance",
        None,
        None,
    )


def test_pick_checks_robot_reachability_and_dual_arm_capacity() -> None:
    env = ClusterEnv(
        _problem(
            robot_count=2,
            wafer_count=2,
            arm_type="dual_arm",
            robot_modules={
                "TM1": ["LP"],
                "TM2": ["LP", "PM1", "PM2"],
            },
        )
    )
    env.reset()

    assert not env._can_pick(0, 0)
    assert env._can_pick(0, 1)

    env._wafers[0].module_id = None
    env._wafers[0].robot_id = "TM2"
    env._robots[1].holding = [0]
    env._wafers[1].module_id = "PM2"
    assert env._can_pick(1, 1)

    env._robots[1].holding.append(1)
    assert not env._can_pick(1, 1)


def test_place_uses_first_matching_wafer_in_holding_order() -> None:
    env = ClusterEnv(_problem(wafer_count=2, arm_type="dual_arm"))
    env.reset()
    for wafer in env._wafers:
        wafer.module_id = None
        wafer.robot_id = "TM1"
    env._robots[0].holding = [1, 0]
    assert env._observation()["robot_holding"].tolist() == [[1, 0]]

    pm_index = env.module_ids.index("PM1")
    env._place(pm_index, 0)

    assert env._pending_queue[-1].wafer_index == 1
    assert env.actions[-1]["wafer_index"] == 1


def test_pick_requires_capacity_in_at_least_one_next_module() -> None:
    env = ClusterEnv(_problem(wafer_count=2))
    env.reset()
    env._wafers[1].module_id = "PM1"

    assert not env._can_pick(0, 0)
