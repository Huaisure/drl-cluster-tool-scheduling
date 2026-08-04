from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cluster_rl.cluster_env import ClusterEnv
from cluster_rl.hetero_graph import GraphEnvAdapter
from cluster_rl.hetero_graph.builder import ClusterHeteroGraphBuilder
from cluster_rl.hetero_graph.feature_schema import (
    GLOBAL_FEATURE_NAMES,
    ROBOT_FEATURE_NAMES,
    ROUTE_STEP_FEATURE_NAMES,
    TIME_SCALE_SECONDS,
    WAFER_FEATURE_NAMES,
)
from problem import load_problem, parse_problem

SCENARIO = (
    Path(__file__).parents[1]
    / "examples"
    / "scenarios"
    / "long_route_1w.json"
)


def _builder_and_observation():
    env = ClusterEnv(load_problem(SCENARIO))
    builder = ClusterHeteroGraphBuilder.from_env(env)
    observation, _ = env.reset()
    action_mask = np.zeros(env.action_space.n, dtype=np.bool_)
    action_mask[0] = True
    action_mask[-1] = True
    observation["action_mask"] = action_mask
    return env, builder, observation


def test_builder_uses_new_observation_and_action_shapes() -> None:
    env, builder, observation = _builder_and_observation()

    graph = builder.build(observation)

    assert graph.nodes["wafer"].ids == env.wafer_keys
    assert graph.nodes["wafer"].feature_names == WAFER_FEATURE_NAMES
    assert graph.nodes["wafer"].features.shape[1] == len(WAFER_FEATURE_NAMES)
    assert graph.nodes["module"].ids == env.module_ids
    assert graph.nodes["robot"].ids == ("TM1",)
    assert graph.nodes["global"].ids == ("system",)
    assert graph.nodes["global"].feature_names == GLOBAL_FEATURE_NAMES
    assert graph.nodes["route_step"].ids == builder.route_step_ids
    assert graph.nodes["route_step"].feature_names == ROUTE_STEP_FEATURE_NAMES
    assert graph.edges[
        ("global", "contextualizes", "wafer")
    ].edge_index.tolist() == [[0], [0]]
    assert graph.edges[
        ("wafer", "summarizes_into", "global")
    ].edge_index.tolist() == [[0], [0]]
    assert graph.action_mask.shape == (
        len(env.wafer_keys) + len(env.module_ids),
        1,
    )
    assert graph.action_mask[:, 0].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert graph.can_advance


def test_builder_creates_bidirectional_dynamic_edges() -> None:
    env, builder, observation = _builder_and_observation()
    module_count = len(env.module_ids)
    lp_index = env.module_ids.index("LP")

    graph = builder.build(observation)
    assert graph.edges[
        ("wafer", "located_in", "module")
    ].edge_index.tolist() == [[0], [lp_index]]
    assert graph.edges[
        ("module", "contains", "wafer")
    ].edge_index.tolist() == [[lp_index], [0]]

    observation["wafer_loc"] = np.asarray(
        [module_count], dtype=np.int64
    )
    observation["robot_loc"] = np.asarray([lp_index], dtype=np.int64)
    observation["robot_holding"][0, 0] = 0
    graph = builder.build(observation)

    assert graph.edges[
        ("wafer", "located_in", "module")
    ].edge_index.shape == (2, 0)
    assert graph.edges[
        ("wafer", "held_by", "robot")
    ].edge_index.tolist() == [[0], [0]]
    assert graph.edges[
        ("robot", "holds", "wafer")
    ].edge_index.tolist() == [[0], [0]]
    assert graph.edges[
        ("robot", "located_at", "module")
    ].edge_index.tolist() == [[0], [lp_index]]


def test_route_steps_connect_wafers_and_candidate_modules() -> None:
    env, builder, observation = _builder_and_observation()
    graph = builder.build(observation)
    first_step = builder.route_step_ids.index(("A", 1))
    final_step_number = len(env.problem.routes["A"].visits) + 1
    final_step = builder.route_step_ids.index(("A", final_step_number))
    pm1 = env.module_ids.index("PM1")
    lp = env.module_ids.index("LP")

    assert graph.edges[
        ("wafer", "at_step", "route_step")
    ].edge_index.shape == (2, 0)
    assert graph.edges[
        ("wafer", "next_step", "route_step")
    ].edge_index.tolist() == [[0], [first_step]]
    can_run_on = graph.edges[
        ("route_step", "can_run_on", "module")
    ].edge_index
    assert [first_step, pm1] in can_run_on.T.tolist()
    assert [final_step, lp] in can_run_on.T.tolist()

    observation["wafer_step"][0] = 1
    graph = builder.build(observation)
    assert graph.edges[
        ("wafer", "at_step", "route_step")
    ].edge_index.tolist() == [[0], [first_step]]
    assert graph.edges[
        ("route_step", "current_for", "wafer")
    ].edge_index.tolist() == [[first_step], [0]]


def test_route_step_features_include_process_and_residency_times() -> None:
    raw_problem = json.loads(SCENARIO.read_text())
    raw_problem["routes"]["A"][0]["residency_time"] = 2.5
    env = ClusterEnv(parse_problem(raw_problem))
    builder = ClusterHeteroGraphBuilder.from_env(env)
    observation, _ = env.reset()
    observation["action_mask"] = np.zeros(
        env.action_space.n, dtype=np.bool_
    )

    features = builder.build(observation).nodes["route_step"].features
    process_time = ROUTE_STEP_FEATURE_NAMES.index("process_time")
    residency_time = ROUTE_STEP_FEATURE_NAMES.index("residency_time")
    has_residency = ROUTE_STEP_FEATURE_NAMES.index("has_residency_limit")
    is_return = ROUTE_STEP_FEATURE_NAMES.index("is_return_to_lp")
    first_step = builder.route_step_ids.index(("A", 1))
    final_step = builder.route_step_ids.index(
        ("A", len(env.problem.routes["A"].visits) + 1)
    )

    assert features[first_step, process_time] == pytest.approx(
        4.0 / TIME_SCALE_SECONDS
    )
    assert features[first_step, residency_time] == pytest.approx(
        2.5 / TIME_SCALE_SECONDS
    )
    assert features[first_step, has_residency] == 1.0
    assert features[final_step, process_time] == 0.0
    assert features[final_step, is_return] == 1.0


def test_dynamic_and_robot_times_use_the_shared_100_second_scale() -> None:
    _, builder, observation = _builder_and_observation()
    observation["process_remaining"][0] = 50.0

    graph = builder.build(observation)
    wafer_features = graph.nodes["wafer"].features[0]
    robot_features = graph.nodes["robot"].features[0]

    assert wafer_features[
        WAFER_FEATURE_NAMES.index("process_remaining")
    ] == pytest.approx(0.5)
    assert wafer_features[
        WAFER_FEATURE_NAMES.index("remaining_process_time")
    ] == pytest.approx(34.0 / TIME_SCALE_SECONDS)
    assert robot_features[
        ROBOT_FEATURE_NAMES.index("pick_time")
    ] == pytest.approx(0.01)
    assert robot_features[
        ROBOT_FEATURE_NAMES.index("place_time")
    ] == pytest.approx(0.01)
    assert robot_features[
        ROBOT_FEATURE_NAMES.index("travel_time")
    ] == pytest.approx(0.01)


def test_global_node_tracks_normalized_completion() -> None:
    env, builder, observation = _builder_and_observation()
    graph = builder.build(observation)
    features = graph.nodes["global"].features[0]

    assert features[
        GLOBAL_FEATURE_NAMES.index("completed_wafer_ratio")
    ] == 0.0
    assert features[
        GLOBAL_FEATURE_NAMES.index("completed_step_ratio")
    ] == 0.0
    assert features[
        GLOBAL_FEATURE_NAMES.index("remaining_process_ratio")
    ] == 1.0

    route_id, _ = env.wafer_keys[0]
    observation["wafer_step"][0] = (
        len(env.problem.routes[route_id].visits) + 1
    )
    graph = builder.build(observation)
    features = graph.nodes["global"].features[0]
    assert features[
        GLOBAL_FEATURE_NAMES.index("completed_wafer_ratio")
    ] == 1.0
    assert features[
        GLOBAL_FEATURE_NAMES.index("completed_step_ratio")
    ] == 1.0
    assert features[
        GLOBAL_FEATURE_NAMES.index("remaining_process_ratio")
    ] == 0.0


def test_adapter_maps_entity_and_robot_action_indexes() -> None:
    env, _, _ = _builder_and_observation()
    graph_env = GraphEnvAdapter(env)
    pm_index = env.module_ids.index("PM1")

    assert graph_env.decode_action(0).entity_id == ("A", 0)
    assert graph_env.decode_action(0).robot_id == "TM1"
    assert graph_env.encode_action("place", pm_index, 0) == (
        len(env.wafer_keys) + pm_index
    )
    assert graph_env.encode_action("advance") == env.action_space.n - 1
    assert graph_env.decode_action(env.action_space.n - 1).kind == "advance"


def test_is_ready_requires_processing_complete_and_fifo_head() -> None:
    env = ClusterEnv(
        load_problem(
            Path(__file__).parents[1]
            / "examples"
            / "scenarios"
            / "mixed_3pm_20w.json"
        )
    )
    builder = ClusterHeteroGraphBuilder.from_env(env)
    wafer_count = len(env.wafer_keys)
    observation, _ = env.reset()
    observation["wafer_loc"] = np.full(
        wafer_count,
        env.module_ids.index("LP"),
        dtype=np.int64,
    )
    observation["wafer_step"] = np.zeros(wafer_count, dtype=np.int64)
    observation["process_remaining"] = np.zeros(
        wafer_count, dtype=np.float32
    )
    observation["action_mask"] = np.zeros(
        env.action_space.n, dtype=np.bool_
    )
    is_ready_index = WAFER_FEATURE_NAMES.index("is_ready")

    graph = builder.build(observation)
    assert graph.nodes["wafer"].features[:, is_ready_index].tolist() == [
        1.0,
        *([0.0] * (wafer_count - 1)),
    ]

    observation["process_remaining"][0] = 1.0
    graph = builder.build(observation)
    assert not graph.nodes["wafer"].features[:, is_ready_index].any()

    route_id, _ = env.wafer_keys[0]
    observation["process_remaining"][0] = 0.0
    observation["wafer_step"][0] = (
        len(env.problem.routes[route_id].visits) + 1
    )
    graph = builder.build(observation)
    next_head = min(
        range(1, wafer_count),
        key=lambda index: (
            env.wafer_keys[index][1],
            env.wafer_keys[index][0],
        ),
    )
    assert graph.nodes["wafer"].features[next_head, is_ready_index] == 1.0


def test_graph_exposes_pending_robot_operation_and_pick_start_holding() -> None:
    env = ClusterEnv(load_problem(SCENARIO))
    builder = ClusterHeteroGraphBuilder.from_env(env)
    observation, _ = env.reset()
    env._robots[0].module_id = "PM4"

    observation, *_ = env.step(0)
    graph = builder.build(observation)
    robot_features = graph.nodes["robot"].features[0]
    assert robot_features[
        ROBOT_FEATURE_NAMES.index("is_travel_to_pick")
    ] == 1.0
    assert robot_features[
        ROBOT_FEATURE_NAMES.index("time_to_operation_start")
    ] == pytest.approx(1.0 / TIME_SCALE_SECONDS)
    assert robot_features[
        ROBOT_FEATURE_NAMES.index("time_to_operation_end")
    ] == pytest.approx(2.0 / TIME_SCALE_SECONDS)
    assert graph.edges[
        ("robot", "operates_on", "wafer")
    ].edge_index.tolist() == [[0], [0]]

    observation, *_ = env.step(int(env.action_space.n) - 1)
    graph = builder.build(observation)
    robot_features = graph.nodes["robot"].features[0]
    lp_index = env.module_ids.index("LP")
    assert robot_features[ROBOT_FEATURE_NAMES.index("is_picking")] == 1.0
    assert graph.edges[
        ("wafer", "located_in", "module")
    ].edge_index.tolist() == [[0], [lp_index]]
    assert graph.edges[
        ("wafer", "held_by", "robot")
    ].edge_index.tolist() == [[0], [0]]
