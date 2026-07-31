from __future__ import annotations

from pathlib import Path

from cluster_rl.cluster_env import ClusterEnv
from cluster_rl.hetero_graph import GraphEnvAdapter
from problem import load_problem

SCENARIO = (
    Path(__file__).parents[1]
    / "examples"
    / "scenarios"
    / "long_route_1w.json"
)


def test_adapter_builds_graph_with_stable_action_indexes() -> None:
    env = ClusterEnv(load_problem(SCENARIO))
    graph_env = GraphEnvAdapter(env)

    graph, info = graph_env.reset()

    assert info == {"time": 0.0}
    assert graph.nodes["wafer"].ids == env.wafer_keys
    assert graph.nodes["module"].ids == env.module_ids
    assert graph.nodes["robot"].ids == ("TM1",)
    assert graph.action_mask.tolist() == [True, False, False, False, False, False]
    assert graph_env.decode_action(0).entity_id == ("A", 0)
    assert graph_env.encode_action(
        "place", env.module_ids.index("PM1")
    ) == 2


def test_dynamic_edges_change_after_env_step() -> None:
    graph_env = GraphEnvAdapter(ClusterEnv(load_problem(SCENARIO)))
    graph, _ = graph_env.reset()

    assert graph.edges[
        ("wafer", "located_in", "module")
    ].edge_index.shape == (2, 1)
    assert graph.edges[
        ("wafer", "held_by", "robot")
    ].edge_index.shape == (2, 0)

    graph, _, terminated, truncated, info = graph_env.step(0)

    assert not terminated and not truncated
    assert graph.graph_features.tolist() == [info["time"]]
    assert graph.edges[
        ("wafer", "located_in", "module")
    ].edge_index.shape == (2, 0)
    assert graph.edges[
        ("wafer", "held_by", "robot")
    ].edge_index.tolist() == [[0], [0]]
    assert graph.action_mask.tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
