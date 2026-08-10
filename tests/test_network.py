from __future__ import annotations

from pathlib import Path

import torch

from cluster_rl.cluster_env import ClusterEnv
from cluster_rl.network import (
    ADVANCE_ACTION,
    PICK_ACTION,
    ClusterActorCritic,
    ClusterObservationEncoder,
    TransformerConfig,
    collate_encoded_observations,
    collate_encoded_observations_fast,
    collate_observations,
)
from problem import load_problem, parse_problem
from tests.problem_fixtures import load_lock_problem

SCENARIO_DIR = Path(__file__).parents[1] / "examples" / "scenarios"


def _model() -> ClusterActorCritic:
    return ClusterActorCritic(
        TransformerConfig(
            model_dim=24,
            num_heads=4,
            hgt_layers=1,
            num_layers=1,
            feedforward_dim=48,
            dropout=0.0,
        )
    )


def _env(path: str = "long_route_1w.json"):
    env = ClusterEnv(load_problem(SCENARIO_DIR / path))
    observation, _ = env.reset()
    return env, observation


def test_collate_keeps_only_legal_actions_and_pads_scenarios() -> None:
    first_env, first_observation = _env("long_route_1w.json")
    second_env, second_observation = _env("mixed_3pm_20w.json")
    batch = collate_observations(
        [
            ClusterObservationEncoder.from_env(first_env),
            ClusterObservationEncoder.from_env(second_env),
        ],
        [first_observation, second_observation],
    )

    assert batch.batch_size == 2
    first_legal = torch.from_numpy(first_observation["action_mask"].nonzero()[0])
    second_legal = torch.from_numpy(second_observation["action_mask"].nonzero()[0])
    assert batch.action_mask.shape == (
        2,
        max(len(first_legal), len(second_legal)),
    )
    assert batch.action_valid[0].sum() == len(first_legal)
    assert batch.action_valid[1].sum() == len(second_legal)
    assert torch.equal(batch.env_action_indices[0, : len(first_legal)], first_legal)
    assert torch.equal(
        batch.env_action_indices[1, : len(second_legal)],
        second_legal,
    )
    assert batch.action_kind[0, 0] == PICK_ACTION


def test_forward_masks_illegal_and_padded_actions() -> None:
    first_env, first_observation = _env("long_route_1w.json")
    second_env, second_observation = _env("mixed_3pm_20w.json")
    batch = collate_observations(
        [
            ClusterObservationEncoder.from_env(first_env),
            ClusterObservationEncoder.from_env(second_env),
        ],
        [first_observation, second_observation],
    )

    output = _model()(batch)

    assert output.logits.shape == batch.action_mask.shape
    assert output.value.shape == (2,)
    assert torch.isfinite(output.logits[batch.action_mask]).all()
    assert torch.isneginf(output.logits[~batch.action_mask]).all()
    assert torch.isfinite(output.value).all()


def test_forward_accepts_conversion_load_lock_relations() -> None:
    env = ClusterEnv(load_lock_problem())
    observation, _ = env.reset()
    batch = collate_observations(
        [ClusterObservationEncoder.from_env(env)],
        [observation],
    )

    output = _model()(batch)

    assert torch.isfinite(output.logits[batch.action_mask]).all()
    assert torch.isfinite(output.value).all()


def test_fast_collate_matches_generic_pyg_batch() -> None:
    first_env, first_observation = _env("long_route_1w.json")
    second_env, second_observation = _env("mixed_3pm_20w.json")
    encoded = [
        ClusterObservationEncoder.from_env(env).encode(observation)
        for env, observation in (
            (first_env, first_observation),
            (second_env, second_observation),
        )
    ]

    generic = collate_encoded_observations(encoded)
    fast = collate_encoded_observations_fast(encoded)

    for name in (
        "action_mask",
        "action_valid",
        "env_action_indices",
        "action_kind",
        "action_entity",
        "action_target",
    ):
        assert torch.equal(getattr(fast, name), getattr(generic, name))
    for node_type in generic.graph.node_types:
        assert torch.equal(fast.graph[node_type].x, generic.graph[node_type].x)
        assert torch.equal(
            fast.graph[node_type].batch,
            generic.graph[node_type].batch,
        )
        assert torch.equal(
            fast.graph[node_type].ptr,
            generic.graph[node_type].ptr,
        )
    for edge_type in generic.graph.edge_types:
        assert torch.equal(
            fast.graph[edge_type].edge_index,
            generic.graph[edge_type].edge_index,
        )

    model = _model().eval()
    with torch.no_grad():
        generic_output = model(generic)
        fast_output = model(fast)
    assert torch.equal(fast_output.logits, generic_output.logits)
    assert torch.equal(fast_output.value, generic_output.value)


def test_compact_action_indexes_map_to_environment_actions() -> None:
    env, observation = _env()
    observation["action_mask"][:] = 0
    observation["action_mask"][2] = 1
    observation["action_mask"][-1] = 1
    batch = collate_observations(
        [ClusterObservationEncoder.from_env(env)],
        [observation],
    )
    env_action = torch.tensor([env.action_space.n - 1])
    model_action = torch.tensor([1])

    assert batch.env_action_indices.tolist() == [[2, env.action_space.n - 1]]
    assert batch.to_model_actions(env_action).tolist() == [1]
    assert batch.to_env_actions(model_action).tolist() == [env.action_space.n - 1]


def test_hgt_decoder_actor_and_value_receive_gradients() -> None:
    env, observation = _env()
    batch = collate_observations(
        [ClusterObservationEncoder.from_env(env)],
        [observation],
    )
    model = _model()
    output = model(batch)

    (-output.logits[0, 0] + output.value.square().mean()).backward()

    assert model.hgt_layers[0].kqv_lin.lins["wafer"].weight.grad is not None
    assert model.decoder.layers[0].multihead_attn.in_proj_weight.grad is not None
    assert model.actor_head[-1].weight.grad is not None
    assert model.value_head[-1].weight.grad is not None


def test_pick_uses_robot_targets_and_place_uses_module_targets() -> None:
    problem = parse_problem(
        {
            "Modules": {"LP": {"type": "LP"}, "PM1": {"type": "PM"}},
            "ClusterTool": {
                robot_id: {
                    "module_ids": ["LP", "PM1"],
                    "arm_type": "single_arm",
                    "travel_times": 1,
                    "pick_time": 1,
                    "place_time": 1,
                }
                for robot_id in ("TM1", "TM2")
            },
            "routes": {"A": [{"module_id": "PM1", "process_time": 1}]},
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 0,
                        "location": {"kind": "module", "module_id": "LP"},
                    }
                ]
            },
        }
    )
    env = ClusterEnv(problem)
    observation, _ = env.reset()
    batch = collate_observations(
        [ClusterObservationEncoder.from_env(env)],
        [observation],
    )
    output = _model()(batch)

    assert output.logits.shape == (1, 2)
    assert batch.env_action_indices.tolist() == [[0, 1]]
    assert torch.isfinite(output.logits).all()
