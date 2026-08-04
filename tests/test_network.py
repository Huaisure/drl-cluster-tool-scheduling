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
    collate_observations,
)
from problem import load_problem, parse_problem

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


def test_collate_uses_environment_action_layout_and_pads_scenarios() -> None:
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
    assert batch.action_mask.shape == (2, second_env.action_space.n)
    assert batch.action_valid[0].sum() == first_env.action_space.n
    assert batch.action_valid[1].sum() == second_env.action_space.n
    assert batch.action_mask[0, 0]
    assert not batch.action_mask[0, first_env.action_space.n :].any()
    assert batch.action_kind[0, 0] == PICK_ACTION
    assert batch.action_kind[0, first_env.action_space.n - 1] == ADVANCE_ACTION


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


def test_action_indexes_are_identical_between_model_and_environment() -> None:
    env, observation = _env()
    batch = collate_observations(
        [ClusterObservationEncoder.from_env(env)],
        [observation],
    )
    action = torch.tensor([0])

    assert batch.to_model_actions(action).tolist() == [0]
    assert batch.to_env_actions(action).tolist() == [0]


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


def test_multiple_robots_expand_entity_major_actions() -> None:
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

    assert output.logits.shape == (1, (1 + 2) * 2 + 1)
    assert torch.isfinite(output.logits[0, :2]).all()
