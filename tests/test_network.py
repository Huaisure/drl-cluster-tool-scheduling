from __future__ import annotations

from collections import Counter

import torch

from cluster_env import ClusterEnv
from network import (
    ClusterActorCritic,
    ClusterObservationEncoder,
    TransformerConfig,
    collate_observations,
)
from problem import parse_problem


def _problem(
    *,
    wafer_routes: tuple[str, ...] = ("A",),
    include_pm2: bool = True,
):
    modules = {
        "LP": {"type": "LP"},
        "PM1": {"type": "PM"},
    }
    candidates = ["PM1"]
    if include_pm2:
        modules["PM2"] = {"type": "PM"}
        candidates.append("PM2")

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

    return parse_problem(
        {
            "Modules": modules,
            "ClusterTool": {
                "TM1": {
                    "module_ids": list(modules),
                    "arm_type": "single_arm",
                    "travel_times": 1,
                    "pick_time": 1,
                    "place_time": 1,
                }
            },
            "routes": {
                "A": [
                    {
                        "module_ids": candidates,
                        "process_time": 2,
                    }
                ]
            },
            "initial_state": {"wafers": wafers},
        }
    )


def _env_and_observation(**kwargs):
    env = ClusterEnv(_problem(**kwargs))
    observation, _ = env.reset()
    return env, observation


def test_collate_builds_relations_and_pads_action_sections() -> None:
    first_env, first_observation = _env_and_observation(
        wafer_routes=("A", "A")
    )
    second_env, second_observation = _env_and_observation(
        include_pm2=False
    )
    encoders = [
        ClusterObservationEncoder.from_env(first_env),
        ClusterObservationEncoder.from_env(second_env),
    ]

    batch = collate_observations(
        encoders, [first_observation, second_observation]
    )

    assert batch.wafer_features.shape == (2, 2, 5)
    assert batch.module_features.shape == (2, 3, 7)
    assert batch.wafer_valid.tolist() == [[True, True], [True, False]]
    assert batch.module_valid.tolist() == [
        [True, True, True],
        [True, True, False],
    ]
    assert batch.action_mask.tolist() == [
        [True, True, False, False, False],
        [True, False, False, False, False],
    ]

    first_modules = {
        module_id: index
        for index, module_id in enumerate(first_env.module_ids)
    }
    assert batch.candidate_modules[
        0, 0, first_modules["PM1"]
    ]
    assert batch.candidate_modules[
        0, 0, first_modules["PM2"]
    ]
    assert not batch.candidate_modules[
        0, 0, first_modules["LP"]
    ]
    assert batch.wafer_locations[
        0, 0, first_modules["LP"]
    ]


def test_forward_masks_illegal_and_padded_actions() -> None:
    first_env, first_observation = _env_and_observation(
        wafer_routes=("A", "A")
    )
    second_env, second_observation = _env_and_observation(
        include_pm2=False
    )
    encoders = [
        ClusterObservationEncoder.from_env(first_env),
        ClusterObservationEncoder.from_env(second_env),
    ]
    batch = collate_observations(
        encoders, [first_observation, second_observation]
    )
    model = ClusterActorCritic(
        TransformerConfig(
            model_dim=32,
            num_heads=4,
            num_layers=2,
            feedforward_dim=64,
            dropout=0.0,
        )
    )

    output = model(batch)

    assert output.logits.shape == (2, 5)
    assert output.value.shape == (2,)
    assert torch.isfinite(output.logits[batch.action_mask]).all()
    assert torch.isneginf(output.logits[~batch.action_mask]).all()
    assert torch.isfinite(output.value).all()


def test_action_indexes_round_trip_through_padded_layout() -> None:
    first_env, first_observation = _env_and_observation(
        wafer_routes=("A", "A")
    )
    second_env, second_observation = _env_and_observation(
        include_pm2=False
    )
    batch = collate_observations(
        [
            ClusterObservationEncoder.from_env(first_env),
            ClusterObservationEncoder.from_env(second_env),
        ],
        [first_observation, second_observation],
    )
    env_actions = torch.tensor([3, 2])

    model_actions = batch.to_model_actions(env_actions)

    assert model_actions.tolist() == [3, 3]
    assert batch.to_env_actions(model_actions).tolist() == [3, 2]


def test_pick_place_and_value_heads_receive_gradients() -> None:
    env, observation = _env_and_observation()
    encoder = ClusterObservationEncoder.from_env(env)
    model = ClusterActorCritic(
        TransformerConfig(
            model_dim=24,
            num_heads=4,
            num_layers=1,
            feedforward_dim=48,
            dropout=0.0,
        )
    )

    pick_batch = collate_observations([encoder], [observation])
    pick_output = model(pick_batch)
    pick_loss = -pick_output.logits[0, 0] + pick_output.value.square().mean()

    place_observation, *_ = env.step(0)
    place_batch = collate_observations([encoder], [place_observation])
    place_output = model(place_batch)
    place_loss = -place_output.logits[
        0, pick_batch.wafer_features.shape[1] + 1
    ]

    (pick_loss + place_loss).backward()

    assert model.pick_head.weight.grad is not None
    assert model.place_head.weight.grad is not None
    assert model.value_head[-1].weight.grad is not None
    assert model.relation_bias.grad is not None
    assert torch.isfinite(model.relation_bias.grad).all()


def test_wafer_tokens_are_permutation_equivariant() -> None:
    env, observation = _env_and_observation(
        wafer_routes=("A", "A")
    )
    encoder = ClusterObservationEncoder.from_env(env)
    batch = collate_observations([encoder], [observation])
    model = ClusterActorCritic(
        TransformerConfig(
            model_dim=32,
            num_heads=4,
            num_layers=2,
            feedforward_dim=64,
            dropout=0.0,
        )
    ).eval()

    permutation = torch.tensor([1, 0])
    permuted = type(batch)(
        robot_features=batch.robot_features,
        wafer_features=batch.wafer_features[:, permutation],
        module_features=batch.module_features,
        candidate_modules=batch.candidate_modules[:, permutation],
        wafer_locations=batch.wafer_locations[:, permutation],
        robot_location=batch.robot_location,
        robot_holds=batch.robot_holds[:, permutation],
        wafer_valid=batch.wafer_valid[:, permutation],
        module_valid=batch.module_valid,
        action_mask=torch.cat(
            (
                batch.action_mask[:, :2][:, permutation],
                batch.action_mask[:, 2:],
            ),
            dim=1,
        ),
    )

    original_output = model(batch)
    permuted_output = model(permuted)

    torch.testing.assert_close(
        permuted_output.logits[:, :2],
        original_output.logits[:, :2][:, permutation],
    )
    torch.testing.assert_close(
        permuted_output.logits[:, 2:],
        original_output.logits[:, 2:],
    )
    torch.testing.assert_close(
        permuted_output.value,
        original_output.value,
    )
