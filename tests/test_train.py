from __future__ import annotations

from pathlib import Path

import torch

from examples.run_scenarios import SCENARIO_DIR
from train import PPOConfig, _advantages, train


def test_advantages_stop_bootstrapping_at_episode_end() -> None:
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    dones = torch.tensor([[False], [True], [False]])
    values = torch.zeros_like(rewards)

    result = _advantages(
        rewards,
        dones,
        values,
        last_values=torch.tensor([4.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(
        result,
        torch.tensor([[3.0], [2.0], [7.0]]),
    )


def test_short_ppo_training_writes_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    config = PPOConfig(
        scenario_paths=tuple(sorted(SCENARIO_DIR.glob("*.json"))),
        checkpoint=checkpoint,
        total_steps=12,
        rollout_steps=4,
        epochs=1,
        minibatch_size=12,
        model_dim=16,
        num_heads=4,
        num_layers=1,
        feedforward_dim=32,
        checkpoint_interval=1,
        evaluate=False,
    )

    summary = train(config)

    assert summary["global_step"] == 12
    assert summary["updates"] == 1
    assert summary["checkpoint"] == str(checkpoint)
    assert checkpoint.is_file()
