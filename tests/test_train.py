from __future__ import annotations

from pathlib import Path
import json

import pytest
import torch

from cluster_generator import ProblemGenerator
from examples.run_scenarios import SCENARIO_DIR
from cluster_rl.network import ClusterActorCritic, TransformerConfig
from cluster_rl.train import (
    GeneratorEnvFactory,
    PPOConfig,
    _advantages,
    _collect_rollout,
    _evaluation_problems,
    _first_legal_reference,
    _manifest_problem_paths,
    _normalized_reward,
    train,
)
from tests.test_cluster_env import _problem


class _ProblemGeneratorStub:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, dict[str, float] | None]] = []

    def sample_curriculum(self, seed, *, split, weights):
        self.calls.append((seed, split, weights))
        return _problem()


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


def test_normalized_reward_centers_reference_schedule_at_zero() -> None:
    raw_rewards = [-40.0, -60.0, -45.0]
    normalized = [
        _normalized_reward(reward, 145.0, index == 2)
        for index, reward in enumerate(raw_rewards)
    ]

    assert sum(normalized) == pytest.approx(0.0)


@pytest.mark.parametrize("seed", [43, 49])
def test_fifo_serial_reference_completes_previous_deadlock_seeds(seed: int) -> None:
    problem = ProblemGenerator().sample_curriculum(seed=seed, split="train")

    assert _first_legal_reference(problem, f"seed_{seed}") > 0


def test_generator_factory_uses_disjoint_deterministic_episode_seeds() -> None:
    generator = _ProblemGeneratorStub()
    config = PPOConfig(
        train_mode="generator",
        num_envs=3,
        generator_seed=100,
        evaluate=False,
    )
    factory = GeneratorEnvFactory(config, generator=generator)

    first = factory.make(slot_index=1, episode_index=0)
    second = factory.make(slot_index=1, episode_index=2)

    assert first.episode_index == 0
    assert second.episode_index == 2
    assert [call[:2] for call in generator.calls] == [
        (101, "train"),
        (107, "train"),
    ]
    assert generator.calls[0][2] == config.difficulty_weights


def test_rollout_replaces_completed_generator_slot() -> None:
    generator = _ProblemGeneratorStub()
    config = PPOConfig(
        train_mode="generator",
        num_envs=1,
        generator_seed=5,
        evaluate=False,
        rollout_steps=12,
    )
    factory = GeneratorEnvFactory(config, generator=generator)
    slots = [factory.make(0)]
    model = ClusterActorCritic(
        TransformerConfig(
            model_dim=16,
            num_heads=4,
            hgt_layers=1,
            num_layers=1,
            feedforward_dim=32,
            dropout=0.0,
        )
    )

    _, stats = _collect_rollout(
        model,
        slots,
        config,
        torch.device("cpu"),
        factory,
    )

    assert len(stats) == 1
    assert slots[0].episode_index == 1
    assert [call[0] for call in generator.calls] == [5, 6]


def test_manifest_problem_paths_load_only_materialized_local_files(
    tmp_path: Path,
) -> None:
    problem_path = tmp_path / "validation-00000.json"
    problem_path.write_text(
        (SCENARIO_DIR / "long_route_1w.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generator": {"mode": "ppo"},
                "instances": [{"problem_file": problem_path.name}],
            }
        ),
        encoding="utf-8",
    )

    assert _manifest_problem_paths(manifest_path) == (problem_path.resolve(),)

    cases = _evaluation_problems(
        PPOConfig(
            train_mode="generator",
            validation_manifest=manifest_path,
            test_manifest=manifest_path,
        )
    )
    assert [split for split, _ in cases] == ["validation", "test"]


def test_short_ppo_training_writes_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "model.pt"
    config = PPOConfig(
        scenario_paths=tuple(sorted(SCENARIO_DIR.glob("*.json"))),
        run_dir=run_dir,
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
    assert (run_dir / "updates.csv").is_file()
    assert (run_dir / "episodes.csv").is_file()
    assert (run_dir / "evaluation.csv").is_file()
    assert (run_dir / "config.json").is_file()
    assert "Masked PPO training" in (run_dir / "train.log").read_text()
    assert (run_dir / "training_curves.png").is_file()
