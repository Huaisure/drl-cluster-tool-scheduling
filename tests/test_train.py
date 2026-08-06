from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from cluster_generator import ProblemGenerator
from examples.run_scenarios import SCENARIO_DIR
from cluster_rl.network import ClusterActorCritic, TransformerConfig
from cluster_rl.train import (
    GeneratorEnvFactory,
    ParallelEnvPool,
    PPOConfig,
    _advantages,
    _collect_rollout,
    _evaluation_problems,
    _first_legal_reference,
    _manifest_problem_paths,
    _normalized_reward,
    _step_env_slot,
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


def test_normalized_reward_has_bounded_telescoping_time_cost() -> None:
    raw_rewards = [-40.0, -60.0, -45.0]
    current_times = [40.0, 100.0, 145.0]
    normalized = [
        _normalized_reward(
            reward,
            145.0,
            index == 2,
            current_time=current_times[index],
            deadlocked=False,
            completed_step_ratio=1.0 if index == 2 else 0.0,
        )
        for index, reward in enumerate(raw_rewards)
    ]

    assert sum(normalized) == pytest.approx(0.75)


def test_normalized_reward_strictly_separates_success_and_deadlock() -> None:
    success = _normalized_reward(
        -1000.0,
        100.0,
        True,
        current_time=1000.0,
        deadlocked=False,
        completed_step_ratio=1.0,
    )
    deadlock = _normalized_reward(
        0.0,
        100.0,
        False,
        current_time=0.0,
        deadlocked=True,
        completed_step_ratio=1.0,
    )

    assert success > 0.5
    assert deadlock == -1.0
    assert success > deadlock


def test_deadlock_reward_prefers_more_completed_route_steps() -> None:
    early = _normalized_reward(
        -100.0,
        100.0,
        False,
        current_time=100.0,
        deadlocked=True,
        completed_step_ratio=0.2,
    )
    late = _normalized_reward(
        -100.0,
        100.0,
        False,
        current_time=100.0,
        deadlocked=True,
        completed_step_ratio=0.8,
    )

    assert late > early


def test_cpu_workers_require_generator_slots() -> None:
    with pytest.raises(ValueError, match="only supported in generator mode"):
        PPOConfig(
            scenario_paths=(SCENARIO_DIR / "long_route_1w.json",),
            cpu_workers=1,
        )

    with pytest.raises(ValueError, match="must not exceed num_envs"):
        PPOConfig(
            train_mode="generator",
            num_envs=1,
            cpu_workers=2,
            evaluate=False,
        )


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

    rollout_batch, stats = _collect_rollout(
        model,
        slots,
        config,
        torch.device("cpu"),
        factory,
    )

    assert len(stats) == 1
    assert len(rollout_batch.states) == config.rollout_steps
    for encoded, action in zip(
        rollout_batch.states,
        rollout_batch.actions.tolist(),
    ):
        assert encoded.action_mask[action]
    assert slots[0].episode_index == 1
    assert [call[0] for call in generator.calls] == [5, 6]


def test_parallel_env_pool_advances_and_replaces_completed_slot() -> None:
    generator = _ProblemGeneratorStub()
    config = PPOConfig(
        train_mode="generator",
        num_envs=1,
        cpu_workers=1,
        generator_seed=5,
        evaluate=False,
    )
    slot = GeneratorEnvFactory(config, generator=generator).make(0)
    pool = ParallelEnvPool([slot], config)

    try:
        completed = []
        for _ in range(100):
            action = int(np.flatnonzero(pool.encoded[0].action_mask)[0])
            _, _, stats = pool.step([action])
            completed.extend(stats)
            if completed:
                break
    finally:
        pool.close()

    assert len(completed) == 1
    assert pool.states[0].episode_index == 1


def test_parallel_env_pool_matches_serial_transition() -> None:
    generator = _ProblemGeneratorStub()
    config = PPOConfig(
        train_mode="generator",
        num_envs=1,
        cpu_workers=1,
        generator_seed=5,
        evaluate=False,
    )
    slot = GeneratorEnvFactory(config, generator=generator).make(0)
    serial_slot = pickle.loads(pickle.dumps(slot))
    action = int(np.flatnonzero(slot.observation["action_mask"])[0])
    pool = ParallelEnvPool([slot], config)

    try:
        parallel_rewards, parallel_dones, _ = pool.step([action])
        serial_slot, serial_reward, serial_done, _ = _step_env_slot(
            serial_slot,
            action,
            0,
            None,
        )
        serial_encoded = serial_slot.encoder.encode(serial_slot.observation)
        parallel_encoded = pool.encoded[0]
    finally:
        pool.close()

    assert parallel_rewards == pytest.approx([serial_reward])
    assert parallel_dones == [serial_done]
    np.testing.assert_array_equal(
        parallel_encoded.action_mask,
        serial_encoded.action_mask,
    )
    for node_type in serial_encoded.graph.node_types:
        torch.testing.assert_close(
            parallel_encoded.graph[node_type].x,
            serial_encoded.graph[node_type].x,
        )


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
