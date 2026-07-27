from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from cluster_env import ClusterEnv
from examples.run_scenarios import (
    SCENARIO_DIR,
    network_greedy_selector,
    rollout,
)
from network import (
    ClusterActorCritic,
    ClusterObservationEncoder,
    EntityBatch,
    TransformerConfig,
    collate_observations,
)
from problem import load_problem


@dataclass(frozen=True, slots=True)
class PPOConfig:
    scenario_paths: tuple[Path, ...]
    checkpoint: Path = Path("checkpoints/ppo_cluster.pt")
    resume: Path | None = None
    total_steps: int = 100_000
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    reward_scale: float = 0.01
    model_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    feedforward_dim: int = 128
    seed: int = 0
    device: str = "cpu"
    log_interval: int = 1
    checkpoint_interval: int = 25
    evaluate: bool = True

    def __post_init__(self) -> None:
        positive_integers = {
            "total_steps": self.total_steps,
            "rollout_steps": self.rollout_steps,
            "epochs": self.epochs,
            "minibatch_size": self.minibatch_size,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.scenario_paths:
            raise ValueError("at least one scenario is required")
        if not 0.0 < self.learning_rate:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 < self.clip_coefficient < 1.0:
            raise ValueError("clip_coefficient must be in (0, 1)")
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive")


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    states: EntityBatch
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor


@dataclass(frozen=True, slots=True)
class EpisodeStat:
    scenario: str
    makespan: float
    reward: float
    success: bool


def _default_scenarios() -> tuple[Path, ...]:
    return tuple(sorted(SCENARIO_DIR.glob("*.json")))


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def _stack_entity_batches(batches: list[EntityBatch]) -> EntityBatch:
    tensors = {}
    for field in fields(EntityBatch):
        values = torch.stack(
            [getattr(batch, field.name) for batch in batches],
            dim=0,
        )
        tensors[field.name] = values.flatten(0, 1)
    return EntityBatch(**tensors)


def _select_states(states: EntityBatch, indexes: Tensor) -> EntityBatch:
    return EntityBatch(
        **{
            field.name: getattr(states, field.name).index_select(0, indexes)
            for field in fields(EntityBatch)
        }
    )


def _advantages(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    last_values: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tensor:
    result = torch.zeros_like(rewards)
    advantage = torch.zeros_like(last_values)

    for step in reversed(range(rewards.shape[0])):
        next_values = (
            last_values if step == rewards.shape[0] - 1
            else values[step + 1]
        )
        next_non_terminal = ~dones[step]
        delta = (
            rewards[step]
            + gamma * next_values * next_non_terminal
            - values[step]
        )
        advantage = (
            delta
            + gamma * gae_lambda * next_non_terminal * advantage
        )
        result[step] = advantage
    return result


def _collect_rollout(
    model: ClusterActorCritic,
    envs: list[ClusterEnv],
    encoders: list[ClusterObservationEncoder],
    observations: list[dict[str, Any]],
    episode_rewards: list[float],
    config: PPOConfig,
    device: torch.device,
) -> tuple[RolloutBatch, list[dict[str, Any]], list[EpisodeStat]]:
    state_steps = []
    action_steps = []
    log_probability_steps = []
    value_steps = []
    reward_steps = []
    done_steps = []
    episode_stats = []

    model.eval()
    for _ in range(config.rollout_steps):
        states = collate_observations(
            encoders,
            observations,
            device=device,
        )
        with torch.inference_mode():
            output = model(states)
            distribution = Categorical(logits=output.logits)
            model_actions = distribution.sample()

        env_actions = states.to_env_actions(model_actions).cpu().tolist()
        next_observations = []
        rewards = []
        dones = []
        for index, (env, action) in enumerate(zip(envs, env_actions)):
            observation, reward, terminated, truncated, info = env.step(
                action
            )
            done = terminated or truncated
            episode_rewards[index] += reward
            rewards.append(reward * config.reward_scale)
            dones.append(done)

            if done:
                episode_stats.append(
                    EpisodeStat(
                        scenario=str(
                            env.problem.meta.get("name", f"env_{index}")
                        ),
                        makespan=float(info["time"]),
                        reward=episode_rewards[index],
                        success=bool(info.get("is_success")),
                    )
                )
                observation, _ = env.reset()
                episode_rewards[index] = 0.0
            next_observations.append(observation)

        state_steps.append(states)
        action_steps.append(model_actions)
        log_probability_steps.append(
            distribution.log_prob(model_actions)
        )
        value_steps.append(output.value)
        reward_steps.append(
            torch.tensor(rewards, dtype=torch.float32, device=device)
        )
        done_steps.append(
            torch.tensor(dones, dtype=torch.bool, device=device)
        )
        observations = next_observations

    with torch.inference_mode():
        last_states = collate_observations(
            encoders,
            observations,
            device=device,
        )
        last_values = model(last_states).value

    rewards = torch.stack(reward_steps)
    dones = torch.stack(done_steps)
    values = torch.stack(value_steps)
    advantages = _advantages(
        rewards,
        dones,
        values,
        last_values,
        config.gamma,
        config.gae_lambda,
    )
    returns = advantages + values

    return (
        RolloutBatch(
            states=_stack_entity_batches(state_steps),
            actions=torch.stack(action_steps).flatten(),
            old_log_probabilities=torch.stack(
                log_probability_steps
            ).flatten(),
            old_values=values.flatten(),
            advantages=advantages.flatten(),
            returns=returns.flatten(),
        ),
        observations,
        episode_stats,
    )


def _ppo_update(
    model: ClusterActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_batch: RolloutBatch,
    config: PPOConfig,
) -> dict[str, float]:
    advantages = rollout_batch.advantages
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)
    sample_count = advantages.shape[0]
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "grad_norm": 0.0,
    }
    minibatch_count = 0

    model.train()
    for _ in range(config.epochs):
        indexes = torch.randperm(
            sample_count,
            device=rollout_batch.actions.device,
        )
        for start in range(0, sample_count, config.minibatch_size):
            minibatch_indexes = indexes[
                start : start + config.minibatch_size
            ]
            output = model(
                _select_states(rollout_batch.states, minibatch_indexes)
            )
            distribution = Categorical(logits=output.logits)
            new_log_probabilities = distribution.log_prob(
                rollout_batch.actions[minibatch_indexes]
            )
            log_ratio = (
                new_log_probabilities
                - rollout_batch.old_log_probabilities[minibatch_indexes]
            )
            ratio = log_ratio.exp()
            minibatch_advantages = advantages[minibatch_indexes]

            unclipped_policy_loss = -minibatch_advantages * ratio
            clipped_policy_loss = -minibatch_advantages * ratio.clamp(
                1.0 - config.clip_coefficient,
                1.0 + config.clip_coefficient,
            )
            policy_loss = torch.maximum(
                unclipped_policy_loss,
                clipped_policy_loss,
            ).mean()

            old_values = rollout_batch.old_values[minibatch_indexes]
            returns = rollout_batch.returns[minibatch_indexes]
            clipped_values = old_values + (
                output.value - old_values
            ).clamp(
                -config.clip_coefficient,
                config.clip_coefficient,
            )
            value_loss = 0.5 * torch.maximum(
                (output.value - returns).square(),
                (clipped_values - returns).square(),
            ).mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                config.max_grad_norm,
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (ratio - 1.0).abs() > config.clip_coefficient
                ).float().mean()
            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "approx_kl": approx_kl,
                "clip_fraction": clip_fraction,
                "grad_norm": grad_norm,
            }
            for name, value in metrics.items():
                totals[name] += value.detach().item()
            minibatch_count += 1

    return {
        name: value / minibatch_count
        for name, value in totals.items()
    }


def _checkpoint_payload(
    model: ClusterActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    global_step: int,
    update: int,
) -> dict[str, object]:
    serialized_config = asdict(config)
    serialized_config["scenario_paths"] = [
        str(path) for path in config.scenario_paths
    ]
    serialized_config["checkpoint"] = str(config.checkpoint)
    serialized_config["resume"] = (
        None if config.resume is None else str(config.resume)
    )
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": serialized_config,
        "global_step": global_step,
        "update": update,
    }


def _save_checkpoint(
    model: ClusterActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    global_step: int,
    update: int,
) -> None:
    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config.checkpoint.with_suffix(
        config.checkpoint.suffix + ".tmp"
    )
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            config,
            global_step,
            update,
        ),
        temporary_path,
    )
    temporary_path.replace(config.checkpoint)


def _evaluation(
    model: ClusterActorCritic,
    envs: list[ClusterEnv],
) -> list[dict[str, object]]:
    model.eval()
    results = []
    for env in envs:
        scenario = str(env.problem.meta.get("name", "scenario"))
        result = rollout(
            env,
            scenario,
            "trained_network_greedy",
            network_greedy_selector(env, model),
        )
        results.append(asdict(result))
    return results


def train(config: PPOConfig) -> dict[str, object]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    problems = [load_problem(path) for path in config.scenario_paths]
    envs = [ClusterEnv(problem) for problem in problems]
    encoders = [
        ClusterObservationEncoder.from_env(env) for env in envs
    ]
    observations = [
        env.reset(seed=config.seed + index)[0]
        for index, env in enumerate(envs)
    ]
    episode_rewards = [0.0] * len(envs)

    model_config = TransformerConfig(
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        feedforward_dim=config.feedforward_dim,
        dropout=0.0,
    )
    model = ClusterActorCritic(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
    )
    global_step = 0
    first_update = 1

    if config.resume is not None:
        checkpoint = torch.load(
            config.resume,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["global_step"])
        first_update = int(checkpoint["update"]) + 1

    steps_per_update = config.rollout_steps * len(envs)
    remaining_steps = max(0, config.total_steps - global_step)
    update_count = math.ceil(remaining_steps / steps_per_update)
    last_update = first_update + update_count - 1

    for update in range(first_update, last_update + 1):
        rollout_batch, observations, episode_stats = _collect_rollout(
            model,
            envs,
            encoders,
            observations,
            episode_rewards,
            config,
            device,
        )
        metrics = _ppo_update(
            model,
            optimizer,
            rollout_batch,
            config,
        )
        global_step += steps_per_update

        if update % config.log_interval == 0 or update == last_update:
            completed = len(episode_stats)
            metrics.update(
                update=update,
                global_step=global_step,
                completed_episodes=completed,
                success_rate=(
                    sum(stat.success for stat in episode_stats) / completed
                    if completed
                    else None
                ),
                mean_makespan=(
                    sum(stat.makespan for stat in episode_stats) / completed
                    if completed
                    else None
                ),
            )
            print(json.dumps(metrics, sort_keys=True))

        if (
            update % config.checkpoint_interval == 0
            or update == last_update
        ):
            _save_checkpoint(
                model,
                optimizer,
                config,
                global_step,
                update,
            )

    if update_count == 0:
        _save_checkpoint(
            model,
            optimizer,
            config,
            global_step,
            first_update - 1,
        )

    evaluation = _evaluation(model, envs) if config.evaluate else []
    for result in evaluation:
        print(json.dumps({"evaluation": result}, sort_keys=True))

    return {
        "checkpoint": str(config.checkpoint),
        "device": str(device),
        "global_step": global_step,
        "updates": update_count,
        "evaluation": evaluation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the entity-token Transformer with masked PPO."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=Path,
        default=_default_scenarios(),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ppo_cluster.pt"),
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coefficient", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
    )
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--no-eval", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PPOConfig(
        scenario_paths=tuple(args.scenarios),
        checkpoint=args.checkpoint,
        resume=args.resume,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coefficient=args.clip_coefficient,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        reward_scale=args.reward_scale,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        evaluate=not args.no_eval,
    )
    summary = train(config)
    print(json.dumps({"training": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
