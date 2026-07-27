from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from cluster_env import ClusterEnv
from network import (
    ClusterActorCritic,
    ClusterObservationEncoder,
    TransformerConfig,
    collate_observations,
)
from problem import load_problem
from validator import ValidatorSuite


SCENARIO_DIR = Path(__file__).with_name("scenarios")


@dataclass(frozen=True, slots=True)
class RolloutResult:
    scenario: str
    policy: str
    wafer_count: int
    route_count: int
    pm_count: int
    action_count: int
    makespan: float
    total_reward: float
    valid: bool


ActionSelector = Callable[[ClusterEnv, dict[str, object]], int]


def first_legal_action(
    env: ClusterEnv,
    observation: dict[str, object],
) -> int:
    del env
    return int(np.flatnonzero(observation["action_mask"])[0])


def network_greedy_selector(
    env: ClusterEnv,
    model: ClusterActorCritic,
) -> ActionSelector:
    encoder = ClusterObservationEncoder.from_env(env)
    device = next(model.parameters()).device

    def select(
        current_env: ClusterEnv,
        observation: dict[str, object],
    ) -> int:
        del current_env
        batch = collate_observations(
            [encoder],
            [observation],
            device=device,
        )
        with torch.inference_mode():
            model_action = model(batch).logits.argmax(dim=1)
        return int(batch.to_env_actions(model_action).item())

    return select


def rollout(
    env: ClusterEnv,
    scenario: str,
    policy: str,
    select_action: ActionSelector,
) -> RolloutResult:
    problem = env.problem
    observation, _ = env.reset()
    total_reward = 0.0

    while True:
        action = select_action(env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            if not info.get("is_success"):
                raise RuntimeError(
                    f"{scenario}/{policy} ended with {info.get('reason')}"
                )
            break

    validation = ValidatorSuite(problem).validate(env.actions)
    if not validation.ok:
        raise RuntimeError(f"{scenario}/{policy} produced invalid actions")

    return RolloutResult(
        scenario=scenario,
        policy=policy,
        wafer_count=len(env.wafer_keys),
        route_count=len(problem.routes),
        pm_count=sum(
            module.type.value == "PM"
            for module in problem.Modules.values()
        ),
        action_count=len(env.actions),
        makespan=float(info["time"]),
        total_reward=total_reward,
        valid=True,
    )


def run_all(seed: int = 0) -> list[RolloutResult]:
    torch.manual_seed(seed)
    model = ClusterActorCritic(
        TransformerConfig(
            model_dim=64,
            num_heads=4,
            num_layers=2,
            feedforward_dim=128,
            dropout=0.0,
        )
    ).eval()
    results = []

    for path in sorted(SCENARIO_DIR.glob("*.json")):
        problem = load_problem(path)
        env = ClusterEnv(problem)
        results.append(
            rollout(
                env,
                path.stem,
                "first_legal",
                first_legal_action,
            )
        )
        results.append(
            rollout(
                env,
                path.stem,
                "untrained_network_greedy",
                network_greedy_selector(env, model),
            )
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run simple scenarios through ClusterEnv."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    for result in run_all(args.seed):
        print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
