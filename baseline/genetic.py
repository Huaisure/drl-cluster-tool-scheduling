"""Random-key genetic algorithm for the simplified scheduling environment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cluster_env import ClusterEnv
from problem import ClusterProblem, load_problem


@dataclass(frozen=True, slots=True)
class GeneticResult:
    """Best successful schedule found by the genetic algorithm."""

    actions: tuple[Mapping[str, object], ...]
    makespan: float
    generations_run: int
    evaluations: int
    seed: int


@dataclass(frozen=True, slots=True)
class _Evaluation:
    cost: float
    success: bool
    makespan: float
    actions: tuple[Mapping[str, object], ...]


def solve(
    problem: ClusterProblem,
    *,
    population_size: int = 64,
    generations: int = 100,
    patience: int = 20,
    seed: int = 0,
) -> GeneticResult:
    """Minimize makespan with a random-key genetic algorithm."""

    _require_integer("population_size", population_size, minimum=4)
    _require_integer("generations", generations)
    _require_integer("patience", patience)
    _require_integer("seed", seed, minimum=0)

    env = ClusterEnv(problem)
    gene_count = sum(
        2 * (len(problem.routes[route_id].visits) + 1)
        for route_id, _ in env.wafer_keys
    )
    rng = np.random.default_rng(seed)
    population = rng.random((population_size, gene_count))
    population[0] = 0.0
    population[1] = np.nextafter(1.0, 0.0)

    best: _Evaluation | None = None
    best_cost = np.inf
    stale_generations = 0
    evaluations = 0

    for generation in range(1, generations + 1):
        decoded = [_decode(env, chromosome) for chromosome in population]
        evaluations += population_size
        costs = np.asarray([item.cost for item in decoded])
        order = np.argsort(costs, kind="stable")
        current = decoded[int(order[0])]

        if current.cost < best_cost:
            best = current
            best_cost = current.cost
            stale_generations = 0
        else:
            stale_generations += 1

        if stale_generations >= patience or generation == generations:
            break

        elite_count = max(1, population_size // 16)
        next_population = [
            chromosome.copy()
            for chromosome in population[order[:elite_count]]
        ]
        while len(next_population) < population_size:
            left = _select_parent(population, costs, rng)
            right = _select_parent(population, costs, rng)
            child = np.where(rng.random(gene_count) < 0.5, left, right)
            child[rng.integers(gene_count)] = rng.random()
            next_population.append(child)
        population = np.asarray(next_population)

    if best is None or not best.success:
        raise RuntimeError(
            "Genetic algorithm did not find a successful schedule "
            "within the configured budget"
        )
    return GeneticResult(
        actions=best.actions,
        makespan=best.makespan,
        generations_run=generation,
        evaluations=evaluations,
        seed=seed,
    )


def _decode(env: ClusterEnv, chromosome: np.ndarray) -> _Evaluation:
    observation, _ = env.reset()
    total_reward = 0.0

    for gene in chromosome:
        legal_actions = np.flatnonzero(observation["action_mask"])
        action_index = min(int(gene * len(legal_actions)), len(legal_actions) - 1)
        observation, reward, terminated, truncated, info = env.step(
            int(legal_actions[action_index])
        )
        total_reward += reward
        if terminated or truncated:
            return _Evaluation(
                cost=-total_reward,
                success=bool(info.get("is_success")),
                makespan=float(info["time"]),
                actions=env.actions,
            )

    raise RuntimeError("Chromosome ended before the environment terminated")


def _select_parent(
    population: np.ndarray,
    costs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    competitors = rng.integers(len(population), size=3)
    return population[competitors[int(np.argmin(costs[competitors]))]]


def _require_integer(name: str, value: int, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m baseline",
        description="Solve a simplified cluster-tool problem with a genetic algorithm."
    )
    parser.add_argument("problem", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if args.problem.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the problem file")

    try:
        result = solve(
            load_problem(args.problem),
            population_size=args.population_size,
            generations=args.generations,
            patience=args.patience,
            seed=args.seed,
        )
        args.output.write_text(
            json.dumps(
                [dict(action) for action in result.actions],
                indent=2,
            ),
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "makespan": result.makespan,
                "generations": result.generations_run,
                "evaluations": result.evaluations,
                "action_count": len(result.actions),
                "output": str(args.output),
            }
        )
    )
    return 0
