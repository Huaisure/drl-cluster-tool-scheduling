"""Random-key genetic algorithm for the simplified scheduling environment."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cluster_rl.cluster_env import ClusterEnv
from cluster_toolkit.problem import ClusterProblem, load_problem
from cluster_toolkit.cluster_generator.pipeline_models import SchedulingInstance
from cluster_toolkit.cluster_generator.problem_adapter import to_cluster_problem
from cluster_toolkit.validator import ValidatorSuite


@dataclass(frozen=True, slots=True)
class GeneticResult:
    """Best successful schedule found by the genetic algorithm."""

    actions: tuple[Mapping[str, object], ...]
    makespan: float
    generations_run: int
    evaluations: int
    seed: int
    runtime_seconds: float
    termination_reason: str


def solve_instance(
    instance: SchedulingInstance,
    **kwargs,
) -> GeneticResult:
    return solve(to_cluster_problem(instance), **kwargs)


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
    time_limit_seconds: float | None = None,
) -> GeneticResult:
    """Minimize makespan with a random-key genetic algorithm."""

    _require_integer("population_size", population_size, minimum=4)
    _require_integer("generations", generations)
    _require_integer("patience", patience)
    _require_integer("seed", seed, minimum=0)
    if time_limit_seconds is not None and (
        isinstance(time_limit_seconds, bool)
        or not isinstance(time_limit_seconds, (int, float))
        or not math.isfinite(float(time_limit_seconds))
        or float(time_limit_seconds) <= 0
    ):
        raise ValueError("time_limit_seconds must be a positive finite number")

    started = time.monotonic()
    deadline = (
        None if time_limit_seconds is None else started + float(time_limit_seconds)
    )
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
    timed_out = False

    for generation in range(1, generations + 1):
        decoded = []
        evaluated_population = []
        for chromosome in population:
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            decoded.append(_decode(env, chromosome))
            evaluated_population.append(chromosome)
            evaluations += 1
        if not decoded:
            break
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
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
        if timed_out:
            break

        elite_count = max(1, population_size // 16)
        next_population = [
            chromosome.copy()
            for chromosome in np.asarray(evaluated_population)[order[:elite_count]]
        ]
        while len(next_population) < population_size:
            evaluated_array = np.asarray(evaluated_population)
            left = _select_parent(evaluated_array, costs, rng)
            right = _select_parent(evaluated_array, costs, rng)
            child = np.where(rng.random(gene_count) < 0.5, left, right)
            child[rng.integers(gene_count)] = rng.random()
            next_population.append(child)
        population = np.asarray(next_population)

    if best is None or not best.success:
        error_type = TimeoutError if timed_out else RuntimeError
        raise error_type(
            "Genetic algorithm did not find a successful schedule "
            "within the configured budget"
        )
    report = ValidatorSuite(problem).validate(
        best.actions,
        require_complete=True,
        exact_action_durations=True,
    )
    if not report.ok:
        details = "; ".join(issue.message for issue in report.issues[:5])
        raise RuntimeError(f"genetic algorithm produced an invalid schedule: {details}")
    return GeneticResult(
        actions=best.actions,
        makespan=best.makespan,
        generations_run=generation,
        evaluations=evaluations,
        seed=seed,
        runtime_seconds=time.monotonic() - started,
        termination_reason="TIME_LIMIT" if timed_out else "NORMAL",
    )


def _decode(env: ClusterEnv, chromosome: np.ndarray) -> _Evaluation:
    observation, _ = env.reset()
    total_reward = 0.0
    advance_action = env.action_space.n - 1

    for gene in chromosome:
        while True:
            legal_actions = np.flatnonzero(observation["action_mask"])
            transfer_actions = legal_actions[legal_actions != advance_action]
            if len(transfer_actions):
                break
            if not len(legal_actions):
                return _failed_result(env, total_reward)
            observation, reward, terminated, truncated, info = env.step(
                advance_action
            )
            total_reward += reward
            if terminated or truncated:
                return _decoded_result(env, total_reward, info)

        legal_actions = transfer_actions
        action_index = min(int(gene * len(legal_actions)), len(legal_actions) - 1)
        observation, reward, terminated, truncated, info = env.step(
            int(legal_actions[action_index])
        )
        total_reward += reward
        if terminated or truncated:
            return _decoded_result(env, total_reward, info)

    while np.array_equal(
        np.flatnonzero(observation["action_mask"]),
        np.asarray([advance_action]),
    ):
        observation, reward, terminated, truncated, info = env.step(
            advance_action
        )
        total_reward += reward
        if terminated or truncated:
            return _decoded_result(env, total_reward, info)

    return _failed_result(env, total_reward)


def _decoded_result(
    env: ClusterEnv,
    total_reward: float,
    info: Mapping[str, object],
) -> _Evaluation:
    if not info.get("is_success"):
        return _failed_result(env, total_reward)
    return _Evaluation(
        cost=-total_reward,
        success=bool(info.get("is_success")),
        makespan=float(info["time"]),
        actions=env.actions,
    )


def _failed_result(env: ClusterEnv, total_reward: float) -> _Evaluation:
    return _Evaluation(
        cost=1e12 - total_reward,
        success=False,
        makespan=-total_reward,
        actions=env.actions,
    )


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
    parser.add_argument("--time-limit-seconds", type=float)
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
            time_limit_seconds=args.time_limit_seconds,
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
