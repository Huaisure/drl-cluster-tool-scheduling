"""Select bounded stochastic SFT rollout search on validation, then test once."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import torch

from .env import IRSchedulingEnv
from .graph import FEATURE_VERSION
from .network import IRActorCritic, collate_graphs
from .sft import SFTConfig, _runtime_config
from .sft_data import SFTCase, load_sft_cases


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def _rollout(
    model: IRActorCritic,
    case: SFTCase,
    config: SFTConfig,
    *,
    temperature: float,
    seed: int,
) -> dict[str, object]:
    env = IRSchedulingEnv(case.problem, **_runtime_config(config).env_options())
    observation, _ = env.reset(seed=seed)
    generator = torch.Generator(device=config.device).manual_seed(seed)
    while env.reason is None:
        logits = model(collate_graphs([observation], config.device)).logits[0]
        if temperature == 0:
            action = int(logits.argmax())
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            action = int(torch.multinomial(
                probabilities, 1, generator=generator,
            )[0])
        observation, _, _, _, info = env.step(action)
    audit = env.audit()
    if not audit.ok:
        raise RuntimeError(f"sampled trajectory audit failed for {case.path}")
    return {
        "success": bool(info["success"]),
        "termination_reason": info["termination_reason"],
        "makespan": info["elapsed_seconds"] if info["success"] else None,
        "elapsed_seconds": info["elapsed_seconds"],
        "decisions": info["decisions"],
        "seed": seed,
    }


def _report(
    outcomes: list[list[dict[str, object]]],
    cases: list[SFTCase],
    budget: int,
) -> dict[str, object]:
    rows = []
    for attempts, case in zip(outcomes, cases):
        considered = attempts[:budget]
        successful = [item for item in considered if item["success"]]
        selected = (
            min(successful, key=lambda item: float(item["makespan"]))
            if successful else considered[0]
        )
        rows.append({
            "path": str(case.path),
            "problem_hash": case.problem.problem_hash,
            "success": bool(selected["success"]),
            "termination_reason": selected["termination_reason"],
            "makespan": selected["makespan"],
            "elapsed_seconds": selected["elapsed_seconds"],
            "decisions": selected["decisions"],
            "selected_seed": selected["seed"],
            "successful_rollouts": len(successful),
            "rollout_budget": budget,
        })
    successful_rows = [row for row in rows if row["success"]]
    reasons: dict[str, int] = {}
    for row in rows:
        reason = str(row["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1

    def ratio(field: str) -> float | None:
        values = [
            float(row["makespan"]) / float(case.metadata[field])
            for row, case in zip(rows, cases)
            if row["success"] and case.metadata.get(field) is not None
        ]
        return sum(values) / len(values) if values else None

    return {
        "success_rate": len(successful_rows) / len(rows),
        "deadlock_rate": reasons.get("deadlock", 0) / len(rows),
        "mean_makespan": (
            sum(float(row["makespan"]) for row in successful_rows)
            / len(successful_rows) if successful_rows else None
        ),
        "mean_ratio_to_genetic": ratio("genetic_makespan"),
        "mean_ratio_to_branch_search": ratio("branch_search_makespan"),
        "termination_counts": reasons,
        "cases": rows,
    }


def _portfolio_attempts(
    greedy: dict[str, object],
    sampled: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep deterministic greedy as the first, non-regressing attempt."""
    return [greedy, *sampled]


def _score(report: dict[str, object]) -> tuple[float, float, float, float]:
    branch = report["mean_ratio_to_branch_search"]
    genetic = report["mean_ratio_to_genetic"]
    return (
        float(report["success_rate"]),
        -float(report["deadlock_rate"]),
        -(float(branch) if branch is not None else math.inf),
        -(float(genetic) if genetic is not None else math.inf),
    )


def evaluate_sample_search(
    checkpoint: Path,
    validation_manifest: Path,
    test_manifest: Path,
    run_dir: Path,
    *,
    temperatures: tuple[float, ...],
    budgets: tuple[int, ...],
    max_evaluation_cases: int,
) -> dict[str, object]:
    if not temperatures or any(
        not math.isfinite(value) or value <= 0 for value in temperatures
    ):
        raise ValueError("temperatures must contain finite positive values")
    if not budgets or any(type(value) is not int or value < 1 for value in budgets):
        raise ValueError("budgets must contain positive integers")
    if max_evaluation_cases < 1:
        raise ValueError("max_evaluation_cases must be positive")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("feature_version") != FEATURE_VERSION:
        raise ValueError("checkpoint feature protocol mismatch")
    config = SFTConfig(**saved["sft_config"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        model = IRActorCritic(config.width, config.layers).to(config.device)
        model.load_state_dict(saved["model"])
        model.eval()
        validation_cases = load_sft_cases(
            [validation_manifest], expected_split="validation",
            limit=max_evaluation_cases, max_wafer_count=config.max_wafer_count,
        )
        test_cases = load_sft_cases(
            [test_manifest], expected_split="test",
            limit=max_evaluation_cases, max_wafer_count=config.max_wafer_count,
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "config.json", {
            "source_checkpoint": str(checkpoint),
            "temperatures": list(temperatures),
            "budgets": list(budgets),
            "max_evaluation_cases": max_evaluation_cases,
            "sft_config": asdict(config),
        })
        candidates = []
        greedy = [
            [_rollout(
                model, case, config, temperature=0,
                seed=config.seed + case_index * 10000,
            )]
            for case_index, case in enumerate(validation_cases)
        ]
        candidates.append({
            "temperature": 0.0,
            "budget": 1,
            "report": _report(greedy, validation_cases, 1),
        })
        maximum_budget = max(budgets)
        for temperature_index, temperature in enumerate(temperatures):
            outcomes = []
            for case_index, case in enumerate(validation_cases):
                attempts = []
                for rollout_index in range(maximum_budget - 1):
                    seed = (
                        config.seed + 100000
                        + temperature_index * 1000000
                        + case_index * 10000 + rollout_index
                    )
                    attempts.append(_rollout(
                        model, case, config, temperature=temperature, seed=seed,
                    ))
                outcomes.append(_portfolio_attempts(
                    greedy[case_index][0], attempts,
                ))
            for budget in sorted(set(budgets)):
                report = _report(outcomes, validation_cases, budget)
                candidates.append({
                    "temperature": temperature,
                    "budget": budget,
                    "report": report,
                })
                print(json.dumps({
                    "temperature": temperature,
                    "budget": budget,
                    "validation_success_rate": report["success_rate"],
                    "validation_deadlock_rate": report["deadlock_rate"],
                    "validation_ratio_to_genetic": report["mean_ratio_to_genetic"],
                }), flush=True)
        selected = max(candidates, key=lambda item: _score(item["report"]))
        temperature = float(selected["temperature"])
        budget = int(selected["budget"])
        test_outcomes = []
        for case_index, case in enumerate(test_cases):
            greedy_attempt = _rollout(
                model, case, config, temperature=0,
                seed=config.seed + 9000000 + case_index * 10000,
            )
            sampled_attempts = []
            for rollout_index in range(budget - 1):
                seed = (
                    config.seed + 9100000
                    + case_index * 10000 + rollout_index
                )
                sampled_attempts.append(_rollout(
                    model, case, config, temperature=temperature, seed=seed,
                ))
            test_outcomes.append(_portfolio_attempts(
                greedy_attempt, sampled_attempts,
            ))
        result = {
            "validation_candidates": candidates,
            "selected_temperature": temperature,
            "selected_budget": budget,
            "test": _report(test_outcomes, test_cases, budget),
        }
        _write_json(run_dir / "result.json", result)
        return result
    finally:
        torch.set_num_threads(previous_threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.5])
    parser.add_argument("--budgets", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--max-evaluation-cases", type=int, default=4)
    args = parser.parse_args(argv)
    result = evaluate_sample_search(
        args.checkpoint, args.validation, args.test, args.run_dir,
        temperatures=tuple(args.temperatures), budgets=tuple(args.budgets),
        max_evaluation_cases=args.max_evaluation_cases,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
