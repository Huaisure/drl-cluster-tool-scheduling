"""Select a bounded shield budget on validation, then evaluate test once."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import torch

from .graph import FEATURE_VERSION
from .network import IRActorCritic
from .sft import SFTConfig, _policy_metrics
from .sft_data import load_sft_cases


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _score(report: dict[str, object]) -> tuple[float, float, float, float]:
    ratio = report["mean_ratio_to_branch_search"]
    makespan = report["mean_makespan"]
    return (
        float(report["success_rate"]),
        -float(report["deadlock_rate"]),
        -(float(ratio) if ratio is not None else math.inf),
        -(float(makespan) if makespan is not None else math.inf),
    )


def evaluate_budgets(
    checkpoint: Path,
    validation_manifest: Path,
    test_manifest: Path,
    run_dir: Path,
    *,
    budgets: tuple[int, ...],
    strides: tuple[int, ...],
    max_evaluation_cases: int,
) -> dict[str, object]:
    if not budgets or any(type(value) is not int or value < 1 for value in budgets):
        raise ValueError("budgets must contain positive integers")
    if max_evaluation_cases < 1:
        raise ValueError("max_evaluation_cases must be positive")
    if not strides or any(type(value) is not int or value < 1 for value in strides):
        raise ValueError("strides must contain positive integers")
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
        if (
            {case.problem.problem_hash for case in validation_cases}
            & {case.problem.problem_hash for case in test_cases}
        ):
            raise ValueError("validation and test cases must be disjoint")
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "config.json", {
            "source_checkpoint": str(checkpoint),
            "budgets": list(budgets),
            "strides": list(strides),
            "max_evaluation_cases": max_evaluation_cases,
            "sft_config": asdict(config),
            "validation_instances": [
                str(case.metadata["instance_id"]) for case in validation_cases
            ],
            "test_instances": [str(case.metadata["instance_id"]) for case in test_cases],
        })
        candidates = []
        for budget in budgets:
            for stride in strides:
                candidate_config = replace(config, max_shield_backtracks=budget)
                report = _policy_metrics(
                    model, validation_cases, candidate_config, shield=True,
                    backtrack_stride=stride,
                )
                candidates.append({
                    "budget": budget,
                    "stride": stride,
                    "report": report,
                })
                print(json.dumps({
                    "budget": budget,
                    "stride": stride,
                    "validation_success_rate": report["success_rate"],
                    "validation_deadlock_rate": report["deadlock_rate"],
                    "validation_ratio_to_branch_search": report["mean_ratio_to_branch_search"],
                }), flush=True)
        selected = max(candidates, key=lambda item: _score(item["report"]))
        selected_budget = int(selected["budget"])
        selected_stride = int(selected["stride"])
        test = _policy_metrics(
            model,
            test_cases,
            replace(config, max_shield_backtracks=selected_budget),
            shield=True,
            backtrack_stride=selected_stride,
        )
        result = {
            "validation_candidates": candidates,
            "selected_budget": selected_budget,
            "selected_stride": selected_stride,
            "test": test,
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
    parser.add_argument("--budgets", type=int, nargs="+", default=[25, 100])
    parser.add_argument("--strides", type=int, nargs="+", default=[1])
    parser.add_argument("--max-evaluation-cases", type=int, default=3)
    args = parser.parse_args(argv)
    result = evaluate_budgets(
        args.checkpoint, args.validation, args.test, args.run_dir,
        budgets=tuple(args.budgets),
        strides=tuple(args.strides),
        max_evaluation_cases=args.max_evaluation_cases,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
