"""Select a generic wait-logit penalty on validation, then test once."""

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


def select_wait_control_action(
    logits: torch.Tensor,
    *,
    intent_count: int,
    penalty: float,
) -> int:
    """Penalize only the generic Wait action while preserving intent order."""
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")
    if intent_count < 0 or intent_count > len(logits):
        raise ValueError("intent_count is outside the action vector")
    if math.isnan(penalty) or penalty < 0:
        raise ValueError("penalty must be non-negative")
    if intent_count == len(logits):
        return int(logits.argmax())
    if intent_count + 1 != len(logits):
        raise ValueError("Wait must be the sole action after the intents")
    adjusted = logits.clone()
    adjusted[intent_count] -= penalty
    return int(adjusted.argmax())


@torch.no_grad()
def _evaluate(
    model: IRActorCritic,
    cases: list[SFTCase],
    config: SFTConfig,
    penalty: float,
) -> dict[str, object]:
    model.eval()
    rows = []
    for case_index, case in enumerate(cases):
        env = IRSchedulingEnv(case.problem, **_runtime_config(config).env_options())
        observation, _ = env.reset(seed=config.seed + case_index)
        penalized_waits = 0
        while env.reason is None:
            logits = model(collate_graphs([observation], config.device)).logits[0]
            raw_action = int(logits.argmax())
            action = select_wait_control_action(
                logits, intent_count=len(env.frame.intents), penalty=penalty,
            )
            penalized_waits += int(raw_action != action)
            observation, _, _, _, info = env.step(action)
        audit = env.audit()
        if not audit.ok:
            raise RuntimeError(f"wait-controlled trajectory audit failed for {case.path}")
        rows.append({
            "path": str(case.path),
            "problem_hash": case.problem.problem_hash,
            "audit_ok": True,
            "success": bool(info["success"]),
            "termination_reason": info["termination_reason"],
            "makespan": info["elapsed_seconds"] if info["success"] else None,
            "elapsed_seconds": info["elapsed_seconds"],
            "decisions": info["decisions"],
            "penalized_wait_choices": penalized_waits,
        })
    successful = [row for row in rows if row["success"]]
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
        "success_rate": len(successful) / len(rows),
        "deadlock_rate": reasons.get("deadlock", 0) / len(rows),
        "mean_makespan": (
            sum(float(row["makespan"]) for row in successful) / len(successful)
            if successful else None
        ),
        "mean_ratio_to_genetic": ratio("genetic_makespan"),
        "mean_ratio_to_branch_search": ratio("branch_search_makespan"),
        "termination_counts": reasons,
        "cases": rows,
    }


def _score(report: dict[str, object]) -> tuple[float, float, float, float]:
    branch = report["mean_ratio_to_branch_search"]
    genetic = report["mean_ratio_to_genetic"]
    return (
        float(report["success_rate"]),
        -float(report["deadlock_rate"]),
        -(float(branch) if branch is not None else math.inf),
        -(float(genetic) if genetic is not None else math.inf),
    )


def evaluate_wait_control(
    checkpoint: Path,
    validation_manifest: Path,
    test_manifest: Path,
    run_dir: Path,
    *,
    penalties: tuple[float, ...],
    max_evaluation_cases: int,
) -> dict[str, object]:
    if not penalties or any(math.isnan(item) or item < 0 for item in penalties):
        raise ValueError("penalties must contain non-negative values")
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
            "penalties": ["infinity" if math.isinf(item) else item for item in penalties],
            "max_evaluation_cases": max_evaluation_cases,
            "sft_config": asdict(config),
        })
        candidates = []
        for penalty in penalties:
            report = _evaluate(model, validation_cases, config, penalty)
            candidates.append({
                "penalty": "infinity" if math.isinf(penalty) else penalty,
                "report": report,
            })
            print(json.dumps({
                "penalty": "infinity" if math.isinf(penalty) else penalty,
                "validation_success_rate": report["success_rate"],
                "validation_deadlock_rate": report["deadlock_rate"],
                "validation_ratio_to_genetic": report["mean_ratio_to_genetic"],
            }), flush=True)
        selected_index = max(
            range(len(candidates)), key=lambda index: _score(candidates[index]["report"])
        )
        selected_penalty = penalties[selected_index]
        result = {
            "validation_candidates": candidates,
            "selected_penalty": (
                "infinity" if math.isinf(selected_penalty) else selected_penalty
            ),
            "test": _evaluate(model, test_cases, config, selected_penalty),
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
    parser.add_argument(
        "--penalties", type=float, nargs="+", default=[0.0, 0.5, 1.0, math.inf],
    )
    parser.add_argument("--max-evaluation-cases", type=int, default=4)
    args = parser.parse_args(argv)
    result = evaluate_wait_control(
        args.checkpoint, args.validation, args.test, args.run_dir,
        penalties=tuple(args.penalties),
        max_evaluation_cases=args.max_evaluation_cases,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
