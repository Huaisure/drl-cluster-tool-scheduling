"""Train an independent action-safety head without changing the SFT actor."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .env import IRSchedulingEnv
from .graph import EDGE_TYPES, FEATURE_VERSION, NODE_TYPES, IRGraph
from .network import IRActorCritic, collate_graphs
from .sft import SFTConfig, _runtime_config
from .sft_data import SFTCase, load_sft_cases, replay_expert


@dataclass(frozen=True)
class SafetyHeadConfig:
    max_train_cases: int = 12
    max_evaluation_cases: int = 3
    collection_max_wafers: int | None = 25
    collection_max_decisions: int = 400
    positive_cap: int = 256
    negative_tail: int = 32
    epochs: int = 4
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    thresholds: tuple[float, ...] = ()
    relative_margins: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "max_train_cases", "max_evaluation_cases", "positive_cap",
            "negative_tail", "collection_max_decisions", "epochs", "batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.collection_max_wafers is not None and self.collection_max_wafers < 1:
            raise ValueError("collection_max_wafers must be positive")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not all(math.isfinite(item) for item in self.thresholds):
            raise ValueError("thresholds must contain finite values")
        if not self.relative_margins or not all(
            math.isfinite(item) and item >= 0 for item in self.relative_margins
        ):
            raise ValueError("relative_margins must contain finite non-negative values")


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def select_safe_action(
    actor_logits: torch.Tensor,
    safety_logits: torch.Tensor,
    threshold: float | None,
    relative_margin: float | None = None,
) -> int:
    """Keep actor ordering, removing candidates classified below threshold."""
    if actor_logits.ndim != 1 or safety_logits.shape != actor_logits.shape:
        raise ValueError("actor and safety logits must be equal one-dimensional tensors")
    if threshold is not None and relative_margin is not None:
        raise ValueError("absolute threshold and relative margin are mutually exclusive")
    if relative_margin is not None:
        if not math.isfinite(relative_margin) or relative_margin < 0:
            raise ValueError("relative margin must be finite and non-negative")
        safe = safety_logits >= safety_logits.max() - relative_margin
    elif threshold is None:
        return int(actor_logits.argmax())
    else:
        safe = safety_logits >= threshold
        if not bool(safe.any()):
            return int(actor_logits.argmax())
    filtered = actor_logits.masked_fill(~safe, -torch.inf)
    return int(filtered.argmax())


@torch.no_grad()
def _collect_actor_trajectory(
    actor: IRActorCritic,
    case: SFTCase,
    sft_config: SFTConfig,
    max_decisions: int,
    trace_tail: int,
) -> tuple[list[tuple[IRGraph, int]], dict[str, object]]:
    actor.eval()
    options = _runtime_config(sft_config).env_options()
    options["max_decisions"] = max_decisions
    env = IRSchedulingEnv(case.problem, **options)
    observation, _ = env.reset(seed=sft_config.seed)
    trace: deque[tuple[IRGraph, int]] = deque(maxlen=trace_tail)
    decisions = 0
    started = time.perf_counter()
    while env.reason is None:
        output = actor(collate_graphs([observation], sft_config.device))
        action = int(output.logits[0].argmax())
        trace.append((observation, action))
        decisions += 1
        observation, _, _, _, info = env.step(action)
    audit = env.audit()
    if not audit.ok:
        raise RuntimeError(f"actor trajectory audit failed for {case.path}")
    return list(trace), {
        "instance_id": case.metadata["instance_id"],
        "success": bool(info["success"]),
        "termination_reason": info["termination_reason"],
        "decisions": env.decisions,
        "trace_length": decisions,
        "seconds": time.perf_counter() - started,
    }


def _collect_samples(
    actor: IRActorCritic,
    cases: list[SFTCase],
    sft_config: SFTConfig,
    config: SafetyHeadConfig,
) -> tuple[list[tuple[IRGraph, int, float, float]], list[dict[str, object]]]:
    rng = random.Random(sft_config.seed + 15000)
    positives: list[tuple[IRGraph, int, float, float]] = []

    class _EnoughPositives(Exception):
        pass

    negatives: list[tuple[IRGraph, int, float, float]] = []
    collection = []
    per_case_positive_cap = math.ceil(config.positive_cap / len(cases))
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        case_positive_count = 0

        def positive(observation: IRGraph, acceptable: tuple[int, ...]) -> None:
            nonlocal case_positive_count
            for action in acceptable:
                if (
                    len(positives) >= config.positive_cap
                    or case_positive_count >= per_case_positive_cap
                ):
                    break
                positives.append((observation, action, 1.0, 1.0))
                case_positive_count += 1
            if (
                len(positives) >= config.positive_cap
                or case_positive_count >= per_case_positive_cap
            ):
                raise _EnoughPositives

        if len(positives) < config.positive_cap:
            try:
                replay_expert(
                    case.problem,
                    case.actions,
                    on_choice_set=positive,
                    max_decisions=sft_config.max_decisions,
                )
            except _EnoughPositives:
                # The materialized dataset has already completed full expert
                # replay validation. Stop early here to bound graph encoding.
                pass
        trace, row = _collect_actor_trajectory(
            actor, case, sft_config, config.collection_max_decisions,
            config.negative_tail,
        )
        if not row["success"]:
            tail = trace[-config.negative_tail:]
            # Actions nearer the terminal failure receive more weight, while
            # retaining the preceding context needed to recognize traps.
            for offset, (observation, action) in enumerate(tail, 1):
                weight = 0.5 + 0.5 * offset / len(tail)
                negatives.append((observation, action, 0.0, weight))
        row.update({
            "case": index,
            "wafer_count": case.metadata["wafer_count"],
            "positive_samples": case_positive_count,
            "negative_samples": 0 if row["success"] else min(len(trace), config.negative_tail),
            "total_seconds": time.perf_counter() - started,
        })
        collection.append(row)
        print(json.dumps(row), flush=True)
    if not positives:
        raise RuntimeError("no expert positive samples were collected")
    if not negatives:
        raise RuntimeError("the frozen actor produced no failed train trajectories")
    # Equalize classes without discarding rare failed-prefix examples.
    target = max(len(positives), len(negatives))
    balanced = [positives[index % len(positives)] for index in range(target)]
    balanced.extend(negatives[index % len(negatives)] for index in range(target))
    rng.shuffle(balanced)
    return balanced, collection


def _enrich_report(rows: list[dict[str, object]], cases: list[SFTCase]) -> dict[str, object]:
    successful = [float(row["makespan"]) for row in rows if row["success"]]
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
        "mean_makespan": sum(successful) / len(successful) if successful else None,
        "termination_counts": reasons,
        "deadlock_rate": reasons.get("deadlock", 0) / len(rows),
        "mean_ratio_to_genetic": ratio("genetic_makespan"),
        "mean_ratio_to_branch_search": ratio("branch_search_makespan"),
        "cases": rows,
    }


@torch.no_grad()
def evaluate_safety_reranker(
    actor: IRActorCritic,
    safety: IRActorCritic,
    cases: list[SFTCase],
    sft_config: SFTConfig,
    threshold: float | None,
    relative_margin: float | None = None,
) -> dict[str, object]:
    actor.eval()
    safety.eval()
    rows = []
    for case_index, case in enumerate(cases):
        env = IRSchedulingEnv(case.problem, **_runtime_config(sft_config).env_options())
        observation, _ = env.reset(seed=sft_config.seed + case_index)
        filtered_choices = fallback_choices = 0
        while env.reason is None:
            batch = collate_graphs([observation], sft_config.device)
            actor_logits = actor(batch).logits[0]
            safety_logits = safety(batch).logits[0]
            action = select_safe_action(
                actor_logits, safety_logits, threshold, relative_margin,
            )
            if threshold is not None or relative_margin is not None:
                safe = (
                    safety_logits >= safety_logits.max() - relative_margin
                    if relative_margin is not None
                    else safety_logits >= threshold
                )
                fallback_choices += int(not bool(safe.any()))
                filtered_choices += int(bool((~safe).any()))
            observation, _, _, _, info = env.step(action)
        audit = env.audit()
        if not audit.ok:
            raise RuntimeError(f"safety-reranked trajectory audit failed for {case.path}")
        rows.append({
            "path": str(case.path),
            "problem_hash": case.problem.problem_hash,
            "audit_ok": True,
            "success": bool(info["success"]),
            "termination_reason": info["termination_reason"],
            "makespan": info["elapsed_seconds"] if info["success"] else None,
            "elapsed_seconds": info["elapsed_seconds"],
            "decisions": info["decisions"],
            "filtered_choices": filtered_choices,
            "fallback_choices": fallback_choices,
        })
    return _enrich_report(rows, cases)


def _validation_score(report: dict[str, object]) -> tuple[float, float, float, float]:
    ratio = report["mean_ratio_to_branch_search"]
    makespan = report["mean_makespan"]
    return (
        float(report["success_rate"]),
        -float(report["deadlock_rate"]),
        -(float(ratio) if ratio is not None else math.inf),
        -(float(makespan) if makespan is not None else math.inf),
    )


def train_safety_head(
    checkpoint: Path,
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    run_dir: Path,
    config: SafetyHeadConfig = SafetyHeadConfig(),
) -> dict[str, object]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("feature_version") != FEATURE_VERSION:
        raise ValueError("source checkpoint feature protocol mismatch")
    sft_config = SFTConfig(**saved["sft_config"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(sft_config.cpu_threads)
    try:
        actor = IRActorCritic(sft_config.width, sft_config.layers).to(sft_config.device)
        actor.load_state_dict(saved["model"])
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        safety = IRActorCritic(sft_config.width, sft_config.layers).to(sft_config.device)
        safety.load_state_dict(saved["model"])

        train_cases = load_sft_cases(
            [train_manifest], expected_split="train", limit=config.max_train_cases,
            max_wafer_count=config.collection_max_wafers,
        )
        validation_cases = load_sft_cases(
            [validation_manifest], expected_split="validation",
            limit=config.max_evaluation_cases, max_wafer_count=sft_config.max_wafer_count,
        )
        test_cases = load_sft_cases(
            [test_manifest], expected_split="test", limit=config.max_evaluation_cases,
            max_wafer_count=sft_config.max_wafer_count,
        )
        hashes = [set(case.problem.problem_hash for case in split)
                  for split in (train_cases, validation_cases, test_cases)]
        if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("safety-head train, validation, and test cases must be disjoint")

        run_dir.mkdir(parents=True, exist_ok=False)
        random.seed(sft_config.seed)
        np.random.seed(sft_config.seed)
        torch.manual_seed(sft_config.seed)
        _write_json(run_dir / "config.json", {
            **asdict(config),
            "thresholds": list(config.thresholds),
            "relative_margins": list(config.relative_margins),
            "source_checkpoint": str(checkpoint),
            "feature_version": FEATURE_VERSION,
            "selected_instances": {
                name: [str(case.metadata["instance_id"]) for case in split]
                for name, split in (
                    ("train", train_cases),
                    ("validation", validation_cases),
                    ("test", test_cases),
                )
            },
        })

        samples, collection = _collect_samples(actor, train_cases, sft_config, config)
        optimizer = torch.optim.AdamW(
            safety.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        rng = random.Random(sft_config.seed + 16000)
        losses = []
        optimizer_steps = 0
        for epoch in range(1, config.epochs + 1):
            rng.shuffle(samples)
            epoch_loss = 0.0
            for start in range(0, len(samples), config.batch_size):
                chunk = samples[start:start + config.batch_size]
                batch = collate_graphs([item[0] for item in chunk], sft_config.device)
                logits = safety(batch).logits
                selected = torch.tensor([item[1] for item in chunk], device=sft_config.device)
                rows = torch.arange(len(chunk), device=sft_config.device)
                targets = torch.tensor([item[2] for item in chunk], device=sft_config.device)
                weights = torch.tensor([item[3] for item in chunk], device=sft_config.device)
                loss = F.binary_cross_entropy_with_logits(
                    logits[rows, selected], targets, weight=weights,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite safety-head loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    safety.parameters(), config.max_grad_norm, error_if_nonfinite=True,
                )
                optimizer.step()
                epoch_loss += float(loss.detach()) * len(chunk)
                optimizer_steps += 1
            mean_loss = epoch_loss / len(samples)
            losses.append(mean_loss)
            print(json.dumps({
                "epoch": epoch,
                "mean_loss": mean_loss,
                "optimizer_steps": optimizer_steps,
            }), flush=True)

        candidates = [(None, None)]
        candidates.extend((threshold, None) for threshold in config.thresholds)
        candidates.extend((None, margin) for margin in config.relative_margins)
        validation_reports = []
        for threshold, relative_margin in candidates:
            report = evaluate_safety_reranker(
                actor, safety, validation_cases, sft_config, threshold,
                relative_margin,
            )
            validation_reports.append({
                "threshold": threshold,
                "relative_margin": relative_margin,
                "report": report,
            })
            print(json.dumps({
                "threshold": threshold,
                "relative_margin": relative_margin,
                "validation_success_rate": report["success_rate"],
                "validation_deadlock_rate": report["deadlock_rate"],
                "validation_ratio_to_branch_search": report["mean_ratio_to_branch_search"],
            }), flush=True)
        selected = max(
            validation_reports,
            key=lambda item: _validation_score(item["report"]),
        )
        selected_threshold = selected["threshold"]
        selected_relative_margin = selected["relative_margin"]
        test_report = evaluate_safety_reranker(
            actor, safety, test_cases, sft_config, selected_threshold,
            selected_relative_margin,
        )
        temporary = run_dir / "last.tmp"
        torch.save({
            "feature_version": FEATURE_VERSION,
            "node_types": NODE_TYPES,
            "edge_types": EDGE_TYPES,
            "sft_config": asdict(sft_config),
            "safety_config": asdict(config),
            "actor": actor.state_dict(),
            "safety": safety.state_dict(),
            "selected_threshold": selected_threshold,
            "selected_relative_margin": selected_relative_margin,
        }, temporary)
        temporary.replace(run_dir / "last.pt")
        result = {
            "collection": collection,
            "sample_count": len(samples),
            "positive_balanced_count": len(samples) // 2,
            "negative_balanced_count": len(samples) // 2,
            "optimizer_steps": optimizer_steps,
            "epoch_losses": losses,
            "validation_candidates": validation_reports,
            "selected_threshold": selected_threshold,
            "selected_relative_margin": selected_relative_margin,
            "test": test_report,
        }
        _write_json(run_dir / "result.json", result)
        return result
    finally:
        torch.set_num_threads(previous_threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-train-cases", type=int, default=12)
    parser.add_argument("--max-evaluation-cases", type=int, default=3)
    parser.add_argument("--collection-max-wafers", type=int, default=25)
    parser.add_argument("--collection-max-decisions", type=int, default=400)
    parser.add_argument("--positive-cap", type=int, default=256)
    parser.add_argument("--negative-tail", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[])
    parser.add_argument(
        "--relative-margins", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0],
    )
    args = parser.parse_args(argv)
    config = SafetyHeadConfig(**{
        key: tuple(value) if key in {"thresholds", "relative_margins"} else value
        for key, value in vars(args).items()
        if key in SafetyHeadConfig.__dataclass_fields__
    })
    result = train_safety_head(
        args.checkpoint, args.train, args.validation, args.test, args.run_dir, config,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
