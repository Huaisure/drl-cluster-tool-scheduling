"""Fit graph3 action features while freezing a migrated graph2 policy."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .graph import EDGE_TYPES, FEATURE_VERSION, NODE_TYPES
from .network import IRActorCritic, collate_graphs
from .sft import SFTConfig, _policy_metrics
from .sft_data import load_sft_cases, replay_expert


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _score(report: dict[str, object]) -> tuple[float, float, float, float]:
    branch = report["mean_ratio_to_branch_search"]
    genetic = report["mean_ratio_to_genetic"]
    return (
        float(report["success_rate"]),
        -float(report["deadlock_rate"]),
        -(float(branch) if branch is not None else math.inf),
        -(float(genetic) if genetic is not None else math.inf),
    )


def train_feature_adapter(
    checkpoint: Path,
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    run_dir: Path,
    *,
    max_train_cases: int,
    max_evaluation_cases: int,
    max_wafer_count: int,
    epochs: int,
    learning_rate: float,
    anchor_kl_weight: float,
) -> dict[str, object]:
    if any(type(value) is not int or value < 1 for value in (
        max_train_cases, max_evaluation_cases, max_wafer_count, epochs,
    )):
        raise ValueError("case, wafer, and epoch limits must be positive integers")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(anchor_kl_weight) or anchor_kl_weight < 0:
        raise ValueError("anchor_kl_weight must be finite and non-negative")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("feature_version") != FEATURE_VERSION:
        raise ValueError("feature adapter requires a graph3 checkpoint")
    config = SFTConfig(**saved["sft_config"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        model = IRActorCritic(config.width, config.layers).to(config.device)
        model.load_state_dict(saved["model"])
        anchor = copy.deepcopy(model).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.numeric.weight.requires_grad_(True)
        original_scalar_column = model.numeric.weight[:, :1].detach().clone()
        train_cases = load_sft_cases(
            [train_manifest], expected_split="train", limit=max_train_cases,
            max_wafer_count=max_wafer_count,
        )
        validation_cases = load_sft_cases(
            [validation_manifest], expected_split="validation",
            limit=max_evaluation_cases, max_wafer_count=config.max_wafer_count,
        )
        test_cases = load_sft_cases(
            [test_manifest], expected_split="test",
            limit=max_evaluation_cases, max_wafer_count=config.max_wafer_count,
        )
        hashes = [set(case.problem.problem_hash for case in split)
                  for split in (train_cases, validation_cases, test_cases)]
        if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("adapter train, validation, and test cases must be disjoint")
        run_dir.mkdir(parents=True, exist_ok=False)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        _write_json(run_dir / "config.json", {
            "source_checkpoint": str(checkpoint),
            "max_train_cases": max_train_cases,
            "max_evaluation_cases": max_evaluation_cases,
            "max_wafer_count": max_wafer_count,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "anchor_kl_weight": anchor_kl_weight,
            "sft_config": asdict(config),
            "train_instances": [
                str(case.metadata["instance_id"]) for case in train_cases
            ],
        })
        optimizer = torch.optim.Adam([model.numeric.weight], lr=learning_rate)
        losses = []
        optimizer_steps = supervised_choices = 0
        for epoch in range(1, epochs + 1):
            order = list(train_cases)
            random.Random(config.seed + 19000 + epoch).shuffle(order)
            observations = []
            targets: list[tuple[int, ...]] = []

            def flush() -> None:
                nonlocal optimizer_steps
                if not observations:
                    return
                batch = collate_graphs(observations, config.device)
                model.train()
                output = model(batch)
                log_probabilities = F.log_softmax(output.logits, dim=-1)
                imitation = torch.stack([
                    -torch.logsumexp(
                        log_probabilities[row, torch.tensor(
                            acceptable, device=config.device,
                        )],
                        dim=0,
                    )
                    for row, acceptable in enumerate(targets)
                ]).mean()
                with torch.no_grad():
                    anchor_logits = anchor(batch).logits
                finite = torch.isfinite(anchor_logits)
                student = torch.where(
                    finite, output.logits, torch.full_like(output.logits, -1e9),
                )
                teacher = torch.where(
                    finite, anchor_logits, torch.full_like(anchor_logits, -1e9),
                )
                anchor_loss = F.kl_div(
                    F.log_softmax(student, dim=-1),
                    F.softmax(teacher, dim=-1),
                    reduction="batchmean",
                )
                loss = imitation + anchor_kl_weight * anchor_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if model.numeric.weight.grad is None:
                    raise RuntimeError("numeric feature adapter received no gradient")
                model.numeric.weight.grad[:, 0] = 0
                torch.nn.utils.clip_grad_norm_(
                    [model.numeric.weight], config.max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                with torch.no_grad():
                    model.numeric.weight[:, :1].copy_(original_scalar_column)
                losses.append(float(loss.detach()))
                optimizer_steps += 1
                observations.clear()
                targets.clear()

            def learn(observation, acceptable: tuple[int, ...]) -> None:
                observations.append(observation)
                targets.append(acceptable)
                if len(observations) >= config.batch_size:
                    flush()

            for case in order:
                replay = replay_expert(
                    case.problem, case.actions, on_choice_set=learn,
                    max_decisions=config.max_decisions,
                )
                supervised_choices += int(replay["supervised_choice_count"])
                flush()
                print(json.dumps({
                    "epoch": epoch,
                    "instance_id": case.metadata["instance_id"],
                    "supervised_choices": replay["supervised_choice_count"],
                    "optimizer_steps": optimizer_steps,
                }), flush=True)
        validation_base = _policy_metrics(anchor, validation_cases, config)
        validation_adapter = _policy_metrics(model, validation_cases, config)
        selected_adapter = _score(validation_adapter) > _score(validation_base)
        selected_model = model if selected_adapter else anchor
        test = _policy_metrics(selected_model, test_cases, config)
        checkpoint_value = {
            "feature_version": FEATURE_VERSION,
            "node_types": NODE_TYPES,
            "edge_types": EDGE_TYPES,
            "config": saved["config"],
            "sft_config": asdict(config),
            "model": selected_model.state_dict(),
            "feature_adapter": {
                "source_checkpoint": str(checkpoint),
                "selected_adapter": selected_adapter,
            },
        }
        temporary = run_dir / "best.tmp"
        torch.save(checkpoint_value, temporary)
        temporary.replace(run_dir / "best.pt")
        result = {
            "supervised_choices": supervised_choices,
            "optimizer_steps": optimizer_steps,
            "mean_loss": sum(losses) / len(losses),
            "validation_base": validation_base,
            "validation_adapter": validation_adapter,
            "selected_adapter": selected_adapter,
            "test": test,
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
    parser.add_argument("--max-train-cases", type=int, default=4)
    parser.add_argument("--max-evaluation-cases", type=int, default=3)
    parser.add_argument("--max-wafer-count", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--anchor-kl-weight", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = train_feature_adapter(
        args.checkpoint, args.train, args.validation, args.test, args.run_dir,
        **{key: value for key, value in vars(args).items() if key not in {
            "checkpoint", "train", "validation", "test", "run_dir",
        }},
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
