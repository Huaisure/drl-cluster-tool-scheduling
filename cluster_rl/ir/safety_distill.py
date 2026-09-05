"""Distill successful IR-shielded train rollouts into an SFT checkpoint."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .env import IRSchedulingEnv
from .network import IRActorCritic, collate_graphs
from .sft import SFTConfig, _immediate_failure, _policy_metrics, _save_checkpoint
from .sft_data import SFTCase, load_sft_cases


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def collect_safe_trajectory(
    model: IRActorCritic,
    case: SFTCase,
    config: SFTConfig,
) -> tuple[list[tuple[object, int]], dict[str, object]]:
    model.eval()
    env = IRSchedulingEnv(case.problem, **config_runtime(config).env_options())
    observation, _ = env.reset(seed=config.seed)
    samples: list[tuple[object, int]] = []
    rejected = 0
    started = time.perf_counter()
    while env.reason is None:
        output = model(collate_graphs([observation], config.device))
        selected = None
        rejected_before_choice = 0
        for action in output.logits[0].argsort(descending=True).tolist():
            if not _immediate_failure(env, action):
                selected = action
                break
            rejected += 1
            rejected_before_choice += 1
        if selected is None:
            selected = int(output.logits[0].argmax())
        elif rejected_before_choice:
            # Preserve the expert-imitation policy everywhere its first choice
            # is already locally safe; distill only actual shield corrections.
            samples.append((observation, selected))
        observation, _, _, _, info = env.step(selected)
    audit = env.audit()
    if not audit.ok:
        raise RuntimeError(f"shield trajectory audit failed for {case.path}")
    row = {
        "instance_id": case.metadata["instance_id"],
        "success": bool(info["success"]),
        "termination_reason": info["termination_reason"],
        "decisions": env.decisions,
        "makespan": float(env.snapshot.tick) if info["success"] else None,
        "shield_rejections": rejected,
        "correction_samples": len(samples),
        "seconds": time.perf_counter() - started,
    }
    return (samples if info["success"] else []), row


def config_runtime(config: SFTConfig):
    # Keep one canonical conversion in sft.py without making it public API.
    from .sft import _runtime_config

    return _runtime_config(config)


def distill(
    checkpoint: Path,
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    run_dir: Path,
    *,
    max_train_cases: int,
    max_evaluation_cases: int,
    epochs: int,
    collection_max_wafers: int | None,
    learning_rate: float,
    anchor_kl_weight: float,
) -> dict[str, object]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = SFTConfig(**saved["sft_config"])
    torch.set_num_threads(config.cpu_threads)
    model = IRActorCritic(config.width, config.layers).to(config.device)
    model.load_state_dict(saved["model"])
    anchor = copy.deepcopy(model).eval()
    for parameter in anchor.parameters():
        parameter.requires_grad_(False)
    train_cases = load_sft_cases(
        [train_manifest], expected_split="train", limit=max_train_cases,
        max_wafer_count=collection_max_wafers,
    )
    validation_cases = load_sft_cases(
        [validation_manifest], expected_split="validation",
        limit=max_evaluation_cases, max_wafer_count=config.max_wafer_count,
    )
    test_cases = load_sft_cases(
        [test_manifest], expected_split="test", limit=max_evaluation_cases,
        max_wafer_count=config.max_wafer_count,
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", {
        "source_checkpoint": str(checkpoint),
        "sft_config": asdict(config),
        "max_train_cases": max_train_cases,
        "max_evaluation_cases": max_evaluation_cases,
        "epochs": epochs,
        "collection_max_wafers": collection_max_wafers,
        "learning_rate": learning_rate,
        "anchor_kl_weight": anchor_kl_weight,
    })

    samples: list[tuple[object, int]] = []
    collection = []
    for index, case in enumerate(train_cases, 1):
        case_samples, row = collect_safe_trajectory(model, case, config)
        row["case"] = index
        collection.append(row)
        print(json.dumps(row), flush=True)
        samples.extend(case_samples)
    if not samples:
        raise RuntimeError("no successful shield trajectories were collected")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = random.Random(config.seed + 9000)
    updates = 0
    losses = []
    for epoch in range(epochs):
        rng.shuffle(samples)
        for start in range(0, len(samples), config.batch_size):
            batch = samples[start:start + config.batch_size]
            model.train()
            graph_batch = collate_graphs(
                [item[0] for item in batch], config.device
            )
            output = model(graph_batch)
            targets = torch.tensor([item[1] for item in batch], device=config.device)
            correction_loss = F.cross_entropy(output.logits, targets)
            with torch.no_grad():
                anchor_logits = anchor(graph_batch).logits
            finite = torch.isfinite(anchor_logits)
            student_logits = torch.where(
                finite, output.logits, torch.full_like(output.logits, -1e9)
            )
            teacher_logits = torch.where(
                finite, anchor_logits, torch.full_like(anchor_logits, -1e9)
            )
            anchor_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction="batchmean",
            )
            loss = correction_loss + anchor_kl_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
        print(json.dumps({"epoch": epoch + 1, "updates": updates}), flush=True)

    _save_checkpoint(
        run_dir / "last.pt", model, optimizer, config,
        epoch=epochs, completed_cases=sum(bool(row["success"]) for row in collection),
        supervised_choices=len(samples),
    )
    result = {
        "collection": collection,
        "successful_train_cases": sum(bool(row["success"]) for row in collection),
        "sample_count": len(samples),
        "optimizer_steps": updates,
        "mean_loss": sum(losses) / len(losses),
        "validation_raw": _policy_metrics(model, validation_cases, config),
        "test_raw": _policy_metrics(model, test_cases, config),
        "validation_shield": _policy_metrics(model, validation_cases, config, shield=True),
        "test_shield": _policy_metrics(model, test_cases, config, shield=True),
    }
    _write_json(run_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-train-cases", type=int, default=5)
    parser.add_argument("--max-evaluation-cases", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--collection-max-wafers", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--anchor-kl-weight", type=float, default=0.0)
    args = parser.parse_args(argv)
    if (
        args.max_train_cases <= 0
        or args.max_evaluation_cases <= 0
        or args.epochs <= 0
        or args.learning_rate <= 0
        or args.anchor_kl_weight < 0
        or (
            args.collection_max_wafers is not None
            and args.collection_max_wafers <= 0
        )
    ):
        parser.error("case limits, epochs, wafer limit, and learning rate must be positive")
    result = distill(
        args.checkpoint, args.train, args.validation, args.test, args.run_dir,
        max_train_cases=args.max_train_cases,
        max_evaluation_cases=args.max_evaluation_cases,
        epochs=args.epochs,
        collection_max_wafers=args.collection_max_wafers,
        learning_rate=args.learning_rate,
        anchor_kl_weight=args.anchor_kl_weight,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
