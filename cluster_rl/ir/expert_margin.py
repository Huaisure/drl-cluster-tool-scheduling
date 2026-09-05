"""Train expert-over-unsafe action margins on train-only IR states."""

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

from .network import IRActorCritic, collate_graphs
from .sft import SFTConfig, _immediate_failure, _policy_metrics, _save_checkpoint
from .sft_data import SFTCase, load_sft_cases, replay_expert


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def collect_margins(
    model: IRActorCritic,
    case: SFTCase,
    config: SFTConfig,
) -> tuple[list[tuple[object, tuple[int, ...], int]], dict[str, object]]:
    model.eval()
    samples: list[tuple[object, tuple[int, ...], int]] = []
    inspected = mistakes = 0
    started = time.perf_counter()

    def inspect(observation, acceptable: tuple[int, ...], env) -> None:
        nonlocal inspected, mistakes
        inspected += 1
        output = model(collate_graphs([observation], config.device))
        action = int(output.logits[0].argmax())
        if action in acceptable:
            return
        mistakes += 1
        if _immediate_failure(env, action):
            samples.append((observation, acceptable, action))

    replay = replay_expert(
        case.problem,
        case.actions,
        on_choice_context=inspect,
        max_decisions=config.max_decisions,
    )
    return samples, {
        "instance_id": case.metadata["instance_id"],
        "expert_solver": case.metadata["expert_solver"],
        "wafer_count": case.metadata["wafer_count"],
        "inspected_states": inspected,
        "top1_nonexpert_states": mistakes,
        "unsafe_top1_states": len(samples),
        "ir_makespan": replay["ir_makespan"],
        "seconds": time.perf_counter() - started,
    }


def train_expert_margins(
    checkpoint: Path,
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    run_dir: Path,
    *,
    max_train_cases: int,
    collection_max_wafers: int | None,
    max_evaluation_cases: int,
    epochs: int,
    learning_rate: float,
    margin: float,
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
        "collection_max_wafers": collection_max_wafers,
        "max_evaluation_cases": max_evaluation_cases,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "margin": margin,
        "anchor_kl_weight": anchor_kl_weight,
    })

    samples = []
    collection = []
    for index, case in enumerate(train_cases, 1):
        case_samples, row = collect_margins(model, case, config)
        row["case"] = index
        collection.append(row)
        samples.extend(case_samples)
        print(json.dumps(row), flush=True)
    if not samples:
        raise RuntimeError("no unsafe expert-state top choices were found")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = random.Random(config.seed + 12000)
    losses = []
    updates = 0
    for epoch in range(epochs):
        rng.shuffle(samples)
        for start in range(0, len(samples), config.batch_size):
            batch = samples[start:start + config.batch_size]
            graph_batch = collate_graphs(
                [sample[0] for sample in batch], config.device
            )
            model.train()
            logits = model(graph_batch).logits
            margin_losses = []
            for row, (_, acceptable, unsafe) in enumerate(batch):
                positive = torch.logsumexp(
                    logits[row, torch.tensor(acceptable, device=config.device)],
                    dim=0,
                )
                margin_losses.append(F.softplus(logits[row, unsafe] - positive + margin))
            margin_loss = torch.stack(margin_losses).mean()
            with torch.no_grad():
                anchor_logits = anchor(graph_batch).logits
            finite = torch.isfinite(anchor_logits)
            student = torch.where(finite, logits, torch.full_like(logits, -1e9))
            teacher = torch.where(
                finite, anchor_logits, torch.full_like(anchor_logits, -1e9)
            )
            anchor_loss = F.kl_div(
                F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1),
                reduction="batchmean",
            )
            loss = margin_loss + anchor_kl_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
            updates += 1
        print(json.dumps({"epoch": epoch + 1, "updates": updates}), flush=True)

    _save_checkpoint(
        run_dir / "last.pt", model, optimizer, config,
        epoch=epochs, completed_cases=len(train_cases),
        supervised_choices=len(samples),
    )
    result = {
        "collection": collection,
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
    parser.add_argument("--max-train-cases", type=int, default=4)
    parser.add_argument("--collection-max-wafers", type=int, default=12)
    parser.add_argument("--max-evaluation-cases", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--anchor-kl-weight", type=float, default=1.0)
    args = parser.parse_args(argv)
    if (
        min(args.max_train_cases, args.collection_max_wafers,
            args.max_evaluation_cases, args.epochs) <= 0
        or args.learning_rate <= 0
        or args.margin < 0
        or args.anchor_kl_weight < 0
    ):
        parser.error("limits, epochs and learning rate must be positive; weights nonnegative")
    result = train_expert_margins(
        args.checkpoint, args.train, args.validation, args.test, args.run_dir,
        max_train_cases=args.max_train_cases,
        collection_max_wafers=args.collection_max_wafers,
        max_evaluation_cases=args.max_evaluation_cases,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        margin=args.margin,
        anchor_kl_weight=args.anchor_kl_weight,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
