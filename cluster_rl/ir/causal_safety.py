"""Mine successful counterfactual branches and train an independent safety head."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .env import IRSchedulingEnv
from .graph import FEATURE_VERSION, IRGraph
from .network import IRActorCritic, collate_graphs
from .safety_head import evaluate_safety_reranker
from .sft import SFTConfig, _runtime_config
from .sft_data import SFTCase, load_sft_cases


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _restore_env(
    source: IRSchedulingEnv,
    config: SFTConfig,
    session,
    snapshot,
    frame,
    wait_tick,
    observation: IRGraph,
    decisions: int,
    reward_tick: int,
) -> IRSchedulingEnv:
    trial = IRSchedulingEnv(
        source.problem, **_runtime_config(config).env_options()
    )
    trial.session = session.fork()
    trial.reason = None
    trial.decisions = decisions
    trial.reward_tick = reward_tick
    trial.snapshot = snapshot
    trial.frame = frame
    trial.wait_tick = wait_tick
    trial.observation = observation
    return trial


@torch.no_grad()
def _rollout_actor(
    actor: IRActorCritic,
    env: IRSchedulingEnv,
    device: str,
    max_steps: int,
) -> bool:
    steps = 0
    observation = env.observation
    while env.reason is None and steps < max_steps:
        output = actor(collate_graphs([observation], device))
        observation, _, _, _, _ = env.step(int(output.logits[0].argmax()))
        steps += 1
    return env.reason == "success" and env.audit().ok


@torch.no_grad()
def collect_causal_pairs(
    actor: IRActorCritic,
    continuation: IRActorCritic,
    case: SFTCase,
    config: SFTConfig,
    *,
    collection_max_decisions: int,
    causal_tail: int,
    alternatives_per_state: int,
    continuation_max_decisions: int,
) -> tuple[list[tuple[IRGraph, int, int]], dict[str, object]]:
    actor.eval()
    continuation.eval()
    options = _runtime_config(config).env_options()
    options["max_decisions"] = collection_max_decisions
    env = IRSchedulingEnv(case.problem, **options)
    observation, _ = env.reset(seed=config.seed)
    trace = deque(maxlen=causal_tail)
    started = time.perf_counter()
    while env.reason is None:
        output = actor(collate_graphs([observation], config.device))
        order = output.logits[0].argsort(descending=True).tolist()
        action = int(order[0])
        trace.append((
            observation,
            action,
            tuple(int(item) for item in order[1:1 + alternatives_per_state]),
            env.session.fork(),
            env.snapshot,
            env.frame,
            env.wait_tick,
            env.decisions,
            env.reward_tick,
        ))
        observation, _, _, _, info = env.step(action)
    if not env.audit().ok:
        raise RuntimeError(f"failed actor trajectory audit failed for {case.path}")
    pairs = []
    attempts = 0
    if env.reason in {"deadlock", "deadline_missed"}:
        for state in reversed(trace):
            graph, failed_action, alternatives = state[:3]
            original = _restore_env(
                env, config, state[3], state[4], state[5], state[6],
                graph, state[7], state[8],
            )
            original.step(failed_action)
            attempts += 1
            if _rollout_actor(
                continuation, original, config.device,
                continuation_max_decisions,
            ):
                continue
            for alternative in alternatives:
                attempts += 1
                trial = _restore_env(
                    env, config, state[3], state[4], state[5], state[6],
                    graph, state[7], state[8],
                )
                trial.step(alternative)
                if _rollout_actor(
                    continuation, trial, config.device,
                    continuation_max_decisions,
                ):
                    pairs.append((graph, alternative, failed_action))
                    break
            if pairs:
                break
    return pairs, {
        "instance_id": case.metadata["instance_id"],
        "wafer_count": case.metadata["wafer_count"],
        "actor_success": bool(info["success"]),
        "actor_reason": info["termination_reason"],
        "actor_decisions": env.decisions,
        "counterfactual_attempts": attempts,
        "causal_pairs": len(pairs),
        "seconds": time.perf_counter() - started,
    }


def train_causal_safety(
    checkpoint: Path,
    continuation_checkpoint: Path | None,
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    run_dir: Path,
    *,
    max_train_cases: int,
    max_evaluation_cases: int,
    collection_max_wafers: int,
    collection_max_decisions: int,
    causal_tail: int,
    alternatives_per_state: int,
    continuation_max_decisions: int,
    epochs: int,
    learning_rate: float,
    margin: float,
) -> dict[str, object]:
    positive_ints = (
        max_train_cases, max_evaluation_cases, collection_max_wafers,
        collection_max_decisions, causal_tail, alternatives_per_state,
        continuation_max_decisions, epochs,
    )
    if any(type(value) is not int or value < 1 for value in positive_ints):
        raise ValueError("case, collection, and epoch limits must be positive integers")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("feature_version") != FEATURE_VERSION:
        raise ValueError("checkpoint feature protocol mismatch")
    config = SFTConfig(**saved["sft_config"])
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        actor = IRActorCritic(config.width, config.layers).to(config.device)
        actor.load_state_dict(saved["model"])
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        safety = IRActorCritic(config.width, config.layers).to(config.device)
        safety.load_state_dict(saved["model"])
        continuation = IRActorCritic(config.width, config.layers).to(config.device)
        if continuation_checkpoint is None:
            continuation.load_state_dict(saved["model"])
        else:
            continuation_saved = torch.load(
                continuation_checkpoint, map_location="cpu", weights_only=True,
            )
            if continuation_saved.get("feature_version") != FEATURE_VERSION:
                raise ValueError("continuation checkpoint feature protocol mismatch")
            continuation.load_state_dict(continuation_saved["model"])
        continuation.eval()
        for parameter in continuation.parameters():
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
        hashes = [set(case.problem.problem_hash for case in split)
                  for split in (train_cases, validation_cases, test_cases)]
        if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError("causal train, validation, and test cases must be disjoint")
        run_dir.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        parameters = {
            "source_checkpoint": str(checkpoint),
            "continuation_checkpoint": (
                None if continuation_checkpoint is None
                else str(continuation_checkpoint)
            ),
            "max_train_cases": max_train_cases,
            "max_evaluation_cases": max_evaluation_cases,
            "collection_max_wafers": collection_max_wafers,
            "collection_max_decisions": collection_max_decisions,
            "causal_tail": causal_tail,
            "alternatives_per_state": alternatives_per_state,
            "continuation_max_decisions": continuation_max_decisions,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "margin": margin,
            "sft_config": asdict(config),
        }
        _write_json(run_dir / "config.json", parameters)
        pairs = []
        collection = []
        for index, case in enumerate(train_cases, 1):
            case_pairs, row = collect_causal_pairs(
                actor, continuation, case, config,
                collection_max_decisions=collection_max_decisions,
                causal_tail=causal_tail,
                alternatives_per_state=alternatives_per_state,
                continuation_max_decisions=continuation_max_decisions,
            )
            row["case"] = index
            collection.append(row)
            pairs.extend(case_pairs)
            print(json.dumps(row), flush=True)
        if not pairs:
            _write_json(run_dir / "result.json", {
                "status": "no_causal_pairs",
                "collection": collection,
                "causal_pair_count": 0,
                "validation_evaluated": False,
                "test_evaluated": False,
            })
            raise RuntimeError("no successful counterfactual train branches were found")
        optimizer = torch.optim.AdamW(
            safety.parameters(), lr=learning_rate, weight_decay=config.weight_decay,
        )
        losses = []
        for epoch in range(1, epochs + 1):
            safety.train()
            batch = collate_graphs([item[0] for item in pairs], config.device)
            logits = safety(batch).logits
            positive = torch.tensor([item[1] for item in pairs], device=config.device)
            negative = torch.tensor([item[2] for item in pairs], device=config.device)
            rows = torch.arange(len(pairs), device=config.device)
            loss = F.softplus(logits[rows, negative] - logits[rows, positive] + margin).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                safety.parameters(), config.max_grad_norm, error_if_nonfinite=True,
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            print(json.dumps({"epoch": epoch, "loss": losses[-1]}), flush=True)
        temporary = run_dir / "last.tmp"
        torch.save({
            "feature_version": FEATURE_VERSION,
            "sft_config": asdict(config),
            "causal_config": parameters,
            "actor": actor.state_dict(),
            "safety": safety.state_dict(),
        }, temporary)
        temporary.replace(run_dir / "last.pt")
        validation_base = evaluate_safety_reranker(
            actor, safety, validation_cases, config, None,
        )
        validation_causal = evaluate_safety_reranker(
            actor, safety, validation_cases, config, None, 0.0,
        )
        def score(report):
            return (
                float(report["success_rate"]),
                -float(report["deadlock_rate"]),
                -(float(report["mean_ratio_to_branch_search"])
                  if report["mean_ratio_to_branch_search"] is not None else math.inf),
            )
        use_causal = score(validation_causal) > score(validation_base)
        test = evaluate_safety_reranker(
            actor, safety, test_cases, config, None, 0.0 if use_causal else None,
        )
        result = {
            "collection": collection,
            "causal_pair_count": len(pairs),
            "losses": losses,
            "validation_base": validation_base,
            "validation_causal": validation_causal,
            "selected_causal_head": use_causal,
            "test": test,
        }
        _write_json(run_dir / "result.json", result)
        return result
    finally:
        torch.set_num_threads(previous_threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-train-cases", type=int, default=4)
    parser.add_argument("--max-evaluation-cases", type=int, default=3)
    parser.add_argument("--collection-max-wafers", type=int, default=25)
    parser.add_argument("--collection-max-decisions", type=int, default=50)
    parser.add_argument("--causal-tail", type=int, default=8)
    parser.add_argument("--alternatives-per-state", type=int, default=2)
    parser.add_argument("--continuation-max-decisions", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args(argv)
    result = train_causal_safety(
        args.checkpoint, args.continuation_checkpoint,
        args.train, args.validation, args.test, args.run_dir,
        **{key: value for key, value in vars(args).items() if key not in {
            "checkpoint", "continuation_checkpoint", "train", "validation",
            "test", "run_dir",
        }},
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
