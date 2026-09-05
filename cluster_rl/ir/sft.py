"""Supervised warm-up of the IR policy from validated solver trajectories."""

from __future__ import annotations

import argparse
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
from .graph import EDGE_TYPES, FEATURE_VERSION, NODE_TYPES
from .network import IRActorCritic, collate_graphs
from .sft_data import SFTCase, load_sft_cases, replay_expert
from .train import IRTrainConfig, evaluate


@dataclass(frozen=True)
class SFTConfig:
    epochs: int = 1
    batch_size: int = 8
    width: int = 64
    layers: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    max_decisions: int = 5000
    seed: int = 1701
    device: str = "cpu"
    cpu_threads: int = 8
    max_train_cases: int | None = None
    max_validation_cases: int | None = None
    max_test_cases: int | None = None
    max_wafer_count: int | None = None
    shield_evaluation: bool = False
    max_shield_backtracks: int = 25

    def __post_init__(self) -> None:
        for name in (
            "epochs", "batch_size", "width", "layers", "max_decisions",
            "cpu_threads", "max_shield_backtracks",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def _runtime_config(config: SFTConfig) -> IRTrainConfig:
    return IRTrainConfig(
        width=config.width,
        layers=config.layers,
        max_decisions=config.max_decisions,
        seed=config.seed,
        device=config.device,
        cpu_threads=config.cpu_threads,
    )


def _save_checkpoint(
    path: Path,
    model: IRActorCritic,
    optimizer: torch.optim.Optimizer,
    config: SFTConfig,
    *,
    epoch: int,
    completed_cases: int,
    supervised_choices: int,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "feature_version": FEATURE_VERSION,
        "node_types": NODE_TYPES,
        "edge_types": EDGE_TYPES,
        "config": asdict(_runtime_config(config)),
        "sft_config": asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": supervised_choices,
        "sft_epoch": epoch,
        "completed_cases": completed_cases,
    }, temporary)
    temporary.replace(path)


def _select_cases(cases: list[SFTCase], limit: int | None, max_wafers: int | None) -> list[SFTCase]:
    selected = [
        case for case in cases
        if max_wafers is None or int(case.metadata["wafer_count"]) <= max_wafers
    ]
    selected.sort(key=lambda case: (
        int(case.metadata["wafer_count"]),
        int(case.metadata["topology_cell_count"]),
        str(case.metadata["instance_id"]),
    ))
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("case filters selected no instances")
    return selected


def _reference_metrics(cases: list[SFTCase]) -> dict[str, object]:
    def summarize(field: str) -> dict[str, float | int | None]:
        values = [float(case.metadata[field]) for case in cases if case.metadata.get(field) is not None]
        return {
            "count": len(values),
            "mean_makespan": sum(values) / len(values) if values else None,
        }
    return {
        "case_count": len(cases),
        "selected_expert": summarize("expert_makespan"),
        "branch_search": summarize("branch_search_makespan"),
        "genetic": summarize("genetic_makespan"),
    }


@torch.no_grad()
def imitation_metrics(model: IRActorCritic, cases: list[SFTCase], config: SFTConfig) -> dict[str, object]:
    model.eval()
    correct = choices = 0
    nll = 0.0

    def score(observation, actions: tuple[int, ...]) -> None:
        nonlocal correct, choices, nll
        output = model(collate_graphs([observation], config.device))
        log_probabilities = F.log_softmax(output.logits[0], dim=-1)
        selected = torch.tensor(actions, device=config.device)
        nll -= float(torch.logsumexp(log_probabilities[selected], dim=0))
        correct += int(int(output.logits[0].argmax()) in actions)
        choices += 1

    case_rows = []
    for case in cases:
        started = time.perf_counter()
        replay = replay_expert(
            case.problem,
            case.actions,
            on_choice_set=score,
            max_decisions=config.max_decisions,
        )
        case_rows.append({
            "instance_id": case.metadata["instance_id"],
            "choices": replay["supervised_choice_count"],
            "seconds": time.perf_counter() - started,
        })
    return {
        "choice_count": choices,
        "top1_accuracy": correct / choices if choices else 1.0,
        "mean_nll": nll / choices if choices else 0.0,
        "cases": case_rows,
    }


def _immediate_failure(env: IRSchedulingEnv, action: int) -> bool:
    clone = IRSchedulingEnv(
        env.problem,
        max_decisions=env.max_decisions,
        max_time_seconds=(
            None
            if env.max_tick is None
            else env.max_tick / env.problem.time_domain.ticks_per_unit
        ),
        reward_scale_seconds=env.reward_scale / env.problem.time_domain.ticks_per_unit,
        encode_observations=False,
    )
    clone.session = env.session.fork()
    clone.reason = None
    clone.decisions = env.decisions
    clone.reward_tick = env.reward_tick
    clone.snapshot = env.snapshot
    clone.frame = env.frame
    clone.wait_tick = env.wait_tick
    clone.observation = None
    clone.step(action)
    return clone.reason in {
        "deadlock", "deadline_missed", "decision_limit", "time_limit",
    }


@torch.no_grad()
def _policy_metrics(
    model: IRActorCritic,
    cases: list[SFTCase],
    config: SFTConfig,
    *,
    shield: bool = False,
    backtrack_stride: int = 1,
) -> dict:
    if type(backtrack_stride) is not int or backtrack_stride < 1:
        raise ValueError("backtrack_stride must be a positive integer")
    if not shield:
        report = evaluate(
            model,
            [(case.path, case.problem) for case in cases],
            _runtime_config(config),
        )
    else:
        model.eval()
        rows = []
        for case in cases:
            env = IRSchedulingEnv(case.problem, **_runtime_config(config).env_options())
            observation, _ = env.reset(seed=config.seed)
            rejected = backtracks = 0
            stack: list[tuple[object, int, int, list[str | None]]] = []

            def resolve(reference: str | None) -> int:
                if reference is None:
                    if env.wait_tick is None:
                        raise ValueError("restored state no longer offers Wait")
                    return len(env.frame.intents)
                return next(
                    index for index, candidate in enumerate(env.frame.intents)
                    if candidate.candidate_key == reference
                )

            def try_references(references: list[str | None]):
                nonlocal rejected
                while references:
                    reference = references.pop(0)
                    action_index = resolve(reference)
                    if not _immediate_failure(env, action_index):
                        return action_index, reference
                    rejected += 1
                return None

            while env.reason is None:
                output = model(collate_graphs([observation], config.device))
                order = output.logits[0].argsort(descending=True).tolist()
                references = [
                    None if action == len(env.frame.intents)
                    else env.frame.intents[action].candidate_key
                    for action in order
                ]
                selected = try_references(references)
                while (
                    selected is None
                    and stack
                    and backtracks < config.max_shield_backtracks
                ):
                    popped = None
                    for _ in range(min(
                        backtrack_stride,
                        len(stack),
                        config.max_shield_backtracks - backtracks,
                    )):
                        popped = stack.pop()
                        backtracks += 1
                    assert popped is not None
                    snapshot, decisions, reward_tick, references = popped
                    env.session = snapshot.fork()
                    env.reason = None
                    env.decisions = decisions
                    env.reward_tick = reward_tick
                    env._settle()
                    observation = env.observation
                    selected = try_references(references)
                if selected is None:
                    # No one-step-safe continuation exists even after
                    # chronological backtracking. Execute the highest-logit
                    # action so the audited terminal reason remains explicit.
                    fallback = model(collate_graphs([env.observation], config.device))
                    action = int(fallback.logits[0].argmax())
                else:
                    action, _ = selected
                    stack.append((
                        env.session.fork(),
                        env.decisions,
                        env.reward_tick,
                        references,
                    ))
                observation, _, _, _, info = env.step(action)
            audit = env.audit()
            if not audit.ok:
                raise RuntimeError(f"shielded trajectory audit failed for {case.path}")
            rows.append({
                "path": str(case.path),
                "problem_hash": case.problem.problem_hash,
                "audit_ok": True,
                "success": info["success"],
                "termination_reason": info["termination_reason"],
                "makespan": info["elapsed_seconds"] if info["success"] else None,
                "elapsed_seconds": info["elapsed_seconds"],
                "decisions": info["decisions"],
                "shield_rejections": rejected,
                "shield_backtracks": backtracks,
            })
        successful = [float(row["makespan"]) for row in rows if row["success"]]
        report = {
            "success_rate": len(successful) / len(rows),
            "mean_makespan": sum(successful) / len(successful) if successful else None,
            "cases": rows,
        }
    reasons: dict[str, int] = {}
    for row in report["cases"]:
        reason = str(row["termination_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    ga_pairs = [
        (float(row["makespan"]), float(case.metadata["genetic_makespan"]))
        for row, case in zip(report["cases"], cases)
        if row["success"] and case.metadata.get("genetic_makespan") is not None
    ]
    branch_pairs = [
        (float(row["makespan"]), float(case.metadata["branch_search_makespan"]))
        for row, case in zip(report["cases"], cases)
        if row["success"] and case.metadata.get("branch_search_makespan") is not None
    ]
    report["termination_counts"] = reasons
    report["deadlock_rate"] = reasons.get("deadlock", 0) / len(cases)
    report["mean_ratio_to_genetic"] = (
        sum(model_ms / expert_ms for model_ms, expert_ms in ga_pairs) / len(ga_pairs)
        if ga_pairs else None
    )
    report["mean_ratio_to_branch_search"] = (
        sum(model_ms / expert_ms for model_ms, expert_ms in branch_pairs) / len(branch_pairs)
        if branch_pairs else None
    )
    return report


def _policy_score(report: dict[str, object]) -> tuple[float, float, float, float]:
    """Rank validation rollouts before consulting the held-out test split."""
    branch = report["mean_ratio_to_branch_search"]
    genetic = report["mean_ratio_to_genetic"]
    return (
        float(report["success_rate"]),
        -float(report["deadlock_rate"]),
        -(float(branch) if branch is not None else math.inf),
        -(float(genetic) if genetic is not None else math.inf),
    )


def train_sft(
    train_paths: list[Path],
    validation_paths: list[Path],
    test_paths: list[Path],
    run_dir: Path,
    config: SFTConfig = SFTConfig(),
    *,
    initialize_from: Path | None = None,
    resume_from: Path | None = None,
) -> dict[str, object]:
    if initialize_from is not None and resume_from is not None:
        raise ValueError("initialize_from and resume_from are mutually exclusive")
    if run_dir.exists() and resume_from is None:
        raise FileExistsError(run_dir)
    if resume_from is not None and not run_dir.exists():
        raise FileNotFoundError(run_dir)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        return _train_sft(
            train_paths, validation_paths, test_paths, run_dir, config,
            initialize_from=initialize_from, resume_from=resume_from,
        )
    finally:
        torch.set_num_threads(previous_threads)


def _train_sft(
    train_paths, validation_paths, test_paths, run_dir, config, *,
    initialize_from, resume_from,
):
    train_cases = load_sft_cases(
        train_paths,
        expected_split="train",
        limit=config.max_train_cases,
        max_wafer_count=config.max_wafer_count,
    )
    validation_cases = load_sft_cases(
        validation_paths,
        expected_split="validation",
        limit=config.max_validation_cases,
        max_wafer_count=config.max_wafer_count,
    )
    test_cases = load_sft_cases(
        test_paths,
        expected_split="test",
        limit=config.max_test_cases,
        max_wafer_count=config.max_wafer_count,
    )
    hashes = [set(case.problem.problem_hash for case in split)
              for split in (train_cases, validation_cases, test_cases)]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("SFT train, validation, and test cases must be disjoint")
    run_dir.mkdir(parents=True, exist_ok=resume_from is not None)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = IRActorCritic(config.width, config.layers).to(config.device)
    saved = None
    checkpoint_path = initialize_from or resume_from
    if checkpoint_path is not None:
        saved = torch.load(
            checkpoint_path, map_location=config.device, weights_only=True
        )
        if saved.get("feature_version") != FEATURE_VERSION:
            raise ValueError("initial checkpoint feature protocol mismatch")
        if resume_from is not None and saved.get("sft_config") != asdict(config):
            raise ValueError("resume checkpoint SFT config mismatch")
        model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if resume_from is not None:
        assert saved is not None
        optimizer.load_state_dict(saved["optimizer"])
    _write_json(run_dir / "config.json", {
        **asdict(config),
        "feature_version": FEATURE_VERSION,
        "initialize_from": None if initialize_from is None else str(initialize_from),
        "resume_from": None if resume_from is None else str(resume_from),
        "selected_instances": {
            name: [str(case.metadata["instance_id"]) for case in split]
            for name, split in (
                ("train", train_cases),
                ("validation", validation_cases),
                ("test", test_cases),
            )
        },
    })
    _write_json(run_dir / "references.json", {
        "validation": _reference_metrics(validation_cases),
        "test": _reference_metrics(test_cases),
    })

    completed_cases = (
        0 if resume_from is None else int(saved.get("completed_cases", 0))
    )
    supervised_choices = (
        0 if resume_from is None else int(saved.get("step", 0))
    )
    optimizer_steps = 0
    if resume_from is not None:
        optimizer_steps = max(
            (
                int(state.get("step", 0).item())
                if hasattr(state.get("step", 0), "item")
                else int(state.get("step", 0))
            )
            for state in optimizer.state.values()
        )
        if int(saved.get("sft_epoch", 0)) != 1 or config.epochs != 1:
            raise ValueError("resume currently supports a partial first epoch only")
        if not 0 <= completed_cases < len(train_cases):
            raise ValueError("resume checkpoint completed_cases is not a partial epoch")
    resume_skip = completed_cases
    best_accuracy = -1.0
    best_policy_score: tuple[float, float, float, float] | None = None
    best_validation_policy: dict[str, object] | None = None
    best_epoch: int | None = None
    if initialize_from is not None:
        initial_policy = _policy_metrics(model, validation_cases, config)
        _write_json(run_dir / "validation_policy_initial.json", initial_policy)
        best_policy_score = _policy_score(initial_policy)
        best_validation_policy = initial_policy
        best_epoch = 0
        _save_checkpoint(
            run_dir / "best.pt", model, optimizer, config,
            epoch=0, completed_cases=completed_cases,
            supervised_choices=supervised_choices,
        )
    for epoch in range(1, config.epochs + 1):
        order = list(train_cases)
        random.Random(config.seed + epoch).shuffle(order)
        if resume_from is not None:
            order = order[resume_skip:]
        epoch_loss = epoch_correct = epoch_choices = 0.0
        observations = []
        targets: list[tuple[int, ...]] = []

        def flush() -> None:
            nonlocal optimizer_steps, epoch_loss, epoch_correct, epoch_choices
            if not observations:
                return
            model.train()
            output = model(collate_graphs(observations, config.device))
            log_probabilities = F.log_softmax(output.logits, dim=-1)
            losses = []
            for row, acceptable in enumerate(targets):
                selected = torch.tensor(acceptable, device=config.device)
                losses.append(-torch.logsumexp(
                    log_probabilities[row, selected], dim=0
                ))
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite SFT loss")
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm, error_if_nonfinite=True
            )
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(targets)
            predictions = output.logits.argmax(dim=1).tolist()
            epoch_correct += sum(
                prediction in acceptable
                for prediction, acceptable in zip(predictions, targets)
            )
            epoch_choices += len(targets)
            optimizer_steps += 1
            observations.clear()
            targets.clear()
            if not math.isfinite(float(grad_norm)):
                raise FloatingPointError("non-finite SFT gradient norm")

        def learn(observation, actions: tuple[int, ...]) -> None:
            observations.append(observation)
            targets.append(actions)
            if len(observations) >= config.batch_size:
                flush()

        for case_index, case in enumerate(order, 1):
            started = time.perf_counter()
            replay = replay_expert(
                case.problem,
                case.actions,
                on_choice_set=learn,
                max_decisions=config.max_decisions,
            )
            flush()
            completed_cases += 1
            supervised_choices += int(replay["supervised_choice_count"])
            row = {
                "epoch": epoch,
                "case": case_index,
                "instance_id": case.metadata["instance_id"],
                "wafer_count": case.metadata["wafer_count"],
                "expert_solver": case.metadata["expert_solver"],
                "supervised_choices": replay["supervised_choice_count"],
                "inserted_waits": replay["inserted_wait_count"],
                "ir_makespan": replay["ir_makespan"],
                "seconds": time.perf_counter() - started,
                "optimizer_steps": optimizer_steps,
            }
            _append_jsonl(run_dir / "cases.jsonl", row)
            print(json.dumps(row), flush=True)
            _save_checkpoint(
                run_dir / "last.pt", model, optimizer, config,
                epoch=epoch, completed_cases=completed_cases,
                supervised_choices=supervised_choices,
            )
        train_row = {
            "epoch": epoch,
            "mean_loss": epoch_loss / epoch_choices,
            "online_top1_accuracy": epoch_correct / epoch_choices,
            "choice_count": int(epoch_choices),
            "optimizer_steps": optimizer_steps,
        }
        _append_jsonl(run_dir / "metrics.jsonl", train_row)
        validation_imitation = imitation_metrics(model, validation_cases, config)
        _write_json(run_dir / f"validation_imitation_epoch_{epoch}.json", validation_imitation)
        best_accuracy = max(
            best_accuracy, float(validation_imitation["top1_accuracy"]),
        )
        validation_policy = _policy_metrics(model, validation_cases, config)
        _write_json(run_dir / f"validation_policy_epoch_{epoch}.json", validation_policy)
        score = _policy_score(validation_policy)
        if best_policy_score is None or score > best_policy_score:
            best_policy_score = score
            best_validation_policy = validation_policy
            best_epoch = epoch
            _save_checkpoint(
                run_dir / "best.pt", model, optimizer, config,
                epoch=epoch, completed_cases=completed_cases,
                supervised_choices=supervised_choices,
            )

    assert best_validation_policy is not None and best_epoch is not None
    selected = torch.load(run_dir / "best.pt", map_location=config.device, weights_only=True)
    model.load_state_dict(selected["model"])
    test_policy = _policy_metrics(model, test_cases, config)
    result = {
        "completed_cases": completed_cases,
        "supervised_choices": supervised_choices,
        "optimizer_steps": optimizer_steps,
        "best_validation_imitation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "validation_policy": best_validation_policy,
        "test_policy": test_policy,
    }
    if config.shield_evaluation:
        result["validation_shielded_policy"] = _policy_metrics(
            model, validation_cases, config, shield=True
        )
        result["test_shielded_policy"] = _policy_metrics(
            model, test_cases, config, shield=True
        )
    _write_json(run_dir / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume-from", type=Path)
    for name in ("epochs", "batch_size", "width", "layers", "max_decisions", "seed",
                 "cpu_threads", "max_train_cases", "max_validation_cases", "max_test_cases",
                 "max_wafer_count", "max_shield_backtracks"):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=getattr(SFTConfig(), name),
        )
    parser.add_argument("--learning-rate", type=float, default=SFTConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=SFTConfig.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=SFTConfig.max_grad_norm)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shield-evaluation", action="store_true")
    args = parser.parse_args(argv)
    config = SFTConfig(**{
        key: value for key, value in vars(args).items()
        if key in SFTConfig.__dataclass_fields__
    })
    result = train_sft(
        args.train, args.validation, args.test, args.run_dir, config,
        initialize_from=args.initialize_from, resume_from=args.resume_from,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
