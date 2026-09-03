"""PPO training on anonymous IR graphs; no legacy observation/action adapter."""

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
from torch.distributions import Categorical

from cluster_rl.ppo import advantages, clipped_losses, normalize_choice_advantages

from .data import load_cases
from .env import IRSchedulingEnv
from .graph import EDGE_TYPES, FEATURE_VERSION, NODE_TYPES
from .network import IRActorCritic, collate_graphs


@dataclass(frozen=True)
class IRTrainConfig:
    total_steps: int = 256
    rollout_steps: int = 64
    minibatch_size: int = 16
    epochs: int = 4
    width: int = 64
    layers: int = 4
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.99
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    max_decisions: int = 128
    max_time_seconds: float | None = None
    reward_scale_seconds: float = 100.0
    evaluation_interval: int = 1
    seed: int = 17
    device: str = "cpu"
    cpu_threads: int = 1

    def __post_init__(self) -> None:
        for name in ("total_steps", "rollout_steps", "minibatch_size", "epochs", "layers",
                     "max_decisions", "evaluation_interval", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width < 8:
            raise ValueError("width must be >= 8")
        for name in ("learning_rate", "max_grad_norm", "target_kl", "reward_scale_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 < self.clip_coefficient < 1:
            raise ValueError("clip_coefficient must be in (0, 1)")
        for name in ("value_coefficient", "entropy_coefficient"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def env_options(self) -> dict:
        return {"max_decisions": self.max_decisions, "max_time_seconds": self.max_time_seconds,
                "reward_scale_seconds": self.reward_scale_seconds}


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _checkpoint(path: Path, model: IRActorCritic, optimizer, config: IRTrainConfig, step: int) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({"feature_version": FEATURE_VERSION, "node_types": NODE_TYPES, "edge_types": EDGE_TYPES,
                "config": asdict(config), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "step": step}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[IRActorCritic, IRTrainConfig]:
    saved = torch.load(path, map_location=device, weights_only=True)
    if (saved.get("feature_version") != FEATURE_VERSION or tuple(saved.get("node_types", ())) != NODE_TYPES
            or tuple(saved.get("edge_types", ())) != EDGE_TYPES):
        raise ValueError("checkpoint does not match the IR graph feature protocol")
    config = IRTrainConfig(**{**saved["config"], "device": device})
    model = IRActorCritic(config.width, config.layers).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, config


@torch.no_grad()
def evaluate(model: IRActorCritic | None, cases: list, config: IRTrainConfig, *,
             baseline: str = "random", trace_dir: Path | None = None) -> dict:
    if baseline not in {"random", "shortest"}:
        raise ValueError("unknown baseline")
    if model is not None:
        model.eval()
    rng, rows = random.Random(config.seed), []
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=False)
    for index, (path, problem) in enumerate(cases):
        env = IRSchedulingEnv(problem, **config.env_options())
        observation, info = env.reset(seed=config.seed + index)
        total_reward = 0.0
        while env.reason is None:
            if model is not None:
                output = model(collate_graphs([observation], config.device))
                action = int(output.logits[0].argmax())
            elif baseline == "random":
                action = rng.randrange(observation.action_count)
            else:
                durations = [item.duration_ticks for item in env.frame.intents]
                if env.wait_tick is not None:
                    durations.append(env.wait_tick - env.snapshot.tick)
                action = min(range(len(durations)), key=durations.__getitem__)
            observation, reward, _, _, info = env.step(action)
            total_reward += reward
        report = env.audit()
        if not report.ok:
            raise RuntimeError(f"independent trajectory audit failed for {path}: {report.issues}")
        if trace_dir is not None:
            (trace_dir / f"{index:04d}.snapshot.json").write_text(env.snapshot.canonical_json(), encoding="utf-8")
        rows.append({"path": str(path), "problem_hash": problem.problem_hash, "audit_ok": True,
                     "success": info["success"], "termination_reason": info["termination_reason"],
                     "makespan": info["elapsed_seconds"] if info["success"] else None,
                     "elapsed_seconds": info["elapsed_seconds"], "decisions": info["decisions"],
                     "return": total_reward})
    successful = [row["makespan"] for row in rows if row["success"]]
    return {"success_rate": len(successful) / len(rows),
            "mean_makespan": sum(successful) / len(successful) if successful else None,
            "cases": rows}


def _score(report: dict) -> tuple[float, float]:
    return report["success_rate"], -(report["mean_makespan"] if report["mean_makespan"] is not None else math.inf)


def _update(model, optimizer, observations, actions_taken, logps, old_values, returns, adv, config):
    device = config.device
    actions_tensor = torch.tensor(actions_taken, device=device)
    old_logps = torch.tensor(logps, device=device)
    choice = torch.tensor([o.action_count > 1 for o in observations], device=device)
    normalized = normalize_choice_advantages(adv, choice)
    totals, count = {name: 0.0 for name in ("policy_loss", "value_loss", "entropy", "kl", "grad_norm")}, 0
    model.train()
    for _ in range(config.epochs):
        epoch_kl = []
        for indexes in torch.randperm(len(observations)).split(config.minibatch_size):
            batch = collate_graphs([observations[i] for i in indexes.tolist()], device)
            ix = indexes.to(device)
            output = model(batch)
            distribution = Categorical(logits=output.logits)
            log_ratio = distribution.log_prob(actions_tensor[ix]) - old_logps[ix]
            policy, value, entropy = clipped_losses(
                log_ratio, output.value, old_values[ix], returns[ix], normalized[ix],
                distribution.entropy(), choice[ix], config.clip_coefficient,
            )
            loss = policy + config.value_coefficient * value - config.entropy_coefficient * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO objective")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("non-finite model parameters")
            kl = (log_ratio[choice[ix]].exp() - 1 - log_ratio[choice[ix]]).mean() if choice[ix].any() else loss * 0
            if choice[ix].any():
                epoch_kl.append(float(kl.detach()))
            for key, value in zip(totals, (policy, value, entropy, kl, norm)):
                totals[key] += float(value.detach())
            count += 1
        if epoch_kl and sum(epoch_kl) / len(epoch_kl) > config.target_kl:
            break
    return {**{key: value / count for key, value in totals.items()}, "choice_fraction": float(choice.float().mean())}


def train(train_paths: list[Path], validation_paths: list[Path], run_dir: Path,
          config: IRTrainConfig = IRTrainConfig(), *, test_paths: list[Path] | None = None) -> dict:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.cpu_threads)
    try:
        return _train(train_paths, validation_paths, run_dir, config, test_paths or [])
    finally:
        torch.set_num_threads(previous_threads)


def _train(train_paths, validation_paths, run_dir, config, test_paths):
    cases = load_cases(train_paths, expected_split="train")
    validation = load_cases(validation_paths, expected_split="validation")
    test = load_cases(test_paths, expected_split="test") if test_paths else []
    hashes = [{problem.problem_hash for _, problem in split} for split in (cases, validation, test)]
    if any(hashes[i] & hashes[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("training, validation and test problems must be disjoint")
    # Validate every observation program before creating any training artifacts.
    envs = [IRSchedulingEnv(problem, **config.env_options()) for _, problem in cases]
    for env in envs:
        env.reset(seed=config.seed)
        if env.reason is not None:
            raise ValueError("training problem has no initial decision; provide a nontrivial feasible task")
    for _, problem in [*validation, *test]:
        IRSchedulingEnv(problem, **config.env_options())
    run_dir.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    model = IRActorCritic(config.width, config.layers).to(config.device)
    initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    _write_json(run_dir / "config.json", {**asdict(config), "feature_version": FEATURE_VERSION,
                                         "splits": {name: [{"path": str(p), "problem_hash": ir.problem_hash}
                                                           for p, ir in split]
                                                    for name, split in (("train", cases), ("validation", validation), ("test", test))}})
    baselines = {name: evaluate(None, validation, config, baseline=name) for name in ("random", "shortest")}
    initial_report = evaluate(model, validation, config)
    _write_json(run_dir / "baselines.json", {**baselines, "initial_policy": initial_report})
    best_score = _score(initial_report)
    best_step = 0
    _checkpoint(run_dir / "best.pt", model, optimizer, config, 0)
    episode, steps, update, episode_return = 0, 0, 0, 0.0
    env = envs[0]
    observation, _ = env.reset(seed=config.seed)
    if env.reason is not None:
        raise ValueError("training problem has no initial decision; provide a nontrivial feasible task")
    while steps < config.total_steps:
        started = time.perf_counter()
        observations, actions_taken, logps, values, rewards, dones, episodes = [], [], [], [], [], [], []
        for _ in range(min(config.rollout_steps, config.total_steps - steps)):
            model.eval()
            with torch.no_grad():
                output = model(collate_graphs([observation], config.device))
                distribution = Categorical(logits=output.logits)
                action = distribution.sample()
            observations.append(observation)
            actions_taken.append(int(action[0]))
            logps.append(float(distribution.log_prob(action)[0]))
            values.append(float(output.value[0]))
            observation, reward, terminated, truncated, info = env.step(int(action[0]))
            rewards.append(reward)
            dones.append(terminated or truncated)
            episode_return += reward
            steps += 1
            if dones[-1]:
                episodes.append({"episode": episode, "step": steps, "return": episode_return,
                                 "success": info["success"], "reason": info["termination_reason"],
                                 "elapsed_seconds": info["elapsed_seconds"], "decisions": info["decisions"]})
                episode += 1
                episode_return = 0.0
                env = envs[episode % len(envs)]
                observation, _ = env.reset(seed=config.seed + episode)
                if env.reason is not None:
                    raise ValueError("training problem has no initial decision")
        with torch.no_grad():
            bootstrap = model(collate_graphs([observation], config.device)).value
            old_values = torch.tensor(values, device=config.device)
            adv = advantages(torch.tensor(rewards, device=config.device)[:, None],
                             torch.tensor(dones, device=config.device)[:, None], old_values[:, None],
                             bootstrap, config.gamma, config.gae_lambda).squeeze(1)
        rollout_seconds = time.perf_counter() - started
        update_started = time.perf_counter()
        metrics = _update(model, optimizer, observations, actions_taken, logps, old_values,
                          adv + old_values, adv, config)
        update += 1
        row = {"update": update, "step": steps, "rollout_seconds": rollout_seconds,
               "ppo_seconds": time.perf_counter() - update_started,
               "steps_per_second": len(observations) / rollout_seconds, **metrics}
        if update % config.evaluation_interval == 0 or steps == config.total_steps:
            report = evaluate(model, validation, config)
            row.update(validation_success_rate=report["success_rate"], validation_makespan=report["mean_makespan"])
            if _score(report) > best_score:
                best_score = _score(report)
                best_step = steps
                _checkpoint(run_dir / "best.pt", model, optimizer, config, steps)
        _checkpoint(run_dir / "last.pt", model, optimizer, config, steps)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        with (run_dir / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            for item in episodes:
                stream.write(json.dumps(item, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)
    parameter_change = math.sqrt(sum(float((p.detach() - initial_parameters[name]).square().sum())
                                     for name, p in model.named_parameters()))
    actor_change = math.sqrt(sum(float((p.detach() - initial_parameters[name]).square().sum())
                                 for name, p in model.named_parameters() if name.startswith("actor.")))
    best, _ = load_checkpoint(run_dir / "best.pt", config.device)
    result = {"steps": steps, "updates": update, "episodes": episode, "best_step": best_step,
              "parameter_change_l2": parameter_change,
              "actor_change_l2": actor_change, "validation": evaluate(best, validation, config,
                                                                       trace_dir=run_dir / "validation_traces")}
    if test:
        result["test"] = evaluate(best, test, config, trace_dir=run_dir / "test_traces")
    _write_json(run_dir / "result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+")
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    for name in ("total_steps", "rollout_steps", "minibatch_size", "epochs", "width", "layers",
                 "max_decisions", "evaluation_interval", "seed", "cpu_threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(IRTrainConfig(), name))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-time-seconds", type=float)
    args = parser.parse_args(argv)
    if args.evaluate_checkpoint:
        torch.set_num_threads(args.cpu_threads)
        model, config = load_checkpoint(args.evaluate_checkpoint, args.device)
        cases = load_cases(args.validation, expected_split="validation")
        args.run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(args.run_dir / "evaluation.json", evaluate(model, cases, config, trace_dir=args.run_dir / "traces"))
    else:
        if not args.train:
            parser.error("--train is required for training")
        config = IRTrainConfig(**{key: value for key, value in vars(args).items()
                                  if key in IRTrainConfig.__dataclass_fields__})
        train(args.train, args.validation, args.run_dir, config, test_paths=args.test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
