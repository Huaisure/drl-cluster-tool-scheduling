from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext, redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical

from cluster_rl.cluster_env import ClusterEnv
from examples.run_scenarios import (
    SCENARIO_DIR,
    network_greedy_selector,
    rollout,
)
from cluster_rl.network import (
    ClusterActorCritic,
    ClusterObservationEncoder,
    EncodedObservation,
    TransformerConfig,
    collate_encoded_observations_fast,
)
from cluster_generator import ProblemGenerator
from problem import ClusterProblem, load_problem


CHECKPOINT_VERSION = 4
UPDATE_FIELDS = (
    "update",
    "global_step",
    "completed_episodes",
    "success_rate",
    "mean_makespan",
    "mean_normalized_return",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "choice_fraction",
    "grad_norm",
)
EPISODE_FIELDS = (
    "update",
    "global_step",
    "scenario",
    "makespan",
    "reference_makespan",
    "normalized_cost",
    "normalized_return",
    "reward",
    "success",
)
EVALUATION_FIELDS = (
    "split",
    "scenario",
    "reference_makespan",
    "makespan",
    "normalized_cost",
    "relative_gain",
    "action_count",
    "valid",
)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    scenario_paths: tuple[Path, ...] = ()
    train_mode: Literal["scenarios", "generator"] = "scenarios"
    num_envs: int = 8
    cpu_workers: int = 0
    generator_seed: int = 42
    generator_max_attempts: int = 64
    difficulty_weights: dict[str, float] = field(
        default_factory=lambda: {
            "easy": 0.20,
            "medium": 0.50,
            "hard": 0.25,
            "edge": 0.05,
        }
    )
    validation_manifest: Path | None = None
    test_manifest: Path | None = None
    run_dir: Path = Path("runs/ppo_cluster")
    checkpoint: Path = Path("runs/ppo_cluster/checkpoint.pt")
    resume: Path | None = None
    total_steps: int = 100_000
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 128
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    model_dim: int = 64
    num_heads: int = 4
    hgt_layers: int = 2
    num_layers: int = 2
    feedforward_dim: int = 128
    seed: int = 0
    device: str = "cpu"
    profile_timing: bool = False
    log_interval: int = 1
    checkpoint_interval: int = 25
    evaluate: bool = True

    def __post_init__(self) -> None:
        positive_integers = {
            "total_steps": self.total_steps,
            "rollout_steps": self.rollout_steps,
            "epochs": self.epochs,
            "minibatch_size": self.minibatch_size,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        if self.train_mode == "generator":
            positive_integers["num_envs"] = self.num_envs
            positive_integers["generator_max_attempts"] = self.generator_max_attempts
        for name, value in positive_integers.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.train_mode == "scenarios" and not self.scenario_paths:
            raise ValueError("at least one scenario is required")
        if self.train_mode not in ("scenarios", "generator"):
            raise ValueError("train_mode must be 'scenarios' or 'generator'")
        if self.cpu_workers < 0:
            raise ValueError("cpu_workers must be non-negative")
        if self.cpu_workers and self.train_mode != "generator":
            raise ValueError("cpu_workers are only supported in generator mode")
        if self.cpu_workers > self.num_envs:
            raise ValueError("cpu_workers must not exceed num_envs")
        if self.train_mode == "generator":
            expected_difficulties = {"easy", "medium", "hard", "edge"}
            if set(self.difficulty_weights) != expected_difficulties:
                raise ValueError(
                    "difficulty_weights must contain easy, medium, hard, and edge"
                )
            if any(
                not math.isfinite(weight) or weight < 0
                for weight in self.difficulty_weights.values()
            ) or sum(self.difficulty_weights.values()) <= 0:
                raise ValueError(
                    "difficulty_weights must be finite, non-negative, and have a positive sum"
                )
            if (
                self.evaluate
                and self.validation_manifest is None
                and self.test_manifest is None
            ):
                raise ValueError(
                    "generator training evaluation requires a validation or test manifest"
                )
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0.0 < self.clip_coefficient < 1.0:
            raise ValueError("clip_coefficient must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    states: tuple[EncodedObservation, ...]
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor


@dataclass(frozen=True, slots=True)
class EpisodeStat:
    scenario: str
    makespan: float
    reference_makespan: float
    normalized_return: float
    reward: float
    success: bool


@dataclass(slots=True)
class EnvSlot:
    env: ClusterEnv
    encoder: ClusterObservationEncoder
    observation: dict[str, Any]
    reference_makespan: float
    problem_seed: int
    episode_index: int = 0
    episode_reward: float = 0.0


@dataclass(frozen=True, slots=True)
class EnvSlotState:
    reference_makespan: float
    problem_seed: int
    episode_index: int


@dataclass(frozen=True, slots=True)
class ParallelStepResult:
    index: int
    encoded: EncodedObservation
    normalized_reward: float
    done: bool
    episode_stat: EpisodeStat | None
    slot_state: EnvSlotState


class PhaseTimer:
    """Collect synchronized wall times for one diagnostic PPO update."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, name: str, seconds: float, count: int = 1) -> None:
        self.totals[name] += seconds
        self.counts[name] += count

    @contextmanager
    def measure(self, name: str):
        self._synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.add(name, time.perf_counter() - started)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _measure(timer: PhaseTimer | None, name: str):
    return nullcontext() if timer is None else timer.measure(name)


class GeneratorEnvFactory:
    """Create deterministic online-training environment slots."""

    def __init__(
        self,
        config: PPOConfig,
        *,
        generator: ProblemGenerator | None = None,
    ) -> None:
        self.config = config
        self.generator = generator or ProblemGenerator()

    def episode_seed(self, slot_index: int, episode_index: int) -> int:
        return (
            self.config.generator_seed
            + slot_index
            + episode_index * self.config.num_envs
        )

    def make(self, slot_index: int, episode_index: int = 0) -> EnvSlot:
        return make_env_slot(
            self.generator,
            seed=self.episode_seed(slot_index, episode_index),
            episode_index=episode_index,
            difficulty_weights=self.config.difficulty_weights,
            max_attempts=self.config.generator_max_attempts,
        )


class _Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def _training_log(path: Path, *, append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as stream:
        with redirect_stdout(_Tee(sys.stdout, stream)):
            yield


def _default_scenarios() -> tuple[Path, ...]:
    return tuple(sorted(SCENARIO_DIR.glob("*.json")))


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")
    return device


def _fifo_serial_action(
    env: ClusterEnv,
    observation: dict[str, Any],
) -> int:
    legal_actions = np.flatnonzero(observation["action_mask"])
    if not len(legal_actions):
        raise RuntimeError("FIFO serial reference reached a deadlock")

    robot_count = len(env._robot_ids)
    pick_action_count = len(env.wafer_keys) * robot_count
    advance_action = env.action_space.n - 1
    active_wafers = [
        wafer_index
        for wafer_index, wafer in enumerate(env._wafers)
        if env._is_pick_reserved(wafer_index)
        or wafer.module_id is None
        or 0 < wafer.step_index < (
            len(env.problem.routes[env.wafer_keys[wafer_index][0]].visits) + 1
        )
    ]
    if active_wafers:
        wafer_index = min(active_wafers)
        wafer_pick_actions = legal_actions[
            (legal_actions >= wafer_index * robot_count)
            & (legal_actions < (wafer_index + 1) * robot_count)
        ]
        if len(wafer_pick_actions):
            return int(wafer_pick_actions[0])

        place_actions = legal_actions[
            (legal_actions >= pick_action_count)
            & (legal_actions < advance_action)
        ]
        if len(place_actions):
            return int(place_actions[0])
        if advance_action in legal_actions:
            return advance_action
        raise RuntimeError("FIFO serial reference cannot advance the active wafer")

    pick_actions = legal_actions[legal_actions < pick_action_count]
    if len(pick_actions):
        return int(pick_actions[0])
    if advance_action in legal_actions:
        return advance_action
    raise RuntimeError("FIFO serial reference has no legal Pick")


def _first_legal_reference(problem: ClusterProblem, scenario: str) -> float:
    return rollout(
        ClusterEnv(problem),
        scenario,
        "reference_fifo_serial",
        _fifo_serial_action,
    ).makespan


def _retry_seed(seed: int, attempt: int) -> int:
    if attempt == 0:
        return seed
    digest = hashlib.sha256(
        f"cluster-rl:{seed}:{attempt}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def make_env_slot(
    generator: ProblemGenerator,
    *,
    seed: int,
    episode_index: int = 0,
    difficulty_weights: Mapping[str, float] | None = None,
    max_attempts: int = 64,
) -> EnvSlot:
    last_error: RuntimeError | IndexError | None = None
    for attempt in range(max_attempts):
        problem_seed = _retry_seed(seed, attempt)
        problem = generator.sample_curriculum(
            seed=problem_seed,
            split="train",
            weights=(
                None
                if difficulty_weights is None
                else dict(difficulty_weights)
            ),
        )
        scenario = str(problem.meta.get("name", f"generated_{problem_seed}"))
        try:
            reference_makespan = _first_legal_reference(problem, scenario)
        except (RuntimeError, IndexError) as exc:
            last_error = exc
            continue

        env = ClusterEnv(problem)
        encoder = ClusterObservationEncoder.from_env(env)
        observation, _ = env.reset(seed=problem_seed)
        return EnvSlot(
            env=env,
            encoder=encoder,
            observation=observation,
            reference_makespan=reference_makespan,
            problem_seed=problem_seed,
            episode_index=episode_index,
        )
    raise RuntimeError(
        f"failed to generate a FIFO-compatible problem after {max_attempts} attempts"
    ) from last_error


def _reference_makespans(
    problems: list[ClusterProblem],
) -> list[float]:
    return [
        _first_legal_reference(
            problem,
            str(problem.meta.get("name", f"env_{index}")),
        )
        for index, problem in enumerate(problems)
    ]


def _fixed_env_slots(
    problems: list[ClusterProblem],
    references: list[float],
    seed: int,
) -> list[EnvSlot]:
    slots = []
    for index, (problem, reference) in enumerate(zip(problems, references)):
        env = ClusterEnv(problem)
        slots.append(
            EnvSlot(
                env=env,
                encoder=ClusterObservationEncoder.from_env(env),
                observation=env.reset(seed=seed + index)[0],
                reference_makespan=reference,
                problem_seed=seed + index,
            )
        )
    return slots


def _manifest_problem_paths(manifest_path: Path) -> tuple[Path, ...]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluation manifest: {manifest_path}") from exc
    if manifest.get("generator", {}).get("mode") != "ppo":
        raise ValueError(f"not a PPO dataset manifest: {manifest_path}")

    paths = []
    dataset_dir = manifest_path.parent.resolve()
    for entry in manifest.get("instances", []):
        problem_file = entry.get("problem_file")
        if not isinstance(problem_file, str):
            raise ValueError(
                f"evaluation manifest contains an unmaterialized problem: {manifest_path}"
            )
        problem_path = (dataset_dir / problem_file).resolve()
        if problem_path.parent != dataset_dir:
            raise ValueError(
                f"evaluation manifest contains an unsafe problem path: {problem_file}"
            )
        paths.append(problem_path)
    if not paths:
        raise ValueError(f"evaluation manifest has no instances: {manifest_path}")
    return tuple(paths)


def _evaluation_problems(
    config: PPOConfig,
) -> list[tuple[str, ClusterProblem]]:
    manifests = tuple(
        (split, path)
        for split, path in (
            ("validation", config.validation_manifest),
            ("test", config.test_manifest),
        )
        if path is not None
    )
    if manifests:
        return [
            (split, load_problem(problem_path))
            for split, manifest_path in manifests
            for problem_path in _manifest_problem_paths(manifest_path)
        ]
    return [
        ("scenario", load_problem(path))
        for path in config.scenario_paths
    ]


def _flatten_encoded_steps(
    steps: Sequence[Sequence[EncodedObservation]],
) -> tuple[EncodedObservation, ...]:
    return tuple(encoded for step in steps for encoded in step)


def _advantages(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    last_values: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tensor:
    result = torch.zeros_like(rewards)
    advantage = torch.zeros_like(last_values)

    for step in reversed(range(rewards.shape[0])):
        next_values = (
            last_values
            if step == rewards.shape[0] - 1
            else values[step + 1]
        )
        next_non_terminal = ~dones[step]
        delta = (
            rewards[step]
            + gamma * next_values * next_non_terminal
            - values[step]
        )
        advantage = (
            delta
            + gamma * gae_lambda * next_non_terminal * advantage
        )
        result[step] = advantage
    return result


def _normalized_reward(
    reward: float,
    reference_makespan: float,
    success: bool,
) -> float:
    return reward / reference_makespan + float(success)


def _env_slot_state(slot: EnvSlot) -> EnvSlotState:
    return EnvSlotState(
        reference_makespan=slot.reference_makespan,
        problem_seed=slot.problem_seed,
        episode_index=slot.episode_index,
    )


def _step_env_slot(
    slot: EnvSlot,
    action: int,
    index: int,
    env_factory: GeneratorEnvFactory | None,
    timings: dict[str, float] | None = None,
) -> tuple[EnvSlot, float, bool, EpisodeStat | None]:
    step_started = time.perf_counter()
    observation, reward, terminated, truncated, info = slot.env.step(action)
    if timings is not None:
        name = "rollout.worker_env_step_cpu"
        timings[name] = timings.get(name, 0.0) + (
            time.perf_counter() - step_started
        )
    done = terminated or truncated
    success = bool(info.get("is_success"))
    slot.episode_reward += reward
    normalized_reward = _normalized_reward(
        reward,
        slot.reference_makespan,
        done and success,
    )
    episode_stat = None

    if done:
        episode_stat = EpisodeStat(
            scenario=str(slot.env.problem.meta.get("name", f"env_{index}")),
            makespan=float(info["time"]),
            reference_makespan=slot.reference_makespan,
            normalized_return=(
                slot.episode_reward / slot.reference_makespan + float(success)
            ),
            reward=slot.episode_reward,
            success=success,
        )
        if env_factory is None:
            reset_started = time.perf_counter()
            slot.observation, _ = slot.env.reset()
            slot.episode_reward = 0.0
            if timings is not None:
                name = "rollout.worker_reset_reference_cpu"
                timings[name] = timings.get(name, 0.0) + (
                    time.perf_counter() - reset_started
                )
        else:
            reset_started = time.perf_counter()
            slot = env_factory.make(index, slot.episode_index + 1)
            if timings is not None:
                name = "rollout.worker_reset_reference_cpu"
                timings[name] = timings.get(name, 0.0) + (
                    time.perf_counter() - reset_started
                )
    else:
        slot.observation = observation
    return slot, normalized_reward, done, episode_stat


def _parallel_env_worker(
    connection: Connection,
    config: PPOConfig,
    indexed_slots: list[tuple[int, EnvSlot]],
) -> None:
    try:
        torch.set_num_threads(1)
        slots = dict(indexed_slots)
        env_factory = GeneratorEnvFactory(config)
        initial = [
            (
                index,
                slot.encoder.encode(slot.observation),
                _env_slot_state(slot),
            )
            for index, slot in slots.items()
        ]
        connection.send(("ok", initial))

        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command != "step":
                raise ValueError(f"unknown parallel environment command: {command}")

            profile = bool(payload["profile"])
            actions = payload["actions"]
            worker_timings: dict[str, float] = {}
            results = []
            for index, action in actions:
                slot, reward, done, episode_stat = _step_env_slot(
                    slots[index],
                    action,
                    index,
                    env_factory,
                    worker_timings if profile else None,
                )
                slots[index] = slot
                encode_started = time.perf_counter()
                encoded = slot.encoder.encode(slot.observation)
                if profile:
                    name = "rollout.worker_encode_cpu"
                    worker_timings[name] = worker_timings.get(name, 0.0) + (
                        time.perf_counter() - encode_started
                    )
                results.append(
                    ParallelStepResult(
                        index=index,
                        encoded=encoded,
                        normalized_reward=reward,
                        done=done,
                        episode_stat=episode_stat,
                        slot_state=_env_slot_state(slot),
                    )
                )
            connection.send(("ok", (results, worker_timings)))
    except (EOFError, KeyboardInterrupt):
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ParallelEnvPool:
    """Keep independent generator environments in persistent CPU processes."""

    def __init__(
        self,
        slots: list[EnvSlot],
        config: PPOConfig,
    ) -> None:
        worker_count = min(config.cpu_workers, len(slots))
        if worker_count <= 0:
            raise ValueError("parallel environment pool requires cpu_workers")

        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._indexes = [
            list(range(offset, len(slots), worker_count))
            for offset in range(worker_count)
        ]
        self._encoded: list[EncodedObservation | None] = [None] * len(slots)
        self._states: list[EnvSlotState | None] = [None] * len(slots)
        self._closed = False

        try:
            for worker_index, indexes in enumerate(self._indexes):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_parallel_env_worker,
                    args=(
                        child,
                        config,
                        [(index, slots[index]) for index in indexes],
                    ),
                    name=f"cluster-env-{worker_index}",
                )
                process.daemon = True
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)

            for connection in self._connections:
                for index, encoded, state in self._receive(connection):
                    self._encoded[index] = encoded
                    self._states[index] = state
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _receive(connection: Connection) -> Any:
        try:
            status, payload = connection.recv()
        except EOFError as exc:
            raise RuntimeError("parallel environment worker exited unexpectedly") from exc
        if status == "error":
            raise RuntimeError(f"parallel environment worker failed:\n{payload}")
        return payload

    @property
    def encoded(self) -> list[EncodedObservation]:
        if any(item is None for item in self._encoded):
            raise RuntimeError("parallel environment pool has incomplete observations")
        return [item for item in self._encoded if item is not None]

    @property
    def states(self) -> list[EnvSlotState]:
        if any(item is None for item in self._states):
            raise RuntimeError("parallel environment pool has incomplete states")
        return [item for item in self._states if item is not None]

    def step(
        self,
        actions: Sequence[int],
        timer: PhaseTimer | None = None,
    ) -> tuple[list[float], list[bool], list[EpisodeStat]]:
        if len(actions) != len(self._encoded):
            raise ValueError("actions must contain one action per environment")

        for connection, indexes in zip(self._connections, self._indexes):
            connection.send(
                (
                    "step",
                    {
                        "actions": [
                            (index, int(actions[index])) for index in indexes
                        ],
                        "profile": timer is not None,
                    },
                )
            )

        rewards = [0.0] * len(actions)
        dones = [False] * len(actions)
        episode_stats = []
        for connection in self._connections:
            results, worker_timings = self._receive(connection)
            if timer is not None:
                for name, seconds in worker_timings.items():
                    timer.add(name, seconds)
            for result in results:
                self._encoded[result.index] = result.encoded
                self._states[result.index] = result.slot_state
                rewards[result.index] = result.normalized_reward
                dones[result.index] = result.done
                if result.episode_stat is not None:
                    episode_stats.append(result.episode_stat)
        return rewards, dones, episode_stats

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for connection in self._connections:
            connection.close()
        for process in self._processes:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)


def _collect_rollout(
    model: ClusterActorCritic,
    slots: list[EnvSlot],
    config: PPOConfig,
    device: torch.device,
    env_factory: GeneratorEnvFactory | None = None,
    timer: PhaseTimer | None = None,
) -> tuple[RolloutBatch, list[EpisodeStat]]:
    state_steps = []
    action_steps = []
    log_probability_steps = []
    value_steps = []
    reward_steps = []
    done_steps = []
    episode_stats = []

    model.eval()
    for _ in range(config.rollout_steps):
        with _measure(timer, "rollout.encode_cpu"):
            encoded = [
                slot.encoder.encode(slot.observation)
                for slot in slots
            ]
        with _measure(timer, "rollout.collate_h2d"):
            states = collate_encoded_observations_fast(encoded, device=device)
        with _measure(timer, "rollout.policy_gpu"):
            with torch.inference_mode():
                output = model(states)
                distribution = Categorical(logits=output.logits)
                model_actions = distribution.sample()

        with _measure(timer, "rollout.action_d2h"):
            env_actions = states.to_env_actions(model_actions).cpu().tolist()
        normalized_rewards = []
        dones = []
        step_timings: dict[str, float] = {}
        with _measure(timer, "rollout.env_wait"):
            for index, action in enumerate(env_actions):
                slot, normalized_reward, done, episode_stat = _step_env_slot(
                    slots[index],
                    action,
                    index,
                    env_factory,
                    step_timings if timer is not None else None,
                )
                slots[index] = slot
                normalized_rewards.append(normalized_reward)
                dones.append(done)
                if episode_stat is not None:
                    episode_stats.append(episode_stat)
        if timer is not None:
            for name, seconds in step_timings.items():
                timer.add(name, seconds)

        with _measure(timer, "rollout.buffer"):
            state_steps.append(tuple(encoded))
            action_steps.append(model_actions)
            log_probability_steps.append(
                distribution.log_prob(model_actions)
            )
            value_steps.append(output.value)
            reward_steps.append(
                torch.tensor(
                    normalized_rewards,
                    dtype=torch.float32,
                    device=device,
                )
            )
            done_steps.append(
                torch.tensor(dones, dtype=torch.bool, device=device)
            )
    with _measure(timer, "rollout.bootstrap"):
        encoded = [
            slot.encoder.encode(slot.observation)
            for slot in slots
        ]
        with torch.inference_mode():
            last_states = collate_encoded_observations_fast(
                encoded,
                device=device,
            )
            last_values = model(last_states).value

    with _measure(timer, "rollout.gae"):
        rewards = torch.stack(reward_steps)
        dones = torch.stack(done_steps)
        values = torch.stack(value_steps)
        advantages = _advantages(
            rewards,
            dones,
            values,
            last_values,
            config.gamma,
            config.gae_lambda,
        )
        returns = advantages + values

    with _measure(timer, "rollout.flatten_states"):
        rollout_states = _flatten_encoded_steps(state_steps)

    return (
        RolloutBatch(
            states=rollout_states,
            actions=torch.stack(action_steps).flatten(),
            old_log_probabilities=torch.stack(
                log_probability_steps
            ).flatten(),
            old_values=values.flatten(),
            advantages=advantages.flatten(),
            returns=returns.flatten(),
        ),
        episode_stats,
    )


def _collect_parallel_rollout(
    model: ClusterActorCritic,
    env_pool: ParallelEnvPool,
    config: PPOConfig,
    device: torch.device,
    timer: PhaseTimer | None = None,
) -> tuple[RolloutBatch, list[EpisodeStat]]:
    state_steps = []
    action_steps = []
    log_probability_steps = []
    value_steps = []
    reward_steps = []
    done_steps = []
    episode_stats = []

    model.eval()
    for _ in range(config.rollout_steps):
        encoded = env_pool.encoded
        with _measure(timer, "rollout.collate_h2d"):
            states = collate_encoded_observations_fast(
                encoded,
                device=device,
            )
        with _measure(timer, "rollout.policy_gpu"):
            with torch.inference_mode():
                output = model(states)
                distribution = Categorical(logits=output.logits)
                model_actions = distribution.sample()

        with _measure(timer, "rollout.action_d2h"):
            env_actions = states.to_env_actions(model_actions).cpu().tolist()
        with _measure(timer, "rollout.env_wait"):
            normalized_rewards, dones, completed = env_pool.step(
                env_actions,
                timer,
            )
        episode_stats.extend(completed)

        with _measure(timer, "rollout.buffer"):
            state_steps.append(tuple(encoded))
            action_steps.append(model_actions)
            log_probability_steps.append(
                distribution.log_prob(model_actions)
            )
            value_steps.append(output.value)
            reward_steps.append(
                torch.tensor(
                    normalized_rewards,
                    dtype=torch.float32,
                    device=device,
                )
            )
            done_steps.append(
                torch.tensor(dones, dtype=torch.bool, device=device)
            )

    with _measure(timer, "rollout.bootstrap"):
        with torch.inference_mode():
            last_states = collate_encoded_observations_fast(
                env_pool.encoded,
                device=device,
            )
            last_values = model(last_states).value

    with _measure(timer, "rollout.gae"):
        rewards = torch.stack(reward_steps)
        dones = torch.stack(done_steps)
        values = torch.stack(value_steps)
        advantages = _advantages(
            rewards,
            dones,
            values,
            last_values,
            config.gamma,
            config.gae_lambda,
        )
        returns = advantages + values

    with _measure(timer, "rollout.flatten_states"):
        rollout_states = _flatten_encoded_steps(state_steps)

    return (
        RolloutBatch(
            states=rollout_states,
            actions=torch.stack(action_steps).flatten(),
            old_log_probabilities=torch.stack(
                log_probability_steps
            ).flatten(),
            old_values=values.flatten(),
            advantages=advantages.flatten(),
            returns=returns.flatten(),
        ),
        episode_stats,
    )


def _ppo_update(
    model: ClusterActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_batch: RolloutBatch,
    config: PPOConfig,
    timer: PhaseTimer | None = None,
) -> dict[str, float]:
    with _measure(timer, "ppo.prepare"):
        advantages = rollout_batch.advantages
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)
        sample_count = advantages.shape[0]
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "choice_fraction": 0.0,
        "grad_norm": 0.0,
    }
    minibatch_count = 0

    model.train()
    for _ in range(config.epochs):
        indexes = torch.randperm(sample_count)
        for start in range(0, sample_count, config.minibatch_size):
            cpu_indexes = indexes[
                start : start + config.minibatch_size
            ]
            with _measure(timer, "ppo.minibatch_rebatch"):
                minibatch_states = [
                    rollout_batch.states[index]
                    for index in cpu_indexes.tolist()
                ]
                states = collate_encoded_observations_fast(
                    minibatch_states,
                    device=rollout_batch.actions.device,
                )
                minibatch_indexes = cpu_indexes.to(
                    rollout_batch.actions.device
                )
            with _measure(timer, "ppo.forward_gpu"):
                output = model(states)
                distribution = Categorical(logits=output.logits)
                new_log_probabilities = distribution.log_prob(
                    rollout_batch.actions[minibatch_indexes]
                )
                log_ratio = (
                    new_log_probabilities
                    - rollout_batch.old_log_probabilities[minibatch_indexes]
                )
                ratio = log_ratio.exp()
                has_choice = states.action_mask.sum(dim=1) > 1
                choice_fraction = has_choice.float().mean()

            with _measure(timer, "ppo.loss_gpu"):
                if has_choice.any():
                    minibatch_advantages = advantages[
                        minibatch_indexes
                    ][has_choice]
                    choice_ratio = ratio[has_choice]
                    unclipped_policy_loss = (
                        -minibatch_advantages * choice_ratio
                    )
                    clipped_policy_loss = (
                        -minibatch_advantages
                        * choice_ratio.clamp(
                            1.0 - config.clip_coefficient,
                            1.0 + config.clip_coefficient,
                        )
                    )
                    policy_loss = torch.maximum(
                        unclipped_policy_loss,
                        clipped_policy_loss,
                    ).mean()
                    entropy = distribution.entropy()[has_choice].mean()
                else:
                    policy_loss = output.value.sum() * 0.0
                    entropy = output.value.sum() * 0.0

                old_values = rollout_batch.old_values[minibatch_indexes]
                returns = rollout_batch.returns[minibatch_indexes]
                clipped_values = old_values + (
                    output.value - old_values
                ).clamp(
                    -config.clip_coefficient,
                    config.clip_coefficient,
                )
                value_loss = 0.5 * torch.maximum(
                    (output.value - returns).square(),
                    (clipped_values - returns).square(),
                ).mean()
                loss = (
                    policy_loss
                    + config.value_coefficient * value_loss
                    - config.entropy_coefficient * entropy
                )

            with _measure(timer, "ppo.backward_optimizer_gpu"):
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_grad_norm,
                )
                optimizer.step()

            with _measure(timer, "ppo.metrics_gpu"):
                with torch.no_grad():
                    if has_choice.any():
                        choice_log_ratio = log_ratio[has_choice]
                        choice_ratio = ratio[has_choice]
                        approx_kl = (
                            (choice_ratio - 1.0) - choice_log_ratio
                        ).mean()
                        clip_fraction = (
                            (choice_ratio - 1.0).abs()
                            > config.clip_coefficient
                        ).float().mean()
                    else:
                        approx_kl = torch.zeros((), device=ratio.device)
                        clip_fraction = torch.zeros((), device=ratio.device)
                metrics = {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approx_kl,
                    "clip_fraction": clip_fraction,
                    "choice_fraction": choice_fraction,
                    "grad_norm": grad_norm,
                }
                for name, value in metrics.items():
                    totals[name] += value.detach().item()
            minibatch_count += 1

    return {
        name: value / minibatch_count
        for name, value in totals.items()
    }


def _serialize_config(config: PPOConfig) -> dict[str, object]:
    serialized = asdict(config)
    serialized["scenario_paths"] = [
        str(path) for path in config.scenario_paths
    ]
    for name in (
        "run_dir",
        "checkpoint",
        "resume",
        "validation_manifest",
        "test_manifest",
    ):
        value = getattr(config, name)
        serialized[name] = None if value is None else str(value)
    return serialized


def _save_checkpoint(
    model: ClusterActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOConfig,
    references: list[float],
    episode_indexes: list[int],
    global_step: int,
    update: int,
) -> None:
    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config.checkpoint.with_suffix(
        config.checkpoint.suffix + ".tmp"
    )
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": _serialize_config(config),
            "reference_makespans": references,
            "episode_indexes": episode_indexes,
            "global_step": global_step,
            "update": update,
        },
        temporary_path,
    )
    temporary_path.replace(config.checkpoint)


def _append_csv(
    path: Path,
    fieldnames: Sequence[str],
    row: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _ensure_csv(path: Path, fieldnames: Sequence[str]) -> None:
    if path.exists() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fieldnames).writeheader()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _rolling_mean(values: list[float], window: int = 10) -> np.ndarray:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(sum(values[start : index + 1]) / (index - start + 1))
    return np.asarray(result)


def _plot_training_curves(run_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    update_rows = _read_csv(run_dir / "updates.csv")
    episode_rows = _read_csv(run_dir / "episodes.csv")
    if not update_rows:
        raise RuntimeError("cannot plot training curves without update data")

    steps = np.asarray(
        [int(row["global_step"]) for row in update_rows]
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = plt.get_cmap("tab10")

    grouped_episodes: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in episode_rows:
        grouped_episodes[row["scenario"]].append(row)
    for index, (scenario, rows) in enumerate(
        sorted(grouped_episodes.items())
    ):
        color = colors(index)
        episode_steps = [
            int(row["global_step"]) for row in rows
        ]
        makespans = [float(row["makespan"]) for row in rows]
        normalized_returns = [
            float(row["normalized_return"]) for row in rows
        ]
        axes[0, 0].plot(
            episode_steps,
            _rolling_mean(makespans),
            label=scenario,
            color=color,
            marker=".",
        )
        axes[0, 1].plot(
            episode_steps,
            _rolling_mean(normalized_returns),
            label=scenario,
            color=color,
            marker=".",
        )

    axes[0, 0].set_title("Episode makespan (rolling mean)")
    axes[0, 0].set_ylabel("Makespan")
    axes[0, 1].set_title("Normalized return (rolling mean)")
    axes[0, 1].axhline(0.0, color="0.5", linewidth=1)
    axes[0, 1].set_ylabel("1 - makespan / reference")
    if grouped_episodes:
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(
        steps,
        [float(row["value_loss"]) for row in update_rows],
        label="value loss",
        marker=".",
    )
    axes[1, 0].plot(
        steps,
        [abs(float(row["policy_loss"])) for row in update_rows],
        label="|policy loss|",
        marker=".",
    )
    axes[1, 0].set_title("PPO losses")
    axes[1, 0].set_yscale("symlog", linthresh=1e-5)
    axes[1, 0].legend(fontsize=8)

    entropy_line = axes[1, 1].plot(
        steps,
        [float(row["entropy"]) for row in update_rows],
        label="entropy",
        marker=".",
    )[0]
    kl_axis = axes[1, 1].twinx()
    kl_line = kl_axis.plot(
        steps,
        [
            max(float(row["approx_kl"]), 1e-12)
            for row in update_rows
        ],
        label="approx KL",
        marker=".",
        color="tab:orange",
    )[0]
    axes[1, 1].set_title("Policy diagnostics")
    axes[1, 1].set_ylabel("Entropy")
    kl_axis.set_ylabel("Approx KL")
    kl_axis.set_yscale("log")
    axes[1, 1].legend(
        [entropy_line, kl_line],
        ["entropy", "approx KL"],
        fontsize=8,
    )

    for axis in axes.flat:
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    curve_path = run_dir / "training_curves.png"
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)
    return curve_path


def _episode_rows(
    stats: list[EpisodeStat],
    update: int,
    global_step: int,
) -> list[dict[str, object]]:
    return [
        {
            "update": update,
            "global_step": global_step,
            "scenario": stat.scenario,
            "makespan": stat.makespan,
            "reference_makespan": stat.reference_makespan,
            "normalized_cost": stat.makespan / stat.reference_makespan,
            "normalized_return": stat.normalized_return,
            "reward": stat.reward,
            "success": stat.success,
        }
        for stat in stats
    ]


def _update_row(
    metrics: dict[str, float],
    stats: list[EpisodeStat],
    update: int,
    global_step: int,
) -> dict[str, object]:
    completed = len(stats)
    return {
        "update": update,
        "global_step": global_step,
        "completed_episodes": completed,
        "success_rate": (
            sum(stat.success for stat in stats) / completed
            if completed
            else ""
        ),
        "mean_makespan": (
            sum(stat.makespan for stat in stats) / completed
            if completed
            else ""
        ),
        "mean_normalized_return": (
            sum(stat.normalized_return for stat in stats) / completed
            if completed
            else ""
        ),
        **metrics,
    }


def _print_update(
    row: dict[str, object],
    stats: list[EpisodeStat],
    last_update: int,
    *,
    rollout_seconds: float,
    ppo_seconds: float,
) -> None:
    success = row["success_rate"]
    success_text = (
        f"{100 * float(success):5.1f}%" if success != "" else "  n/a "
    )
    scenario_values: dict[str, list[float]] = defaultdict(list)
    for stat in stats:
        scenario_values[stat.scenario].append(stat.makespan)
    scenario_text = ", ".join(
        f"{name}={sum(values) / len(values):.1f}"
        for name, values in sorted(scenario_values.items())
    )
    if not scenario_text:
        scenario_text = "no completed episode"
    normalized_return = row["mean_normalized_return"]
    normalized_text = (
        f"{float(normalized_return):+.4f}"
        if normalized_return != ""
        else "n/a"
    )

    print(
        f"[Update {int(row['update']):4d}/{last_update}] "
        f"steps={int(row['global_step']):7d}  "
        f"episodes={int(row['completed_episodes']):2d}  "
        f"success={success_text}"
    )
    print(
        f"  schedule: {scenario_text}  |  "
        f"normalized_return={normalized_text}"
    )
    print(
        f"  PPO: policy={float(row['policy_loss']):+.5f}  "
        f"value={float(row['value_loss']):.5f}  "
        f"entropy={float(row['entropy']):.4f}  "
        f"KL={float(row['approx_kl']):.2e}  "
        f"choice={100 * float(row['choice_fraction']):.1f}%"
    )
    print(
        f"  timing: rollout={rollout_seconds:.2f}s  "
        f"PPO={ppo_seconds:.2f}s  total={rollout_seconds + ppo_seconds:.2f}s"
    )


def _print_timing_profile(timer: PhaseTimer) -> None:
    labels = (
        ("rollout.encode_cpu", "rollout graph encode (serial)"),
        ("rollout.collate_h2d", "rollout collate + CPU->GPU"),
        ("rollout.policy_gpu", "rollout policy inference"),
        ("rollout.action_d2h", "rollout action GPU->CPU"),
        ("rollout.env_wait", "rollout environment/IPC wait"),
        ("rollout.worker_env_step_cpu", "worker env.step CPU-sum"),
        (
            "rollout.worker_reset_reference_cpu",
            "worker generation/reference CPU-sum",
        ),
        ("rollout.worker_encode_cpu", "worker graph encode CPU-sum"),
        ("rollout.buffer", "rollout buffer writes"),
        ("rollout.bootstrap", "rollout bootstrap value"),
        ("rollout.gae", "rollout GAE"),
        ("rollout.flatten_states", "rollout flatten CPU graph list"),
        ("ppo.prepare", "PPO advantage preparation"),
        ("ppo.minibatch_rebatch", "PPO minibatch fast rebatch + H2D"),
        ("ppo.forward_gpu", "PPO model forward"),
        ("ppo.loss_gpu", "PPO loss construction"),
        ("ppo.backward_optimizer_gpu", "PPO backward + optimizer"),
        ("ppo.metrics_gpu", "PPO metrics + scalar sync"),
    )
    print("  timing profile (CUDA-synchronized diagnostic):")
    for name, label in labels:
        if name not in timer.totals:
            continue
        seconds = timer.totals[name]
        count = timer.counts[name]
        average_ms = 1000.0 * seconds / count
        print(
            f"    {label:<39} {seconds:>9.3f}s  "
            f"calls={count:<5d} avg={average_ms:>8.2f}ms"
        )


def _evaluation(
    model: ClusterActorCritic,
    envs: list[ClusterEnv],
    references: list[float],
    splits: list[str] | None = None,
) -> list[dict[str, object]]:
    model.eval()
    results = []
    evaluation_splits = splits or ["scenario"] * len(envs)
    for split, env, reference in zip(evaluation_splits, envs, references):
        scenario = str(env.problem.meta.get("name", "scenario"))
        result = rollout(
            env,
            scenario,
            "trained_network_greedy",
            network_greedy_selector(env, model, reference),
        )
        results.append(
            {
                "split": split,
                "scenario": scenario,
                "reference_makespan": reference,
                "makespan": result.makespan,
                "normalized_cost": result.makespan / reference,
                "relative_gain": 1.0 - result.makespan / reference,
                "action_count": result.action_count,
                "valid": result.valid,
            }
        )
    return results


def _print_evaluation(results: list[dict[str, object]]) -> None:
    if not results:
        return
    print("\nFinal greedy evaluation")
    print("-" * 78)
    print(
        f"{'Split':<12} {'Scenario':<20} {'Reference':>10} {'Model':>10} "
        f"{'Gain':>10} {'Valid':>8}"
    )
    print("-" * 78)
    for result in results:
        print(
            f"{str(result['split']):<12} "
            f"{str(result['scenario']):<20} "
            f"{float(result['reference_makespan']):>10.1f} "
            f"{float(result['makespan']):>10.1f} "
            f"{100 * float(result['relative_gain']):>9.2f}% "
            f"{str(result['valid']):>8}"
        )
    print("-" * 78)


def _prepare_run_dir(config: PPOConfig) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if config.resume is None:
        for filename in (
            "updates.csv",
            "episodes.csv",
            "evaluation.csv",
            "training_curves.png",
            "train.log",
            "config.json",
        ):
            path = config.run_dir / filename
            if path.exists():
                path.unlink()


def train(config: PPOConfig) -> dict[str, object]:
    _prepare_run_dir(config)
    config.run_dir.joinpath("config.json").write_text(
        json.dumps(_serialize_config(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _ensure_csv(config.run_dir / "updates.csv", UPDATE_FIELDS)
    _ensure_csv(config.run_dir / "episodes.csv", EPISODE_FIELDS)
    _ensure_csv(config.run_dir / "evaluation.csv", EVALUATION_FIELDS)
    with _training_log(
        config.run_dir / "train.log",
        append=config.resume is not None,
    ):
        return _train(config)


def _train(config: PPOConfig) -> dict[str, object]:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)

    env_factory = (
        GeneratorEnvFactory(config)
        if config.train_mode == "generator"
        else None
    )
    if env_factory is None:
        problems = [load_problem(path) for path in config.scenario_paths]
        slots = _fixed_env_slots(
            problems,
            _reference_makespans(problems),
            config.seed,
        )
    else:
        slots = [
            env_factory.make(index)
            for index in range(config.num_envs)
        ]

    model_config = TransformerConfig(
        model_dim=config.model_dim,
        num_heads=config.num_heads,
        hgt_layers=config.hgt_layers,
        num_layers=config.num_layers,
        feedforward_dim=config.feedforward_dim,
        dropout=0.0,
    )
    model = ClusterActorCritic(model_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        eps=1e-5,
    )
    global_step = 0
    first_update = 1

    if config.resume is not None:
        checkpoint = torch.load(
            config.resume,
            map_location=device,
            weights_only=True,
        )
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            raise ValueError(
                "checkpoint uses an older network/reward format; "
                "start a new training run"
            )
        saved_references = checkpoint.get("reference_makespans")
        if env_factory is not None:
            saved_episode_indexes = checkpoint.get("episode_indexes")
            if (
                isinstance(saved_episode_indexes, list)
                and len(saved_episode_indexes) == len(slots)
            ):
                slots = [
                    env_factory.make(index, int(episode_index) + 1)
                    for index, episode_index in enumerate(saved_episode_indexes)
                ]
        references = [slot.reference_makespan for slot in slots]
        if config.train_mode == "scenarios" and saved_references != references:
            raise ValueError(
                "checkpoint reference makespans do not match the scenarios"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["global_step"])
        first_update = int(checkpoint["update"]) + 1

    steps_per_update = config.rollout_steps * len(slots)
    remaining_steps = max(0, config.total_steps - global_step)
    update_count = math.ceil(remaining_steps / steps_per_update)
    last_update = first_update + update_count - 1

    print("=" * 78)
    print("Masked PPO training")
    print(
        f"device={device}  mode={config.train_mode}  envs={len(slots)}  "
        f"cpu_workers={config.cpu_workers}  "
        f"target_steps={config.total_steps}  run_dir={config.run_dir}"
    )
    print("Reference makespans:")
    for slot in slots:
        print(
            f"  - {slot.env.problem.meta.get('name', 'scenario')}: "
            f"{slot.reference_makespan:.1f}"
        )
    print("=" * 78)

    env_pool = (
        ParallelEnvPool(slots, config)
        if config.cpu_workers
        else None
    )

    try:
        for update in range(first_update, last_update + 1):
            timer = PhaseTimer(device) if config.profile_timing else None
            rollout_started = time.perf_counter()
            if env_pool is None:
                rollout_batch, episode_stats = _collect_rollout(
                    model,
                    slots,
                    config,
                    device,
                    env_factory,
                    timer,
                )
            else:
                rollout_batch, episode_stats = _collect_parallel_rollout(
                    model,
                    env_pool,
                    config,
                    device,
                    timer,
                )
            rollout_seconds = time.perf_counter() - rollout_started
            ppo_started = time.perf_counter()
            metrics = _ppo_update(
                model,
                optimizer,
                rollout_batch,
                config,
                timer,
            )
            ppo_seconds = time.perf_counter() - ppo_started
            global_step += steps_per_update
            update_row = _update_row(
                metrics,
                episode_stats,
                update,
                global_step,
            )
            _append_csv(
                config.run_dir / "updates.csv",
                UPDATE_FIELDS,
                update_row,
            )
            for episode_row in _episode_rows(
                episode_stats,
                update,
                global_step,
            ):
                _append_csv(
                    config.run_dir / "episodes.csv",
                    EPISODE_FIELDS,
                    episode_row,
                )

            if update % config.log_interval == 0 or update == last_update:
                _print_update(
                    update_row,
                    episode_stats,
                    last_update,
                    rollout_seconds=rollout_seconds,
                    ppo_seconds=ppo_seconds,
                )
                if timer is not None:
                    _print_timing_profile(timer)

            if (
                update % config.checkpoint_interval == 0
                or update == last_update
            ):
                slot_states = (
                    [_env_slot_state(slot) for slot in slots]
                    if env_pool is None
                    else env_pool.states
                )
                _save_checkpoint(
                    model,
                    optimizer,
                    config,
                    [state.reference_makespan for state in slot_states],
                    [state.episode_index for state in slot_states],
                    global_step,
                    update,
                )
                _plot_training_curves(config.run_dir)

        if update_count == 0:
            slot_states = (
                [_env_slot_state(slot) for slot in slots]
                if env_pool is None
                else env_pool.states
            )
            _save_checkpoint(
                model,
                optimizer,
                config,
                [state.reference_makespan for state in slot_states],
                [state.episode_index for state in slot_states],
                global_step,
                first_update - 1,
            )
            if _read_csv(config.run_dir / "updates.csv"):
                _plot_training_curves(config.run_dir)
    finally:
        if env_pool is not None:
            env_pool.close()

    final_slot_states = (
        [_env_slot_state(slot) for slot in slots]
        if env_pool is None
        else env_pool.states
    )

    if config.evaluate:
        evaluation_cases = _evaluation_problems(config)
        evaluation_problems = [problem for _, problem in evaluation_cases]
        evaluation_references = _reference_makespans(evaluation_problems)
        evaluation = _evaluation(
            model,
            [ClusterEnv(problem) for problem in evaluation_problems],
            evaluation_references,
            [split for split, _ in evaluation_cases],
        )
    else:
        evaluation = []
    for result in evaluation:
        _append_csv(
            config.run_dir / "evaluation.csv",
            EVALUATION_FIELDS,
            result,
        )
    _print_evaluation(evaluation)

    curve_path = config.run_dir / "training_curves.png"
    print("\nSaved outputs")
    print(f"  checkpoint : {config.checkpoint}")
    print(f"  updates    : {config.run_dir / 'updates.csv'}")
    print(f"  episodes   : {config.run_dir / 'episodes.csv'}")
    print(f"  curves     : {curve_path}")
    print(f"  console log: {config.run_dir / 'train.log'}")

    return {
        "checkpoint": str(config.checkpoint),
        "run_dir": str(config.run_dir),
        "curves": str(curve_path),
        "device": str(device),
        "global_step": global_step,
        "updates": update_count,
        "reference_makespans": [
            state.reference_makespan for state in final_slot_states
        ],
        "evaluation": evaluation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the entity-token Transformer with masked PPO."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        type=Path,
        default=_default_scenarios(),
    )
    parser.add_argument(
        "--train-mode",
        choices=("scenarios", "generator"),
        default="scenarios",
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=0,
        help="persistent CPU environment workers; 0 keeps serial rollout",
    )
    parser.add_argument("--generator-seed", type=int, default=42)
    parser.add_argument("--generator-max-attempts", type=int, default=64)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coefficient", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--hgt-layers", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps", "auto"),
        default="cpu",
    )
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="synchronize CUDA and print a detailed per-update timing profile",
    )
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--no-eval", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_dir is not None:
        run_dir = args.run_dir
    elif args.resume is not None:
        run_dir = args.resume.parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / f"ppo_cluster_{timestamp}"
    checkpoint = args.checkpoint or run_dir / "checkpoint.pt"

    config = PPOConfig(
        scenario_paths=tuple(args.scenarios),
        train_mode=args.train_mode,
        num_envs=args.num_envs,
        cpu_workers=args.cpu_workers,
        generator_seed=args.generator_seed,
        generator_max_attempts=args.generator_max_attempts,
        validation_manifest=args.validation_manifest,
        test_manifest=args.test_manifest,
        run_dir=run_dir,
        checkpoint=checkpoint,
        resume=args.resume,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coefficient=args.clip_coefficient,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        hgt_layers=args.hgt_layers,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        seed=args.seed,
        device=args.device,
        profile_timing=args.profile_timing,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        evaluate=not args.no_eval,
    )
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
