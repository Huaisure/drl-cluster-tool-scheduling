from __future__ import annotations

import hashlib
import itertools
import json
import random
import statistics
import warnings
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

from cluster_toolkit.problem import ClusterProblem, ModuleType, parse_problem

from .heuristic import HeuristicResult, build_safe_reference_schedule
from .models import (
    GENERATOR_NAME,
    GENERATOR_VERSION,
    GenerationAudit,
    ProblemGenerationConfig,
)
from .validation import validate_generated_instance


Difficulty = Literal["easy", "medium", "hard", "edge"]
Split = Literal["train", "validation", "test"]
TopologyFamily = Literal["simple", "single_vacuum", "dual_vacuum"]

_DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard", "edge")
_DIFFICULTY_ALIASES = {
    "level0": "easy",
    "level1": "medium",
    "level2": "hard",
    "level3": "edge",
}
_DEFAULT_CURRICULUM_WEIGHTS: dict[str, float] = {
    "easy": 0.30,
    "medium": 0.40,
    "hard": 0.20,
    "edge": 0.10,
}
_SPLIT_BUCKETS: dict[Split, range] = {
    "train": range(0, 70),
    "validation": range(70, 85),
    "test": range(85, 100),
}


@dataclass(frozen=True, slots=True)
class GeneratedBenchmark:
    problem: ClusterProblem
    raw_problem: dict[str, Any]
    actions: tuple[dict[str, object], ...]
    metadata: dict[str, Any]
    audit: GenerationAudit


@dataclass(frozen=True, slots=True)
class _ProcessUnit:
    unit_index: int
    pm_ids: tuple[str, ...]
    pm_processes: dict[str, tuple[str, ...]]
    process_pms: dict[str, tuple[str, ...]]
    process_times: dict[str, int]


class _RetryGeneration(RuntimeError):
    pass


class ProblemGenerator:
    """Generate domain-shaped semiconductor Cluster Tool RL problems.

    The public API intentionally matches the previous procedural generator,
    while all topology, capability, Recipe, timing, and wafer logic is new.
    """

    def __init__(
        self,
        *,
        config: ProblemGenerationConfig | None = None,
        max_split_attempts: int | None = None,
    ) -> None:
        selected = config or ProblemGenerationConfig()
        if max_split_attempts is not None:
            if (
                not isinstance(max_split_attempts, int)
                or isinstance(max_split_attempts, bool)
                or max_split_attempts <= 0
            ):
                raise ValueError("max_split_attempts must be positive")
            selected = selected.model_copy(update={"max_attempts": max_split_attempts})
        self.config = selected
        self.max_split_attempts = selected.max_attempts

    def sample(
        self,
        seed: int,
        difficulty: str = "medium",
        *,
        split: Split = "train",
    ) -> ClusterProblem:
        return self.generate(seed, difficulty=difficulty, split=split).problem

    def sample_curriculum(
        self,
        seed: int,
        *,
        split: Split = "train",
        weights: dict[str, float] | None = None,
    ) -> ClusterProblem:
        return self.generate_curriculum(seed, split=split, weights=weights).problem

    def generate_curriculum(
        self,
        seed: int,
        *,
        split: Split = "train",
        weights: dict[str, float] | None = None,
    ) -> GeneratedBenchmark:
        difficulty = self.select_curriculum_difficulty(seed, split=split, weights=weights)
        return self.generate(seed, difficulty=difficulty, split=split)

    def select_curriculum_difficulty(
        self,
        seed: int,
        *,
        split: Split = "train",
        weights: dict[str, float] | None = None,
    ) -> Difficulty:
        if split not in _SPLIT_BUCKETS:
            raise ValueError(f"unsupported split: {split!r}")
        selected_weights = weights or _DEFAULT_CURRICULUM_WEIGHTS
        if set(selected_weights) != set(_DIFFICULTIES):
            raise ValueError("curriculum weights must contain every difficulty exactly")
        if any(weight < 0 for weight in selected_weights.values()):
            raise ValueError("curriculum weights must be non-negative")
        if sum(selected_weights.values()) <= 0:
            raise ValueError("curriculum weights must contain a positive weight")
        selector = random.Random(self._attempt_seed(seed, "curriculum", split, 0))
        return _weighted_choice(selected_weights, selector)  # type: ignore[return-value]

    def generate(
        self,
        seed: int,
        difficulty: str = "medium",
        *,
        split: Split = "train",
    ) -> GeneratedBenchmark:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        normalized = _DIFFICULTY_ALIASES.get(difficulty, difficulty)
        if normalized not in _DIFFICULTIES:
            raise ValueError(f"unsupported difficulty: {difficulty!r}")
        if split not in _SPLIT_BUCKETS:
            raise ValueError(f"unsupported split: {split!r}")

        topology_selector = random.Random(
            self._attempt_seed(seed, f"topology:{normalized}", split, 0)
        )
        topology_family: TopologyFamily = _weighted_choice(
            self.config.topology_weights_by_split[split],
            topology_selector,
        )  # type: ignore[assignment]

        last_error: Exception | None = None
        for attempt in range(self.max_split_attempts):
            attempt_seed = self._attempt_seed(seed, normalized, split, attempt)
            rng = random.Random(attempt_seed)
            try:
                raw_problem, topology_family, process_step_counts = self._construct_problem(
                    seed=seed,
                    attempt_seed=attempt_seed,
                    difficulty=normalized,  # type: ignore[arg-type]
                    split=split,
                    topology_family=topology_family,
                    rng=rng,
                )
                structural_signature = _structural_signature(raw_problem)
                signature_bucket = int(structural_signature[:8], 16) % 100
                if signature_bucket not in _SPLIT_BUCKETS[split]:
                    continue

                audit = validate_generated_instance(raw_problem)
                problem = _parse_generated_problem(raw_problem)
                reference = build_safe_reference_schedule(problem)
                metadata = self._metadata(
                    problem,
                    reference,
                    audit,
                    seed=seed,
                    split=split,
                    difficulty=normalized,  # type: ignore[arg-type]
                    topology_family=topology_family,
                    structural_signature=structural_signature,
                    signature_bucket=signature_bucket,
                    process_step_counts=process_step_counts,
                )
                raw_problem["_meta"]["generator"].update(metadata)
                final_problem = _parse_generated_problem(raw_problem)
                return GeneratedBenchmark(
                    problem=final_problem,
                    raw_problem=raw_problem,
                    actions=reference.actions,
                    metadata=metadata,
                    audit=audit,
                )
            except (_RetryGeneration, RuntimeError, ValueError) as exc:
                last_error = exc
                continue
        detail = f": {last_error}" if last_error is not None else ""
        raise RuntimeError(
            f"could not generate a valid {difficulty} problem in split {split} "
            f"after {self.max_split_attempts} attempts{detail}"
        )

    def _construct_problem(
        self,
        *,
        seed: int,
        attempt_seed: int,
        difficulty: Difficulty,
        split: Split,
        topology_family: TopologyFamily,
        rng: random.Random,
    ) -> tuple[dict[str, Any], TopologyFamily, dict[str, int]]:
        wafer_count = rng.randint(*self._wafer_range(difficulty))
        recipe_count = min(
            int(_weighted_choice(self.config.recipe_count_weights, rng)),
            wafer_count,
        )
        al_time = rng.randint(*self.config.al_time_range)

        modules: dict[str, dict[str, Any]] = {
            "IO1": {"type": "IO", "capacity": wafer_count}
        }
        robots: dict[str, dict[str, Any]] = {}
        buffer_ids: tuple[str, ...] = ()
        ll_ids: tuple[str, ...] = ()
        process_units: dict[int, _ProcessUnit] = {}
        process_counter = 0

        if topology_family == "simple":
            unit, process_counter = self._build_process_unit(
                unit_index=1,
                process_counter=process_counter,
                simple=True,
                rng=rng,
            )
            process_units[1] = unit
            self._add_process_modules(modules, unit)
            robots["TM1"] = self._robot(
                ("IO1", *unit.pm_ids),
                arm_type=_weighted_choice(self.config.vtm_arm_weights, rng),
                rng=rng,
            )
        else:
            modules["AL1"] = {"type": "AL", "capacity": 1}
            ll_count = int(_weighted_choice(self.config.ll_count_weights, rng))
            ll_ids = tuple(f"LL{index}" for index in range(1, ll_count + 1))
            for ll_id in ll_ids:
                pump_time = self._ll_time(rng)
                vent_time = self._ll_time(rng)
                modules[ll_id] = {
                    "type": "LL",
                    "capacity": 1,
                    "load_lock": {
                        "initial_state": "atmosphere",
                        "atmosphere_to_vacuum_time": pump_time,
                        "vacuum_to_atmosphere_time": vent_time,
                        "tm_required_states": {
                            "ATM1": "atmosphere",
                            "VTM1": "vacuum",
                        },
                    },
                }
            unit1, process_counter = self._build_process_unit(
                unit_index=1,
                process_counter=process_counter,
                simple=False,
                rng=rng,
            )
            process_units[1] = unit1
            self._add_process_modules(modules, unit1)
            if topology_family == "dual_vacuum":
                buffer_count = int(_weighted_choice(self.config.buffer_count_weights, rng))
                buffer_ids = tuple(
                    f"BUFFER{index}" for index in range(1, buffer_count + 1)
                )
                for buffer_id in buffer_ids:
                    modules[buffer_id] = {"type": "BUFFER", "capacity": 1}
                unit2, process_counter = self._build_process_unit(
                    unit_index=2,
                    process_counter=process_counter,
                    simple=False,
                    rng=rng,
                )
                process_units[2] = unit2
                self._add_process_modules(modules, unit2)

            robots["ATM1"] = self._robot(
                ("IO1", "AL1", *ll_ids),
                arm_type=_weighted_choice(self.config.atm_arm_weights, rng),
                rng=rng,
            )
            robots["VTM1"] = self._robot(
                (*ll_ids, *unit1.pm_ids, *buffer_ids),
                arm_type=_weighted_choice(self.config.vtm_arm_weights, rng),
                rng=rng,
            )
            if topology_family == "dual_vacuum":
                robots["VTM2"] = self._robot(
                    (*buffer_ids, *process_units[2].pm_ids),
                    arm_type=_weighted_choice(self.config.vtm_arm_weights, rng),
                    rng=rng,
                )

        routes, process_step_counts = self._build_routes(
            recipe_count,
            topology_family=topology_family,
            process_units=process_units,
            ll_ids=ll_ids,
            buffer_ids=buffer_ids,
            al_time=al_time,
            rng=rng,
        )
        raw_problem: dict[str, Any] = {
            "schema_version": 2,
            "_meta": {
                "generator": {
                    "name": GENERATOR_NAME,
                    "version": GENERATOR_VERSION,
                    "mode": "ppo",
                    "instance_id": f"{split}_{difficulty}_{seed}",
                    "seed": seed,
                    "attempt_seed": attempt_seed,
                    "difficulty": difficulty,
                    "split": split,
                }
            },
            "Modules": modules,
            "ClusterTool": robots,
            "routes": routes,
            "initial_state": {
                "robots": {
                    robot_id: {"position_module_id": None}
                    for robot_id in robots
                },
                "wafers": [],
            },
        }
        self._prune_unused_pms(raw_problem, topology_family)
        counts = self._distribute_wafers(wafer_count, recipe_count, rng)
        priorities = self._priorities(counts, rng)
        raw_problem["initial_state"]["wafers"] = [
            {
                "route_id": route_id,
                "wafer_index": str(wafer_index),
                "priority": priorities[(route_id, wafer_index)],
                "step_index": 0,
                "location": {"kind": "module", "module_id": "IO1"},
                "process_end_time": None,
            }
            for route_id, count in zip(sorted(routes), counts, strict=True)
            for wafer_index in range(count)
        ]
        return raw_problem, topology_family, process_step_counts

    def _build_process_unit(
        self,
        *,
        unit_index: int,
        process_counter: int,
        simple: bool,
        rng: random.Random,
    ) -> tuple[_ProcessUnit, int]:
        pm_count = int(_weighted_choice(self.config.pm_count_weights, rng))
        if simple:
            pm_ids = tuple(f"PM{index}" for index in range(1, pm_count + 1))
        else:
            pm_ids = tuple(
                f"PM{unit_index}_{index}" for index in range(1, pm_count + 1)
            )
        target_degrees = {
            pm_id: int(_weighted_choice(self.config.pm_capability_count_weights, rng))
            for pm_id in pm_ids
        }
        remaining = dict(target_degrees)
        pm_processes: dict[str, list[str]] = {pm_id: [] for pm_id in pm_ids}
        process_pms: dict[str, tuple[str, ...]] = {}
        process_times: dict[str, int] = {}

        while any(slots > 0 for slots in remaining.values()):
            available = [pm_id for pm_id, slots in remaining.items() if slots > 0]
            support_weights = {
                count: weight
                for count, weight in self.config.process_support_count_weights.items()
                if count <= len(available)
            }
            support_count = int(_weighted_choice(support_weights, rng))
            selected: list[str] = []
            pool = list(available)
            for _ in range(support_count):
                weights = [remaining[pm_id] for pm_id in pool]
                pm_id = rng.choices(pool, weights=weights, k=1)[0]
                selected.append(pm_id)
                pool.remove(pm_id)

            process_id = f"PROC_{process_counter:03d}"
            process_counter += 1
            selected_tuple = tuple(sorted(selected))
            process_pms[process_id] = selected_tuple
            process_times[process_id] = self._pm_process_time(rng)
            for pm_id in selected_tuple:
                pm_processes[pm_id].append(process_id)
                remaining[pm_id] -= 1

        return (
            _ProcessUnit(
                unit_index=unit_index,
                pm_ids=pm_ids,
                pm_processes={
                    pm_id: tuple(processes)
                    for pm_id, processes in pm_processes.items()
                },
                process_pms=process_pms,
                process_times=process_times,
            ),
            process_counter,
        )

    @staticmethod
    def _add_process_modules(
        modules: dict[str, dict[str, Any]],
        unit: _ProcessUnit,
    ) -> None:
        for pm_id in unit.pm_ids:
            modules[pm_id] = {
                "type": "PM",
                "capacity": 1,
                "process_ids": list(unit.pm_processes[pm_id]),
            }

    def _build_routes(
        self,
        recipe_count: int,
        *,
        topology_family: TopologyFamily,
        process_units: dict[int, _ProcessUnit],
        ll_ids: tuple[str, ...],
        buffer_ids: tuple[str, ...],
        al_time: int,
        rng: random.Random,
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
        routes: dict[str, list[dict[str, Any]]] = {}
        process_step_counts: dict[str, int] = {}
        signatures: set[tuple[object, ...]] = set()
        has_cross_recipe = False

        for recipe_index in range(recipe_count):
            route_id = f"R{recipe_index:02d}"
            force_cross = (
                topology_family == "dual_vacuum"
                and recipe_index == recipe_count - 1
                and not has_cross_recipe
            )
            for _ in range(300):
                built = self._build_one_route(
                    topology_family=topology_family,
                    process_units=process_units,
                    ll_ids=ll_ids,
                    buffer_ids=buffer_ids,
                    al_time=al_time,
                    force_cross=force_cross,
                    rng=rng,
                )
                visits, unit_pattern, cooling_positions = built
                signature = tuple(
                    (
                        tuple(visit["module_ids"]),
                        visit.get("process_id"),
                        int(visit.get("process_time", 0) > 0)
                        if visit.get("process_id") is None
                        else visit["process_time"],
                    )
                    for visit in visits
                )
                if signature in signatures:
                    continue
                signatures.add(signature)
                routes[route_id] = visits
                process_step_counts[route_id] = len(unit_pattern)
                has_cross_recipe = has_cross_recipe or 2 in unit_pattern
                break
            else:
                raise _RetryGeneration(f"could not construct unique Recipe {route_id}")

        if topology_family == "dual_vacuum" and not has_cross_recipe:
            raise _RetryGeneration("dual topology has no VTM2 Recipe")
        return routes, process_step_counts

    def _build_one_route(
        self,
        *,
        topology_family: TopologyFamily,
        process_units: dict[int, _ProcessUnit],
        ll_ids: tuple[str, ...],
        buffer_ids: tuple[str, ...],
        al_time: int,
        force_cross: bool,
        rng: random.Random,
    ) -> tuple[list[dict[str, Any]], tuple[int, ...], frozenset[int]]:
        reentry_mode = str(_weighted_choice(self.config.reentry_weights, rng))
        repeated_count = 1
        minimum_length = 1
        if reentry_mode == "twice":
            repeated_count = 2
            minimum_length = 3
        elif reentry_mode == "deep":
            repeated_count = int(
                _weighted_choice(self.config.deep_reentry_count_weights, rng)
            )
            minimum_length = 2 * repeated_count - 1
        length_weights = {
            length: weight
            for length, weight in self.config.recipe_length_weights.items()
            if length >= minimum_length
        }
        process_step_count = int(_weighted_choice(length_weights, rng))
        unit_pattern = self._unit_pattern(
            topology_family,
            process_step_count,
            force_cross=force_cross,
            rng=rng,
        )

        process_ids = self._assign_processes(
            unit_pattern,
            process_units,
            repeated_count=repeated_count,
            rng=rng,
        )
        cooling_positions: frozenset[int] = frozenset()
        if (
            topology_family == "dual_vacuum"
            and rng.random() < self.config.cooling_probability
        ):
            requested = int(_weighted_choice(self.config.cooling_count_weights, rng))
            count = min(requested, process_step_count)
            cooling_positions = frozenset(rng.sample(range(process_step_count), count))

        visits: list[dict[str, Any]] = []
        if topology_family != "simple":
            visits.append(
                {"module_ids": ["AL1"], "process_time": al_time}
            )
            visits.append(
                {"module_ids": list(ll_ids), "process_time": 0}
            )

        current_unit = 1
        cooling_used: set[int] = set()
        for index, (unit_index, process_id) in enumerate(
            zip(unit_pattern, process_ids, strict=True)
        ):
            if unit_index != current_unit:
                cooling_time = 0
                previous_index = index - 1
                if previous_index in cooling_positions:
                    cooling_time = rng.randint(*self.config.cooling_time_range)
                    cooling_used.add(previous_index)
                visits.append(
                    {"module_ids": list(buffer_ids), "process_time": cooling_time}
                )
                current_unit = unit_index

            unit = process_units[unit_index]
            visits.append(
                {
                    "module_ids": list(unit.process_pms[process_id]),
                    "process_id": process_id,
                    "process_time": unit.process_times[process_id],
                }
            )
            if index not in cooling_positions:
                continue
            next_unit = unit_pattern[index + 1] if index + 1 < len(unit_pattern) else 1
            if next_unit != current_unit:
                continue
            visits.append(
                {
                    "module_ids": list(buffer_ids),
                    "process_time": rng.randint(*self.config.cooling_time_range),
                }
            )
            cooling_used.add(index)

        if current_unit == 2:
            final_index = len(unit_pattern) - 1
            cooling_time = 0
            if final_index in cooling_positions and final_index not in cooling_used:
                cooling_time = rng.randint(*self.config.cooling_time_range)
                cooling_used.add(final_index)
            visits.append(
                {"module_ids": list(buffer_ids), "process_time": cooling_time}
            )
        if cooling_used != set(cooling_positions):
            raise _RetryGeneration("cooling placement lost a required visit")
        if topology_family != "simple":
            visits.append(
                {"module_ids": list(ll_ids), "process_time": 0}
            )
        return visits, unit_pattern, cooling_positions

    def _unit_pattern(
        self,
        topology_family: TopologyFamily,
        length: int,
        *,
        force_cross: bool,
        rng: random.Random,
    ) -> tuple[int, ...]:
        if topology_family != "dual_vacuum":
            return (1,) * length
        crosses = force_cross or rng.random() < self.config.cross_unit_probability
        if not crosses:
            return (1,) * length
        if length == 1 or rng.random() < self.config.vtm2_only_probability_given_cross:
            return (2,) * length

        layout = str(_weighted_choice(self.config.mixed_unit_layout_weights, rng))
        if layout == "both" and length >= 3:
            first_cut = rng.randint(1, length - 2)
            second_cut = rng.randint(first_cut + 1, length - 1)
            return (1,) * first_cut + (2,) * (second_cut - first_cut) + (1,) * (length - second_cut)
        if layout == "post":
            cut = rng.randint(1, length - 1)
            return (2,) * cut + (1,) * (length - cut)
        cut = rng.randint(1, length - 1)
        return (1,) * cut + (2,) * (length - cut)

    @staticmethod
    def _assign_processes(
        unit_pattern: tuple[int, ...],
        process_units: dict[int, _ProcessUnit],
        *,
        repeated_count: int,
        rng: random.Random,
    ) -> tuple[str, ...]:
        length = len(unit_pattern)
        repeat_positions: tuple[int, ...] = ()
        repeated_process: str | None = None
        if repeated_count > 1:
            options = [
                positions
                for positions in itertools.combinations(range(length), repeated_count)
                if len({unit_pattern[index] for index in positions}) == 1
                and all(right - left > 1 for left, right in zip(positions, positions[1:]))
            ]
            if not options:
                raise _RetryGeneration("sampled unit layout cannot place non-adjacent reentry")
            repeat_positions = rng.choice(options)
            repeat_unit = unit_pattern[repeat_positions[0]]
            repeated_process = rng.choice(sorted(process_units[repeat_unit].process_pms))

        result: list[str | None] = [None] * length
        for index in repeat_positions:
            result[index] = repeated_process
        for unit_index in sorted(set(unit_pattern)):
            indexes = [
                index
                for index, selected_unit in enumerate(unit_pattern)
                if selected_unit == unit_index and result[index] is None
            ]
            pool = [
                process_id
                for process_id in sorted(process_units[unit_index].process_pms)
                if process_id != repeated_process
            ]
            if len(pool) < len(indexes):
                raise _RetryGeneration("process capability pool is too small for unique Recipe steps")
            for index, process_id in zip(indexes, rng.sample(pool, len(indexes)), strict=True):
                result[index] = process_id
        if any(process_id is None for process_id in result):
            raise RuntimeError("process assignment is incomplete")
        return tuple(str(process_id) for process_id in result)

    def _prune_unused_pms(
        self,
        raw_problem: dict[str, Any],
        topology_family: TopologyFamily,
    ) -> None:
        modules = raw_problem["Modules"]
        used = {
            module_id
            for visits in raw_problem["routes"].values()
            for visit in visits
            for module_id in visit["module_ids"]
            if modules[module_id]["type"] == "PM"
        }
        for module_id in list(modules):
            if modules[module_id]["type"] == "PM" and module_id not in used:
                del modules[module_id]
        for robot in raw_problem["ClusterTool"].values():
            robot["module_ids"] = [
                module_id for module_id in robot["module_ids"] if module_id in modules
            ]

        unit_pm_counts: list[int]
        if topology_family == "simple":
            unit_pm_counts = [
                sum(module["type"] == "PM" for module in modules.values())
            ]
        else:
            unit_pm_counts = [
                sum(
                    module_id.startswith(f"PM{unit_index}_")
                    for module_id, module in modules.items()
                    if module["type"] == "PM"
                )
                for unit_index in range(1, 3 if topology_family == "dual_vacuum" else 2)
            ]
        if any(count < 3 or count > 6 for count in unit_pm_counts):
            raise _RetryGeneration(
                f"PM pruning left invalid per-unit counts: {unit_pm_counts}"
            )
        self._compact_pm_ids(raw_problem, topology_family)

    @staticmethod
    def _compact_pm_ids(
        raw_problem: dict[str, Any],
        topology_family: TopologyFamily,
    ) -> None:
        modules = raw_problem["Modules"]
        mapping: dict[str, str] = {}
        if topology_family == "simple":
            pm_ids = sorted(
                module_id
                for module_id, module in modules.items()
                if module["type"] == "PM"
            )
            mapping.update(
                {module_id: f"PM{index}" for index, module_id in enumerate(pm_ids, start=1)}
            )
        else:
            unit_count = 2 if topology_family == "dual_vacuum" else 1
            for unit_index in range(1, unit_count + 1):
                pm_ids = sorted(
                    module_id
                    for module_id, module in modules.items()
                    if module["type"] == "PM" and module_id.startswith(f"PM{unit_index}_")
                )
                mapping.update(
                    {
                        module_id: f"PM{unit_index}_{index}"
                        for index, module_id in enumerate(pm_ids, start=1)
                    }
                )
        if all(old == new for old, new in mapping.items()):
            return
        raw_problem["Modules"] = {
            mapping.get(module_id, module_id): module
            for module_id, module in modules.items()
        }
        for robot in raw_problem["ClusterTool"].values():
            robot["module_ids"] = [mapping.get(module_id, module_id) for module_id in robot["module_ids"]]
        for visits in raw_problem["routes"].values():
            for visit in visits:
                visit["module_ids"] = [mapping.get(module_id, module_id) for module_id in visit["module_ids"]]

    def _distribute_wafers(
        self,
        total: int,
        recipe_count: int,
        rng: random.Random,
    ) -> list[int]:
        for _ in range(200):
            weights = [
                rng.uniform(
                    1 - self.config.recipe_count_perturbation,
                    1 + self.config.recipe_count_perturbation,
                )
                for _ in range(recipe_count)
            ]
            remaining = total - recipe_count
            exact = [remaining * weight / sum(weights) for weight in weights]
            extras = [int(value) for value in exact]
            for index in sorted(
                range(recipe_count),
                key=lambda item: (exact[item] - extras[item], rng.random()),
                reverse=True,
            )[: remaining - sum(extras)]:
                extras[index] += 1
            counts = [1 + extra for extra in extras]
            if max(counts) / min(counts) <= self.config.recipe_count_max_ratio:
                return counts
        raise _RetryGeneration("could not distribute wafers near-uniformly across Recipes")

    def _priorities(
        self,
        counts: list[int],
        rng: random.Random,
    ) -> dict[tuple[str, int], int]:
        mode = str(_weighted_choice(self.config.priority_mode_weights, rng))
        recipe_ids = [f"R{index:02d}" for index in range(len(counts))]
        priorities: dict[tuple[str, int], int] = {}
        if mode == "none":
            return {
                (route_id, wafer_index): 0
                for route_id, count in zip(recipe_ids, counts, strict=True)
                for wafer_index in range(count)
            }
        if mode == "recipe":
            shuffled = list(recipe_ids)
            rng.shuffle(shuffled)
            group_count = rng.randint(1, min(3, len(recipe_ids)))
            recipe_priority = {
                route_id: index % group_count
                for index, route_id in enumerate(shuffled)
            }
            return {
                (route_id, wafer_index): recipe_priority[route_id]
                for route_id, count in zip(recipe_ids, counts, strict=True)
                for wafer_index in range(count)
            }
        for route_id, count in zip(recipe_ids, counts, strict=True):
            group_count = min(count, rng.randint(2, 4))
            for wafer_index in range(count):
                priorities[(route_id, wafer_index)] = min(
                    group_count - 1,
                    wafer_index * group_count // count,
                )
        return priorities

    def _robot(
        self,
        module_ids: tuple[str, ...],
        *,
        arm_type: str,
        rng: random.Random,
    ) -> dict[str, Any]:
        return {
            "module_ids": list(module_ids),
            "arm_type": arm_type,
            "travel_times": rng.randint(*self.config.transfer_time_range),
            "pick_time": rng.randint(*self.config.transfer_time_range),
            "place_time": rng.randint(*self.config.transfer_time_range),
        }

    def _pm_process_time(self, rng: random.Random) -> int:
        anchor = int(_weighted_choice(self.config.pm_time_anchor_weights, rng))
        jitter = self.config.process_time_jitter
        return max(1, round(anchor * rng.uniform(1 - jitter, 1 + jitter)))

    def _ll_time(self, rng: random.Random) -> int:
        if rng.random() < self.config.ll_tail_probability:
            return rng.randint(*self.config.ll_tail_time_range)
        return rng.randint(*self.config.ll_common_time_range)

    def _wafer_range(self, difficulty: Difficulty) -> tuple[int, int]:
        if difficulty == "easy":
            return self.config.easy_wafer_range
        if difficulty == "medium":
            return self.config.medium_wafer_range
        return self.config.hard_wafer_range

    @staticmethod
    def _metadata(
        problem: ClusterProblem,
        reference: HeuristicResult,
        audit: GenerationAudit,
        *,
        seed: int,
        split: Split,
        difficulty: Difficulty,
        topology_family: TopologyFamily,
        structural_signature: str,
        signature_bucket: int,
        process_step_counts: dict[str, int],
    ) -> dict[str, Any]:
        visits = [visit for route in problem.routes.values() for visit in route.visits]
        pm_visits = [
            visit
            for visit in visits
            if problem.Modules[visit.module_ids[0]].type is ModuleType.PM
        ]
        process_times = [float(visit.process_time or 0.0) for visit in pm_visits]
        optimality_gap = (
            (reference.makespan - reference.lower_bound) / reference.lower_bound
            if reference.lower_bound > 0
            else 0.0
        )
        return {
            "seed": seed,
            "split": split,
            "difficulty": difficulty,
            "topology_family": topology_family,
            "structural_signature": structural_signature,
            "signature_bucket": signature_bucket,
            "io_count": audit.module_counts.get("IO", 0),
            "pm_count": audit.module_counts.get("PM", 0),
            "ll_count": audit.module_counts.get("LL", 0),
            "buffer_count": audit.module_counts.get("BUFFER", 0),
            "robot_count": audit.robot_count,
            "wafer_count": audit.wafer_count,
            "route_count": audit.route_count,
            "max_route_length": audit.max_route_steps,
            "max_process_steps": max(process_step_counts.values()),
            "candidate_ratio": round(
                sum(len(visit.module_ids) > 1 for visit in visits) / len(visits),
                6,
            ),
            "median_process_time": round(statistics.median(process_times), 6),
            "pm_total_loads": {
                module_id: round(load, 6)
                for module_id, load in sorted(reference.pm_loads.items())
            },
            "average_legal_actions": round(reference.average_legal_actions, 6),
            "multiple_choice_state_ratio": round(reference.multiple_choice_state_ratio, 6),
            "reference_makespan": round(reference.makespan, 6),
            "lower_bound": round(reference.lower_bound, 6),
            "optimality_gap": round(optimality_gap, 6),
            "reference_policy": "serial_domain_feasibility_witness",
            "validator_result": True,
            "feasible": True,
        }

    @staticmethod
    def _attempt_seed(seed: int, difficulty: str, split: str, attempt: int) -> int:
        digest = hashlib.sha256(
            f"{seed}:{difficulty}:{split}:{attempt}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big", signed=False)


_Choice = TypeVar("_Choice")


def _weighted_choice(weights: dict[_Choice, float], rng: random.Random) -> _Choice:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]


def _parse_generated_problem(raw_problem: dict[str, Any]) -> ClusterProblem:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Explicit Module.capacity overrides the type-based default",
            category=UserWarning,
        )
        return parse_problem(raw_problem)


def _structural_signature(raw_problem: dict[str, Any]) -> str:
    modules = raw_problem["Modules"]
    payload = {
        "modules": {
            module_id: {
                "type": module["type"],
                "process_ids": module.get("process_ids", []),
            }
            for module_id, module in sorted(modules.items())
        },
        "robots": {
            robot_id: {
                "module_ids": robot["module_ids"],
                "arm_type": robot["arm_type"],
            }
            for robot_id, robot in sorted(raw_problem["ClusterTool"].items())
        },
        "routes": {
            route_id: [
                {
                    "module_ids": visit["module_ids"],
                    "process_id": visit.get("process_id"),
                    "cooling": (
                        modules[visit["module_ids"][0]]["type"] == "BUFFER"
                        and visit.get("process_time", 0) > 0
                    ),
                }
                for visit in visits
            ]
            for route_id, visits in sorted(raw_problem["routes"].items())
        },
        "route_wafer_counts": {
            route_id: sum(
                wafer["route_id"] == route_id
                for wafer in raw_problem["initial_state"]["wafers"]
            )
            for route_id in sorted(raw_problem["routes"])
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
