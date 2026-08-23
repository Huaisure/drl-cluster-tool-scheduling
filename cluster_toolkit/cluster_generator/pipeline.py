from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass
from typing import TypeVar

from .pipeline_catalog import PipelineCatalog
from .pipeline_models import (
    EquipmentTiming,
    GenerationProvenance,
    InstanceGenerationRequest,
    LoadLockTiming,
    ModuleKind,
    ModuleTag,
    Recipe,
    RecipeGenerationProfile,
    RecipeStep,
    RobotTiming,
    SchedulingInstance,
    TopologyTemplate,
    WorkloadItem,
)


PIPELINE_GENERATOR_NAME = "cluster_data_pipeline"
PIPELINE_GENERATOR_VERSION = "0.3.0"

WAFER_RANGES = {
    "small": (10, 25),
    "medium": (26, 50),
    "large": (51, 100),
    "xlarge": (101, 200),
}
PERIODIC_RATIOS = {
    1: {(1,)},
    2: {(1, 1), (1, 2), (2, 1)},
    3: {
        (1, 1, 1),
        (1, 2, 1),
        (2, 1, 1),
        (1, 1, 2),
        (1, 2, 2),
        (2, 2, 1),
        (2, 1, 2),
    },
}


@dataclass(frozen=True, slots=True)
class GeneratedInstance:
    instance: SchedulingInstance
    metadata: dict[str, object]

    @property
    def instance_id(self) -> str:
        return self.instance.instance_id


class InstanceGenerator:
    """Generate an immutable instance without relying on the legacy Problem schema."""

    def __init__(
        self,
        catalog: PipelineCatalog,
        *,
        wafer_ranges: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        self.catalog = catalog
        self.wafer_ranges = dict(WAFER_RANGES if wafer_ranges is None else wafer_ranges)

    def generate(self, request: InstanceGenerationRequest) -> GeneratedInstance:
        topology = self.catalog.topology(request.topology_id)
        profile = self.catalog.profile(request.profile_id)
        if (
            topology.topology_id not in profile.applies_to
            and topology.family_id not in profile.applies_to_families
        ):
            raise ValueError(
                f"Recipe profile {profile.profile_id} does not apply to "
                f"topology {topology.topology_id}"
            )
        self._require_supported_topology(topology, profile)

        rng = random.Random(request.seed)
        periodic_ratio = self._normalized_periodic_ratio(request)
        wafer_counts = self._wafer_counts(request, periodic_ratio, rng)
        timing = self._generate_timing(topology, profile, rng)
        recipes, recipe_stats = self._generate_recipes(
            topology,
            profile,
            request.recipe_count,
            rng,
            require_periodic_structure=periodic_ratio is not None,
            route_pattern=request.route_pattern,
        )
        instance_id = self._instance_id(request, topology, profile)
        instance = SchedulingInstance(
            schema_version=topology.schema_version,
            instance_id=instance_id,
            topology=topology,
            timing=timing,
            recipes=recipes,
            workload=tuple(
                WorkloadItem(recipe_id=recipe.recipe_id, wafer_count=count)
                for recipe, count in zip(recipes, wafer_counts, strict=True)
            ),
            source_module_id=topology.io_module_id,
            sink_module_id=topology.io_module_id,
            provenance=GenerationProvenance(
                generator_name=PIPELINE_GENERATOR_NAME,
                generator_version=PIPELINE_GENERATOR_VERSION,
                seed=request.seed,
                profile_id=profile.profile_id,
                profile_version=profile.profile_version,
                wafer_scale=request.wafer_scale,
                periodic_ratio=periodic_ratio,
            ),
        )
        content_hash = hashlib.sha256(
            instance.model_dump_json(exclude={"instance_id"}).encode("utf-8")
        ).hexdigest()
        overlap_ok = self._has_exact_or_disjoint_candidate_domains(recipes)
        periodic_eligible = periodic_ratio is not None and overlap_ok
        metadata: dict[str, object] = {
            "instance_id": instance_id,
            "generation_status": "generated",
            "labeling_status": "pending",
            "topology_id": topology.topology_id,
            "topology_version": topology.topology_version,
            "profile_id": profile.profile_id,
            "profile_version": profile.profile_version,
            "recipe_count": request.recipe_count,
            "wafer_scale": request.wafer_scale,
            "wafer_count": sum(wafer_counts),
            "wafer_counts": {
                recipe.recipe_id: count
                for recipe, count in zip(recipes, wafer_counts, strict=True)
            },
            "periodic_ratio": list(periodic_ratio) if periodic_ratio else None,
            "requested_route_pattern": request.route_pattern,
            "periodic_eligible": periodic_eligible,
            "periodic_ineligibility_reasons": (
                []
                if periodic_eligible
                else [
                    reason
                    for condition, reason in (
                        (periodic_ratio is None, "missing_periodic_ratio"),
                        (not overlap_ok, "partially_overlapping_candidate_domains"),
                    )
                    if condition
                ]
            ),
            "content_hash": content_hash,
            **recipe_stats,
        }
        return GeneratedInstance(instance=instance, metadata=metadata)

    @staticmethod
    def _require_supported_topology(
        topology: TopologyTemplate,
        profile: RecipeGenerationProfile,
    ) -> None:
        if profile.compiler == "atmospheric_linear":
            if topology.schema_version != 2:
                raise ValueError("atmospheric_linear requires a schema-v2 topology")
            if topology.family_id != "atmospheric_linear":
                raise ValueError(
                    "atmospheric_linear requires the atmospheric_linear topology family"
                )
            if any(
                module.physical_kind is ModuleKind.LOAD_LOCK
                for module in topology.modules.values()
            ):
                raise NotImplementedError(
                    "atmospheric_linear v2 does not support LOAD_LOCK Modules"
                )
            if not topology.pm_module_ids:
                raise ValueError(
                    "atmospheric_linear topology must contain a PROCESS CHAMBER"
                )
            return
        if profile.compiler != "direct_single_cell":  # pragma: no cover
            raise NotImplementedError(f"unsupported Recipe compiler: {profile.compiler}")
        unsupported = sorted(
            module_id
            for module_id, module in topology.modules.items()
            if module.kind not in {ModuleKind.IO, ModuleKind.PM}
        )
        if unsupported:
            raise NotImplementedError(
                "direct_single_cell compiler only supports IO and PM Modules; "
                f"unsupported Modules: {unsupported}"
            )
        if not topology.pm_module_ids:
            raise ValueError("direct_single_cell topology must contain at least one PM")
        required = {topology.io_module_id, *topology.pm_module_ids}
        if not any(required <= set(robot.module_ids) for robot in topology.robots.values()):
            raise ValueError(
                "direct_single_cell topology requires one Robot that reaches IO and every PM"
            )

    @staticmethod
    def _generate_timing(
        topology: TopologyTemplate,
        profile: RecipeGenerationProfile,
        rng: random.Random,
    ) -> EquipmentTiming:
        robots = {
            robot_id: RobotTiming(
                pick_time=rng.randint(
                    profile.robot_time.minimum,
                    profile.robot_time.maximum,
                ),
                place_time=rng.randint(
                    profile.robot_time.minimum,
                    profile.robot_time.maximum,
                ),
                travel_time=rng.randint(
                    profile.robot_time.minimum,
                    profile.robot_time.maximum,
                ),
            )
            for robot_id in sorted(topology.robots)
        }
        load_locks = {
            module_id: LoadLockTiming(
                atmosphere_to_vacuum_time=rng.randint(
                    profile.ll_transition_time.minimum,
                    profile.ll_transition_time.maximum,
                ),
                vacuum_to_atmosphere_time=rng.randint(
                    profile.ll_transition_time.minimum,
                    profile.ll_transition_time.maximum,
                ),
            )
            for module_id, module in sorted(topology.modules.items())
            if module.kind is ModuleKind.LL
        }
        return EquipmentTiming(robots=robots, load_locks=load_locks)

    def _generate_recipes(
        self,
        topology: TopologyTemplate,
        profile: RecipeGenerationProfile,
        recipe_count: int,
        rng: random.Random,
        *,
        require_periodic_structure: bool,
        route_pattern: str | None,
    ) -> tuple[tuple[Recipe, ...], dict[str, object]]:
        recipes: list[Recipe] = []
        route_stats: list[dict[str, object]] = []
        requested_route_pattern = route_pattern
        for recipe_index in range(recipe_count):
            if profile.compiler == "atmospheric_linear":
                steps, has_reentry, selected_route_pattern, process_step_count = (
                    self._atmospheric_recipe_steps(topology, profile, rng)
                    if requested_route_pattern is None
                    else self._atmospheric_recipe_steps(
                        topology,
                        profile.model_copy(
                            update={
                                "route_pattern_weights": {
                                    requested_route_pattern: 1.0
                                }
                            }
                        ),
                        rng,
                    )
                )
            else:
                steps, has_reentry = self._recipe_steps(topology, profile, rng)
                selected_route_pattern = "local"
                process_step_count = len(steps)
            recipe_id = f"R{recipe_index}"
            recipes.append(Recipe(recipe_id=recipe_id, steps=steps))
            route_stats.append(
                {
                    "recipe_id": recipe_id,
                    "pm_step_count": process_step_count,
                    "route_step_count": len(steps),
                    "route_pattern": selected_route_pattern,
                    "has_reentry": has_reentry,
                }
            )

        if require_periodic_structure:
            recipes = list(self._canonicalize_candidate_domains(tuple(recipes)))

        all_steps = [step for recipe in recipes for step in recipe.steps]
        return tuple(recipes), {
            "recipes": route_stats,
            "max_pm_steps": max(
                int(stats["pm_step_count"]) for stats in route_stats
            ),
            "candidate_step_ratio": round(
                sum(len(step.candidate_module_ids) > 1 for step in all_steps)
                / len(all_steps),
                6,
            ),
            "median_process_time": statistics.median(
                step.process_time for step in all_steps
            ),
        }

    def _recipe_steps(
        self,
        topology: TopologyTemplate,
        profile: RecipeGenerationProfile,
        rng: random.Random,
    ) -> tuple[tuple[RecipeStep, ...], bool]:
        step_count = int(_weighted_choice(profile.pm_step_count_weights, rng))
        has_reentry = rng.random() < profile.reentry_probability and step_count >= 3
        base_step_count = step_count - 1 if has_reentry else step_count
        raw_steps = [
            self._new_pm_step(topology, profile, rng)
            for _ in range(base_step_count)
        ]
        if has_reentry:
            source_index = rng.randrange(len(raw_steps) - 1)
            insertion_index = rng.randint(source_index + 2, len(raw_steps))
            raw_steps.insert(insertion_index, raw_steps[source_index])
        steps = tuple(
            RecipeStep(step_id=f"S{index}", **raw_step)
            for index, raw_step in enumerate(raw_steps)
        )
        return steps, has_reentry

    def _atmospheric_recipe_steps(
        self,
        topology: TopologyTemplate,
        profile: RecipeGenerationProfile,
        rng: random.Random,
    ) -> tuple[tuple[RecipeStep, ...], bool, str, int]:
        process_step_count = int(_weighted_choice(profile.pm_step_count_weights, rng))
        has_reentry = (
            rng.random() < profile.reentry_probability and process_step_count >= 3
        )
        route_pattern = str(_weighted_choice(profile.route_pattern_weights, rng))
        base_step_count = process_step_count - 1 if has_reentry else process_step_count
        process_cells = self._process_cell_sequence(
            topology,
            base_step_count,
            route_pattern,
            rng,
        )

        raw_process_steps = [
            self._new_process_step_for_cell(topology, profile, cell_id, rng)
            for cell_id in process_cells
        ]
        if has_reentry:
            source_index = rng.randrange(len(raw_process_steps) - 1)
            insertion_index = rng.randint(source_index + 2, len(raw_process_steps))
            raw_process_steps.insert(
                insertion_index,
                dict(raw_process_steps[source_index]),
            )
            process_cells.insert(insertion_index, process_cells[source_index])

        raw_route: list[dict[str, object]] = []
        current_cell = topology.cell_order[0]
        alignment_ids = tuple(
            module_id
            for module_id, module in sorted(topology.modules.items())
            if ModuleTag.ALIGN in module.effective_tags
        )
        if alignment_ids and rng.random() < profile.alignment_probability:
            raw_route.append(
                {
                    "candidate_module_ids": alignment_ids,
                    "process_time": rng.randint(
                        profile.alignment_time.minimum,
                        profile.alignment_time.maximum,
                    ),
                }
            )

        for cell_id, process_step in zip(
            process_cells,
            raw_process_steps,
            strict=True,
        ):
            raw_route.extend(
                self._buffer_steps_between(
                    topology,
                    current_cell,
                    cell_id,
                    profile,
                    rng,
                )
            )
            raw_route.append(process_step)
            current_cell = cell_id
        raw_route.extend(
            self._buffer_steps_between(
                topology,
                current_cell,
                topology.cell_order[0],
                profile,
                rng,
            )
        )

        steps = tuple(
            RecipeStep(step_id=f"S{index}", **raw_step)
            for index, raw_step in enumerate(raw_route)
        )
        return steps, has_reentry, route_pattern, process_step_count

    @staticmethod
    def _process_cell_sequence(
        topology: TopologyTemplate,
        step_count: int,
        route_pattern: str,
        rng: random.Random,
    ) -> list[str]:
        cells = list(topology.cell_order)
        if len(cells) == 1 or route_pattern == "local" or step_count == 1:
            cell = rng.choice(cells)
            return [cell] * step_count

        if route_pattern == "single_transition" or step_count == 2:
            first, second = rng.sample(cells, 2)
            split = rng.randint(1, step_count - 1)
            return [first] * split + [second] * (step_count - split)

        # Multi-transition routes intentionally revisit a Cell whenever the
        # number of process visits permits it.  The compiler inserts every
        # intermediate boundary BUFFER, so non-adjacent choices remain valid.
        first, second = rng.sample(cells, 2)
        sequence = [first, second, first]
        while len(sequence) < step_count:
            choices = [cell for cell in cells if cell != sequence[-1]]
            sequence.append(rng.choice(choices))
        return sequence[:step_count]

    @staticmethod
    def _new_process_step_for_cell(
        topology: TopologyTemplate,
        profile: RecipeGenerationProfile,
        cell_id: str,
        rng: random.Random,
    ) -> dict[str, object]:
        process_ids = tuple(
            module_id
            for module_id, module in sorted(topology.modules.items())
            if module.cell_id == cell_id
            and ModuleTag.PROCESS in module.effective_tags
        )
        if not process_ids:
            raise ValueError(f"Cell {cell_id} has no PROCESS CHAMBER")
        candidate_count = min(
            int(_weighted_choice(profile.candidate_pm_count_weights, rng)),
            len(process_ids),
        )
        module_ids = tuple(sorted(rng.sample(process_ids, candidate_count)))
        anchor = int(_weighted_choice(profile.process_time_anchor_weights, rng))
        process_time = max(
            1,
            round(
                anchor
                * rng.uniform(
                    1 - profile.process_time_jitter,
                    1 + profile.process_time_jitter,
                )
            ),
        )
        return {
            "candidate_module_ids": module_ids,
            "process_time": process_time,
        }

    @staticmethod
    def _buffer_steps_between(
        topology: TopologyTemplate,
        source_cell: str,
        target_cell: str,
        profile: RecipeGenerationProfile,
        rng: random.Random,
    ) -> list[dict[str, object]]:
        source_index = topology.cell_order.index(source_cell)
        target_index = topology.cell_order.index(target_cell)
        if source_index == target_index:
            return []
        direction = 1 if target_index > source_index else -1
        result: list[dict[str, object]] = []
        for index in range(source_index, target_index, direction):
            pair = frozenset(
                (topology.cell_order[index], topology.cell_order[index + direction])
            )
            buffer_ids = tuple(
                module_id
                for module_id, module in sorted(topology.modules.items())
                if ModuleTag.BUFFER in module.effective_tags
                and module.connected_cell_ids is not None
                and frozenset(module.connected_cell_ids) == pair
            )
            if not buffer_ids:
                raise ValueError(
                    f"no BUFFER connects Cells {sorted(pair)}"
                )
            result.append(
                {
                    "candidate_module_ids": buffer_ids,
                    "process_time": rng.randint(
                        profile.buffer_hold_time.minimum,
                        profile.buffer_hold_time.maximum,
                    ),
                }
            )
        return result

    @staticmethod
    def _canonicalize_candidate_domains(
        recipes: tuple[Recipe, ...],
    ) -> tuple[Recipe, ...]:
        domains: list[frozenset[str]] = []
        normalized: list[Recipe] = []
        for recipe in recipes:
            steps: list[RecipeStep] = []
            for step in recipe.steps:
                domain = frozenset(step.candidate_module_ids)
                matching = next(
                    (
                        existing
                        for existing in domains
                        if existing & domain and existing != domain
                    ),
                    None,
                )
                if matching is not None:
                    domain = matching
                if domain not in domains:
                    domains.append(domain)
                steps.append(
                    step.model_copy(update={"candidate_module_ids": tuple(sorted(domain))})
                )
            normalized.append(recipe.model_copy(update={"steps": tuple(steps)}))
        return tuple(normalized)

    @staticmethod
    def _has_exact_or_disjoint_candidate_domains(
        recipes: tuple[Recipe, ...],
    ) -> bool:
        domains = [
            frozenset(step.candidate_module_ids)
            for recipe in recipes
            for step in recipe.steps
        ]
        return all(
            not left.intersection(right) or left == right
            for index, left in enumerate(domains)
            for right in domains[index + 1 :]
        )

    @staticmethod
    def _new_pm_step(
        topology: TopologyTemplate,
        profile: RecipeGenerationProfile,
        rng: random.Random,
    ) -> dict[str, object]:
        candidate_count = min(
            int(_weighted_choice(profile.candidate_pm_count_weights, rng)),
            len(topology.pm_module_ids),
        )
        module_ids = tuple(sorted(rng.sample(topology.pm_module_ids, candidate_count)))
        anchor = int(_weighted_choice(profile.process_time_anchor_weights, rng))
        jitter = profile.process_time_jitter
        process_time = max(1, round(anchor * rng.uniform(1 - jitter, 1 + jitter)))
        return {
            "candidate_module_ids": module_ids,
            "process_time": process_time,
        }

    @staticmethod
    def _normalized_periodic_ratio(
        request: InstanceGenerationRequest,
    ) -> tuple[int, ...] | None:
        if request.periodic_ratio is None:
            return None
        divisor = math.gcd(*request.periodic_ratio)
        normalized = tuple(value // divisor for value in request.periodic_ratio)
        if normalized not in PERIODIC_RATIOS[request.recipe_count]:
            raise ValueError(
                f"unsupported periodic ratio for {request.recipe_count} Recipes: {normalized}"
            )
        return normalized

    def _wafer_counts(
        self,
        request: InstanceGenerationRequest,
        periodic_ratio: tuple[int, ...] | None,
        rng: random.Random,
    ) -> tuple[int, ...]:
        try:
            minimum, maximum = self.wafer_ranges[request.wafer_scale]
        except KeyError as exc:
            raise ValueError(
                f"wafer scale has no configured range: {request.wafer_scale}"
            ) from exc
        if minimum <= 0 or maximum < minimum:
            raise ValueError(
                f"invalid wafer range for {request.wafer_scale}: {(minimum, maximum)}"
            )
        if periodic_ratio is not None:
            ratio_total = sum(periodic_ratio)
            minimum_repeats = math.ceil(minimum / ratio_total)
            maximum_repeats = maximum // ratio_total
            if maximum_repeats < minimum_repeats:
                raise ValueError(
                    f"periodic ratio {periodic_ratio} cannot fit wafer scale {request.wafer_scale}"
                )
            repeats = rng.randint(minimum_repeats, maximum_repeats)
            return tuple(value * repeats for value in periodic_ratio)

        total = rng.randint(max(minimum, request.recipe_count), maximum)
        counts = [1] * request.recipe_count
        for _ in range(total - request.recipe_count):
            counts[rng.randrange(request.recipe_count)] += 1
        return tuple(counts)

    @staticmethod
    def _instance_id(
        request: InstanceGenerationRequest,
        topology: TopologyTemplate,
        profile: RecipeGenerationProfile,
    ) -> str:
        payload = {
            "request": request.model_dump(mode="json"),
            "topology_version": topology.topology_version,
            "profile_version": profile.profile_version,
            "generator_version": PIPELINE_GENERATOR_VERSION,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return f"instance-{digest[:16]}"


_Choice = TypeVar("_Choice")


def _weighted_choice(weights: dict[_Choice, float], rng: random.Random) -> _Choice:
    keys = sorted(weights)
    return rng.choices(keys, weights=[weights[key] for key in keys], k=1)[0]
