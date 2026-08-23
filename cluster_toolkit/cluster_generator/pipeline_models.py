from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class ModuleKind(str, Enum):
    """Physical module kinds.

    ``PM``/``AL``/``BUFFER``/``LL`` remain readable for schema-v1 corpus
    compatibility.  New schema-v2 problems use only ``IO``, ``CHAMBER`` and
    ``LOAD_LOCK``; process roles are expressed with :class:`ModuleTag`.
    """

    IO = "IO"
    CHAMBER = "CHAMBER"
    LOAD_LOCK = "LOAD_LOCK"
    PM = "PM"
    AL = "AL"
    LL = "LL"
    BUFFER = "BUFFER"


class ModuleTag(str, Enum):
    PROCESS = "PM"
    ALIGN = "AL"
    BUFFER = "BUFFER"


class RobotArmKind(str, Enum):
    SINGLE = "single_arm"
    DUAL = "dual_arm"


class LoadLockState(str, Enum):
    ATMOSPHERE = "atmosphere"
    VACUUM = "vacuum"


class IntInterval(_StrictModel):
    minimum: int
    maximum: int

    @field_validator("minimum", "maximum", mode="before")
    @classmethod
    def _validate_bound(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("interval bounds must be non-negative integers")
        return value

    @model_validator(mode="after")
    def _validate_order(self) -> "IntInterval":
        if self.maximum < self.minimum:
            raise ValueError("interval maximum must be greater than or equal to minimum")
        return self


class TopologyLoadLock(_StrictModel):
    initial_state: LoadLockState = LoadLockState.ATMOSPHERE
    robot_required_states: dict[str, LoadLockState]

    @field_validator("robot_required_states")
    @classmethod
    def _validate_states(
        cls,
        value: dict[str, LoadLockState],
    ) -> dict[str, LoadLockState]:
        if not value:
            raise ValueError("TopologyLoadLock.robot_required_states must not be empty")
        for robot_id in value:
            _non_empty(robot_id, "TopologyLoadLock.robot_required_states key")
        return value


class TopologyModule(_StrictModel):
    kind: ModuleKind
    cell_id: str | None = None
    connected_cell_ids: tuple[str, str] | None = None
    tags: tuple[ModuleTag, ...] = ()
    capacity: Literal[1] = 1
    load_lock: TopologyLoadLock | None = None

    @field_validator("cell_id")
    @classmethod
    def _validate_cell_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, "TopologyModule.cell_id")

    @field_validator("connected_cell_ids")
    @classmethod
    def _validate_connected_cells(
        cls,
        value: tuple[str, str] | None,
    ) -> tuple[str, str] | None:
        if value is None:
            return None
        if value[0] == value[1]:
            raise ValueError("connected_cell_ids must name two different Cells")
        for cell_id in value:
            _non_empty(cell_id, "TopologyModule.connected_cell_ids item")
        return value

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: tuple[ModuleTag, ...]) -> tuple[ModuleTag, ...]:
        if len(set(value)) != len(value):
            raise ValueError("TopologyModule.tags must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_load_lock(self) -> "TopologyModule":
        is_load_lock = self.kind in {ModuleKind.LL, ModuleKind.LOAD_LOCK}
        if is_load_lock and self.load_lock is None:
            raise ValueError("LL topology Module must define load_lock sides")
        if not is_load_lock and self.load_lock is not None:
            raise ValueError("only LL topology Modules may define load_lock sides")
        if ModuleTag.BUFFER in self.tags:
            if self.kind is not ModuleKind.CHAMBER:
                raise ValueError("BUFFER tag requires a schema-v2 CHAMBER")
            if self.connected_cell_ids is None:
                raise ValueError("BUFFER CHAMBER must define connected_cell_ids")
            if self.cell_id is not None:
                raise ValueError("BUFFER CHAMBER belongs to a Cell boundary, not one Cell")
        elif self.connected_cell_ids is not None:
            raise ValueError("only BUFFER CHAMBER may define connected_cell_ids")
        return self

    @property
    def physical_kind(self) -> ModuleKind:
        if self.kind in {ModuleKind.PM, ModuleKind.AL, ModuleKind.BUFFER}:
            return ModuleKind.CHAMBER
        if self.kind is ModuleKind.LL:
            return ModuleKind.LOAD_LOCK
        return self.kind

    @property
    def effective_tags(self) -> frozenset[ModuleTag]:
        legacy = {
            ModuleKind.PM: ModuleTag.PROCESS,
            ModuleKind.AL: ModuleTag.ALIGN,
            ModuleKind.BUFFER: ModuleTag.BUFFER,
        }
        tag = legacy.get(self.kind)
        return frozenset((*self.tags, *((tag,) if tag is not None else ())))


class TopologyRobot(_StrictModel):
    module_ids: tuple[str, ...]
    arm_kind: RobotArmKind
    cell_id: str | None = None

    @field_validator("module_ids")
    @classmethod
    def _validate_module_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("TopologyRobot.module_ids must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("TopologyRobot.module_ids must not contain duplicates")
        for module_id in value:
            _non_empty(module_id, "TopologyRobot.module_ids item")
        return value

    @field_validator("cell_id")
    @classmethod
    def _validate_cell(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, "TopologyRobot.cell_id")


class TopologyTemplate(_StrictModel):
    schema_version: Literal[1, 2] = 1
    topology_id: str
    topology_version: str
    family_id: str | None = None
    archetype_id: str | None = None
    arm_profile_id: str | None = None
    cell_order: tuple[str, ...] = ()
    modules: dict[str, TopologyModule]
    robots: dict[str, TopologyRobot]

    @field_validator("topology_id", "topology_version")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"TopologyTemplate.{info.field_name}")

    @field_validator("family_id", "archetype_id", "arm_profile_id")
    @classmethod
    def _validate_optional_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _non_empty(value, f"TopologyTemplate.{info.field_name}")

    @model_validator(mode="after")
    def _validate_topology(self) -> "TopologyTemplate":
        if not self.modules:
            raise ValueError("TopologyTemplate.modules must not be empty")
        if not self.robots:
            raise ValueError("TopologyTemplate.robots must not be empty")
        if (self.archetype_id is None) != (self.arm_profile_id is None):
            raise ValueError(
                "TopologyTemplate archetype_id and arm_profile_id must be set together"
            )
        for module_id in self.modules:
            _non_empty(module_id, "TopologyTemplate.modules key")
        for robot_id in self.robots:
            _non_empty(robot_id, "TopologyTemplate.robots key")

        io_ids = [
            module_id
            for module_id, module in self.modules.items()
            if module.kind is ModuleKind.IO
        ]
        if len(io_ids) != 1:
            raise ValueError("pipeline topology must contain exactly one virtual IO")

        if self.schema_version == 2:
            self._validate_v2_cells()

        reachable: set[str] = set()
        for robot_id, robot in self.robots.items():
            for module_id in robot.module_ids:
                if module_id not in self.modules:
                    raise ValueError(
                        f"Topology Robot {robot_id} references unknown Module: {module_id}"
                    )
                reachable.add(module_id)
        missing = sorted(set(self.modules) - reachable)
        if missing:
            raise ValueError(f"topology Modules are unreachable by every Robot: {missing}")

        for module_id, module in self.modules.items():
            if module.load_lock is None:
                continue
            for robot_id in module.load_lock.robot_required_states:
                if robot_id not in self.robots:
                    raise ValueError(
                        f"Topology LL {module_id} references unknown Robot: {robot_id}"
                    )
                if module_id not in self.robots[robot_id].module_ids:
                    raise ValueError(
                        f"Topology LL {module_id} is not reachable by Robot {robot_id}"
                    )
        return self

    def _validate_v2_cells(self) -> None:
        if not self.cell_order or len(set(self.cell_order)) != len(self.cell_order):
            raise ValueError("schema-v2 topology requires a unique non-empty cell_order")
        cells = set(self.cell_order)
        allowed_kinds = {ModuleKind.IO, ModuleKind.CHAMBER, ModuleKind.LOAD_LOCK}
        for module_id, module in self.modules.items():
            if module.kind not in allowed_kinds:
                raise ValueError(
                    f"schema-v2 Module {module_id} must use IO, CHAMBER, or LOAD_LOCK"
                )
            if ModuleTag.BUFFER in module.effective_tags:
                assert module.connected_cell_ids is not None
                left, right = module.connected_cell_ids
                if left not in cells or right not in cells:
                    raise ValueError(f"BUFFER {module_id} references an unknown Cell")
                if abs(self.cell_order.index(left) - self.cell_order.index(right)) != 1:
                    raise ValueError(f"BUFFER {module_id} must connect adjacent Cells")
            elif module.cell_id not in cells:
                raise ValueError(f"Module {module_id} must belong to one declared Cell")

        robots_by_cell: dict[str, list[str]] = {cell_id: [] for cell_id in self.cell_order}
        for robot_id, robot in self.robots.items():
            if robot.cell_id not in cells:
                raise ValueError(f"schema-v2 Robot {robot_id} must belong to one Cell")
            assert robot.cell_id is not None
            robots_by_cell[robot.cell_id].append(robot_id)
        invalid = {
            cell_id: robot_ids
            for cell_id, robot_ids in robots_by_cell.items()
            if len(robot_ids) != 1
        }
        if invalid:
            raise ValueError(f"schema-v2 requires exactly one Robot per Cell: {invalid}")

        robot_by_cell = {
            robot.cell_id: robot_id for robot_id, robot in self.robots.items()
        }
        for module_id, module in self.modules.items():
            reachable = {
                robot_id
                for robot_id, robot in self.robots.items()
                if module_id in robot.module_ids
            }
            if ModuleTag.BUFFER in module.effective_tags:
                assert module.connected_cell_ids is not None
                expected = {robot_by_cell[cell_id] for cell_id in module.connected_cell_ids}
            else:
                assert module.cell_id is not None
                expected = {robot_by_cell[module.cell_id]}
            if reachable != expected:
                raise ValueError(
                    f"Module {module_id} must be reachable exactly by Cell Robot(s) "
                    f"{sorted(expected)}, got {sorted(reachable)}"
                )

    @property
    def io_module_id(self) -> str:
        return next(
            module_id
            for module_id, module in self.modules.items()
            if module.kind is ModuleKind.IO
        )

    @property
    def pm_module_ids(self) -> tuple[str, ...]:
        return tuple(
            module_id
            for module_id, module in sorted(self.modules.items())
            if ModuleTag.PROCESS in module.effective_tags
        )

    @property
    def chamber_module_ids(self) -> tuple[str, ...]:
        return tuple(
            module_id
            for module_id, module in sorted(self.modules.items())
            if module.physical_kind is ModuleKind.CHAMBER
        )


class RecipeGenerationProfile(_StrictModel):
    schema_version: Literal[1, 2] = 1
    profile_id: str
    profile_version: str
    applies_to: tuple[str, ...]
    applies_to_families: tuple[str, ...] = ()
    compiler: Literal["direct_single_cell", "atmospheric_linear"] = "direct_single_cell"
    pm_step_count_weights: dict[int, float]
    candidate_pm_count_weights: dict[int, float]
    reentry_probability: float = 0.1
    process_time_anchor_weights: dict[int, float]
    process_time_jitter: float = 0.1
    robot_time: IntInterval = Field(
        default_factory=lambda: IntInterval(minimum=8, maximum=15)
    )
    ll_transition_time: IntInterval = Field(
        default_factory=lambda: IntInterval(minimum=10, maximum=30)
    )
    alignment_probability: float = 0.0
    alignment_time: IntInterval = Field(
        default_factory=lambda: IntInterval(minimum=10, maximum=30)
    )
    buffer_hold_time: IntInterval = Field(
        default_factory=lambda: IntInterval(minimum=0, maximum=0)
    )
    route_pattern_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "local": 0.30,
            "single_transition": 0.40,
            "multi_transition": 0.30,
        }
    )

    @field_validator("profile_id", "profile_version")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"RecipeGenerationProfile.{info.field_name}")

    @field_validator("applies_to", "applies_to_families")
    @classmethod
    def _validate_applies_to(
        cls,
        value: tuple[str, ...],
        info,
    ) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError(f"RecipeGenerationProfile.{info.field_name} has duplicates")
        for item in value:
            _non_empty(item, f"RecipeGenerationProfile.{info.field_name} item")
        return value

    @model_validator(mode="after")
    def _validate_scope(self) -> "RecipeGenerationProfile":
        if not self.applies_to and not self.applies_to_families:
            raise ValueError(
                "RecipeGenerationProfile must apply to a topology or topology family"
            )
        return self

    @field_validator(
        "pm_step_count_weights",
        "candidate_pm_count_weights",
        "process_time_anchor_weights",
    )
    @classmethod
    def _validate_weights(
        cls,
        value: dict[int, float],
        info,
    ) -> dict[int, float]:
        if not value or any(key <= 0 for key in value):
            raise ValueError(f"{info.field_name} must use positive integer keys")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
            for weight in value.values()
        ):
            raise ValueError(f"{info.field_name} weights must be finite and non-negative")
        if sum(float(weight) for weight in value.values()) <= 0:
            raise ValueError(f"{info.field_name} must contain a positive weight")
        return value

    @field_validator("route_pattern_weights")
    @classmethod
    def _validate_route_weights(cls, value: dict[str, float]) -> dict[str, float]:
        allowed = {"local", "single_transition", "multi_transition"}
        if not value or set(value) - allowed:
            raise ValueError(
                "route_pattern_weights supports local, single_transition, and "
                "multi_transition"
            )
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
            for weight in value.values()
        ) or sum(float(weight) for weight in value.values()) <= 0:
            raise ValueError("route_pattern_weights must contain positive finite weight")
        return value

    @field_validator(
        "reentry_probability",
        "process_time_jitter",
        "alignment_probability",
        mode="before",
    )
    @classmethod
    def _validate_probability(cls, value: object, info) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        result = float(value)
        if not 0 <= result <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return result


WaferScale = Literal["small", "medium", "large", "xlarge"]


class InstanceGenerationRequest(_StrictModel):
    topology_id: str
    profile_id: str
    recipe_count: int
    wafer_scale: WaferScale
    seed: int
    periodic_ratio: tuple[int, ...] | None = None
    route_pattern: Literal["local", "single_transition", "multi_transition"] | None = None

    @field_validator("topology_id", "profile_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"InstanceGenerationRequest.{info.field_name}")

    @field_validator("recipe_count", mode="before")
    @classmethod
    def _validate_recipe_count(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 3:
            raise ValueError("recipe_count must be an integer from 1 to 3")
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("seed must be an integer")
        return value

    @model_validator(mode="after")
    def _validate_periodic_ratio(self) -> "InstanceGenerationRequest":
        if self.periodic_ratio is None:
            return self
        if len(self.periodic_ratio) != self.recipe_count:
            raise ValueError("periodic_ratio length must equal recipe_count")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.periodic_ratio
        ):
            raise ValueError("periodic_ratio values must be positive integers")
        return self


class RobotTiming(_StrictModel):
    pick_time: int
    place_time: int
    travel_time: int

    @field_validator("pick_time", "place_time", "travel_time", mode="before")
    @classmethod
    def _validate_time(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Robot times must be non-negative integers")
        return value


class LoadLockTiming(_StrictModel):
    atmosphere_to_vacuum_time: int
    vacuum_to_atmosphere_time: int

    @field_validator(
        "atmosphere_to_vacuum_time",
        "vacuum_to_atmosphere_time",
        mode="before",
    )
    @classmethod
    def _validate_time(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("Load-lock times must be non-negative integers")
        return value


class EquipmentTiming(_StrictModel):
    robots: dict[str, RobotTiming]
    load_locks: dict[str, LoadLockTiming] = Field(default_factory=dict)


class RecipeStep(_StrictModel):
    step_id: str
    candidate_module_ids: tuple[str, ...]
    process_time: int

    @field_validator("step_id")
    @classmethod
    def _validate_step_id(cls, value: str) -> str:
        return _non_empty(value, "RecipeStep.step_id")

    @field_validator("candidate_module_ids")
    @classmethod
    def _validate_candidates(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("RecipeStep.candidate_module_ids must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("RecipeStep.candidate_module_ids must not contain duplicates")
        return value

    @field_validator("process_time", mode="before")
    @classmethod
    def _validate_process_time(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("RecipeStep.process_time must be a non-negative integer")
        return value


class Recipe(_StrictModel):
    recipe_id: str
    steps: tuple[RecipeStep, ...]

    @field_validator("recipe_id")
    @classmethod
    def _validate_recipe_id(cls, value: str) -> str:
        return _non_empty(value, "Recipe.recipe_id")

    @field_validator("steps")
    @classmethod
    def _validate_steps(cls, value: tuple[RecipeStep, ...]) -> tuple[RecipeStep, ...]:
        if not value:
            raise ValueError("Recipe.steps must not be empty")
        step_ids = [step.step_id for step in value]
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("Recipe.step_id values must be unique within a Recipe")
        return value


class WorkloadItem(_StrictModel):
    recipe_id: str
    wafer_count: int
    release_time: Literal[0] = 0
    priority: Literal[0] = 0

    @field_validator("recipe_id")
    @classmethod
    def _validate_recipe_id(cls, value: str) -> str:
        return _non_empty(value, "WorkloadItem.recipe_id")

    @field_validator("wafer_count", mode="before")
    @classmethod
    def _validate_wafer_count(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("WorkloadItem.wafer_count must be a positive integer")
        return value


class GenerationProvenance(_StrictModel):
    generator_name: str
    generator_version: str
    seed: int
    profile_id: str
    profile_version: str
    wafer_scale: WaferScale
    periodic_ratio: tuple[int, ...] | None = None


class SchedulingInstance(_StrictModel):
    """Canonical input shared by solvers, validators, and learning adapters."""

    schema_version: Literal[1, 2] = 1
    instance_id: str
    topology: TopologyTemplate
    timing: EquipmentTiming
    recipes: tuple[Recipe, ...]
    workload: tuple[WorkloadItem, ...]
    source_module_id: str
    sink_module_id: str
    objective: Literal["makespan"] = "makespan"
    provenance: GenerationProvenance

    @field_validator("instance_id", "source_module_id", "sink_module_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"SchedulingInstance.{info.field_name}")

    @model_validator(mode="after")
    def _validate_references(self) -> "SchedulingInstance":
        module_ids = set(self.topology.modules)
        if self.source_module_id not in module_ids:
            raise ValueError("source_module_id references an unknown Module")
        if self.sink_module_id not in module_ids:
            raise ValueError("sink_module_id references an unknown Module")
        if self.topology.modules[self.source_module_id].kind is not ModuleKind.IO:
            raise ValueError("source_module_id must reference the virtual IO")
        if self.topology.modules[self.sink_module_id].kind is not ModuleKind.IO:
            raise ValueError("sink_module_id must reference the virtual IO")

        if set(self.timing.robots) != set(self.topology.robots):
            raise ValueError("timing.robots must match topology.robots exactly")
        load_lock_ids = {
            module_id
            for module_id, module in self.topology.modules.items()
            if module.physical_kind is ModuleKind.LOAD_LOCK
        }
        if set(self.timing.load_locks) != load_lock_ids:
            raise ValueError("timing.load_locks must match topology LL Modules exactly")

        recipe_ids = [recipe.recipe_id for recipe in self.recipes]
        if not recipe_ids or len(set(recipe_ids)) != len(recipe_ids):
            raise ValueError("SchedulingInstance requires unique Recipes")
        workload_ids = [item.recipe_id for item in self.workload]
        if len(set(workload_ids)) != len(workload_ids):
            raise ValueError("workload must contain each Recipe at most once")
        if set(workload_ids) != set(recipe_ids):
            raise ValueError("workload must contain every Recipe exactly once")

        for recipe in self.recipes:
            for step in recipe.steps:
                unknown = set(step.candidate_module_ids) - module_ids
                if unknown:
                    raise ValueError(
                        f"Recipe {recipe.recipe_id} step {step.step_id} has "
                        f"unknown candidates: {sorted(unknown)}"
                    )
                candidates = [
                    self.topology.modules[module_id]
                    for module_id in step.candidate_module_ids
                ]
                if any(
                    module.physical_kind is not ModuleKind.CHAMBER
                    for module in candidates
                ):
                    raise ValueError(
                        f"Recipe {recipe.recipe_id} step {step.step_id} candidates "
                        "must be ordinary CHAMBER Modules"
                    )
                if self.schema_version == 1:
                    if step.process_time <= 0 or any(
                        ModuleTag.PROCESS not in module.effective_tags
                        for module in candidates
                    ):
                        raise ValueError(
                            f"schema-v1 Recipe {recipe.recipe_id} step {step.step_id} "
                            "requires positive-time PM candidates"
                        )
                self._validate_candidate_domain(recipe.recipe_id, step, candidates)
            self._validate_route_transfers(recipe)
        return self

    def _validate_candidate_domain(
        self,
        recipe_id: str,
        step: RecipeStep,
        candidates: list[TopologyModule],
    ) -> None:
        buffer_candidates = [
            ModuleTag.BUFFER in module.effective_tags for module in candidates
        ]
        if any(buffer_candidates) and not all(buffer_candidates):
            raise ValueError(
                f"Recipe {recipe_id} step {step.step_id} mixes BUFFER and non-BUFFER candidates"
            )
        if all(buffer_candidates):
            connections = {module.connected_cell_ids for module in candidates}
            if len(connections) != 1:
                raise ValueError(
                    f"Recipe {recipe_id} step {step.step_id} BUFFER candidates "
                    "must connect the same Cell pair"
                )
            return
        cells = {module.cell_id for module in candidates}
        if len(cells) != 1:
            raise ValueError(
                f"Recipe {recipe_id} step {step.step_id} candidates must stay in one Cell"
            )

    def _validate_route_transfers(self, recipe: Recipe) -> None:
        candidate_layers: list[tuple[str, ...]] = [
            (self.source_module_id,),
            *(step.candidate_module_ids for step in recipe.steps),
            (self.sink_module_id,),
        ]
        for edge_index, (sources, targets) in enumerate(
            zip(candidate_layers, candidate_layers[1:])
        ):
            for source in sources:
                for target in targets:
                    if not any(
                        source in robot.module_ids and target in robot.module_ids
                        for robot in self.topology.robots.values()
                    ):
                        raise ValueError(
                            f"Recipe {recipe.recipe_id} transfer edge {edge_index} "
                            f"has no Robot for {source} -> {target}"
                        )
