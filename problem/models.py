from __future__ import annotations

import warnings
from enum import Enum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, RootModel, field_validator, model_validator

from .initial_state import InitialState, ModuleLocation, RobotLocation, WaferInitialState


class ModuleType(str, Enum):
    PM = "PM"
    LL = "LL"
    LP = "LP"


class TMArmType(str, Enum):
    SINGLE_ARM = "single_arm"
    DUAL_ARM = "dual_arm"


class LoadLockState(str, Enum):
    ATMOSPHERE = "atmosphere"
    VACUUM = "vacuum"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_optional_duration(value: float | int | None, field_name: str) -> float | None:
    if value is None:
        return None
    duration = float(value)
    if duration < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return duration


def _validate_mapping_keys(section_name: str, values: dict[str, Any]) -> None:
    for key in values:
        _require_non_empty(key, f"{section_name} key")


class LoadLockConstraint(_StrictModel):
    initial_state: LoadLockState
    atmosphere_to_vacuum_time: float
    vacuum_to_atmosphere_time: float
    tm_required_states: dict[str, LoadLockState] = Field(default_factory=dict)

    @field_validator("atmosphere_to_vacuum_time", "vacuum_to_atmosphere_time", mode="before")
    @classmethod
    def _validate_transition_times(cls, value: float | int | None, info) -> float:
        duration = _validate_optional_duration(value, info.field_name)
        if duration is None:
            raise ValueError(f"LoadLockConstraint.{info.field_name} is required")
        return duration

    @field_validator("tm_required_states")
    @classmethod
    def _validate_tm_required_states(
        cls,
        value: dict[str, LoadLockState],
    ) -> dict[str, LoadLockState]:
        _validate_mapping_keys("LoadLockConstraint.tm_required_states", value)
        return value


class JustInTimeConstraint(_StrictModel):
    residency_time: float | None = None
    pm_residency_time: float | None = None
    ll_residency_time: float | None = None
    completion_to_next_load_time: float | None = None

    @field_validator(
        "residency_time",
        "pm_residency_time",
        "ll_residency_time",
        "completion_to_next_load_time",
        mode="before",
    )
    @classmethod
    def _validate_jit_durations(cls, value: float | int | None, info) -> float | None:
        return _validate_optional_duration(value, f"JustInTimeConstraint.{info.field_name}")


class IdleCleaningRule(_StrictModel):
    idle_time_threshold: float
    clean_time: float

    @field_validator("idle_time_threshold", mode="before")
    @classmethod
    def _validate_idle_time_threshold(cls, value: float | int | None) -> float:
        duration = _validate_optional_duration(value, "IdleCleaningRule.idle_time_threshold")
        if duration is None or duration <= 0:
            raise ValueError("IdleCleaningRule.idle_time_threshold must be positive")
        return duration

    @field_validator("clean_time", mode="before")
    @classmethod
    def _validate_clean_time(cls, value: float | int | None) -> float:
        duration = _validate_optional_duration(value, "IdleCleaningRule.clean_time")
        if duration is None:
            raise ValueError("IdleCleaningRule.clean_time is required")
        return duration


class ProcessSwitchCleaningRule(_StrictModel):
    clean_time: float

    @field_validator("clean_time", mode="before")
    @classmethod
    def _validate_clean_time(cls, value: float | int | None) -> float:
        duration = _validate_optional_duration(value, "ProcessSwitchCleaningRule.clean_time")
        if duration is None:
            raise ValueError("ProcessSwitchCleaningRule.clean_time is required")
        return duration


class WaferCountCleaningRule(_StrictModel):
    wafer_count_threshold: int
    clean_time: float

    @field_validator("wafer_count_threshold", mode="before")
    @classmethod
    def _validate_wafer_count_threshold(cls, value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("WaferCountCleaningRule.wafer_count_threshold must be a positive integer")
        if value <= 0:
            raise ValueError("WaferCountCleaningRule.wafer_count_threshold must be positive")
        return value

    @field_validator("clean_time", mode="before")
    @classmethod
    def _validate_clean_time(cls, value: float | int | None) -> float:
        duration = _validate_optional_duration(value, "WaferCountCleaningRule.clean_time")
        if duration is None:
            raise ValueError("WaferCountCleaningRule.clean_time is required")
        return duration


class CleaningConstraint(_StrictModel):
    module_ids: tuple[str, ...] = Field(
        validation_alias=AliasChoices("module_ids", "modules", "Modules")
    )
    idle: IdleCleaningRule | None = None
    process_switch: ProcessSwitchCleaningRule = Field(
        validation_alias=AliasChoices("process_switch", "batch_switch", "route_switch")
    )
    wafer_count: WaferCountCleaningRule | None = None

    @field_validator("module_ids")
    @classmethod
    def _validate_module_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("CleaningConstraint.module_ids must not be empty")
        for module_id in value:
            _require_non_empty(module_id, "CleaningConstraint.module_ids item")
        if len(set(value)) != len(value):
            raise ValueError("CleaningConstraint.module_ids must not contain duplicates")
        return value


class Module(_StrictModel):
    type: ModuleType
    capacity: int
    load_lock: LoadLockConstraint | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_capacity(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        if "capacity" in value:
            warnings.warn(
                "Explicit Module.capacity overrides the type-based default",
                UserWarning,
                stacklevel=2,
            )
            return value

        normalized = dict(value)
        normalized["capacity"] = 25 if value.get("type") == ModuleType.LP else 1
        return normalized

    @field_validator("capacity", mode="before")
    @classmethod
    def _validate_capacity(cls, value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("Module.capacity must be a positive integer")
        return value


class ClusterCell(_StrictModel):
    module_ids: tuple[str, ...] = Field(default_factory=tuple)
    arm_type: TMArmType
    travel_times: float = 0.0
    place_time: float = Field(
        validation_alias=AliasChoices("place_time", "load_time")
    )
    pick_time: float = Field(
        validation_alias=AliasChoices("pick_time", "unload_time")
    )

    @field_validator("module_ids")
    @classmethod
    def _validate_module_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("ClusterCell.module_ids must not be empty")
        for module_id in value:
            _require_non_empty(module_id, "ClusterCell.module_ids item")
        if len(set(value)) != len(value):
            raise ValueError("ClusterCell.module_ids must not contain duplicates")
        return value

    @field_validator("travel_times", mode="before")
    @classmethod
    def _validate_travel_times(cls, value: Any) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError("ClusterCell.travel_times must be a non-negative number")
        duration = float(value)
        if duration < 0:
            raise ValueError("ClusterCell.travel_times must be non-negative")
        return duration

    @field_validator("place_time", "pick_time", mode="before")
    @classmethod
    def _validate_operation_times(cls, value: float | int | None, info) -> float:
        duration = _validate_optional_duration(value, info.field_name)
        if duration is None:
            raise ValueError(f"ClusterCell.{info.field_name} is required")
        return duration

    def travel_time(self, src_module_id: str, dst_module_id: str) -> float:
        return float(self.travel_times)


class RouteVisit(_StrictModel):
    module_ids: tuple[str, ...]
    process_time: float | None = None
    residency_time: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_module_ids(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        has_module_id = "module_id" in value
        has_module_ids = "module_ids" in value
        if has_module_id == has_module_ids:
            raise ValueError("RouteVisit must contain exactly one of module_id or module_ids")

        normalized = dict(value)
        if has_module_id:
            module_id = normalized.pop("module_id")
            normalized["module_ids"] = (
                tuple(module_id) if isinstance(module_id, (list, tuple)) else (module_id,)
            )
        return normalized

    @field_validator("module_ids")
    @classmethod
    def _validate_module_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("RouteVisit.module_ids must not be empty")
        for module_id in value:
            _require_non_empty(module_id, "RouteVisit.module_ids item")
        if len(set(value)) != len(value):
            raise ValueError("RouteVisit.module_ids must not contain duplicates")
        return value

    @field_validator("process_time", "residency_time", mode="before")
    @classmethod
    def _validate_durations(cls, value: float | int | None, info) -> float | None:
        return _validate_optional_duration(value, info.field_name)

    @property
    def module_id(self) -> str:
        if len(self.module_ids) != 1:
            raise ValueError("RouteVisit has multiple module candidates; use module_ids instead")
        return self.module_ids[0]


class Route(RootModel[tuple[RouteVisit, ...]]):
    @model_validator(mode="before")
    @classmethod
    def _normalize_route(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {"visits"}:
                return value["visits"]
            raise ValueError("Route must be a list or an object with only a visits field")
        return value

    @model_validator(mode="after")
    def _validate_visits(self) -> "Route":
        if not self.root:
            raise ValueError("Route must not be empty")
        return self

    @property
    def visits(self) -> tuple[RouteVisit, ...]:
        return self.root


class ClusterProblem(_StrictModel):
    meta: dict[str, Any] = Field(
        default_factory=dict,
        alias="_meta",
        validation_alias=AliasChoices("_meta", "meta"),
    )
    Modules: dict[str, Module] = Field(default_factory=dict)
    ClusterTool: dict[str, ClusterCell] = Field(default_factory=dict)
    routes: dict[str, Route] = Field(default_factory=dict)
    initial_state: InitialState = Field(default_factory=InitialState)
    just_in_time: JustInTimeConstraint | None = None
    cleaning: CleaningConstraint | None = Field(
        default=None,
        validation_alias=AliasChoices("cleaning", "Cleaning"),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_modules(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "Modules" not in value:
            return value

        modules = value["Modules"]
        if not isinstance(modules, dict) or not modules:
            return value

        grouped_values = [isinstance(module_value, (list, tuple)) for module_value in modules.values()]
        if not any(grouped_values):
            return value
        if not all(grouped_values):
            raise ValueError("Modules must use either module-id keys or module-type keys, not both")

        normalized_modules: dict[str, dict[str, str]] = {}
        for module_type_value, module_ids in modules.items():
            try:
                module_type = ModuleType(module_type_value)
            except ValueError as exc:
                allowed = ", ".join(item.value for item in ModuleType)
                raise ValueError(f"Modules type key must be one of: {allowed}") from exc

            if not module_ids:
                raise ValueError(f"Modules.{module_type.value} must not be empty")
            for module_id in module_ids:
                _require_non_empty(module_id, f"Modules.{module_type.value} item")
                if module_id in normalized_modules:
                    raise ValueError(f"Modules contains duplicate module id: {module_id}")
                normalized_modules[module_id] = {"type": module_type.value}

        normalized = dict(value)
        normalized["Modules"] = normalized_modules
        return normalized

    @property
    def ClusterCells(self) -> dict[str, ClusterCell]:
        return self.ClusterTool

    @field_validator("ClusterTool")
    @classmethod
    def _validate_cluster_tool(cls, value: dict[str, ClusterCell]) -> dict[str, ClusterCell]:
        if not value:
            raise ValueError("ClusterTool must not be empty")
        _validate_mapping_keys("ClusterTool", value)
        return value

    @model_validator(mode="after")
    def _validate_references(self) -> "ClusterProblem":
        _validate_mapping_keys("Modules", self.Modules)
        _validate_mapping_keys("routes", self.routes)

        for tm_id, cell in self.ClusterTool.items():
            for module_id in cell.module_ids:
                if module_id not in self.Modules:
                    raise ValueError(f"ClusterTool {tm_id} references unknown Module: {module_id}")

        for route_id, route in self.routes.items():
            for visit in route.visits:
                for module_id in visit.module_ids:
                    if module_id not in self.Modules:
                        raise ValueError(f"Route {route_id} references unknown Module: {module_id}")

        for module_id, module in self.Modules.items():
            if module.load_lock is None:
                continue
            if module.type is not ModuleType.LL:
                raise ValueError(f"Module {module_id} configures load_lock but is not an LL module")
            for tm_id in module.load_lock.tm_required_states:
                if tm_id not in self.ClusterTool:
                    raise ValueError(f"Module {module_id} load_lock references unknown ClusterTool: {tm_id}")

        if self.cleaning is not None:
            for module_id in self.cleaning.module_ids:
                if module_id not in self.Modules:
                    raise ValueError(f"cleaning references unknown Module: {module_id}")
                if self.Modules[module_id].type is not ModuleType.PM:
                    raise ValueError(f"cleaning Module {module_id} must be a PM module")

        self._validate_initial_state()

        return self

    def _validate_initial_state(self) -> None:
        snapshot = self.initial_state.to_snapshot()

        for robot_id, position_module_id in snapshot.tm_positions.items():
            robot = self.ClusterTool.get(robot_id)
            if robot is None:
                raise ValueError(f"initial_state references unknown Robot: {robot_id}")
            if position_module_id is None:
                continue
            if position_module_id not in self.Modules:
                raise ValueError(
                    f"initial_state Robot {robot_id} references unknown position Module: "
                    f"{position_module_id}"
                )
            if position_module_id not in robot.module_ids:
                raise ValueError(
                    f"initial_state Robot {robot_id} cannot reach position Module: "
                    f"{position_module_id}"
                )

        for wafer in snapshot.wafers_by_key.values():
            route = self.routes.get(wafer.route_id)
            if route is None:
                raise ValueError(f"initial_state references unknown Route: {wafer.route_id}")

            if isinstance(wafer.location, ModuleLocation):
                module_id = wafer.location.module_id
                module = self.Modules.get(module_id)
                if module is None:
                    raise ValueError(f"initial_state references unknown Module: {module_id}")
                self._validate_initial_module_step(wafer, route, module_id, module.type)
                continue

            if isinstance(wafer.location, RobotLocation):
                robot_id = wafer.location.robot_id
                robot = self.ClusterTool.get(robot_id)
                if robot is None:
                    raise ValueError(f"initial_state references unknown Robot: {robot_id}")
                if wafer.process_end_time is not None and wafer.process_end_time > 0:
                    raise ValueError(
                        f"initial_state wafer {wafer.wafer_key!r} is on a Robot "
                        "but still has unfinished processing"
                    )
                if wafer.step_index > len(route.visits):
                    raise ValueError(
                        f"initial_state wafer {wafer.wafer_key!r} has invalid step_index: {wafer.step_index}"
                    )

        for module_id, occupants in snapshot.module_occupants.items():
            capacity = self.Modules[module_id].capacity
            if len(occupants) > capacity:
                raise ValueError(
                    f"initial_state Module {module_id} has {len(occupants)} wafers but capacity is {capacity}"
                )

        for robot_id, arms in snapshot.tm_arms.items():
            arm_capacity = 1 if self.ClusterTool[robot_id].arm_type is TMArmType.SINGLE_ARM else 2
            if len(arms) > arm_capacity:
                raise ValueError(
                    f"initial_state Robot {robot_id} holds {len(arms)} wafers but arm capacity is {arm_capacity}"
                )

    @staticmethod
    def _validate_initial_module_step(
        wafer: WaferInitialState,
        route: Route,
        module_id: str,
        module_type: ModuleType,
    ) -> None:
        if module_type is ModuleType.LP:
            valid_lp_steps = {0, len(route.visits) + 1}
            if wafer.step_index not in valid_lp_steps:
                raise ValueError(
                    f"initial_state wafer {wafer.wafer_key!r} in LP {module_id} has invalid step_index: "
                    f"{wafer.step_index}"
                )
            return

        if not 1 <= wafer.step_index <= len(route.visits):
            raise ValueError(
                f"initial_state wafer {wafer.wafer_key!r} has invalid step_index: {wafer.step_index}"
            )
        visit = route.visits[wafer.step_index - 1]
        if module_id not in visit.module_ids:
            raise ValueError(
                f"initial_state wafer {wafer.wafer_key!r} Module {module_id} is not allowed "
                f"at step {wafer.step_index}"
            )


ClusterTool = dict[str, ClusterCell]
