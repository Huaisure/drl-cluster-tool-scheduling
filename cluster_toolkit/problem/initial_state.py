from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_non_empty(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class ModuleLocation(_StrictModel):
    kind: Literal["module"] = "module"
    module_id: str

    @field_validator("module_id")
    @classmethod
    def _validate_module_id(cls, value: str) -> str:
        return _require_non_empty(value, "ModuleLocation.module_id")


class RobotLocation(_StrictModel):
    kind: Literal["robot"] = "robot"
    robot_id: str
    arm_id: str = "arm0"

    @field_validator("robot_id", "arm_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _require_non_empty(value, f"RobotLocation.{info.field_name}")


WaferLocation = Annotated[ModuleLocation | RobotLocation, Field(discriminator="kind")]
WaferKey = tuple[str, int]
_WAFER_INDEX_ITEM_PATTERN = re.compile(
    r"^(?P<start>[0-9]+)(?:\s*-\s*(?P<end>[0-9]+))?$"
)


def _parse_wafer_index_expression(value: object) -> tuple[int, ...]:
    if not isinstance(value, str):
        raise ValueError("wafer_index must be a string expression")

    expression = value.strip()
    if not expression:
        raise ValueError("wafer_index expression must not be empty")

    indexes: list[int] = []
    seen: set[int] = set()
    for raw_item in expression.split(","):
        item = raw_item.strip()
        match = _WAFER_INDEX_ITEM_PATTERN.fullmatch(item)
        if match is None:
            raise ValueError(f"invalid wafer_index item: {item!r}")

        start = int(match.group("start"))
        end_text = match.group("end")
        end = start if end_text is None else int(end_text)
        if end < start:
            raise ValueError(f"wafer_index range must be ascending: {item!r}")

        for wafer_index in range(start, end + 1):
            if wafer_index in seen:
                raise ValueError(
                    f"wafer_index expression contains duplicate index: {wafer_index}"
                )
            seen.add(wafer_index)
            indexes.append(wafer_index)

    return tuple(indexes)


class RobotInitialState(_StrictModel):
    position_module_id: str | None = None

    @field_validator("position_module_id")
    @classmethod
    def _validate_position_module_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty(value, "RobotInitialState.position_module_id")


class WaferInitialState(_StrictModel):
    route_id: str
    wafer_index: int
    priority: int
    step_index: int = 0
    location: WaferLocation
    process_end_time: float | None = None
    return_lp_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("return_lp_id", "return_module_id"),
    )

    @field_validator("route_id")
    @classmethod
    def _validate_route_id(cls, value: str) -> str:
        return _require_non_empty(value, "WaferInitialState.route_id")

    @field_validator("wafer_index", "priority", "step_index", mode="before")
    @classmethod
    def _validate_indexes(cls, value: object, info) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"WaferInitialState.{info.field_name} must be a non-negative integer")
        return value

    @field_validator("process_end_time", mode="before")
    @classmethod
    def _validate_process_end_time(cls, value: float | int | None) -> float | None:
        if value is None:
            return None
        process_end_time = float(value)
        if process_end_time < 0:
            raise ValueError("WaferInitialState.process_end_time must be non-negative")
        return process_end_time

    @field_validator("return_lp_id")
    @classmethod
    def _validate_return_lp_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_non_empty(value, "WaferInitialState.return_lp_id")

    @property
    def wafer_key(self) -> WaferKey:
        return self.route_id, self.wafer_index


@dataclass(frozen=True, slots=True)
class InitialSnapshot:
    """Read-only indexes derived from the initial-state facts."""

    wafers_by_key: Mapping[WaferKey, WaferInitialState]
    module_occupants: Mapping[str, frozenset[WaferKey]]
    tm_arms: Mapping[str, Mapping[str, WaferKey]]
    tm_positions: Mapping[str, str | None]


class InitialState(_StrictModel):
    wafers: tuple[WaferInitialState, ...] = Field(default_factory=tuple)
    robots: dict[str, RobotInitialState] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _expand_wafer_index_expressions(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value

        raw_wafers = value.get("wafers")
        if not isinstance(raw_wafers, (list, tuple)):
            return value

        expanded_wafers: list[object] = []
        for item_index, raw_wafer in enumerate(raw_wafers):
            if not isinstance(raw_wafer, dict) or "wafer_index" not in raw_wafer:
                expanded_wafers.append(raw_wafer)
                continue

            try:
                wafer_indexes = _parse_wafer_index_expression(
                    raw_wafer["wafer_index"]
                )
            except ValueError as exc:
                raise ValueError(
                    f"InitialState.wafers[{item_index}]: {exc}"
                ) from exc

            for wafer_index in wafer_indexes:
                expanded_wafer = dict(raw_wafer)
                expanded_wafer["wafer_index"] = wafer_index
                expanded_wafers.append(expanded_wafer)

        normalized = dict(value)
        normalized["wafers"] = expanded_wafers
        return normalized

    @field_validator("robots")
    @classmethod
    def _validate_robot_ids(
        cls,
        value: dict[str, RobotInitialState],
    ) -> dict[str, RobotInitialState]:
        for robot_id in value:
            _require_non_empty(robot_id, "InitialState.robots key")
        return value

    @model_validator(mode="after")
    def _validate_unique_placements(self) -> "InitialState":
        wafer_keys = [wafer.wafer_key for wafer in self.wafers]
        if len(set(wafer_keys)) != len(wafer_keys):
            raise ValueError("InitialState.wafers must not contain duplicate wafer identities")

        occupied_arms: set[tuple[str, str]] = set()
        for wafer in self.wafers:
            if not isinstance(wafer.location, RobotLocation):
                continue
            arm_key = (wafer.location.robot_id, wafer.location.arm_id)
            if arm_key in occupied_arms:
                raise ValueError(
                    f"initial_state Robot {wafer.location.robot_id} arm "
                    f"{wafer.location.arm_id} holds multiple wafers"
                )
            occupied_arms.add(arm_key)
        return self

    def to_snapshot(self) -> InitialSnapshot:
        """Project the declared facts into immutable validator-facing indexes."""

        wafers_by_key: dict[WaferKey, WaferInitialState] = {}
        module_occupants: dict[str, set[WaferKey]] = defaultdict(set)
        tm_arms: dict[str, dict[str, WaferKey]] = defaultdict(dict)

        for wafer in self.wafers:
            wafers_by_key[wafer.wafer_key] = wafer
            if isinstance(wafer.location, ModuleLocation):
                module_occupants[wafer.location.module_id].add(wafer.wafer_key)
            else:
                tm_arms[wafer.location.robot_id][wafer.location.arm_id] = wafer.wafer_key

        return InitialSnapshot(
            wafers_by_key=MappingProxyType(wafers_by_key),
            module_occupants=MappingProxyType(
                {
                    module_id: frozenset(occupants)
                    for module_id, occupants in module_occupants.items()
                }
            ),
            tm_arms=MappingProxyType(
                {
                    tm_id: MappingProxyType(dict(arms))
                    for tm_id, arms in tm_arms.items()
                }
            ),
            tm_positions=MappingProxyType(
                {
                    tm_id: state.position_module_id
                    for tm_id, state in self.robots.items()
                }
            ),
        )
