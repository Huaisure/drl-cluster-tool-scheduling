from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cluster_toolkit.problem import WaferKey


Side = Literal["atmosphere", "vacuum"]


@dataclass(slots=True)
class WaferState:
    route_id: str
    wafer_index: int
    step_index: int
    module_id: str | None
    robot_id: str | None
    ready_at: float
    return_module_id: str
    last_place_robot_id: str | None = None

    @property
    def wafer_key(self) -> WaferKey:
        return self.route_id, self.wafer_index


@dataclass(slots=True)
class RobotState:
    module_id: str | None
    ready_at: float = 0.0
    holding: list[WaferKey] = field(default_factory=list)


@dataclass(slots=True)
class LoadLockRuntimeState:
    """Minimal lazy state for one conversion Load Lock."""

    last_pick_side: Side
    last_pick_end: float
    occupied_exit_side: Side | None = None
    occupied_ready_at: float = 0.0
    occupied_transition_start: float = 0.0
    occupied_transition_duration: float = 0.0


@dataclass(frozen=True, slots=True)
class LoadLockObservation:
    """Immutable, relative-time view of one conversion Load Lock."""

    pump_time: float
    vent_time: float
    last_pick_side: Side
    empty_transition_progress: float
    occupied_exit_side: Side | None
    occupied_transition_progress: float


@dataclass(slots=True)
class PendingOperation:
    action_type: Literal["pick", "place"]
    robot_id: str
    wafer_key: WaferKey
    module_id: str
    start: float
    end: float
    started: bool = False


@dataclass(slots=True)
class ClusterState:
    time: float
    wafers: dict[WaferKey, WaferState]
    robots: dict[str, RobotState]
    module_occupants: dict[str, set[WaferKey]]
    load_locks: dict[str, LoadLockRuntimeState]
    pending_operations: list[PendingOperation] = field(default_factory=list)
