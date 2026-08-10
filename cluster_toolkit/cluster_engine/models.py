from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from cluster_toolkit.problem import WaferKey


@dataclass(frozen=True, slots=True)
class PickAction:
    """Dispatch one Robot to pick one wafer from its current Module."""

    robot_id: str
    wafer_key: WaferKey


@dataclass(frozen=True, slots=True)
class PlaceAction:
    """Place one held wafer into one allowed target Module."""

    wafer_key: WaferKey
    target_module_id: str


@dataclass(frozen=True, slots=True)
class AdvanceAction:
    """Advance the runtime clock to the next event boundary."""


ADVANCE = AdvanceAction()
EngineAction: TypeAlias = PickAction | PlaceAction | AdvanceAction


@dataclass(frozen=True, slots=True)
class DispatchRecord:
    """One externally meaningful Pick or Place scheduled by the Engine."""

    action_type: Literal["pick", "place"]
    robot_id: str
    module_id: str
    wafer_key: WaferKey
    step_index: int
    start: float
    end: float

    def to_dict(self) -> dict[str, object]:
        return {
            "action_type": self.action_type,
            "tm_id": self.robot_id,
            "module_id": self.module_id,
            "route_id": self.wafer_key[0],
            "wafer_index": self.wafer_key[1],
            "step_index": self.step_index,
            "start": self.start,
            "end": self.end,
        }


class IllegalActionError(ValueError):
    """Raised when an action is absent from the current available-action set."""
