from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping

from problem import WaferKey


TimeValue = int | float
PICK = "pick"
PLACE = "place"

_ACTION_TYPE_ALIASES = {
    "load": PLACE,
    "place": PLACE,
    "pick": PICK,
    "unload": PICK,
}


def normalize_action_type(value: object) -> str:
    """Normalize external action names to the validator's canonical vocabulary."""

    action_type = str(value).strip().lower()
    return _ACTION_TYPE_ALIASES.get(action_type, action_type)


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One immutable action normalized once at the validation boundary."""

    index: int
    action_type: str
    start: TimeValue
    end: TimeValue
    module_id: str | None
    tm_id: str | None
    wafer_key: WaferKey | None
    step_index: Any
    arm_id: str | None

    @property
    def sort_key(self) -> tuple[TimeValue, TimeValue, int]:
        return self.start, self.end, self.index

    @classmethod
    def from_mapping(cls, index: int, action: Mapping[str, Any]) -> "ActionRecord":
        module_id = action.get("module_id", action.get("pm_id"))
        tm_id = action.get("tm_id")
        route_id = action.get("route_id")
        raw_wafer_index = action.get("wafer_index", action.get("count"))
        wafer_key = (
            None
            if route_id is None or raw_wafer_index is None
            else (str(route_id), int(raw_wafer_index))
        )

        return cls(
            index=index,
            action_type=normalize_action_type(action.get("action_type", "")),
            start=action["start"],
            end=action["end"],
            module_id=None if module_id is None else str(module_id),
            tm_id=None if tm_id is None else str(tm_id),
            wafer_key=wafer_key,
            step_index=action.get("step_index"),
            arm_id=None if action.get("arm_id") is None else str(action["arm_id"]),
        )


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    constraint_id: str
    subject_kind: str
    subject_id: Hashable
    message: str
    action_index: int | None = None
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)
    checked_subjects: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    def extend(self, other: "ValidationReport") -> None:
        self.issues.extend(other.issues)
        for kind, count in other.checked_subjects.items():
            self.checked_subjects[kind] = self.checked_subjects.get(kind, 0) + count
