from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from cluster_toolkit.problem import ClusterCell, TMArmType

from .common import Interval, intervals_overlap
from .models import (
    PICK,
    PLACE,
    ActionRecord,
    TimeValue,
    ValidationIssue,
    ValidationReport,
    WaferKey,
)


@dataclass(frozen=True, slots=True)
class _CapacityEvent:
    time: TimeValue
    occupies: bool
    action: ActionRecord
    wafer_key: WaferKey

    @property
    def sort_key(self) -> tuple[TimeValue, int, int]:
        # At the same time, Place.end releases capacity before Pick.start uses it.
        priority = 1 if self.occupies else 0
        return self.time, priority, self.action.index


class RobotValidator:
    """Validate the time-ordered actions of one concrete robot."""

    def __init__(
        self,
        robot_id: str,
        config: ClusterCell,
        actions: Sequence[ActionRecord],
        initial_position_module_id: str | None = None,
        initial_arms: Mapping[str, WaferKey] | None = None,
        exact_action_durations: bool = False,
    ) -> None:
        self.robot_id = robot_id
        self.config = config
        self.actions = tuple(sorted(actions, key=lambda action: action.sort_key))
        self.initial_position_module_id = initial_position_module_id
        self.initial_arms = MappingProxyType(
            dict(initial_arms or {}),
        )
        self.exact_action_durations = exact_action_durations

    @property
    def capacity(self) -> int:
        if self.config.arm_type is TMArmType.SINGLE_ARM:
            return 1
        return 2

    def validate(self) -> ValidationReport:
        """Validate action mutual exclusion and total wafer capacity."""

        report = ValidationReport(checked_subjects={"robot": 1})
        report.issues.extend(self._validate_reachability())
        report.issues.extend(self._validate_action_overlaps())
        report.issues.extend(self._validate_action_durations())
        report.issues.extend(self._validate_movement_times())
        report.issues.extend(self._validate_capacity())
        return report

    def _validate_reachability(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        reachable = set(self.config.module_ids)
        for action in self.actions:
            if action.module_id is None or action.module_id in reachable:
                continue
            issues.append(
                ValidationIssue(
                    constraint_id="robot.reachability",
                    subject_kind="robot",
                    subject_id=self.robot_id,
                    message=(
                        f"Robot {self.robot_id} cannot reach Module {action.module_id}"
                    ),
                    action_index=action.index,
                    context={"module_id": action.module_id},
                )
            )
        return issues

    def _validate_action_overlaps(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        for index, left in enumerate(self.actions):
            for right in self.actions[index + 1 :]:
                if right.start >= left.end:
                    break
                if intervals_overlap(
                    Interval(left.start, left.end),
                    Interval(right.start, right.end),
                ):
                    issues.append(
                        ValidationIssue(
                            constraint_id="robot.action_overlap",
                            subject_kind="robot",
                            subject_id=self.robot_id,
                            message=(
                                f"Robot {self.robot_id} action {left.index} "
                                f"[{left.start}, {left.end}) overlaps action "
                                f"{right.index} [{right.start}, {right.end})"
                            ),
                            action_index=right.index,
                            context={
                                "left_action_index": left.index,
                                "left_interval": (left.start, left.end),
                                "right_action_index": right.index,
                                "right_interval": (right.start, right.end),
                            },
                        )
                    )

        return issues

    def _validate_action_durations(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        for action in self.actions:
            if action.action_type == PLACE:
                required_time = self.config.place_time
            elif action.action_type == PICK:
                required_time = self.config.pick_time
            else:
                continue

            actual_time = action.end - action.start
            invalid = (
                actual_time != required_time
                if self.exact_action_durations
                else actual_time < required_time
            )
            if invalid:
                issues.append(
                    ValidationIssue(
                        constraint_id="robot.action_duration",
                        subject_kind="robot",
                        subject_id=self.robot_id,
                        message=(
                            f"Robot {self.robot_id} {action.action_type} action needs "
                            f"{'exactly' if self.exact_action_durations else 'at least'} "
                            f"{required_time} time, but {actual_time} is provided"
                        ),
                        action_index=action.index,
                        context={
                            "action_type": action.action_type,
                            "actual_time": actual_time,
                            "required_time": required_time,
                        },
                    )
                )

        return issues

    def _validate_movement_times(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        previous_module_id = self.initial_position_module_id
        previous_end: TimeValue = 0
        previous_action_index: int | None = None

        for action in self.actions:
            module_id = action.module_id
            if not module_id:
                continue

            if previous_module_id is not None and module_id != previous_module_id:
                required_time = self.config.travel_time(previous_module_id, module_id)
                available_time = action.start - previous_end
                if available_time < required_time:
                    issues.append(
                        ValidationIssue(
                            constraint_id="robot.movement_time",
                            subject_kind="robot",
                            subject_id=self.robot_id,
                            message=(
                                f"Robot {self.robot_id} needs {required_time} time "
                                f"to move from {previous_module_id} to {module_id}, "
                                f"but only {available_time} is available"
                            ),
                            action_index=action.index,
                            context={
                                "from_module_id": previous_module_id,
                                "to_module_id": module_id,
                                "previous_action_index": previous_action_index,
                                "next_action_index": action.index,
                                "available_time": available_time,
                                "required_time": required_time,
                            },
                        )
                    )

            previous_module_id = module_id
            previous_end = action.end
            previous_action_index = action.index

        return issues

    def _validate_capacity(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        held_wafers = set(self.initial_arms.values())

        for event in self._capacity_events():
            if not event.occupies:
                held_wafers.discard(event.wafer_key)
                continue

            if event.wafer_key not in held_wafers and len(held_wafers) >= self.capacity:
                issues.append(
                    ValidationIssue(
                        constraint_id="robot.capacity",
                        subject_kind="robot",
                        subject_id=self.robot_id,
                        message=(
                            f"Robot {self.robot_id} capacity {self.capacity} "
                            f"is exceeded at time {event.time}"
                        ),
                        action_index=event.action.index,
                        context={
                            "time": event.time,
                            "capacity": self.capacity,
                            "occupancy_before": len(held_wafers),
                            "held_wafers_before": sorted(held_wafers),
                            "picking_wafer": event.wafer_key,
                        },
                    )
                )
            held_wafers.add(event.wafer_key)

        return issues

    def _capacity_events(self) -> list[_CapacityEvent]:
        events: list[_CapacityEvent] = []

        for action in self.actions:
            if action.action_type not in {PICK, PLACE}:
                continue

            if action.wafer_key is None:
                continue

            events.append(
                _CapacityEvent(
                    time=action.start if action.action_type == PICK else action.end,
                    occupies=action.action_type == PICK,
                    action=action,
                    wafer_key=action.wafer_key,
                )
            )

        events.sort(key=lambda event: event.sort_key)
        return events
