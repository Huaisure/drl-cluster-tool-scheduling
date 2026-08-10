from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cluster_toolkit.problem import (
    JustInTimeConstraint,
    ModuleLocation,
    RobotLocation,
    Route,
    WaferInitialState,
)

from .common import Interval, intervals_overlap
from .models import PICK, PLACE, ActionRecord, ValidationIssue, ValidationReport, WaferKey


@dataclass(frozen=True, slots=True)
class _WaferInterval:
    """One interval during which the wafer cannot do anything else."""

    kind: str
    start: float
    end: float
    action_index: int | None


class WaferValidator:
    """Validate the time-ordered actions of one concrete wafer."""

    def __init__(
        self,
        wafer_key: WaferKey,
        route: Route,
        just_in_time: JustInTimeConstraint | None,
        actions: Sequence[ActionRecord],
        initial_wafer: WaferInitialState,
        return_module_id: str | None,
        source_module_ids: frozenset[str],
        pm_module_ids: frozenset[str],
    ) -> None:
        self.wafer_key = wafer_key
        self.route = route
        self.just_in_time = just_in_time
        self.source_module_ids = source_module_ids
        self.pm_module_ids = pm_module_ids
        self.actions = tuple(sorted(actions, key=lambda action: action.sort_key))
        self.initial_wafer = initial_wafer
        self.return_module_id = return_module_id

    def validate(self) -> ValidationReport:
        """Validate route order and mutually exclusive wafer intervals."""

        report = ValidationReport(checked_subjects={"wafer": 1})
        order_issue = self._validate_process_order()
        if order_issue is not None:
            report.issues.append(order_issue)
        report.issues.extend(self._validate_interval_overlaps())
        return report

    def _validate_process_order(self) -> ValidationIssue | None:
        step_index = self.initial_wafer.step_index
        location = self.initial_wafer.location

        for action in self._transfer_actions():
            action_type = action.action_type
            action_step = action.step_index
            module_id = action.module_id
            robot_id = action.tm_id

            if not isinstance(action_step, int) or isinstance(action_step, bool):
                return self._order_issue(action, "step_index must be an integer")
            if not isinstance(module_id, str) or not module_id:
                return self._order_issue(action, "module_id is required")
            if not isinstance(robot_id, str) or not robot_id:
                return self._order_issue(action, "tm_id is required")

            if isinstance(location, ModuleLocation):
                if step_index == len(self.route.visits) + 1:
                    return self._order_issue(action, "the wafer has already completed its route")
                if action_type != PICK:
                    return self._order_issue(
                        action,
                        f"expected Pick from {location.module_id}, got {action_type}",
                    )
                if action_step != step_index or module_id != location.module_id:
                    return self._order_issue(
                        action,
                        f"expected step {step_index} at {location.module_id}",
                    )
                location = RobotLocation(
                    robot_id=robot_id,
                    arm_id=action.arm_id or "arm0",
                )
                continue

            next_step = step_index + 1
            if action_type != PLACE:
                return self._order_issue(
                    action,
                    f"expected Place for step {next_step}, got {action_type}",
                )
            if robot_id != location.robot_id:
                return self._order_issue(
                    action,
                    f"wafer is on Robot {location.robot_id}, not {robot_id}",
                )
            if action_step != next_step:
                return self._order_issue(action, f"expected step {next_step}")
            if next_step <= len(self.route.visits):
                allowed_modules = self.route.visits[next_step - 1].module_ids
                if module_id not in allowed_modules:
                    expected = ", ".join(allowed_modules)
                    return self._order_issue(
                        action,
                        f"step {next_step} must use one of: {expected}",
                    )
            elif next_step == len(self.route.visits) + 1:
                if (
                    self.return_module_id is not None
                    and module_id != self.return_module_id
                ):
                    return self._order_issue(
                        action,
                        "the completed wafer must return to "
                        f"source module {self.return_module_id}",
                    )
                if module_id not in self.source_module_ids:
                    return self._order_issue(
                        action,
                        "the completed wafer must return to a source module",
                    )
            else:
                return self._order_issue(action, "step_index exceeds the route")

            step_index = next_step
            location = ModuleLocation(module_id=module_id)

        return None

    def _validate_interval_overlaps(self) -> list[ValidationIssue]:
        intervals = self._wafer_intervals()
        issues: list[ValidationIssue] = []

        for index, left in enumerate(intervals):
            for right in intervals[index + 1 :]:
                if right.start >= left.end:
                    break
                if intervals_overlap(
                    Interval(left.start, left.end),
                    Interval(right.start, right.end),
                ):
                    issues.append(
                        ValidationIssue(
                            constraint_id="wafer.interval_overlap",
                            subject_kind="wafer",
                            subject_id=self.wafer_key,
                            message=(
                                f"{left.kind} [{left.start}, {left.end}) overlaps "
                                f"{right.kind} [{right.start}, {right.end})"
                            ),
                            action_index=right.action_index,
                            context={
                                "left": self._interval_context(left),
                                "right": self._interval_context(right),
                            },
                        )
                    )
        return issues

    def _wafer_intervals(self) -> list[_WaferInterval]:
        intervals: list[_WaferInterval] = []

        if (
            isinstance(self.initial_wafer.location, ModuleLocation)
            and self.initial_wafer.location.module_id in self.pm_module_ids
            and self.initial_wafer.process_end_time is not None
            and self.initial_wafer.process_end_time > 0
        ):
            intervals.append(
                _WaferInterval(
                    kind="initial process",
                    start=0.0,
                    end=float(self.initial_wafer.process_end_time),
                    action_index=None,
                )
            )

        for action in self._transfer_actions():
            action_type = action.action_type
            if action.end > action.start:
                intervals.append(
                    _WaferInterval(
                        kind="Pick" if action_type == PICK else "Place",
                        start=float(action.start),
                        end=float(action.end),
                        action_index=action.index,
                    )
                )

            if action_type != PLACE:
                continue

            step_index = action.step_index
            module_id = action.module_id
            if (
                not isinstance(step_index, int)
                or isinstance(step_index, bool)
                or not 1 <= step_index <= len(self.route.visits)
            ):
                continue

            visit = self.route.visits[step_index - 1]
            if (
                module_id not in visit.module_ids
                or module_id not in self.pm_module_ids
                or not visit.process_time
            ):
                continue

            process_start = float(action.end)
            process_end = process_start + visit.process_time
            intervals.append(
                _WaferInterval(
                    kind=f"Process step {step_index}",
                    start=process_start,
                    end=process_end,
                    action_index=action.index,
                )
            )

        intervals.sort(key=lambda interval: (interval.start, interval.end))
        return intervals

    def _transfer_actions(self) -> tuple[ActionRecord, ...]:
        return tuple(
            action
            for action in self.actions
            if action.action_type in {PICK, PLACE}
        )

    def _order_issue(self, action: ActionRecord, detail: str) -> ValidationIssue:
        return ValidationIssue(
            constraint_id="wafer.process_order",
            subject_kind="wafer",
            subject_id=self.wafer_key,
            message=f"Invalid wafer process order: {detail}",
            action_index=action.index,
            context={
                "action_type": action.action_type,
                "module_id": action.module_id,
                "step_index": action.step_index,
            },
        )

    @staticmethod
    def _interval_context(interval: _WaferInterval) -> dict[str, object]:
        return {
            "kind": interval.kind,
            "start": interval.start,
            "end": interval.end,
            "action_index": interval.action_index,
        }
