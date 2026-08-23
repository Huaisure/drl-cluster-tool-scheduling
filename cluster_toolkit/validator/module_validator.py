from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from cluster_toolkit.problem import Module

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
        # At the same time, Pick.end releases capacity before Place.start uses it.
        priority = 1 if self.occupies else 0
        return self.time, priority, self.action.index


class ModuleValidator:
    """Validate the time-ordered actions of one concrete module."""

    def __init__(
        self,
        module_id: str,
        config: Module,
        actions: Sequence[ActionRecord],
        initial_occupants: Iterable[WaferKey] = (),
        check_occupant_identity: bool = False,
    ) -> None:
        self.module_id = module_id
        self.config = config
        self.actions = tuple(sorted(actions, key=lambda action: action.sort_key))
        self.initial_occupants = frozenset(initial_occupants)
        self.check_occupant_identity = check_occupant_identity

    @property
    def capacity(self) -> int:
        return self.config.capacity

    def validate(self) -> ValidationReport:
        """Validate this module's wafer capacity."""

        report = ValidationReport(checked_subjects={"module": 1})
        occupants = set(self.initial_occupants)

        for event in self._capacity_events():
            if not event.occupies:
                if event.wafer_key not in occupants:
                    if self.check_occupant_identity:
                        report.issues.append(
                            ValidationIssue(
                                constraint_id="module.occupant_identity",
                                subject_kind="module",
                                subject_id=self.module_id,
                                message=(
                                    f"Module {self.module_id} cannot release wafer "
                                    f"{event.wafer_key!r}; it is not an occupant"
                                ),
                                action_index=event.action.index,
                                context={
                                    "time": event.time,
                                    "occupants_before": sorted(occupants),
                                    "picked_wafer": event.wafer_key,
                                },
                            )
                        )
                    continue
                occupants.remove(event.wafer_key)
                continue

            if event.wafer_key not in occupants and len(occupants) >= self.capacity:
                report.issues.append(
                    ValidationIssue(
                        constraint_id="module.capacity",
                        subject_kind="module",
                        subject_id=self.module_id,
                        message=(
                            f"Module {self.module_id} capacity {self.capacity} "
                            f"is exceeded at time {event.time}"
                        ),
                        action_index=event.action.index,
                        context={
                            "time": event.time,
                            "capacity": self.capacity,
                            "occupancy_before": len(occupants),
                            "occupants_before": sorted(occupants),
                            "placing_wafer": event.wafer_key,
                        },
                    )
                )
            occupants.add(event.wafer_key)

        return report

    def _capacity_events(self) -> list[_CapacityEvent]:
        events: list[_CapacityEvent] = []

        for action in self.actions:
            if action.action_type not in {PICK, PLACE}:
                continue

            if action.wafer_key is None:
                continue

            events.append(
                _CapacityEvent(
                    time=action.start if action.action_type == PLACE else action.end,
                    occupies=action.action_type == PLACE,
                    action=action,
                    wafer_key=action.wafer_key,
                )
            )

        events.sort(key=lambda event: event.sort_key)
        return events
