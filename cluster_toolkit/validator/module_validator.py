from __future__ import annotations

from typing import Iterable, Sequence

from cluster_toolkit.problem import Module

from .common.actions import capacity_events
from .models import (
    PLACE,
    ActionRecord,
    ValidationIssue,
    ValidationReport,
    WaferKey,
)


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

        for event in capacity_events(self.actions, PLACE):
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
