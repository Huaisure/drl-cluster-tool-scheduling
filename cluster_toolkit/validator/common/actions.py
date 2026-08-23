from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..models import PICK, PLACE, ActionRecord, TimeValue, WaferKey


SubjectSelector = Callable[[ActionRecord], Iterable[Hashable]]


@dataclass(frozen=True, slots=True)
class _CapacityEvent:
    time: TimeValue
    occupies: bool
    action: ActionRecord
    wafer_key: WaferKey


def parse_actions(actions: Sequence[Mapping[str, Any]]) -> tuple[ActionRecord, ...]:
    """Normalize every input action exactly once, preserving input order."""

    return tuple(
        ActionRecord.from_mapping(index, raw_action)
        for index, raw_action in enumerate(actions)
    )


def group_actions(
    actions: Sequence[ActionRecord],
    subject_selector: SubjectSelector,
) -> dict[Hashable, list[ActionRecord]]:
    """Group actions by subject and sort each subject's actions by start time."""

    grouped: dict[Hashable, list[ActionRecord]] = defaultdict(list)
    for action in actions:
        for subject_id in subject_selector(action):
            grouped[subject_id].append(action)

    for subject_actions in grouped.values():
        subject_actions.sort(key=lambda action: action.sort_key)
    return dict(grouped)


def capacity_events(
    actions: Sequence[ActionRecord],
    occupying_action_type: str,
) -> list[_CapacityEvent]:
    """Return capacity changes with releases ordered before claims."""

    events = [
        _CapacityEvent(
            time=(
                action.start
                if action.action_type == occupying_action_type
                else action.end
            ),
            occupies=action.action_type == occupying_action_type,
            action=action,
            wafer_key=action.wafer_key,
        )
        for action in actions
        if action.action_type in {PICK, PLACE} and action.wafer_key is not None
    ]
    return sorted(
        events,
        key=lambda event: (event.time, event.occupies, event.action.index),
    )
