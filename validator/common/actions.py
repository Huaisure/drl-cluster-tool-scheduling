from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any

from ..models import ActionRecord


SubjectSelector = Callable[[ActionRecord], Iterable[Hashable]]


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
