from __future__ import annotations

import pytest

from cluster_toolkit.problem import Module
from cluster_toolkit.validator import ActionRecord, ModuleValidator


def _action(
    index: int,
    action_type: str,
    wafer_index: int,
    start: float,
    end: float,
) -> ActionRecord:
    return ActionRecord.from_mapping(
        index,
        {
            "action_type": action_type,
            "module_id": "PM1",
            "route_id": "A",
            "wafer_index": wafer_index,
            "start": start,
            "end": end,
        },
    )


def _validator(
    actions: list[ActionRecord],
    *,
    module_type: str = "PM",
    initial_occupants: set[tuple[str, int]] | None = None,
) -> ModuleValidator:
    return ModuleValidator(
        module_id="PM1",
        config=Module.model_validate({"type": module_type}),
        actions=actions,
        initial_occupants=initial_occupants or set(),
    )


def test_non_lp_module_capacity_is_one() -> None:
    validator = _validator([])

    assert validator.capacity == 1
    assert validator.config.capacity == 1


def test_place_before_pick_end_exceeds_capacity() -> None:
    validator = _validator(
        [
            _action(0, "unload", wafer_index=0, start=0, end=1),
            _action(1, "load", wafer_index=1, start=0.5, end=1.5),
        ],
        initial_occupants={("A", 0)},
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == ["module.capacity"]
    assert report.issues[0].action_index == 1
    assert report.issues[0].context["occupancy_before"] == 1
    assert report.issues[0].context["placing_wafer"] == ("A", 1)


def test_pick_end_releases_capacity_before_place_at_the_same_time() -> None:
    validator = _validator(
        [
            _action(0, "unload", wafer_index=0, start=0, end=1),
            _action(1, "load", wafer_index=1, start=1, end=2),
        ],
        initial_occupants={("A", 0)},
    )

    assert validator.validate().ok


def test_place_uses_capacity_from_its_start_time() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=1),
            _action(1, "load", wafer_index=1, start=0.5, end=1.5),
        ]
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == ["module.capacity"]
    assert report.issues[0].action_index == 1


def test_lp_module_capacity_is_twenty_five_without_a_subclass() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=1),
            _action(1, "load", wafer_index=1, start=0.5, end=1.5),
        ],
        module_type="LP",
    )

    assert validator.capacity == 25
    assert validator.validate().ok


def test_explicit_module_capacity_override_is_used() -> None:
    with pytest.warns(
        UserWarning,
        match="Explicit Module.capacity overrides the type-based default",
    ):
        config = Module.model_validate({"type": "PM", "capacity": 2})
    validator = ModuleValidator(
        module_id="PM1",
        config=config,
        actions=[
            _action(0, "place", wafer_index=0, start=0, end=1),
            _action(1, "place", wafer_index=1, start=0.5, end=1.5),
        ],
    )

    assert validator.capacity == 2
    assert validator.validate().ok


def test_validate_does_not_mutate_initial_occupants() -> None:
    validator = _validator(
        [_action(0, "unload", wafer_index=0, start=0, end=1)],
        initial_occupants={("A", 0)},
    )

    validator.validate()

    assert validator.initial_occupants == frozenset({("A", 0)})
