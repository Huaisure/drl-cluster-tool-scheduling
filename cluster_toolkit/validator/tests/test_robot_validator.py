from __future__ import annotations

from cluster_toolkit.problem import ClusterCell
from cluster_toolkit.validator import ActionRecord, RobotValidator


def _action(
    index: int,
    action_type: str,
    wafer_index: int,
    start: float,
    end: float,
    module_id: str = "PM1",
) -> ActionRecord:
    return ActionRecord.from_mapping(
        index,
        {
            "action_type": action_type,
            "tm_id": "TM1",
            "module_id": module_id,
            "route_id": "A",
            "wafer_index": wafer_index,
            "start": start,
            "end": end,
        },
    )


def _validator(
    actions: list[ActionRecord],
    *,
    dual_arm: bool = False,
    initial_arms: dict[str, tuple[str, int]] | None = None,
    initial_position_module_id: str | None = None,
    travel_time: float = 0,
    load_time: float = 1,
    unload_time: float = 1,
) -> RobotValidator:
    config = ClusterCell.model_validate(
        {
            "module_ids": ["LP", "PM1", "PM2"],
            "arm_type": "dual_arm" if dual_arm else "single_arm",
            "travel_times": travel_time,
            "load_time": load_time,
            "unload_time": unload_time,
        }
    )
    return RobotValidator(
        robot_id="TM1",
        config=config,
        actions=actions,
        initial_position_module_id=initial_position_module_id,
        initial_arms=initial_arms,
    )


def test_single_and_dual_arm_capacity() -> None:
    assert _validator([]).capacity == 1
    assert _validator([], dual_arm=True).capacity == 2


def test_robot_actions_must_not_overlap() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=2),
            _action(1, "load", wafer_index=1, start=1, end=3),
        ],
        dual_arm=True,
        initial_arms={"arm0": ("A", 0), "arm1": ("A", 1)},
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == ["robot.action_overlap"]
    assert report.issues[0].action_index == 1
    assert report.issues[0].context["left_interval"] == (0, 2)
    assert report.issues[0].context["right_interval"] == (1, 3)


def test_touching_robot_actions_do_not_overlap() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=1),
            _action(1, "unload", wafer_index=1, start=1, end=2),
        ],
        initial_arms={"arm0": ("A", 0)},
    )

    assert validator.validate().ok


def test_place_end_releases_arm_before_pick_at_the_same_time() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=1),
            _action(1, "unload", wafer_index=1, start=1, end=2),
        ],
        initial_arms={"arm0": ("A", 0)},
    )

    assert validator.validate().ok


def test_single_arm_rejects_second_held_wafer() -> None:
    validator = _validator(
        [
            _action(0, "unload", wafer_index=0, start=0, end=1),
            _action(1, "unload", wafer_index=1, start=2, end=3),
        ]
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == ["robot.capacity"]
    assert report.issues[0].action_index == 1
    assert report.issues[0].context["occupancy_before"] == 1
    assert report.issues[0].context["picking_wafer"] == ("A", 1)


def test_initial_wafer_already_uses_robot_capacity() -> None:
    validator = _validator(
        [_action(0, "unload", wafer_index=1, start=0, end=1)],
        initial_arms={"arm0": ("A", 0)},
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == ["robot.capacity"]


def test_dual_arm_allows_two_held_wafers() -> None:
    validator = _validator(
        [
            _action(0, "unload", wafer_index=0, start=0, end=1),
            _action(1, "unload", wafer_index=1, start=2, end=3),
        ],
        dual_arm=True,
    )

    assert validator.validate().ok


def test_dual_arm_rejects_third_held_wafer() -> None:
    validator = _validator(
        [
            _action(0, "unload", wafer_index=0, start=0, end=1),
            _action(1, "unload", wafer_index=1, start=2, end=3),
            _action(2, "unload", wafer_index=2, start=4, end=5),
        ],
        dual_arm=True,
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == ["robot.capacity"]
    assert report.issues[0].action_index == 2


def test_validate_does_not_mutate_initial_arms() -> None:
    validator = _validator(
        [_action(0, "load", wafer_index=0, start=0, end=1)],
        initial_arms={"arm0": ("A", 0)},
    )

    validator.validate()

    assert validator.initial_arms == {"arm0": ("A", 0)}


def test_different_modules_require_movement_time() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=1, module_id="PM1"),
            _action(1, "unload", wafer_index=1, start=4, end=5, module_id="PM2"),
        ],
        initial_arms={"arm0": ("A", 0)},
        travel_time=5,
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == ["robot.movement_time"]
    assert report.issues[0].action_index == 1
    assert report.issues[0].context["available_time"] == 3
    assert report.issues[0].context["required_time"] == 5


def test_exact_movement_time_is_allowed() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=1, module_id="PM1"),
            _action(1, "unload", wafer_index=1, start=6, end=7, module_id="PM2"),
        ],
        initial_arms={"arm0": ("A", 0)},
        travel_time=5,
    )

    assert validator.validate().ok


def test_same_module_does_not_require_movement_time() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=1),
            _action(1, "unload", wafer_index=1, start=1, end=2),
        ],
        initial_arms={"arm0": ("A", 0)},
        travel_time=5,
    )

    assert validator.validate().ok


def test_explicit_initial_position_is_checked_before_first_action() -> None:
    validator = _validator(
        [_action(0, "unload", wafer_index=0, start=4, end=5, module_id="PM1")],
        initial_position_module_id="LP",
        travel_time=5,
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == ["robot.movement_time"]
    assert report.issues[0].context["from_module_id"] == "LP"
    assert report.issues[0].context["previous_action_index"] is None
    assert report.issues[0].context["available_time"] == 4


def test_same_initial_position_needs_no_movement_time() -> None:
    validator = _validator(
        [_action(0, "unload", wafer_index=0, start=0, end=1, module_id="PM1")],
        initial_position_module_id="PM1",
        travel_time=5,
    )

    assert validator.validate().ok


def test_exact_initial_movement_time_is_allowed() -> None:
    validator = _validator(
        [_action(0, "unload", wafer_index=0, start=5, end=6, module_id="PM1")],
        initial_position_module_id="LP",
        travel_time=5,
    )

    assert validator.validate().ok


def test_anywhere_skips_only_the_first_movement() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=1, module_id="PM1"),
            _action(1, "unload", wafer_index=1, start=2, end=3, module_id="PM2"),
        ],
        initial_arms={"arm0": ("A", 0)},
        travel_time=5,
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == ["robot.movement_time"]
    assert report.issues[0].action_index == 1


def test_load_and_unload_require_their_own_minimum_duration() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=1),
            _action(1, "unload", wafer_index=1, start=2, end=4),
        ],
        load_time=2,
        unload_time=3,
    )

    report = validator.validate()

    assert [issue.constraint_id for issue in report.issues] == [
        "robot.action_duration",
        "robot.action_duration",
    ]
    assert report.issues[0].context == {
        "action_type": "place",
        "actual_time": 1,
        "required_time": 2.0,
    }
    assert report.issues[1].context == {
        "action_type": "pick",
        "actual_time": 2,
        "required_time": 3.0,
    }


def test_action_duration_equal_to_required_time_is_allowed() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=2),
            _action(1, "unload", wafer_index=1, start=2, end=5),
        ],
        load_time=2,
        unload_time=3,
    )

    assert validator.validate().ok


def test_action_duration_may_exceed_required_time() -> None:
    validator = _validator(
        [
            _action(0, "load", wafer_index=0, start=0, end=3),
            _action(1, "unload", wafer_index=1, start=3, end=7),
        ],
        load_time=2,
        unload_time=3,
    )

    assert validator.validate().ok
