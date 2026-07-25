from __future__ import annotations

import json
from pathlib import Path

import pytest

from problem import InitialState, load_problem, parse_problem
from validator import ActionRecord, ModuleValidator, ValidatorSuite
from validator.common import (
    Interval,
    group_actions,
    intervals_overlap,
    parse_actions,
    within_closed_window,
)


ACTIONS = [
    {
        "action_type": "unload",
        "tm_id": "TM1",
        "module_id": "LP",
        "route_id": "A",
        "wafer_index": 0,
        "step_index": 0,
        "start": 0,
        "end": 4,
    },
    {
        "action_type": "load",
        "tm_id": "TM1",
        "module_id": "PM1",
        "route_id": "A",
        "wafer_index": 0,
        "step_index": 1,
        "start": 5,
        "end": 9,
    },
]


PROBLEM = parse_problem(
    {
        "Modules": {
            "LP": {"type": "LP"},
            "PM1": {"type": "PM"},
        },
        "ClusterTool": {
            "TM1": {
                "module_ids": ["LP", "PM1"],
                "arm_type": "single_arm",
                "travel_times": 1,
                "load_time": 4,
                "unload_time": 4,
            }
        },
        "just_in_time": {"residency_time": 15},
        "routes": {
            "A": [{"module_id": "PM1", "process_time": 10}],
        },
        "initial_state": {
            "robots": {"TM1": {"position_module_id": "LP"}},
            "wafers": [
                {
                    "route_id": "A",
                    "wafer_index": "0",
                    "step_index": 0,
                    "location": {"kind": "module", "module_id": "LP"},
                    "process_end_time": None,
                }
            ]
        },
    }
)

EXAMPLES = Path(__file__).parents[1] / "examples"


def _module_ids(action: ActionRecord) -> tuple[str, ...]:
    return () if action.module_id is None else (action.module_id,)


def test_common_grouping_sorts_each_subjects_actions() -> None:
    actions = parse_actions(
        [
            {"action_type": "load", "module_id": "PM1", "start": 1, "end": 2},
            {"action_type": "unload", "module_id": "PM1", "start": 0, "end": 1},
        ]
    )
    grouped = group_actions(
        actions,
        _module_ids,
    )

    assert set(grouped) == {"PM1"}
    assert [action.index for action in grouped["PM1"]] == [1, 0]
    assert grouped["PM1"][0] is actions[1]


def test_action_names_are_normalized_to_pick_and_place() -> None:
    actions = parse_actions(
        [
            {"action_type": action_type, "start": index, "end": index + 1}
            for index, action_type in enumerate(["pick", "unload", "place", "load"])
        ]
    )

    assert [action.action_type for action in actions] == [
        "pick",
        "pick",
        "place",
        "place",
    ]


def test_one_module_validator_owns_only_one_modules_actions_and_initial_slice() -> None:
    later = ActionRecord.from_mapping(
        1,
        {"action_type": "unload", "module_id": "PM1", "start": 2, "end": 3},
    )
    earlier = ActionRecord.from_mapping(
        0,
        {"action_type": "load", "module_id": "PM1", "start": 1, "end": 2},
    )
    initial_occupants = {("A", 0)}

    validator = ModuleValidator(
        "PM1",
        PROBLEM.Modules["PM1"],
        [later, earlier],
        initial_occupants,
    )

    assert validator.module_id == "PM1"
    assert validator.config is PROBLEM.Modules["PM1"]
    assert validator.config.capacity == 1
    assert [action.index for action in validator.actions] == [0, 1]
    assert validator.initial_occupants == frozenset(initial_occupants)


def test_suite_creates_one_validator_for_each_concrete_subject() -> None:
    suite = ValidatorSuite(PROBLEM)

    report = suite.validate(ACTIONS)

    assert [validator.module_id for validator in suite.module_validators] == ["LP", "PM1"]
    assert all(type(validator) is ModuleValidator for validator in suite.module_validators)
    assert [validator.config.capacity for validator in suite.module_validators] == [25, 1]
    assert suite.module_validators[0].initial_occupants == frozenset({("A", 0)})
    assert suite.module_validators[1].initial_occupants == frozenset()
    assert [[action.index for action in validator.actions] for validator in suite.module_validators] == [
        [0],
        [1],
    ]
    assert [validator.robot_id for validator in suite.robot_validators] == ["TM1"]
    assert suite.robot_validators[0].config is PROBLEM.ClusterTool["TM1"]
    assert suite.robot_validators[0].initial_position_module_id == "LP"
    assert suite.robot_validators[0].initial_arms == {}
    assert [action.index for action in suite.robot_validators[0].actions] == [0, 1]
    assert [validator.wafer_key for validator in suite.wafer_validators] == [("A", 0)]
    assert suite.wafer_validators[0].route is PROBLEM.routes["A"]
    assert suite.wafer_validators[0].just_in_time is PROBLEM.just_in_time
    assert suite.wafer_validators[0].initial_wafer.step_index == 0
    assert suite.wafer_validators[0].initial_wafer.location.module_id == "LP"
    assert suite.wafer_validators[0].initial_wafer.process_end_time is None
    assert [action.index for action in suite.wafer_validators[0].actions] == [0, 1]
    assert report.ok
    assert report.checked_subjects == {"module": 2, "robot": 1, "wafer": 1}


def test_problem_initial_snapshot_creates_subject_slices_without_actions() -> None:
    suite = ValidatorSuite(PROBLEM)

    suite.validate([])

    assert [validator.module_id for validator in suite.module_validators] == ["LP", "PM1"]
    lp_validator = next(validator for validator in suite.module_validators if validator.module_id == "LP")
    assert lp_validator.actions == ()
    assert lp_validator.initial_occupants == frozenset({("A", 0)})
    assert [validator.robot_id for validator in suite.robot_validators] == ["TM1"]
    assert [validator.wafer_key for validator in suite.wafer_validators] == [("A", 0)]


def test_reusing_suite_rebuilds_instead_of_accumulating_validators() -> None:
    suite = ValidatorSuite(PROBLEM)
    suite.validate(ACTIONS)

    suite.validate([])

    assert [validator.module_id for validator in suite.module_validators] == ["LP", "PM1"]
    assert [validator.robot_id for validator in suite.robot_validators] == ["TM1"]
    assert [validator.wafer_key for validator in suite.wafer_validators] == [("A", 0)]


def test_suite_parses_each_action_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ActionRecord.from_mapping.__func__
    parsed_indexes: list[int] = []

    def parse_spy(cls, index, action):
        parsed_indexes.append(index)
        return original(cls, index, action)

    monkeypatch.setattr(ActionRecord, "from_mapping", classmethod(parse_spy))

    ValidatorSuite(PROBLEM).validate(ACTIONS)

    assert parsed_indexes == list(range(len(ACTIONS)))


def test_suite_projects_initial_snapshot_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = InitialState.to_snapshot
    projection_count = 0

    def projection_spy(self):
        nonlocal projection_count
        projection_count += 1
        return original(self)

    monkeypatch.setattr(InitialState, "to_snapshot", projection_spy)

    ValidatorSuite(PROBLEM).validate(ACTIONS)

    assert projection_count == 1


def test_suite_rejects_action_referencing_unknown_configured_subject() -> None:
    suite = ValidatorSuite(PROBLEM)

    with pytest.raises(ValueError, match="Unknown Module.*PM2"):
        suite.validate(
            [
                {
                    "action_type": "load",
                    "module_id": "PM2",
                    "tm_id": "TM1",
                    "route_id": "A",
                    "wafer_index": 0,
                    "start": 0,
                    "end": 1,
                }
            ]
        )


def test_suite_builds_hardware_validators_from_real_problem_file() -> None:
    suite = ValidatorSuite(load_problem(EXAMPLES / "naura_task1.json"))

    report = suite.validate([])

    assert len(suite.module_validators) == 18
    assert len(suite.robot_validators) == 3
    assert suite.wafer_validators == []
    assert report.checked_subjects == {"module": 18, "robot": 3}


def test_suite_derives_robot_arms_from_wafer_location() -> None:
    problem = parse_problem(
        {
            "Modules": {"PM1": {"type": "PM"}},
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["PM1"],
                    "arm_type": "dual_arm",
                    "load_time": 4,
                    "unload_time": 4,
                }
            },
            "routes": {"A": [{"module_id": "PM1", "process_time": 10}]},
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "step_index": 1,
                        "location": {
                            "kind": "robot",
                            "robot_id": "TM1",
                            "arm_id": "arm1",
                        },
                        "process_end_time": None,
                    }
                ]
            },
        }
    )

    suite = ValidatorSuite(problem)
    suite.validate([])

    assert suite.module_validators[0].initial_occupants == frozenset()
    assert suite.robot_validators[0].initial_position_module_id is None
    assert suite.robot_validators[0].initial_arms == {"arm1": ("A", 0)}
    assert suite.wafer_validators[0].initial_wafer.process_end_time is None


def test_complete_action_example_uses_problem_initial_state() -> None:
    problem = load_problem(EXAMPLES / "all_actions_recipe.json")
    actions = json.loads((EXAMPLES / "all_actions_actions.json").read_text(encoding="utf-8"))

    suite = ValidatorSuite(problem)
    report = suite.validate(actions)

    assert report.ok
    assert report.checked_subjects == {"module": 3, "robot": 2, "wafer": 2}
    positions = {
        validator.robot_id: validator.initial_position_module_id
        for validator in suite.robot_validators
    }
    assert positions == {"TM1": "LP", "TM2": None}


def test_common_time_helpers_define_expected_boundaries() -> None:
    assert not intervals_overlap(Interval(0, 1), Interval(1, 2))
    assert intervals_overlap(Interval(0, 1.5), Interval(1, 2))
    assert within_closed_window(10, 10, 15)
    assert within_closed_window(15, 10, 15)
    assert not within_closed_window(15.1, 10, 15)
