from __future__ import annotations

from copy import deepcopy

from cluster_toolkit.problem import parse_problem
from cluster_toolkit.validator import ValidatorSuite


VALID_ACTIONS = [
    {
        "action_type": "unload",
        "tm_id": "TM1",
        "module_id": "LP",
        "route_id": "A",
        "wafer_index": 0,
        "step_index": 0,
        "start": 0,
        "end": 1,
    },
    {
        "action_type": "load",
        "tm_id": "TM1",
        "module_id": "PM1",
        "route_id": "A",
        "wafer_index": 0,
        "step_index": 1,
        "start": 2,
        "end": 3,
    },
    {
        "action_type": "unload",
        "tm_id": "TM1",
        "module_id": "PM1",
        "route_id": "A",
        "wafer_index": 0,
        "step_index": 1,
        "start": 7,
        "end": 8,
    },
    {
        "action_type": "load",
        "tm_id": "TM1",
        "module_id": "PM2",
        "route_id": "A",
        "wafer_index": 0,
        "step_index": 2,
        "start": 9,
        "end": 10,
    },
    {
        "action_type": "unload",
        "tm_id": "TM1",
        "module_id": "PM2",
        "route_id": "A",
        "wafer_index": 0,
        "step_index": 2,
        "start": 13,
        "end": 14,
    },
    {
        "action_type": "load",
        "tm_id": "TM1",
        "module_id": "LP",
        "route_id": "A",
        "wafer_index": 0,
        "step_index": 3,
        "start": 15,
        "end": 16,
    },
]


def _problem(
    *,
    initial_module: str = "LP",
    initial_step: int = 0,
    process_end_time: float | None = None,
    just_in_time: dict[str, float] | None = None,
):
    raw_problem = {
        "Modules": {
            "LP": {"type": "LP"},
            "PM1": {"type": "PM"},
            "PM2": {"type": "PM"},
        },
        "ClusterTool": {
            "TM1": {
                "module_ids": ["LP", "PM1", "PM2"],
                "arm_type": "single_arm",
                "load_time": 1,
                "unload_time": 1,
            }
        },
        "routes": {
            "A": [
                {"module_id": "PM1", "process_time": 4},
                {"module_id": "PM2", "process_time": 3},
            ]
        },
        "initial_state": {
            "wafers": [
                {
                    "route_id": "A",
                    "wafer_index": "0",
                    "priority": 0,
                    "step_index": initial_step,
                    "location": {"kind": "module", "module_id": initial_module},
                    "process_end_time": process_end_time,
                }
            ]
        },
    }
    if just_in_time is not None:
        raw_problem["just_in_time"] = just_in_time
    return parse_problem(raw_problem)


def _issues(actions, *, problem=None):
    report = ValidatorSuite(problem or _problem()).validate(actions)
    return report.issues


def test_valid_wafer_actions_follow_route_without_overlaps() -> None:
    report = ValidatorSuite(_problem()).validate(VALID_ACTIONS)

    assert report.ok
    assert report.checked_subjects == {"module": 3, "robot": 1, "wafer": 1}


def test_canonical_pick_and_place_names_follow_the_same_route() -> None:
    actions = deepcopy(VALID_ACTIONS)
    for action in actions:
        action["action_type"] = (
            "pick" if action["action_type"] == "unload" else "place"
        )

    assert ValidatorSuite(_problem()).validate(actions).ok


def test_completed_wafer_returns_to_resolved_initial_lp() -> None:
    problem = parse_problem(
        {
            "Modules": {
                "LP1": {"type": "LP"},
                "LP2": {"type": "LP"},
                "PM1": {"type": "PM"},
            },
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["LP1", "LP2", "PM1"],
                    "arm_type": "single_arm",
                    "pick_time": 1,
                    "place_time": 1,
                }
            },
            "routes": {"A": [{"module_id": "PM1", "process_time": 1}]},
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "LP2"},
                    }
                ]
            },
        }
    )
    actions = [
        {
            "action_type": action_type,
            "tm_id": "TM1",
            "module_id": module_id,
            "route_id": "A",
            "wafer_index": 0,
            "step_index": step_index,
            "start": start,
            "end": start + 1,
        }
        for action_type, module_id, step_index, start in (
            ("pick", "LP2", 0, 0),
            ("place", "PM1", 1, 1),
            ("pick", "PM1", 1, 3),
            ("place", "LP1", 2, 4),
        )
    ]

    issues = ValidatorSuite(problem).validate(actions).issues

    assert any(
        issue.constraint_id == "wafer.process_order"
        and "must return to source module LP2" in issue.message
        for issue in issues
    )


def test_wafer_process_order_rejects_wrong_module_for_step() -> None:
    actions = deepcopy(VALID_ACTIONS)
    actions[1]["module_id"] = "PM2"

    issues = _issues(actions)

    assert [issue.constraint_id for issue in issues] == ["wafer.process_order"]
    assert issues[0].action_index == 1
    assert "step 1 must use one of: PM1" in issues[0].message


def test_wafer_process_order_rejects_skipped_step() -> None:
    actions = deepcopy(VALID_ACTIONS)
    actions[1]["step_index"] = 2

    issues = _issues(actions)

    assert [issue.constraint_id for issue in issues] == ["wafer.process_order"]
    assert issues[0].action_index == 1
    assert "expected step 1" in issues[0].message


def test_pick_and_place_intervals_must_not_overlap() -> None:
    actions = deepcopy(VALID_ACTIONS)
    actions[1]["start"] = 0.5
    actions[1]["end"] = 1.5

    issues = [
        issue
        for issue in _issues(actions)
        if issue.subject_kind == "wafer"
    ]

    assert [issue.constraint_id for issue in issues] == ["wafer.interval_overlap"]
    assert "Pick [0.0, 1.0) overlaps Place [0.5, 1.5)" in issues[0].message


def test_pick_must_not_start_before_processing_finishes() -> None:
    actions = deepcopy(VALID_ACTIONS)
    actions[2]["start"] = 6.5

    issues = _issues(actions)

    assert [issue.constraint_id for issue in issues] == ["wafer.interval_overlap"]
    assert "Process step 1 [3.0, 7.0) overlaps Pick [6.5, 8.0)" in issues[0].message


def test_pick_may_start_exactly_when_processing_finishes() -> None:
    assert _issues(VALID_ACTIONS) == []


def test_initial_process_interval_blocks_an_early_pick() -> None:
    problem = _problem(initial_module="PM1", initial_step=1, process_end_time=5)
    actions = deepcopy(VALID_ACTIONS[2:])
    actions[0]["start"] = 4
    actions[0]["end"] = 5

    issues = _issues(actions, problem=problem)

    assert [issue.constraint_id for issue in issues] == ["wafer.interval_overlap"]
    assert "initial process [0.0, 5.0) overlaps Pick [4.0, 5.0)" in issues[0].message


def test_just_in_time_is_not_checked_yet() -> None:
    actions = deepcopy(VALID_ACTIONS)
    actions[2]["start"] = 20
    actions[2]["end"] = 21
    actions[3]["start"] = 22
    actions[3]["end"] = 23
    actions[4]["start"] = 26
    actions[4]["end"] = 27
    actions[5]["start"] = 28
    actions[5]["end"] = 29

    report = ValidatorSuite(_problem(just_in_time={"residency_time": 0})).validate(actions)

    assert report.ok


def test_load_lock_process_time_is_not_checked_yet() -> None:
    problem = parse_problem(
        {
            "Modules": {
                "LP": {"type": "LP"},
                "LL": {"type": "LL"},
            },
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["LP", "LL"],
                    "arm_type": "single_arm",
                    "load_time": 1,
                    "unload_time": 1,
                }
            },
            "routes": {"A": [{"module_id": "LL", "process_time": 10}]},
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 0,
                        "location": {"kind": "module", "module_id": "LP"},
                    }
                ]
            },
        }
    )
    actions = [
        {
            "action_type": action_type,
            "tm_id": "TM1",
            "module_id": module_id,
            "route_id": "A",
            "wafer_index": 0,
            "step_index": step_index,
            "start": start,
            "end": end,
        }
        for action_type, module_id, step_index, start, end in [
            ("unload", "LP", 0, 0, 1),
            ("load", "LL", 1, 2, 3),
            ("unload", "LL", 1, 4, 5),
            ("load", "LP", 2, 6, 7),
        ]
    ]

    report = ValidatorSuite(problem).validate(actions)

    assert report.ok
