from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cluster_toolkit.problem import (
    InitialState,
    ModuleLocation,
    ModuleType,
    RobotLocation,
    TMArmType,
    load_problem,
    parse_problem,
)


EXAMPLES = Path(__file__).parents[2] / "validator" / "examples"


def _parse_initial_indexes(expression: object) -> tuple[int, ...]:
    initial_state = InitialState.model_validate(
        {
            "wafers": [
                {
                    "route_id": "A",
                    "wafer_index": expression,
                    "priority": 0,
                    "location": {"kind": "module", "module_id": "LP"},
                }
            ]
        }
    )
    return tuple(wafer.wafer_index for wafer in initial_state.wafers)


def test_initial_wafer_index_expression_expands_single_list_and_ranges() -> None:
    assert _parse_initial_indexes("1, 3-5, 8, 10-10") == (
        1,
        3,
        4,
        5,
        8,
        10,
    )


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        (0, "wafer_index must be a string expression"),
        ("", "wafer_index expression must not be empty"),
        ("1,,3", "invalid wafer_index item"),
        ("5-3", "wafer_index range must be ascending"),
        ("1,a", "invalid wafer_index item"),
        ("-1", "invalid wafer_index item"),
        ("1-", "invalid wafer_index item"),
    ],
)
def test_initial_wafer_index_expression_rejects_invalid_input(
    expression: object,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _parse_initial_indexes(expression)


@pytest.mark.parametrize("expression", ["1,1", "1-3,3-5"])
def test_initial_wafer_index_expression_rejects_duplicates(
    expression: str,
) -> None:
    with pytest.raises(
        ValidationError,
        match="wafer_index expression contains duplicate index",
    ):
        _parse_initial_indexes(expression)


def test_expanded_initial_entries_still_reject_duplicate_wafer_identity() -> None:
    with pytest.raises(
        ValidationError,
        match="InitialState.wafers must not contain duplicate wafer identities",
    ):
        InitialState.model_validate(
            {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "1-2",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "LP"},
                    },
                    {
                        "route_id": "A",
                        "wafer_index": "2-3",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "LP"},
                    },
                ]
            }
        )


def test_load_problem_reads_and_normalizes_naura_task1() -> None:
    problem = load_problem(EXAMPLES / "naura_task1.json")

    assert len(problem.Modules) == 18
    assert len(problem.ClusterTool) == 3
    assert len(problem.routes) == 11
    assert problem.Modules["PM1"].type is ModuleType.PM
    assert problem.Modules["PM1"].capacity == 1
    assert problem.Modules["LP1"].capacity == 25
    assert problem.Modules["LLA"].capacity == 1
    assert problem.ClusterTool["TM1"].arm_type is TMArmType.SINGLE_ARM
    assert problem.ClusterTool["TM1"].place_time == 4.0
    assert problem.ClusterTool["TM1"].pick_time == 4.0
    assert problem.routes["B"].visits[1].module_ids == ("LLA", "LLB")
    assert problem.routes["B"].visits[1].process_time == 0.0


def test_initial_wafer_priority_is_required_and_preserved() -> None:
    wafer = InitialState.model_validate(
        {
            "wafers": [
                {
                    "route_id": "A",
                    "wafer_index": "0",
                    "priority": 7,
                    "location": {"kind": "module", "module_id": "IO1"},
                }
            ]
        }
    ).wafers[0]

    assert wafer.priority == 7
    assert wafer.model_dump()["priority"] == 7


@pytest.mark.parametrize("priority", [None, -1, True, 1.5])
def test_initial_wafer_rejects_missing_or_invalid_priority(priority: object) -> None:
    raw = {
        "route_id": "A",
        "wafer_index": "0",
        "location": {"kind": "module", "module_id": "IO1"},
    }
    if priority is not None:
        raw["priority"] = priority

    with pytest.raises(ValidationError, match="priority"):
        InitialState.model_validate({"wafers": [raw]})


def test_virtual_io_is_unique_and_not_an_explicit_route_visit() -> None:
    base = {
        "Modules": {
            "IO1": {"type": "IO", "capacity": 2},
            "PM1": {"type": "PM"},
        },
        "ClusterTool": {
            "TM1": {
                "module_ids": ["IO1", "PM1"],
                "arm_type": "single_arm",
                "place_time": 1,
                "pick_time": 1,
            }
        },
        "routes": {"A": [{"module_id": "IO1", "process_time": 0}]},
    }

    with pytest.warns(UserWarning), pytest.raises(
        ValidationError,
        match="must not contain virtual IO Module IO1",
    ):
        parse_problem(base)

    base["routes"] = {"A": [{"module_id": "PM1", "process_time": 1}]}
    base["Modules"]["IO2"] = {"type": "IO", "capacity": 2}
    with pytest.warns(UserWarning), pytest.raises(
        ValidationError,
        match="at most one virtual IO",
    ):
        parse_problem(base)


def test_parse_problem_accepts_grouped_module_definition() -> None:
    problem = parse_problem(
        {
            "Modules": {"LP": ["LP1"], "PM": ["PM1"]},
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["LP1", "PM1"],
                    "arm_type": "single_arm",
                    "travel_times": 1,
                    "load_time": 4,
                    "unload_time": 4,
                }
            },
            "routes": {
                "A": [
                    {"module_id": "PM1", "process_time": 10},
                ]
            },
        }
    )

    assert problem.Modules["LP1"].type is ModuleType.LP
    assert problem.Modules["LP1"].capacity == 25
    assert problem.Modules["PM1"].type is ModuleType.PM
    assert problem.Modules["PM1"].capacity == 1
    assert problem.routes["A"].visits[0].module_ids == ("PM1",)


def test_parse_problem_accepts_canonical_pick_and_place_times() -> None:
    problem = parse_problem(
        {
            "Modules": {"PM1": {"type": "PM"}},
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["PM1"],
                    "arm_type": "single_arm",
                    "place_time": 3,
                    "pick_time": 5,
                }
            },
        }
    )

    assert problem.ClusterTool["TM1"].place_time == 3.0
    assert problem.ClusterTool["TM1"].pick_time == 5.0


def test_parse_problem_rejects_route_reference_to_unknown_module() -> None:
    with pytest.raises(ValidationError, match="Route A references unknown Module: PM2"):
        parse_problem(
            {
                "Modules": {"PM1": {"type": "PM"}},
                "ClusterTool": {
                    "TM1": {
                        "module_ids": ["PM1"],
                        "arm_type": "single_arm",
                        "load_time": 4,
                        "unload_time": 4,
                    }
                },
                "routes": {
                    "A": [{"module_id": "PM2", "process_time": 10}],
                },
            }
        )


def test_parse_problem_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        parse_problem(
            {
                "Modules": {
                    "PM1": {
                        "type": "PM",
                        "unexpected": True,
                    }
                },
                "ClusterTool": {
                    "TM1": {
                        "module_ids": ["PM1"],
                        "arm_type": "single_arm",
                        "load_time": 4,
                        "unload_time": 4,
                    }
                },
            }
        )


def test_parse_problem_warns_and_accepts_explicit_module_capacity() -> None:
    with pytest.warns(
        UserWarning,
        match="Explicit Module.capacity overrides the type-based default",
    ):
        problem = parse_problem(
            {
                "Modules": {"PM1": {"type": "PM", "capacity": 2}},
                "ClusterTool": {
                    "TM1": {
                        "module_ids": ["PM1"],
                        "arm_type": "single_arm",
                        "load_time": 4,
                        "unload_time": 4,
                    }
                },
            }
        )

    assert problem.Modules["PM1"].capacity == 2


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_parse_problem_rejects_invalid_module_capacity(capacity: object) -> None:
    with pytest.warns(UserWarning), pytest.raises(
        ValidationError,
        match="Module.capacity must be a positive integer",
    ):
        parse_problem(
            {
                "Modules": {"PM1": {"type": "PM", "capacity": capacity}},
                "ClusterTool": {
                    "TM1": {
                        "module_ids": ["PM1"],
                        "arm_type": "single_arm",
                        "load_time": 4,
                        "unload_time": 4,
                    }
                },
            }
        )


def test_initial_snapshot_projects_wafer_locations_once() -> None:
    problem = parse_problem(
        {
            "Modules": {
                "LP": {"type": "LP"},
                "PM1": {"type": "PM"},
            },
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["LP", "PM1"],
                    "arm_type": "dual_arm",
                    "load_time": 4,
                    "unload_time": 4,
                }
            },
            "routes": {"A": [{"module_id": "PM1", "process_time": 10}]},
            "initial_state": {
                "robots": {"TM1": {"position_module_id": "PM1"}},
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 0,
                        "location": {"kind": "module", "module_id": "LP"},
                    },
                    {
                        "route_id": "A",
                        "wafer_index": "1",
                        "priority": 0,
                        "step_index": 1,
                        "location": {
                            "kind": "robot",
                            "robot_id": "TM1",
                            "arm_id": "arm1",
                        },
                        "process_end_time": None,
                    },
                ]
            },
        }
    )

    snapshot = problem.initial_state.to_snapshot()
    module_wafer = snapshot.wafers_by_key[("A", 0)]
    robot_wafer = snapshot.wafers_by_key[("A", 1)]
    assert isinstance(module_wafer.location, ModuleLocation)
    assert module_wafer.location.module_id == "LP"
    assert isinstance(robot_wafer.location, RobotLocation)
    assert robot_wafer.location.robot_id == "TM1"
    assert robot_wafer.location.arm_id == "arm1"
    assert robot_wafer.process_end_time is None
    assert snapshot.module_occupants == {"LP": frozenset({("A", 0)})}
    assert snapshot.tm_arms == {"TM1": {"arm1": ("A", 1)}}
    assert snapshot.tm_positions == {"TM1": "PM1"}


def test_multi_lp_wafer_defaults_to_its_initial_lp() -> None:
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
                    "place_time": 1,
                    "pick_time": 1,
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
    wafer = problem.initial_state.to_snapshot().wafers_by_key[("A", 0)]

    assert problem.return_module_id(wafer) == "LP2"


def test_return_module_id_input_alias_normalizes_to_return_lp_id() -> None:
    problem = parse_problem(
        {
            "Modules": {"LP1": {"type": "LP"}, "PM1": {"type": "PM"}},
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["LP1", "PM1"],
                    "arm_type": "single_arm",
                    "place_time": 1,
                    "pick_time": 1,
                }
            },
            "routes": {"A": [{"module_id": "PM1", "process_time": 1}]},
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "LP1"},
                        "return_module_id": "LP1",
                    }
                ]
            },
        }
    )
    wafer = problem.initial_state.wafers[0]

    assert wafer.return_lp_id == "LP1"
    assert "return_module_id" not in wafer.model_dump()
    assert wafer.model_dump()["return_lp_id"] == "LP1"


def test_multi_lp_mid_route_wafer_requires_explicit_return_lp() -> None:
    raw = {
        "Modules": {
            "LP1": {"type": "LP"},
            "LP2": {"type": "LP"},
            "PM1": {"type": "PM"},
        },
        "ClusterTool": {
            "TM1": {
                "module_ids": ["LP1", "LP2", "PM1"],
                "arm_type": "single_arm",
                "place_time": 1,
                "pick_time": 1,
            }
        },
        "routes": {"A": [{"module_id": "PM1", "process_time": 1}]},
        "initial_state": {
            "wafers": [
                {
                    "route_id": "A",
                    "wafer_index": "0",
                    "priority": 0,
                    "step_index": 1,
                    "location": {"kind": "module", "module_id": "PM1"},
                }
            ]
        },
    }

    with pytest.raises(ValidationError, match="must define return_lp_id"):
        parse_problem(raw)

    raw["initial_state"]["wafers"][0]["return_module_id"] = "LP2"
    problem = parse_problem(raw)
    wafer = problem.initial_state.to_snapshot().wafers_by_key[("A", 0)]
    assert problem.return_module_id(wafer) == "LP2"


def test_initial_snapshot_indexes_are_read_only() -> None:
    problem = parse_problem(
        {
            "Modules": {"LP": {"type": "LP"}},
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["LP"],
                    "arm_type": "single_arm",
                    "load_time": 1,
                    "unload_time": 1,
                }
            },
            "routes": {"A": [{"module_id": "LP", "process_time": 0}]},
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0-24",
                        "priority": 0,
                        "step_index": 0,
                        "location": {"kind": "module", "module_id": "LP"},
                    }
                ]
            },
        }
    )
    snapshot = problem.initial_state.to_snapshot()

    assert len(snapshot.wafers_by_key) == 25
    assert len(snapshot.module_occupants["LP"]) == 25
    with pytest.raises(TypeError):
        snapshot.module_occupants["LP"] = frozenset()


def test_initial_state_rejects_module_capacity_overflow() -> None:
    with pytest.raises(ValidationError, match="Module PM1 has 2 wafers but capacity is 1"):
        parse_problem(
            {
                "Modules": {"PM1": {"type": "PM"}},
                "ClusterTool": {
                    "TM1": {
                        "module_ids": ["PM1"],
                        "arm_type": "single_arm",
                        "load_time": 4,
                        "unload_time": 4,
                    }
                },
                "routes": {"A": [{"module_id": "PM1", "process_time": 10}]},
                "initial_state": {
                    "wafers": [
                        {
                            "route_id": "A",
                            "wafer_index": "0-1",
                            "priority": 0,
                            "step_index": 1,
                            "location": {"kind": "module", "module_id": "PM1"},
                        }
                    ]
                },
            }
        )


def test_initial_state_rejects_two_wafers_on_the_same_robot_arm() -> None:
    with pytest.raises(ValidationError, match="Robot TM1 arm arm0 holds multiple wafers"):
        parse_problem(
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
                            "wafer_index": "0-1",
                            "priority": 0,
                            "step_index": 1,
                            "location": {
                                "kind": "robot",
                                "robot_id": "TM1",
                                "arm_id": "arm0",
                            },
                        }
                    ]
                },
            }
        )


def test_initial_state_rejects_unfinished_process_on_robot() -> None:
    with pytest.raises(ValidationError, match="is on a Robot but still has unfinished processing"):
        parse_problem(
            {
                "Modules": {"PM1": {"type": "PM"}},
                "ClusterTool": {
                    "TM1": {
                        "module_ids": ["PM1"],
                        "arm_type": "single_arm",
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
                            "priority": 0,
                            "step_index": 1,
                            "location": {
                                "kind": "robot",
                                "robot_id": "TM1",
                            },
                            "process_end_time": 8,
                        }
                    ]
                },
            }
        )


def test_missing_robot_initial_position_means_anywhere() -> None:
    problem = parse_problem(
        {
            "Modules": {"PM1": {"type": "PM"}},
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["PM1"],
                    "arm_type": "single_arm",
                    "load_time": 4,
                    "unload_time": 4,
                }
            },
        }
    )

    assert problem.initial_state.robots == {}


def test_initial_state_rejects_unknown_robot_position_owner() -> None:
    with pytest.raises(ValidationError, match="initial_state references unknown Robot: TM2"):
        parse_problem(
            {
                "Modules": {"PM1": {"type": "PM"}},
                "ClusterTool": {
                    "TM1": {
                        "module_ids": ["PM1"],
                        "arm_type": "single_arm",
                        "load_time": 4,
                        "unload_time": 4,
                    }
                },
                "initial_state": {
                    "robots": {"TM2": {"position_module_id": "PM1"}},
                },
            }
        )


def test_initial_state_rejects_unreachable_robot_position() -> None:
    with pytest.raises(ValidationError, match="Robot TM1 cannot reach position Module: PM1"):
        parse_problem(
            {
                "Modules": {
                    "LP": {"type": "LP"},
                    "PM1": {"type": "PM"},
                },
                "ClusterTool": {
                    "TM1": {
                        "module_ids": ["LP"],
                        "arm_type": "single_arm",
                        "load_time": 4,
                        "unload_time": 4,
                    }
                },
                "initial_state": {
                    "robots": {"TM1": {"position_module_id": "PM1"}},
                },
            }
        )
