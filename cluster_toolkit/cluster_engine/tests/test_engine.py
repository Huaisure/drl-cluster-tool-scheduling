from __future__ import annotations

from cluster_toolkit.cluster_engine import (
    ADVANCE,
    ClusterEngine,
    PickAction,
    PlaceAction,
)
from cluster_toolkit.problem import ClusterProblem, parse_problem


def _simple_problem(
    *,
    wafer_count: int = 1,
    arm_type: str = "single_arm",
) -> ClusterProblem:
    return parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO", "capacity": wafer_count},
                "PM1": {"type": "PM"},
                "PM2": {"type": "PM"},
            },
            "ClusterTool": {
                "TM1": {
                    "module_ids": ["IO1", "PM1", "PM2"],
                    "arm_type": arm_type,
                    "travel_times": 2,
                    "pick_time": 1,
                    "place_time": 1.5,
                }
            },
            "routes": {
                "A": [
                    {
                        "module_ids": ["PM1", "PM2"],
                        "process_time": 5,
                    }
                ]
            },
            "initial_state": {
                "robots": {"TM1": {"position_module_id": "PM2"}},
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": str(index),
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "IO1"},
                    }
                    for index in range(wafer_count)
                ],
            },
        }
    )


def _advance_until(engine: ClusterEngine, predicate) -> None:
    for _ in range(100):
        if predicate():
            return
        assert ADVANCE in engine.available_actions()
        engine.step(ADVANCE)
    raise AssertionError("predicate did not become true")


def _legacy_available_actions(engine: ClusterEngine):
    """Reference the original exhaustive action enumeration."""

    state = engine.state
    if engine.is_complete():
        return ()

    pick_actions = [
        PickAction(robot_id=robot_id, wafer_key=wafer_key)
        for wafer_key in sorted(state.wafers)
        for robot_id in sorted(state.robots)
        if engine._can_pick(wafer_key, robot_id)
    ]
    source_picks = [
        action
        for action in pick_actions
        if engine._is_initial_source_wafer(state.wafers[action.wafer_key])
    ]
    if source_picks:
        minimum_priority = min(
            engine._initial_wafer(action.wafer_key).priority
            for action in source_picks
        )
        pick_actions = [
            action
            for action in pick_actions
            if action not in source_picks
            or engine._initial_wafer(action.wafer_key).priority
            == minimum_priority
        ]

    actions = list(pick_actions)
    for _, robot in sorted(state.robots.items()):
        for wafer_key in tuple(robot.holding):
            wafer = state.wafers[wafer_key]
            for module_id in engine._next_targets(wafer):
                if engine._can_place(wafer_key, module_id):
                    actions.append(PlaceAction(wafer_key, module_id))
    if engine.next_event_time() is not None:
        actions.append(ADVANCE)
    return tuple(actions)


def test_stateful_dispatch_and_advance_keep_original_travel_semantics() -> None:
    engine = ClusterEngine(_simple_problem())
    state = engine.reset()
    pick = PickAction(robot_id="TM1", wafer_key=("A", 0))

    record = engine.step(pick)

    assert record is not None
    assert (record.start, record.end) == (2.0, 3.0)
    assert state.time == 0.0
    assert state.robots["TM1"].holding == []
    assert state.wafers[("A", 0)].module_id == "IO1"
    assert engine.available_actions() == (ADVANCE,)

    engine.step(ADVANCE)
    assert state.time == 2.0
    assert state.robots["TM1"].holding == [("A", 0)]
    assert state.wafers[("A", 0)].module_id == "IO1"

    engine.step(ADVANCE)
    assert state.time == 3.0
    assert state.wafers[("A", 0)].robot_id == "TM1"
    assert PlaceAction(("A", 0), "PM1") in engine.available_actions()


def test_step_validates_one_action_without_enumerating_the_action_space() -> None:
    engine = ClusterEngine(_simple_problem(wafer_count=20))
    engine.reset()

    def fail_if_enumerated():
        raise AssertionError("step() rebuilt the full action set")

    engine.available_actions = fail_if_enumerated  # type: ignore[method-assign]

    record = engine.step(PickAction("TM1", ("A", 0)))

    assert record is not None
    assert record.wafer_key == ("A", 0)


def test_indexed_action_enumeration_matches_original_exhaustive_semantics() -> None:
    engine = ClusterEngine(_simple_problem(wafer_count=4, arm_type="dual_arm"))
    engine.reset()

    for _ in range(500):
        actions = engine.available_actions()
        assert actions == _legacy_available_actions(engine)
        if engine.is_complete():
            break
        places = [action for action in actions if isinstance(action, PlaceAction)]
        engine.step(places[0] if places else actions[0])
    else:
        raise AssertionError("comparison episode did not complete")

    assert engine.is_complete()


def test_place_explicitly_selects_which_dual_arm_wafer_to_move() -> None:
    raw = _simple_problem(wafer_count=2, arm_type="dual_arm").model_dump(
        mode="json",
        by_alias=True,
    )
    raw["initial_state"] = {
        "wafers": [
            {
                "route_id": "A",
                "wafer_index": "0",
                "priority": 0,
                "location": {
                    "kind": "robot",
                    "robot_id": "TM1",
                    "arm_id": "arm0",
                },
            },
            {
                "route_id": "A",
                "wafer_index": "1",
                "priority": 0,
                "location": {
                    "kind": "robot",
                    "robot_id": "TM1",
                    "arm_id": "arm1",
                },
            },
        ]
    }
    engine = ClusterEngine(parse_problem(raw))
    engine.reset()

    assert PlaceAction(("A", 0), "PM1") in engine.available_actions()
    assert PlaceAction(("A", 1), "PM1") in engine.available_actions()

    record = engine.step(PlaceAction(("A", 1), "PM1"))

    assert record is not None and record.wafer_key == ("A", 1)
    assert engine.state.pending_operations[0].wafer_key == ("A", 1)


def test_pick_does_not_require_capacity_in_the_next_module() -> None:
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
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                }
            },
            "routes": {
                "A": [{"module_id": "PM1", "process_time": 1}],
                "B": [{"module_id": "PM1", "process_time": 1}],
            },
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "LP"},
                    },
                    {
                        "route_id": "B",
                        "wafer_index": "0",
                        "priority": 0,
                        "step_index": 1,
                        "location": {"kind": "module", "module_id": "PM1"},
                        "process_end_time": 10,
                    },
                ]
            },
        }
    )
    engine = ClusterEngine(problem)
    engine.reset()

    assert PickAction("TM1", ("A", 0)) in engine.available_actions()


def test_static_priority_masks_only_initial_source_pick_until_dispatch() -> None:
    raw = _simple_problem(wafer_count=2, arm_type="dual_arm").model_dump(
        mode="json",
        by_alias=True,
    )
    for wafer in raw["initial_state"]["wafers"]:
        wafer["wafer_index"] = str(wafer["wafer_index"])
    raw["initial_state"]["wafers"][0]["priority"] = 10
    raw["initial_state"]["wafers"][1]["priority"] = 1
    engine = ClusterEngine(parse_problem(raw))
    engine.reset()

    assert engine.available_actions() == (
        PickAction("TM1", ("A", 1)),
    )
    engine.step(PickAction("TM1", ("A", 1)))
    assert PickAction("TM1", ("A", 0)) not in engine.available_actions()
    _advance_until(
        engine,
        lambda: engine.state.wafers[("A", 1)].robot_id == "TM1",
    )
    assert PickAction("TM1", ("A", 0)) in engine.available_actions()


def test_priority_does_not_filter_internal_pick() -> None:
    raw = _simple_problem(wafer_count=2, arm_type="dual_arm").model_dump(
        mode="json",
        by_alias=True,
    )
    raw["initial_state"]["wafers"] = [
        {
            "route_id": "A",
            "wafer_index": "0",
            "priority": 10,
            "step_index": 1,
            "location": {"kind": "module", "module_id": "PM1"},
            "process_end_time": 0,
        },
        {
            "route_id": "A",
            "wafer_index": "1",
            "priority": 1,
            "location": {"kind": "module", "module_id": "IO1"},
        },
    ]
    engine = ClusterEngine(parse_problem(raw))
    engine.reset()

    assert PickAction("TM1", ("A", 0)) in engine.available_actions()
    assert PickAction("TM1", ("A", 1)) in engine.available_actions()


def test_load_lock_conversion_is_internal_and_route_determined() -> None:
    problem = parse_problem(
        {
            "Modules": {
                "LP": {"type": "LP"},
                "LL": {
                    "type": "LL",
                    "load_lock": {
                        "initial_state": "atmosphere",
                        "atmosphere_to_vacuum_time": 5,
                        "vacuum_to_atmosphere_time": 7,
                        "tm_required_states": {
                            "ATM": "atmosphere",
                            "VTM": "vacuum",
                        },
                    },
                },
                "PM": {"type": "PM"},
            },
            "ClusterTool": {
                "ATM": {
                    "module_ids": ["LP", "LL"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                },
                "VTM": {
                    "module_ids": ["LL", "PM"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                },
            },
            "routes": {
                "A": [
                    {"module_id": "LL", "process_time": 0},
                    {"module_id": "PM", "process_time": 0},
                ]
            },
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0-1",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "LP"},
                    }
                ]
            },
        }
    )
    engine = ClusterEngine(problem)
    engine.reset()

    initial_ll = engine.load_lock_observation("LL")
    assert initial_ll.pump_time == 5
    assert initial_ll.vent_time == 7
    assert initial_ll.last_pick_side == "atmosphere"
    assert initial_ll.empty_transition_progress == 0
    assert initial_ll.occupied_exit_side is None
    assert initial_ll.occupied_transition_progress == 0

    engine.step(PickAction("ATM", ("A", 0)))
    _advance_until(engine, lambda: engine.state.wafers[("A", 0)].robot_id == "ATM")
    place_record = engine.step(PlaceAction(("A", 0), "LL"))
    assert place_record is not None
    _advance_until(engine, lambda: engine.state.wafers[("A", 0)].module_id == "LL")

    ll_state = engine.state.load_locks["LL"]
    assert ll_state.occupied_exit_side == "vacuum"
    assert ll_state.occupied_ready_at == place_record.end + 5
    occupied_ll = engine.load_lock_observation("LL")
    assert occupied_ll.empty_transition_progress == 0
    assert occupied_ll.occupied_exit_side == "vacuum"
    assert occupied_ll.occupied_transition_progress == 0
    assert PickAction("VTM", ("A", 0)) not in engine.available_actions()

    engine.step(PickAction("ATM", ("A", 1)))
    _advance_until(engine, lambda: engine.state.time >= ll_state.occupied_ready_at)
    assert engine.load_lock_observation("LL").occupied_transition_progress == 1
    assert PickAction("VTM", ("A", 0)) in engine.available_actions()
    pick_dispatch_time = engine.state.time
    vacuum_pick = engine.step(PickAction("VTM", ("A", 0)))
    assert vacuum_pick is not None
    assert vacuum_pick.start == pick_dispatch_time
    _advance_until(engine, lambda: engine.state.wafers[("A", 0)].robot_id == "VTM")

    assert PlaceAction(("A", 1), "LL") not in engine.available_actions()
    empty_ll = engine.load_lock_observation("LL")
    assert empty_ll.last_pick_side == "vacuum"
    assert empty_ll.empty_transition_progress == 0
    assert empty_ll.occupied_exit_side is None
    last_pick_end = engine.state.load_locks["LL"].last_pick_end
    empty_atmosphere_ready = last_pick_end + 7
    _advance_until(engine, lambda: engine.state.time >= empty_atmosphere_ready)
    assert engine.state.time - last_pick_end >= 7
    assert engine.load_lock_observation("LL").empty_transition_progress == 1
    assert PlaceAction(("A", 1), "LL") in engine.available_actions()
    place_dispatch_time = engine.state.time
    atmosphere_place = engine.step(PlaceAction(("A", 1), "LL"))
    assert atmosphere_place is not None
    assert atmosphere_place.start == place_dispatch_time


def test_load_lock_observation_rejects_non_conversion_module() -> None:
    engine = ClusterEngine(_simple_problem())
    engine.reset()

    try:
        engine.load_lock_observation("PM1")
    except ValueError as exc:
        assert "not a conversion Load Lock" in str(exc)
    else:
        raise AssertionError("non-conversion modules must not expose LL state")


def test_only_pick_and_place_create_dispatch_records() -> None:
    engine = ClusterEngine(_simple_problem())
    engine.reset()

    pick_record = engine.step(PickAction("TM1", ("A", 0)))
    advance_record = engine.step(ADVANCE)

    assert pick_record is not None
    assert pick_record.action_type == "pick"
    assert advance_record is None
    assert set(pick_record.to_dict()) == {
        "action_type",
        "tm_id",
        "module_id",
        "route_id",
        "wafer_index",
        "step_index",
        "start",
        "end",
    }
