from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from urllib.parse import quote, unquote

import pytest
from pydantic import ValidationError

from cluster_toolkit.cluster_engine import AdvanceAction, ClusterEngine, PickAction, PlaceAction
from cluster_toolkit.problem import ClusterProblem, load_problem, parse_problem
from cluster_toolkit.run_ir_compilation import main
from cluster_toolkit.validator.pipeline import ValidatorSuite
from cluster_toolkit.constraint_ir import (
    BindingDomainSpec, BindingRowSpec, ConstraintIRV1, DiagnosticCode, InitialStateSpec,
    LeaseSpec, ParameterSpec, ReferenceKernel, ReferenceSession, ReferenceValidator,
    ResourceSpec, ScheduleV1, SemanticError, StateAssignment, StateCellSpec,
    TerminalStateSpec, TimeDomain, compile_problem,
)


SCENARIOS = Path(__file__).resolve().parents[3] / "examples" / "scenarios"
TIME = TimeDomain(unit="second", ticks_per_unit=1)


def _raw(*, dual: bool = False, wafers: int = 1) -> dict:
    return {
        "Modules": {"LP": {"type": "LP"}, "PM1": {"type": "PM"}, "PM2": {"type": "PM"}},
        "ClusterTool": {"TM": {"module_ids": ["LP", "PM1", "PM2"],
                              "arm_type": "dual_arm" if dual else "single_arm",
                              "pick_time": 1, "place_time": 1, "travel_times": 1}},
        "routes": {"A": [{"module_ids": ["PM1", "PM2"], "process_time": 3},
                         {"module_id": "PM1", "process_time": 2}]},
        "initial_state": {"robots": {"TM": {"position_module_id": "LP"}},
                          "wafers": [{"route_id": "A", "wafer_index": f"0-{wafers-1}",
                                      "priority": 0, "location": {"kind": "module", "module_id": "LP"}}]},
    }


def _bound(candidate) -> dict[str, str]:
    return {item.parameter: item.value for item in candidate.bindings}


def _run_and_compare(problem: ClusterProblem, ir: ConstraintIRV1) -> ReferenceSession:
    """Serial witness, not an optimizing policy; compare every dispatch to the old Engine."""
    session = ReferenceKernel.start(ir)
    engine = ClusterEngine(problem)
    engine.reset()
    actions = []
    robot_id = next(iter(problem.ClusterTool))
    templates = {item.id: item for item in ir.operator_templates}
    scale = ir.time_domain.ticks_per_unit
    for wafer in sorted(problem.initial_state.wafers, key=lambda item: item.wafer_key):
        owner = f"wafer/{quote(wafer.route_id, safe='')}/{wafer.wafer_index}"
        for _ in range(2 * (len(problem.routes[wafer.route_id].visits) + 1)):
            for _ in range(10):
                frame = session.frame()
                candidates = [item for item in frame.intents if _bound(item)["wafer"] == owner]
                if candidates:
                    break
                advanced = session.advance_next()
                assert advanced is not None, "compiled workflow unexpectedly stuck"
                engine.step(AdvanceAction())
                assert advanced.final_snapshot.tick == round(engine.state.time * scale)
            assert candidates
            candidate = candidates[0]
            bindings = _bound(candidate)
            operation = next(item for item in templates[candidate.operator_template_id].intervals
                             if item.audit_kind in {"Pick", "Place"})
            module_id = unquote(bindings["holder"].split("/", 1)[1])
            action = (PickAction(robot_id=robot_id, wafer_key=wafer.wafer_key)
                      if operation.audit_kind == "Pick" else PlaceAction(
                          wafer_key=wafer.wafer_key, target_module_id=module_id))
            dispatch = engine.step(action)
            result = session.commit(frame.frame_token, (candidate.candidate_key,))
            actual = next(item for item in result.schedule.intervals
                          if item.origin_intent_id == candidate.id and item.audit_kind == operation.audit_kind)
            assert (actual.start_tick, actual.end_tick) == (round(dispatch.start * scale), round(dispatch.end * scale))
            actions.append({
                "action_type": dispatch.action_type, "module_id": module_id, "tm_id": robot_id,
                "route_id": wafer.route_id, "wafer_index": wafer.wafer_index,
                "step_index": dispatch.step_index, "start": dispatch.start, "end": dispatch.end,
                "arm_id": bindings["hand"].split("/")[-1],
            })
    while session.advance_next() is not None:
        engine.step(AdvanceAction())
    assert engine.is_complete()
    assert session.frame().intents == ()
    snapshot = session.snapshot()
    assert set(snapshot.kernel_snapshot.active_leases) == set(ir.terminal_state.leases)
    if scale == 1:
        old_report = ValidatorSuite(problem).validate(actions, require_complete=True, exact_action_durations=True)
        assert old_report.ok, old_report.issues
    # Legacy Validator compares float durations without tolerance. Fractional
    # tests use exact IR replay plus tick-by-tick Engine comparison above.
    report = ReferenceValidator.validate_session(ir, snapshot, require_terminal=True)
    assert report.ok, report.issues
    return session


@pytest.mark.parametrize("name", ["long_route_1w", "mixed_3pm_20w", "mixed_5pm_24w"])
def test_real_json_compiles_runs_and_passes_both_validators(name: str) -> None:
    problem = load_problem(SCENARIOS / f"{name}.json")
    original = problem.model_dump_json()
    ir = compile_problem(problem, TIME)
    assert not ir.intent_seeds  # All choices go through declared dynamic binding rows.
    restored_ir = ConstraintIRV1.model_validate_json(ir.canonical_json())
    assert restored_ir.problem_hash == ir.problem_hash
    session = _run_and_compare(problem, restored_ir)
    assert len(session.commit_log) == sum(2 * (len(problem.routes[w.route_id].visits) + 1)
                                          for w in problem.initial_state.wafers)
    assert problem.model_dump_json() == original


@pytest.mark.parametrize("travel", [0, 0.1])
@pytest.mark.parametrize("position", [None, "LP", "PM2"])
def test_fractional_zero_travel_and_unknown_initial_position(travel: float, position: str | None) -> None:
    raw = _raw()
    raw["ClusterTool"]["TM"].update(travel_times=travel, pick_time=0.2, place_time=0.3)
    raw["initial_state"]["robots"]["TM"]["position_module_id"] = position
    problem = parse_problem(raw)
    _run_and_compare(problem, compile_problem(problem, TimeDomain(unit="second", ticks_per_unit=10)))


def test_dual_arm_prefetch_uses_distinct_hands_and_one_motion_resource() -> None:
    ir = compile_problem(parse_problem(_raw(dual=True, wafers=2)), TIME)
    session = ReferenceKernel.start(ir)
    frame = session.frame()
    first = frame.intents[0]
    other = next(item for item in frame.intents
                 if _bound(item)["wafer"] != _bound(first)["wafer"]
                 and _bound(item)["hand"] != _bound(first)["hand"])
    before = session.snapshot().snapshot_hash
    with pytest.raises(SemanticError) as error:
        session.commit(frame.frame_token, (first.id, other.id))
    assert error.value.code is DiagnosticCode.RESOURCE_OVER_CAPACITY
    assert session.snapshot().snapshot_hash == before
    session.commit(frame.frame_token, (first.id,))
    session.advance_next()
    frame = session.frame()
    second = next(item for item in frame.intents if _bound(item)["wafer"] != _bound(first)["wafer"])
    assert _bound(second)["hand"] != _bound(first)["hand"]
    session.commit(frame.frame_token, (second.id,))
    session.advance_next()
    assert len([item for item in session.snapshot().kernel_snapshot.active_leases
                if item.resource_id.startswith("hand/")]) == 2
    assert ReferenceValidator.validate_session(ir, session.snapshot()).ok

    # A real swap sequence needs no new operation type: prefetch B, Pick A,
    # then Place B into the just-vacated PM while the other hand retains A.
    raw = _raw(dual=True, wafers=2)
    raw["routes"]["A"] = [{"module_id": "PM1", "process_time": 10}]
    ir = compile_problem(parse_problem(raw), TIME)
    session = ReferenceKernel.start(ir)

    def take(owner: str, holder: str):
        frame = session.frame()
        candidate = next(item for item in frame.intents
                         if _bound(item)["wafer"] == owner and _bound(item)["holder"] == holder)
        result = session.commit(frame.frame_token, (candidate.id,))
        action = next(item for item in result.schedule.intervals
                      if item.origin_intent_id == candidate.id and item.audit_kind in {"Pick", "Place"})
        session.advance_to(action.end_tick)
        return candidate, action

    take("wafer/A/0", "module/LP")
    take("wafer/A/0", "module/PM1")
    prefetched, _ = take("wafer/A/1", "module/LP")
    session.advance_to(13)
    picked, pick_out = take("wafer/A/0", "module/PM1")
    assert _bound(picked)["hand"] != _bound(prefetched)["hand"]
    _, place_in = take("wafer/A/1", "module/PM1")
    assert pick_out.end_tick == place_in.start_tick
    take("wafer/A/0", "module/LP")
    session.advance_to(26)
    take("wafer/A/1", "module/PM1")
    take("wafer/A/1", "module/LP")
    assert ReferenceValidator.validate_session(ir, session.snapshot(), require_terminal=True).ok


def test_alternative_choice_reentry_and_midway_restore() -> None:
    ir = compile_problem(parse_problem(_raw()), TIME)
    session = ReferenceKernel.start(ir)
    frame = session.frame()
    session.commit(frame.frame_token, (frame.intents[0].id,))
    session.advance_next()
    frame = session.frame()
    assert {_bound(item)["holder"] for item in frame.intents} == {"module/PM1", "module/PM2"}
    with pytest.raises(SemanticError) as error:
        session.commit(frame.frame_token, tuple(item.id for item in frame.intents))
    assert error.value.code is DiagnosticCode.CHOICE_SCOPE_CONFLICT
    chosen = next(item for item in frame.intents if _bound(item)["holder"] == "module/PM1")
    session.commit(frame.frame_token, (chosen.id,))
    restored = ReferenceKernel.restore(ir, session.snapshot().canonical_json())
    for current in (session, restored):
        for _ in range(30):
            frame = current.frame()
            if frame.intents:
                current.commit(frame.frame_token, (frame.intents[0].id,))
            elif current.advance_next() is None:
                break
    assert session.snapshot().canonical_json() == restored.snapshot().canonical_json()
    processes = [item for item in session.schedule.intervals if item.audit_kind == "Process"]
    assert len(processes) == 2
    assert {dict((b.parameter, b.value) for b in item.bindings)["holder"] for item in processes} == {"module/PM1"}
    assert ReferenceValidator.validate_session(ir, session.snapshot(), require_terminal=True).ok


def test_process_blocks_early_pick_but_not_other_robot_work() -> None:
    raw = _raw(wafers=2)
    raw["routes"]["A"] = [{"module_ids": ["PM1", "PM2"], "process_time": 20}]
    ir = compile_problem(parse_problem(raw), TIME)
    session = ReferenceKernel.start(ir)

    def choose(owner: str, holder: str) -> None:
        frame = session.frame()
        candidate = next(item for item in frame.intents
                         if _bound(item)["wafer"] == owner and _bound(item)["holder"] == holder)
        session.commit(frame.frame_token, (candidate.id,))

    choose("wafer/A/0", "module/LP")
    session.advance_to(1)
    choose("wafer/A/0", "module/PM1")
    session.advance_to(3)  # Place ended, first Process is still running until 23.
    assert all(_bound(item)["wafer"] != "wafer/A/0" for item in session.frame().intents)
    choose("wafer/A/1", "module/LP")
    session.advance_to(5)
    choose("wafer/A/1", "module/PM2")
    session.advance_to(7)
    processes = [item for item in session.schedule.intervals if item.audit_kind == "Process"]
    assert len(processes) == 2
    assert max(item.start_tick for item in processes) < min(item.end_tick for item in processes)
    assert not session.frame().intents
    session.advance_to(23)
    assert {_bound(item)["wafer"] for item in session.frame().intents} == {"wafer/A/0"}
    assert ReferenceValidator.validate_session(ir, session.snapshot()).ok


def test_schema_v2_capabilities_and_reachability_are_lowered_to_bindings() -> None:
    raw = _raw()
    raw["schema_version"] = 2
    raw["Modules"]["LP"]["type"] = "IO"
    raw["Modules"]["PM1"]["process_ids"] = ["etch", "coat"]
    raw["Modules"]["PM2"]["process_ids"] = ["etch"]
    raw["routes"]["A"][0]["process_id"] = "etch"
    raw["routes"]["A"][1]["process_id"] = "coat"
    raw["ClusterTool"]["TM"]["module_ids"].remove("PM2")
    problem = parse_problem(raw)
    ir = compile_problem(problem, TIME)
    assert all("module/PM2" not in row.values for domain in ir.binding_domains for row in domain.rows)
    _run_and_compare(problem, ir)
    raw["ClusterTool"]["TM"]["module_ids"] = ["LP", "PM2"]
    with pytest.raises(SemanticError, match="no candidate reachable"):
        compile_problem(parse_problem(raw), TIME)


def test_explicit_return_and_encoded_names_and_input_order() -> None:
    raw = _raw()
    raw["Modules"]["return/%"] = {"type": "LP"}
    raw["ClusterTool"]["TM"]["module_ids"].append("return/%")
    raw["routes"]["A/%"] = raw["routes"].pop("A")
    raw["initial_state"]["wafers"][0].update(route_id="A/%", return_lp_id="return/%")
    problem = parse_problem(raw)
    ir = compile_problem(problem, TIME)
    assert ir.terminal_state.leases[0].resource_id == "module/return%2F%25"
    _run_and_compare(problem, ir)
    reordered = deepcopy(raw)
    reordered["Modules"] = dict(reversed(tuple(raw["Modules"].items())))
    reordered["routes"]["A/%"][0]["module_ids"].reverse()
    reordered["ClusterTool"]["TM"]["module_ids"].reverse()
    assert compile_problem(parse_problem(reordered), TIME).canonical_json() == ir.canonical_json()


@pytest.mark.parametrize(("mutation", "path"), [
    ("cleaning", "cleaning"), ("jit", "just_in_time"), ("residency", "residency_time"),
    ("zero_pick", "pick_time"), ("zero_process", "process_time"), ("missing_process", "process_time"),
    ("capacity", "capacity"), ("ll", "type"), ("priority", "priority"),
    ("inflight", "wafers"), ("robots", "ClusterTool"),
])
def test_unsupported_semantics_fail_explicitly(mutation: str, path: str) -> None:
    raw = _raw(wafers=2)
    if mutation == "cleaning":
        raw["cleaning"] = {"module_ids": ["PM1"], "process_switch": {"clean_time": 1}}
    elif mutation == "jit":
        raw["just_in_time"] = {"residency_time": 5}
    elif mutation == "residency":
        raw["routes"]["A"][0]["residency_time"] = 5
    elif mutation == "zero_pick":
        raw["ClusterTool"]["TM"]["pick_time"] = 0
    elif mutation in {"zero_process", "missing_process"}:
        raw["routes"]["A"][0]["process_time"] = 0 if mutation == "zero_process" else None
    elif mutation == "capacity":
        raw["Modules"]["PM1"]["capacity"] = 2
    elif mutation == "ll":
        raw["Modules"]["LL"] = {"type": "LL"}
        raw["ClusterTool"]["TM"]["module_ids"].append("LL")
    elif mutation == "priority":
        wafer = raw["initial_state"]["wafers"][0]
        wafer["wafer_index"] = "0"
        raw["initial_state"]["wafers"].append({**wafer, "wafer_index": "1", "priority": 1})
    elif mutation == "inflight":
        raw["initial_state"]["wafers"][0]["process_end_time"] = 2
    else:
        raw["ClusterTool"]["TM2"] = raw["ClusterTool"]["TM"].copy()
    problem = parse_problem(raw)
    with pytest.raises(SemanticError) as error:
        compile_problem(problem, TIME)
    assert error.value.code is DiagnosticCode.UNSUPPORTED_FEATURE
    assert path in error.value.path


def test_precision_loss_is_not_rounded() -> None:
    raw = _raw()
    raw["ClusterTool"]["TM"]["pick_time"] = 0.1
    with pytest.raises(SemanticError) as error:
        compile_problem(parse_problem(raw), TIME)
    assert error.value.code is DiagnosticCode.TIME_PRECISION_LOSS


def test_binding_canonicalization_preserves_columns_and_detects_changed_assignments() -> None:
    parameters = (ParameterSpec(name="z", kind="resource"), ParameterSpec(name="a", kind="resource"))
    domain = BindingDomainSpec(id="table", parameters=parameters, rows=(BindingRowSpec(values=("a", "z")),))
    ir = ConstraintIRV1(time_domain=TIME, resources=(ResourceSpec(id="a", capacity=1), ResourceSpec(id="z", capacity=1)),
                        binding_domains=(domain,))
    parsed = ConstraintIRV1.model_validate_json(ir.canonical_json())
    assert dict(zip((p.name for p in parsed.binding_domains[0].parameters),
                    parsed.binding_domains[0].rows[0].values)) == {"z": "a", "a": "z"}
    assert parsed.problem_hash == ir.problem_hash
    reordered = ir.model_copy(update={"binding_domains": (domain.model_copy(update={
        "parameters": tuple(reversed(parameters)), "rows": (BindingRowSpec(values=("z", "a")),),
    }),)})
    assert reordered.problem_hash == ir.problem_hash
    changed = ir.model_copy(update={"binding_domains": (domain.model_copy(update={
        "rows": (BindingRowSpec(values=("z", "a")),),
    }),)})
    assert changed.problem_hash != ir.problem_hash


def test_terminal_requires_goal_progress_and_exact_ownership() -> None:
    ir = compile_problem(parse_problem(_raw()), TIME)
    initial = ReferenceKernel.start(ir).snapshot()
    assert not ReferenceValidator.validate_session(ir, initial, require_terminal=True).ok
    # Open return leases are permitted only when declared; exact ownership is
    # checked in addition to progress, not discarded to fake completion.
    goal = TerminalStateSpec(leases=(LeaseSpec(resource_id="slot", owner_id="A"),),
                             state_values=(StateAssignment(cell_id="done", value=True),))
    problem = ConstraintIRV1(time_domain=TIME, resources=(ResourceSpec(id="slot", capacity=2),),
                            state_cells=(StateCellSpec(id="done", value_type="bool"),),
                            initial_state=InitialStateSpec(state_values=goal.state_values, leases=goal.leases),
                            terminal_state=goal)
    assert ReferenceValidator.validate(problem, ScheduleV1(), require_terminal=True).ok
    assert ReferenceValidator.validate_session(problem, ReferenceKernel.start(problem).snapshot(),
                                               require_terminal=True).ok
    for state in (
        problem.initial_state.model_copy(update={"state_values": (StateAssignment(cell_id="done", value=False),)}),
        problem.initial_state.model_copy(update={"leases": ()}),
        problem.initial_state.model_copy(update={"leases": (LeaseSpec(resource_id="slot", owner_id="B"),)}),
    ):
        assert not ReferenceValidator.validate(problem.model_copy(update={"initial_state": state}),
                                               ScheduleV1(), require_terminal=True).ok
        changed = problem.model_copy(update={"initial_state": state})
        assert not ReferenceValidator.validate_session(changed, ReferenceKernel.start(changed).snapshot(),
                                                       require_terminal=True).ok
    with pytest.raises(ValidationError, match="terminal"):
        ConstraintIRV1.model_validate(problem.model_copy(update={
            "terminal_state": TerminalStateSpec(leases=(LeaseSpec(resource_id="unknown", owner_id="A"),)),
        }).model_dump())


def test_cli_writes_loadable_ir_without_overwriting(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    output = tmp_path / "problem.ir.json"
    args = [str(SCENARIOS / "long_route_1w.json"), "--output", str(output)]
    assert main(args) == 0
    saved = output.read_text()
    assert ConstraintIRV1.model_validate_json(saved).terminal_state is not None
    assert main(args) == 2
    assert output.read_text() == saved
    raw = _raw()
    raw["just_in_time"] = {"residency_time": 4}
    source = tmp_path / "unsupported.json"
    source.write_text(json.dumps(raw))
    rejected = tmp_path / "rejected.ir.json"
    assert main([str(source), "--output", str(rejected)]) == 2
    assert not rejected.exists()
    assert "UNSUPPORTED_FEATURE" in capsys.readouterr().err
