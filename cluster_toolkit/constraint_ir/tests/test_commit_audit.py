"""Audit persisted evidence, including internally rehashed forgeries."""

from collections.abc import Callable
from itertools import permutations

import pytest

from cluster_toolkit.constraint_ir import (
    AutomaticRuleSpec,
    ChoiceScopeClaimSpec,
    CommitRecordSpec,
    ConstraintIRV1,
    DiagnosticCode,
    InitialStateSpec,
    ObligationInstanceSpec,
    ReferenceKernel,
    ReferenceSession,
    ReferenceValidator,
    SatisfyObligationTemplateEffect,
    ScheduleV1,
    SemanticError,
    SessionSnapshot,
    StateAssignment,
    StateCondition,
)
from cluster_toolkit.constraint_ir.schema import canonical_digest
from cluster_toolkit.constraint_ir.tests.test_continuous_session import _activity, _repeated_problem
from cluster_toolkit.constraint_ir.tests.test_golden_cases import (
    _composite_exchange_problem,
    _dynamic_pick_problem,
    _g05_problem,
    _g06_problem,
    _g07_problem,
    _g08_problem,
    _g09_problem,
    _g10_problem,
)


def _commit_first(problem: ConstraintIRV1) -> ReferenceSession:
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    session.commit(frame.frame_token, (frame.intents[0].candidate_key,))
    return session


def _rehash(record: CommitRecordSpec, **changes: object) -> CommitRecordSpec:
    changed = record.model_copy(update=changes)
    return changed.model_copy(update={
        "commit_id": canonical_digest(changed.model_dump(exclude={"commit_id"})),
    })


def _replace_selection(snapshot: SessionSnapshot, **changes: object) -> SessionSnapshot:
    record = snapshot.commit_log[0]
    selection = record.selections[0].model_copy(update=changes)
    return snapshot.model_copy(update={
        "commit_log": (_rehash(record, selections=(selection,)),),
    })


def _assert_rejected(
    problem: ConstraintIRV1, snapshot: SessionSnapshot | str, code: DiagnosticCode,
) -> None:
    report = ReferenceValidator.validate_session(problem, snapshot)
    assert not report.ok
    assert report.issues[0].code is code, report.issues
    assert report.final_snapshot is None


@pytest.mark.parametrize("factory", [
    _g05_problem, _g06_problem, _g07_problem,
    lambda: _g08_problem(motion_capacity=2, geometry_compatible=False),
    _g09_problem, _g10_problem, _composite_exchange_problem, _dynamic_pick_problem,
])
def test_audit_legacy_and_dynamic_golden_artifacts_at_each_boundary(
    factory: Callable[[], ConstraintIRV1],
) -> None:
    problem = factory()
    session = _commit_first(problem)
    while True:
        snapshot = session.snapshot()
        original = snapshot.canonical_json()
        for artifact in (snapshot, original):
            report = ReferenceValidator.validate_session(problem, artifact)
            assert report.ok, report.issues
            assert report.final_snapshot.canonical_dict() == snapshot.kernel_snapshot.canonical_dict()
        assert session.snapshot().canonical_json() == original
        try:
            if session.advance_next() is None:
                break
        except SemanticError as error:
            # G07 intentionally leaves a newly created obligation for another
            # decision; the valid checkpoint must not be rejected prematurely.
            assert error.code is DiagnosticCode.DEADLINE_MISSED
            break


def test_audit_repeated_commits_uses_chain_not_wire_array_order() -> None:
    problem = _repeated_problem(with_lease=True)
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    session.commit(frame.frame_token, tuple(item.id for item in frame.intents))
    for _ in range(2):
        session.advance_next()
        frame = session.frame()
        session.commit(frame.frame_token, (frame.intents[0].id,))
    snapshot = session.snapshot()
    for records in permutations(snapshot.commit_log):
        artifact = snapshot.model_copy(update={"commit_log": records})
        report = ReferenceValidator.validate_session(problem, artifact)
        assert report.ok, report.issues
        restored = ReferenceKernel.restore(problem, artifact)
        assert restored.commit_log == session.commit_log
        assert restored.frame() == session.frame()
    restored = ReferenceKernel.restore(problem, snapshot.canonical_json())
    restored.advance_to(15)
    session.advance_to(15)
    assert restored.frame() == session.frame()
    for current in (session, restored):
        frame = current.frame()
        current.commit(frame.frame_token, (frame.intents[0].id,))
    assert session.snapshot().canonical_json() == restored.snapshot().canonical_json()
    report = ReferenceValidator.validate_session(problem, snapshot.canonical_json())
    assert report.ok, report.issues
    session.advance_to(20)
    report = ReferenceValidator.validate_session(problem, session.snapshot(), require_terminal=True)
    assert report.ok, report.issues


def test_audit_same_tick_commits_preserve_decision_rounds() -> None:
    problem = _repeated_problem(with_lease=True)
    session = ReferenceKernel.start(problem)
    session.commit(session.frame().frame_token, ("background",))
    frame = session.frame()
    session.commit(frame.frame_token, (frame.intents[0].id,))
    assert {event.decision_round for event in session.schedule.events if event.tick == 0} == {1, 2}
    report = ReferenceValidator.validate_session(problem, session.snapshot().canonical_json())
    assert report.ok, report.issues


@pytest.mark.parametrize(("changes", "code"), [
    ({"candidate_digest": "0" * 64}, DiagnosticCode.CANDIDATE_DIGEST_MISMATCH),
    ({"candidate_key": "0" * 64}, DiagnosticCode.CANDIDATE_DIGEST_MISMATCH),
    ({"intent_instance_id": "0" * 64}, DiagnosticCode.COMMIT_LOG_MISMATCH),
    ({"operator_instance_ids": ("fabricated",)}, DiagnosticCode.COMMIT_LOG_MISMATCH),
    ({"source_intent_id": "undeclared"}, DiagnosticCode.INTENT_NOT_COMMITTABLE),
    ({"earliest_start_offset": 1}, DiagnosticCode.INTENT_NOT_COMMITTABLE),
    ({"choice_scope_claims": ()}, DiagnosticCode.CHOICE_SCOPE_CONFLICT),
])
def test_audit_rejects_rehashed_selection_fields(
    changes: dict[str, object], code: DiagnosticCode,
) -> None:
    problem = _dynamic_pick_problem()
    snapshot = _commit_first(problem).snapshot()
    _assert_rejected(problem, _replace_selection(snapshot, **changes), code)


@pytest.mark.parametrize("field", ["frame_token", "tick", "expanded_schedule_digest"])
def test_audit_rejects_rehashed_commit_fields(field: str) -> None:
    problem = _g05_problem()
    session = _commit_first(problem)
    snapshot = session.snapshot(1)
    record = _rehash(snapshot.commit_log[0], **{field: 1 if field == "tick" else "0" * 64})
    artifact = snapshot.model_copy(update={"commit_log": (record,)})
    code = DiagnosticCode.COMMIT_LOG_MISMATCH if field == "expanded_schedule_digest" else DiagnosticCode.STALE_FRAME
    _assert_rejected(problem, artifact, code)


def test_audit_rejects_missing_and_disconnected_history() -> None:
    problem = _repeated_problem()
    session = _commit_first(problem)
    snapshot = session.snapshot()
    _assert_rejected(problem, snapshot.model_copy(update={"commit_log": ()}), DiagnosticCode.COMMIT_LOG_MISMATCH)
    _assert_rejected(problem, snapshot.model_copy(update={
        "commit_log": (), "revision": 0,
    }), DiagnosticCode.OPERATOR_CONFORMANCE_MISMATCH)
    record = _rehash(snapshot.commit_log[0], previous_commit_id="0" * 64)
    _assert_rejected(problem, snapshot.model_copy(update={
        "commit_log": (record,),
    }), DiagnosticCode.COMMIT_LOG_MISMATCH)
    _assert_rejected(problem, snapshot.model_copy(update={
        "commit_log": (snapshot.commit_log[0],) * 2, "revision": 2,
    }), DiagnosticCode.COMMIT_LOG_MISMATCH)


@pytest.mark.parametrize("mutation", ["missing_event", "duration", "claim", "effect", "round", "origin"])
def test_audit_rejects_schedule_forgery_after_rehashing(mutation: str) -> None:
    problem = _g05_problem()
    snapshot = _commit_first(problem).snapshot()
    events, intervals = list(snapshot.schedule.events), list(snapshot.schedule.intervals)
    if mutation == "missing_event":
        events.pop()
    elif mutation == "duration":
        intervals[0] = intervals[0].model_copy(update={"end_tick": intervals[0].end_tick + 1})
    elif mutation == "claim":
        use = intervals[0].resource_uses[0]
        intervals[0] = intervals[0].model_copy(update={
            "resource_uses": (use.model_copy(update={"amount": use.amount + 1}),),
        })
    elif mutation == "effect":
        index = next(i for i, event in enumerate(events) if event.effects)
        events[index] = events[index].model_copy(update={
            "effects": (), "effect_digest": canonical_digest([]),
        })
    elif mutation == "round":
        events[0] = events[0].model_copy(update={"decision_round": 5})
    else:
        events[0] = events[0].model_copy(update={"origin_intent_id": "fabricated"})
    schedule = ScheduleV1(events=tuple(events), intervals=tuple(intervals))
    artifact = snapshot.model_copy(update={"schedule": schedule, "schedule_hash": schedule.schedule_hash})
    _assert_rejected(problem, artifact, DiagnosticCode.OPERATOR_CONFORMANCE_MISMATCH)
    artifact = artifact.model_copy(update={"commit_log": (
        _rehash(snapshot.commit_log[0], expanded_schedule_digest=schedule.schedule_hash),
    )})
    _assert_rejected(problem, artifact, DiagnosticCode.COMMIT_LOG_MISMATCH)


def _retarget_first_commit(
    snapshot: SessionSnapshot, problem: ConstraintIRV1,
) -> SessionSnapshot:
    """Rehash all outer evidence after changing a prerequisite in the Problem."""
    empty = ReferenceKernel.start(problem).snapshot()
    record = snapshot.commit_log[0]
    selections = tuple(item.model_copy(update={
        "intent_instance_id": canonical_digest({
            "frame_token": empty.frame_token, "candidate_key": item.candidate_key,
        }),
    }) for item in record.selections)
    return snapshot.model_copy(update={
        "problem_hash": problem.problem_hash,
        "commit_log": (_rehash(record, frame_token=empty.frame_token, selections=selections),),
    })


@pytest.mark.parametrize("prerequisite", ["guard", "obligation"])
def test_audit_checks_prerequisites_not_just_physical_schedule(prerequisite: str) -> None:
    problem = _repeated_problem()
    session = ReferenceKernel.start(problem)
    session.commit(session.frame().frame_token, ("background",))
    seed = problem.intent_seeds[0]
    updates = {"guards": (StateCondition(cell_id="completed", operator="equal", value=1),)}
    if prerequisite == "obligation":
        updates = {"required_obligation_ids": ("missing",)}
    changed = problem.model_copy(update={"intent_seeds": (seed.model_copy(update=updates),)})
    artifact = _retarget_first_commit(session.snapshot(), changed)
    _assert_rejected(changed, artifact, DiagnosticCode.INTENT_NOT_COMMITTABLE)


def test_audit_rejects_rehashed_final_state_and_scope() -> None:
    problem = _repeated_problem()
    snapshot = _commit_first(problem).snapshot()
    state = snapshot.kernel_snapshot.model_copy(update={
        "state_values": (StateAssignment(cell_id="completed", value=999),),
    })
    _assert_rejected(problem, snapshot.model_copy(update={
        "kernel_snapshot": state, "kernel_state_hash": state.state_hash,
    }), DiagnosticCode.SNAPSHOT_STATE_MISMATCH)
    _assert_rejected(problem, snapshot.model_copy(update={
        "active_choice_scope_keys": (),
    }), DiagnosticCode.SNAPSHOT_STATE_MISMATCH)


@pytest.mark.parametrize("factory", [_g06_problem, _dynamic_pick_problem])
def test_audit_is_independent_of_kernel_generator_and_session_expansion(
    monkeypatch: pytest.MonkeyPatch, factory: Callable[[], ConstraintIRV1],
) -> None:
    from cluster_toolkit.constraint_ir import reference_session

    problem = factory()
    snapshot = _commit_first(problem).snapshot()

    def forbidden(*args, **kwargs):
        raise AssertionError("audit used the production execution path")

    for method in ("execute", "execute_until", "start", "restore"):
        monkeypatch.setattr(ReferenceKernel, method, forbidden)
    monkeypatch.setattr(reference_session, "_expand_intents", forbidden)
    monkeypatch.setattr(reference_session.ExhaustiveReferenceCandidateGenerator, "generate", forbidden)
    report = ReferenceValidator.validate_session(problem, snapshot.canonical_json())
    assert report.ok, report.issues
    _assert_rejected(problem, _replace_selection(snapshot, candidate_digest="0" * 64),
                     DiagnosticCode.CANDIDATE_DIGEST_MISMATCH)


def test_audit_terminal_rejects_future_work_and_unbounded_obligations() -> None:
    base = _repeated_problem()
    problem = base.model_copy(update={"intent_seeds": (
        base.intent_seeds[0].model_copy(update={"earliest_start_offset": 3}),
    )})
    session = ReferenceKernel.start(problem)
    session.commit(session.frame().frame_token, ("background",))
    snapshot = session.snapshot()
    assert snapshot.kernel_snapshot.active_interval_ids == ()
    report = ReferenceValidator.validate_session(problem, snapshot, require_terminal=True)
    assert report.issues[0].code is DiagnosticCode.NON_TERMINAL_STATE
    session.advance_to(23)
    assert ReferenceValidator.validate_session(problem, session.snapshot(), require_terminal=True).ok

    problem = base.model_copy(update={"initial_state": InitialStateSpec(
        state_values=base.initial_state.state_values,
        obligations=(ObligationInstanceSpec(id="required"),),
    )})
    snapshot = ReferenceKernel.start(problem).snapshot()
    assert ReferenceValidator.validate_session(problem, snapshot).ok
    report = ReferenceValidator.validate_session(problem, snapshot, require_terminal=True)
    assert report.issues[0].code is DiagnosticCode.NON_TERMINAL_STATE


def test_audit_reports_malformed_wire_input() -> None:
    _assert_rejected(_g05_problem(), "not json", DiagnosticCode.INVALID_SCHEDULE)


def test_restore_uses_independent_audit_before_trusting_selection() -> None:
    problem = _dynamic_pick_problem()
    snapshot = _commit_first(problem).snapshot()
    artifact = _replace_selection(snapshot, candidate_digest="0" * 64)
    with pytest.raises(SemanticError) as error:
        ReferenceKernel.restore(problem, artifact)
    assert error.value.code is DiagnosticCode.CANDIDATE_DIGEST_MISMATCH


@pytest.mark.parametrize("constraint", ["capacity", "scope", "alternative"])
def test_audit_checks_batch_legality_after_individual_candidates(constraint: str) -> None:
    problem = _g08_problem(motion_capacity=1 if constraint == "capacity" else 2,
                           geometry_compatible=True)
    if constraint == "scope":
        problem = problem.model_copy(update={"intent_seeds": tuple(seed.model_copy(update={
            "choice_scope_claims": (ChoiceScopeClaimSpec(scope_key="shared"),),
        }) for seed in problem.intent_seeds)})
    elif constraint == "alternative":
        problem = _g09_problem()
    frame = ReferenceKernel.start(problem).frame()
    choices = list(frame.intents)
    if constraint == "alternative":
        choices = [item for item in choices
                   if item.alternative_group_id == choices[0].alternative_group_id]
    assert len(choices) >= 2
    snapshots = []
    for candidate in choices[:2]:
        session = ReferenceKernel.start(problem)
        session.commit(frame.frame_token, (candidate.candidate_key,))
        snapshots.append(session.snapshot())
    combined = ScheduleV1(
        events=tuple(event for snapshot in snapshots for event in snapshot.schedule.events),
        intervals=tuple(item for snapshot in snapshots for item in snapshot.schedule.intervals),
    )
    record = _rehash(snapshots[0].commit_log[0],
                     selections=tuple(snapshot.commit_log[0].selections[0] for snapshot in snapshots),
                     expanded_schedule_digest=combined.schedule_hash)
    artifact = snapshots[0].model_copy(update={
        "commit_log": (record,), "schedule": combined, "schedule_hash": combined.schedule_hash,
    })
    _assert_rejected(problem, artifact, {
        "capacity": DiagnosticCode.RESOURCE_OVER_CAPACITY,
        "scope": DiagnosticCode.CHOICE_SCOPE_CONFLICT,
        "alternative": DiagnosticCode.ALTERNATIVE_GROUP_CONFLICT,
    }[constraint])


@pytest.mark.parametrize("constraint", ["guard", "domain"])
def test_dynamic_source_must_satisfy_declared_guard_and_domain(constraint: str) -> None:
    problem = _dynamic_pick_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    candidate = next(item for item in frame.intents
                     if any(binding.value == "wafer.B" for binding in item.bindings))
    session.commit(frame.frame_token, (candidate.id,))
    if constraint == "guard":
        changed = _dynamic_pick_problem(second_ready=False)
    else:
        domain = problem.binding_domains[0]
        changed = problem.model_copy(update={"binding_domains": (domain.model_copy(update={
            "rows": tuple(row for row in domain.rows if "wafer.B" not in row.values),
        }),)})
    artifact = _retarget_first_commit(session.snapshot(), changed)
    _assert_rejected(changed, artifact, DiagnosticCode.INTENT_NOT_COMMITTABLE)


def test_audit_repeated_dynamic_operations_with_automatic_child_scopes() -> None:
    base = _repeated_problem()
    child = _activity("child", 2, "background").model_copy(update={"origin": "automatic"})
    problem = base.model_copy(update={
        "operator_templates": base.operator_templates + (child,),
        "automatic_rules": (AutomaticRuleSpec(
            id="child_after_start", trigger_operator_template_id="work",
            trigger_boundary_id="step.start", emit_operator_template_id="child",
        ),),
    })
    session = ReferenceKernel.start(problem)
    for _ in range(3):
        frame = session.frame()
        candidate = next(item for item in frame.intents if item.operator_template_id == "work")
        session.commit(frame.frame_token, (candidate.id,))
        for _ in range(2):
            session.advance_next()
            report = ReferenceValidator.validate_session(problem, session.snapshot().canonical_json())
            assert report.ok, report.issues


def test_audit_actual_advancement_does_not_defer_finite_deadlines() -> None:
    base = _repeated_problem()
    problem = base.model_copy(update={"initial_state": InitialStateSpec(
        state_values=base.initial_state.state_values,
        obligations=(ObligationInstanceSpec(id="required", deadline_tick=5),),
    )})
    session = ReferenceKernel.start(problem)
    session.commit(session.frame().frame_token, ("background",))
    snapshot = session.snapshot(4)
    assert ReferenceValidator.validate_session(problem, snapshot).ok
    state = snapshot.kernel_snapshot.model_copy(update={"tick": 5})
    artifact = snapshot.model_copy(update={
        "tick": 5, "kernel_snapshot": state, "kernel_state_hash": state.state_hash,
    })
    _assert_rejected(problem, artifact, DiagnosticCode.DEADLINE_MISSED)


def test_audit_conditional_obligation_creation_and_gated_completion() -> None:
    problem = _g07_problem()
    session = ReferenceKernel.start(problem)
    for name in ("process.B", "clean.PM1"):
        session.commit(session.frame().frame_token, (name,))
        report = ReferenceValidator.validate_session(problem, session.snapshot())
        assert report.ok, report.issues
        session.advance_next()
    report = ReferenceValidator.validate_session(problem, session.snapshot(), require_terminal=True)
    assert report.ok, report.issues


@pytest.mark.parametrize("deadline", [None, 20])
def test_audit_obligation_satisfaction_and_explicit_late_projection(deadline: int | None) -> None:
    base = _repeated_problem()
    template = _activity("background", 20, "background", end_effects=(
        SatisfyObligationTemplateEffect(obligation_id="required"),
    ))
    problem = base.model_copy(update={
        "operator_templates": (base.operator_templates[0], template),
        "initial_state": InitialStateSpec(
            state_values=base.initial_state.state_values,
            obligations=(ObligationInstanceSpec(id="required", deadline_tick=deadline),),
        ),
    })
    session = ReferenceKernel.start(problem)
    session.commit(session.frame().frame_token, ("background",))
    snapshot = session.snapshot()
    assert ReferenceValidator.validate_session(problem, snapshot).ok
    session.advance_to(20)
    report = ReferenceValidator.validate_session(problem, session.snapshot(), require_terminal=True)
    assert report.ok, report.issues

    late_problem = problem.model_copy(update={"initial_state": InitialStateSpec(
        state_values=base.initial_state.state_values,
        obligations=(ObligationInstanceSpec(id="required", deadline_tick=19),),
    )})
    _assert_rejected(late_problem, _retarget_first_commit(snapshot, late_problem),
                     DiagnosticCode.DEADLINE_MISSED)
