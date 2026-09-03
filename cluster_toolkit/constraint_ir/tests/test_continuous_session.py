from itertools import permutations

import pytest

from cluster_toolkit.constraint_ir import (
    AcquireLeaseEffect,
    AcquireLeaseTemplateEffect,
    AutomaticRuleSpec,
    BindingDomainSpec,
    BindingRowSpec,
    ChoiceScopeTemplateSpec,
    ConstraintIRV1,
    CreateObligationEffect,
    CreateObligationTemplateEffect,
    DiagnosticCode,
    DynamicIntentSpec,
    EventSpec,
    IncrementStateTemplateEffect,
    InitialStateSpec,
    IntentSeedSpec,
    IntervalTemplateSpec,
    LiteralIdRef,
    ObligationInstanceSpec,
    OperatorTemplateSpec,
    ParameterIdRef,
    ParameterSpec,
    ReferenceKernel,
    ReferenceValidator,
    ReleaseLeaseEffect,
    ReleaseLeaseTemplateEffect,
    ResourceSpec,
    ResourceUseTemplate,
    SatisfyObligationTemplateEffect,
    ScheduleV1,
    SemanticError,
    SetStateEffect,
    StateAssignment,
    StateCellSpec,
    StateCondition,
    TimeDomain,
)


def _activity(
    template_id: str, duration: int, resource_id: str, **kwargs: object,
) -> OperatorTemplateSpec:
    return OperatorTemplateSpec(
        id=template_id,
        origin="selectable",
        intervals=(IntervalTemplateSpec(
            id="step", start_offset=0, duration=duration, audit_kind="Activity",
            resource_uses=(ResourceUseTemplate(resource=LiteralIdRef(value=resource_id)),),
            **kwargs,
        ),),
    )


def _repeated_problem(*, release: bool = True, with_lease: bool = False) -> ConstraintIRV1:
    parameters = (ParameterSpec(name="machine", kind="resource"),)
    start_effects = ()
    end_effects = (IncrementStateTemplateEffect(cell=LiteralIdRef(value="completed"), delta=1),)
    if with_lease:
        start_effects = (AcquireLeaseTemplateEffect(
            resource=LiteralIdRef(value="holder"), owner=LiteralIdRef(value="item"),
        ),)
        end_effects += (ReleaseLeaseTemplateEffect(
            resource=LiteralIdRef(value="holder"), owner=LiteralIdRef(value="item"),
        ),)
    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=tuple(ResourceSpec(id=name, capacity=1) for name in (
            "machine", "background", "holder",
        )),
        state_cells=(StateCellSpec(id="completed", value_type="int"),),
        initial_state=InitialStateSpec(
            state_values=(StateAssignment(cell_id="completed", value=0),),
        ),
        operator_templates=(
            OperatorTemplateSpec(
                id="work", origin="selectable", parameters=parameters,
                intervals=(IntervalTemplateSpec(
                    id="step", start_offset=0, duration=5, audit_kind="Activity",
                    resource_uses=(ResourceUseTemplate(
                        resource=ParameterIdRef(parameter="machine"),
                    ),),
                    start_effects=start_effects, end_effects=end_effects,
                ),),
            ),
            _activity("background", 20, "background"),
        ),
        binding_domains=(BindingDomainSpec(
            id="machines", parameters=parameters,
            rows=(BindingRowSpec(values=("machine",)),),
        ),),
        dynamic_intents=(DynamicIntentSpec(
            id="work", operator_template_id="work", binding_domain_id="machines",
            choice_scope_templates=(ChoiceScopeTemplateSpec(
                scope_prefix="work", identity_parameters=("machine",),
                release_boundary_id="step.end" if release else None,
            ),),
        ),),
        intent_seeds=(IntentSeedSpec(
            id="background", operator_template_id="background", bindings=(),
        ),),
    )


def test_repeated_bindings_get_new_instances_while_background_work_continues() -> None:
    problem = _repeated_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    committed = session.commit(frame.frame_token, tuple(item.id for item in frame.intents))
    assert committed.execution.final_snapshot.tick == 0
    assert session.frame().tick == 0
    assert session.frame().intents == ()
    assert session.advance_next().final_snapshot.tick == 5
    assert session.snapshot().kernel_snapshot.active_interval_ids == (
        "intent.background.step",
    )
    first_id = next(item.id for item in frame.intents if item.operator_template_id == "work")
    next_frame = session.frame()
    assert len(next_frame.intents) == 1
    second = next_frame.intents[0]
    assert second.id != first_id
    assert second.bindings == next(item.bindings for item in frame.intents if item.id == first_id)
    assert second.state_delta.completion_tick == 10
    checkpoint = session.snapshot()
    restored = ReferenceKernel.restore(problem, checkpoint.canonical_json())
    assert restored.frame() == next_frame
    for current in (session, restored):
        for finish in (10, 15):
            frame = current.frame()
            current.commit(frame.frame_token, (frame.intents[0].candidate_key,))
            assert current.advance_next().final_snapshot.tick == finish
        assert current.advance_next().final_snapshot.tick == 20
        # There may still be eligible work: no next boundary is not completion.
        assert current.advance_next() is None
        assert current.frame().intents
    assert session.snapshot().canonical_json() == restored.snapshot().canonical_json()
    roots = [item for item in session.schedule.intervals if item.operator_template_id == "work"]
    assert [(item.start_tick, item.end_tick) for item in roots] == [(0, 5), (5, 10), (10, 15)]
    assert len({item.operator_instance_id for item in roots}) == 3
    assert session.snapshot().kernel_snapshot.state_values == (
        StateAssignment(cell_id="completed", value=3),
    )


def test_scope_without_release_still_prevents_repetition() -> None:
    session = ReferenceKernel.start(_repeated_problem(release=False))
    frame = session.frame()
    work = next(item for item in frame.intents if item.operator_template_id == "work")
    session.commit(frame.frame_token, (work.id,))
    session.advance_next()
    assert all(item.operator_template_id != "work" for item in session.frame().intents)


def test_repeated_same_tick_lease_handoff_preserves_future_reservation() -> None:
    problem = _repeated_problem(with_lease=True)
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    work = next(item for item in frame.intents if item.operator_template_id == "work")
    session.commit(frame.frame_token, (work.id,))
    session.advance_next()
    frame = session.frame()
    repeated = next(item for item in frame.intents if item.operator_template_id == "work")
    result = session.commit(frame.frame_token, (repeated.id,))
    assert ("holder", 5, 10) in {
        (item.resource_id, item.start_tick, item.end_tick) for item in result.reservations
    }
    assert session.snapshot().kernel_snapshot.active_leases[0].owner_id == "item"
    assert ReferenceKernel.restore(problem, session.snapshot()).frame() == session.frame()
    assert session.advance_next().final_snapshot.active_leases == ()


def test_midway_decision_can_satisfy_deadline_before_background_completion() -> None:
    problem = ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=tuple(ResourceSpec(id=name, capacity=1) for name in ("machine", "background")),
        initial_state=InitialStateSpec(obligations=(
            ObligationInstanceSpec(id="finish", deadline_tick=8),
        )),
        operator_templates=(
            _activity("short", 5, "machine"),
            _activity("long", 20, "background"),
            _activity("finish", 2, "machine", end_effects=(
                SatisfyObligationTemplateEffect(obligation_id="finish"),
            )),
        ),
        intent_seeds=tuple(IntentSeedSpec(id=name, operator_template_id=name, bindings=())
                           for name in ("short", "long", "finish")),
    )
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    session.commit(frame.frame_token, ("short", "long"))
    assert session.advance_next().final_snapshot.tick == 5
    frame = session.frame()
    assert [item.id for item in frame.intents] == ["finish"]
    session.commit(frame.frame_token, ("finish",))
    assert session.advance_next().final_snapshot.tick == 7
    assert session.snapshot().kernel_snapshot.active_obligations == ()
    assert session.advance_next().final_snapshot.tick == 20
    assert ReferenceValidator.validate(problem, session.schedule, require_terminal=True).ok

    missed = ReferenceKernel.start(problem)
    missed.commit(missed.frame().frame_token, ("short", "long"))
    missed.advance_next()
    before = missed.snapshot().snapshot_hash
    with pytest.raises(SemanticError) as exc_info:
        missed.advance_next()
    assert exc_info.value.code is DiagnosticCode.DEADLINE_MISSED
    assert missed.snapshot().snapshot_hash == before


def test_future_preview_rejects_explicit_late_satisfaction() -> None:
    problem = ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(ResourceSpec(id="machine", capacity=1),),
        initial_state=InitialStateSpec(obligations=(
            ObligationInstanceSpec(id="work", deadline_tick=2),
        )),
        operator_templates=(_activity("late", 3, "machine", end_effects=(
            SatisfyObligationTemplateEffect(obligation_id="work"),
        )),),
        intent_seeds=(IntentSeedSpec(id="late", operator_template_id="late", bindings=()),),
    )
    assert ReferenceKernel.start(problem).frame().intents == ()


def test_decision_rounds_do_not_retroactively_change_an_earlier_condition() -> None:
    problem = ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        state_cells=(StateCellSpec(id="count", value_type="int"),),
        initial_state=InitialStateSpec(state_values=(StateAssignment(cell_id="count", value=0),)),
    )
    events = (
        EventSpec(id="earlier", tick=5, effects=(CreateObligationEffect(
            obligation_id="unexpected",
            condition=StateCondition(cell_id="count", operator="equal", value=1),
        ),)),
        EventSpec(id="later", tick=5, decision_round=1, effects=(
            SetStateEffect(cell_id="count", value=1),
        )),
    )
    for ordering in permutations(events):
        schedule = ScheduleV1(events=ordering)
        result = ReferenceKernel.execute(problem, schedule)
        report = ReferenceValidator.validate(problem, schedule, require_terminal=True)
        assert report.ok
        assert report.final_snapshot == result.final_snapshot
        assert result.final_snapshot.active_obligations == ()


def test_later_round_cannot_hide_an_earlier_capacity_violation() -> None:
    problem = ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(ResourceSpec(id="holder", capacity=1),),
    )
    schedule = ScheduleV1(events=(
        EventSpec(id="acquire", tick=0, effects=(
            AcquireLeaseEffect(resource_id="holder", owner_id="item", amount=2),
        )),
        EventSpec(id="release", tick=0, decision_round=1, effects=(
            ReleaseLeaseEffect(resource_id="holder", owner_id="item"),
        )),
    ))
    with pytest.raises(SemanticError) as exc_info:
        ReferenceKernel.execute(problem, schedule)
    assert exc_info.value.code is DiagnosticCode.RESOURCE_OVER_CAPACITY
    report = ReferenceValidator.validate(problem, schedule)
    assert [issue.code for issue in report.issues] == [DiagnosticCode.RESOURCE_OVER_CAPACITY]


def test_empty_next_boundary_is_noop_and_advance_rejects_invalid_ticks() -> None:
    problem = ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        initial_state=InitialStateSpec(obligations=(ObligationInstanceSpec(id="pending"),)),
    )
    session = ReferenceKernel.start(problem)
    before = session.snapshot().snapshot_hash
    assert session.advance_next() is None
    assert session.snapshot().snapshot_hash == before
    for tick in (-1, True, 0.5):
        with pytest.raises(SemanticError) as exc_info:
            session.advance_to(tick)
        assert exc_info.value.code is DiagnosticCode.INVALID_TIME_VALUE


def test_old_frames_and_snapshots_before_latest_decision_are_rejected() -> None:
    session = ReferenceKernel.start(_repeated_problem())
    frame = session.frame()
    session.commit(frame.frame_token, tuple(item.id for item in frame.intents))
    session.advance_next()
    before = session.snapshot().snapshot_hash
    with pytest.raises(SemanticError) as exc_info:
        session.commit(frame.frame_token, (frame.intents[0].id,))
    assert exc_info.value.code is DiagnosticCode.STALE_FRAME
    assert session.snapshot().snapshot_hash == before
    frame = session.frame()
    session.commit(frame.frame_token, (frame.intents[0].id,))
    with pytest.raises(SemanticError) as exc_info:
        session.snapshot(3)
    assert exc_info.value.code is DiagnosticCode.INVALID_TIME_VALUE


def test_advance_next_visits_delayed_start_before_finish() -> None:
    problem = ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(ResourceSpec(id="machine", capacity=1),),
        operator_templates=(_activity("delayed", 2, "machine"),),
        intent_seeds=(IntentSeedSpec(
            id="delayed", operator_template_id="delayed", bindings=(),
            earliest_start_offset=3,
        ),),
    )
    session = ReferenceKernel.start(problem)
    result = session.commit(session.frame().frame_token, ("delayed",))
    assert result.execution.final_snapshot.tick == 0
    assert result.execution.final_snapshot.active_interval_ids == ()
    started = session.advance_next().final_snapshot
    assert started.tick == 3
    assert started.active_interval_ids == ("intent.delayed.step",)
    assert session.advance_next().final_snapshot.tick == 5
    assert session.advance_next() is None


def test_preview_does_not_relax_current_tick_obligation_checks() -> None:
    problem = ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(ResourceSpec(id="machine", capacity=1),),
        operator_templates=(_activity("invalid", 2, "machine", start_effects=(
            CreateObligationTemplateEffect(obligation_id="now", deadline_offset=0),
        )),),
        intent_seeds=(IntentSeedSpec(
            id="invalid", operator_template_id="invalid", bindings=(),
        ),),
    )
    session = ReferenceKernel.start(problem)
    assert session.frame().intents == ()


def test_automatic_child_cannot_release_parent_scope_with_same_boundary_name() -> None:
    base = _repeated_problem()
    child = _activity("child", 2, "background").model_copy(
        update={"origin": "automatic"},
    )
    problem = base.model_copy(update={
        "operator_templates": base.operator_templates + (child,),
        "automatic_rules": (AutomaticRuleSpec(
            id="child_after_start",
            trigger_operator_template_id="work",
            trigger_boundary_id="step.start",
            emit_operator_template_id="child",
        ),),
    })
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    work = next(item for item in frame.intents if item.operator_template_id == "work")
    session.commit(frame.frame_token, (work.id,))
    scopes = session.snapshot().active_choice_scope_keys
    assert len(scopes) == 1
    assert session.advance_next().final_snapshot.tick == 2
    assert session.snapshot().active_choice_scope_keys == scopes
    assert session.advance_next().final_snapshot.tick == 5
    assert session.snapshot().active_choice_scope_keys == ()
