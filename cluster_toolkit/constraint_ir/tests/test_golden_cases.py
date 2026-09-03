from __future__ import annotations

from decimal import Decimal
from itertools import permutations

import pytest
from pydantic import ValidationError

from cluster_toolkit.constraint_ir import (
    AcquireLeaseEffect,
    AcquireLeaseTemplateEffect,
    AutomaticRuleSpec,
    BindingAssignment,
    BindingDomainSpec,
    BindingForwardSpec,
    BindingRowSpec,
    ChoiceScopeClaimSpec,
    ChoiceScopeTemplateSpec,
    ConstraintIRV1,
    CreateObligationEffect,
    CreateObligationTemplateEffect,
    DiagnosticCode,
    DynamicIntentSpec,
    EventSpec,
    InitialStateSpec,
    IncrementStateTemplateEffect,
    IntentSeedSpec,
    IntervalSpec,
    IntervalTemplateSpec,
    LeaseCondition,
    LeaseConditionTemplate,
    LeaseSpec,
    LegacyIntentSeedCandidateGenerator,
    ReferenceKernel,
    ReferenceValidator,
    ReleaseLeaseEffect,
    ReleaseLeaseTemplateEffect,
    ResourceSpec,
    ResourceUseSpec,
    ResourceUseTemplate,
    SatisfyObligationEffect,
    SatisfyObligationTemplateEffect,
    ScheduleV1,
    SemanticError,
    SessionSnapshot,
    SetStateEffect,
    SetCurrentTickTemplateEffect,
    SetStateTemplateEffect,
    StateAssignment,
    StateCellSpec,
    StateCondition,
    StateConditionTemplate,
    StepDependencySpec,
    TimeDomain,
    OperatorTemplateSpec,
    ParameterIdRef,
    ParameterSpec,
    canonical_effect_digest,
    compile_ticks,
)


def _problem(
    *,
    resources: tuple[ResourceSpec, ...] = (),
    state_cells: tuple[StateCellSpec, ...] = (),
    initial_state: InitialStateSpec | None = None,
) -> ConstraintIRV1:
    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1000),
        resources=resources,
        state_cells=state_cells,
        initial_state=initial_state or InitialStateSpec(),
    )


def test_g01_converts_exact_external_time_to_integer_ticks() -> None:
    domain = TimeDomain(unit="second", ticks_per_unit=1000)

    assert compile_ticks(domain, Decimal("0.125"), path="pick_time") == 125


def test_g01_rejects_time_precision_loss_without_rounding() -> None:
    domain = TimeDomain(unit="second", ticks_per_unit=1000)

    with pytest.raises(SemanticError) as exc_info:
        compile_ticks(domain, Decimal("0.0005"), path="process_time")

    assert exc_info.value.code is DiagnosticCode.TIME_PRECISION_LOSS
    assert exc_info.value.path == "process_time"


def test_g01_compiled_schema_rejects_float_ticks() -> None:
    with pytest.raises(ValidationError):
        EventSpec(id="event", tick=1.5)


def test_compiled_schema_requires_an_initial_value_for_every_state_cell() -> None:
    with pytest.raises(ValidationError, match="missing values for cells"):
        _problem(
            state_cells=(StateCellSpec(id="counter", value_type="int"),),
        )


def test_g02_adjacent_half_open_intervals_do_not_conflict() -> None:
    problem = _problem(resources=(ResourceSpec(id="pm.slot", capacity=1),))
    schedule = ScheduleV1(
        intervals=(
            IntervalSpec(
                id="process.A",
                start_tick=0,
                end_tick=10,
                resource_uses=(ResourceUseSpec(resource_id="pm.slot"),),
            ),
            IntervalSpec(
                id="place.B",
                start_tick=10,
                end_tick=12,
                resource_uses=(ResourceUseSpec(resource_id="pm.slot"),),
            ),
        )
    )

    result = ReferenceKernel.execute(problem, schedule)
    report = ReferenceValidator.validate(problem, schedule)

    assert report.ok
    assert report.final_snapshot is not None
    tick_ten = next(snapshot for snapshot in result.snapshots if snapshot.tick == 10)
    assert tick_ten.active_interval_ids == ("place.B",)
    assert report.final_snapshot.state_hash == result.final_snapshot.state_hash


def test_g02_overlapping_intervals_exceed_capacity() -> None:
    problem = _problem(resources=(ResourceSpec(id="pm.slot", capacity=1),))
    schedule = ScheduleV1(
        intervals=(
            IntervalSpec(
                id="process.A",
                start_tick=0,
                end_tick=10,
                resource_uses=(ResourceUseSpec(resource_id="pm.slot"),),
            ),
            IntervalSpec(
                id="place.B",
                start_tick=9,
                end_tick=12,
                resource_uses=(ResourceUseSpec(resource_id="pm.slot"),),
            ),
        )
    )

    with pytest.raises(SemanticError) as exc_info:
        ReferenceKernel.execute(problem, schedule)
    report = ReferenceValidator.validate(problem, schedule)

    assert exc_info.value.code is DiagnosticCode.RESOURCE_OVER_CAPACITY
    assert [issue.code for issue in report.issues] == [
        DiagnosticCode.RESOURCE_OVER_CAPACITY
    ]


def _deadline_schedule(satisfy_tick: int) -> ScheduleV1:
    return ScheduleV1(
        events=(
            EventSpec(
                id="process.end",
                tick=10,
                effects=(
                    CreateObligationEffect(
                        obligation_id="wafer.residency",
                        deadline_tick=15,
                    ),
                ),
            ),
            EventSpec(
                id="pick.start",
                tick=satisfy_tick,
                effects=(
                    SatisfyObligationEffect(
                        obligation_id="wafer.residency"
                    ),
                ),
            ),
        )
    )


def test_g03_deadline_is_inclusive() -> None:
    problem = _problem()
    schedule = _deadline_schedule(satisfy_tick=15)

    result = ReferenceKernel.execute(problem, schedule)
    report = ReferenceValidator.validate(problem, schedule)

    assert report.ok
    assert result.final_snapshot.tick == 15
    assert result.final_snapshot.active_obligations == ()


def test_g03_satisfaction_after_deadline_is_rejected_at_deadline() -> None:
    problem = _problem()
    schedule = _deadline_schedule(satisfy_tick=16)

    with pytest.raises(SemanticError) as exc_info:
        ReferenceKernel.execute(problem, schedule)
    report = ReferenceValidator.validate(problem, schedule)

    assert exc_info.value.code is DiagnosticCode.DEADLINE_MISSED
    assert [issue.code for issue in report.issues] == [DiagnosticCode.DEADLINE_MISSED]
    assert report.issues[0].tick == 15


def _same_tick_problem() -> ConstraintIRV1:
    return _problem(
        resources=(ResourceSpec(id="slot", capacity=1),),
        state_cells=(
            StateCellSpec(
                id="machine.mode",
                value_type="enum",
                enum_values=("idle", "ready"),
            ),
        ),
        initial_state=InitialStateSpec(
            state_values=(StateAssignment(cell_id="machine.mode", value="idle"),),
            leases=(LeaseSpec(resource_id="slot", owner_id="wafer.A"),),
        ),
    )


def _same_tick_events() -> tuple[EventSpec, ...]:
    return (
        EventSpec(
            id="release.A",
            tick=20,
            effects=(ReleaseLeaseEffect(resource_id="slot", owner_id="wafer.A"),),
        ),
        EventSpec(
            id="acquire.B",
            tick=20,
            effects=(AcquireLeaseEffect(resource_id="slot", owner_id="wafer.B"),),
        ),
        EventSpec(
            id="create.obligation",
            tick=20,
            effects=(
                CreateObligationEffect(obligation_id="handoff", deadline_tick=20),
                SetStateEffect(cell_id="machine.mode", value="ready"),
            ),
        ),
        EventSpec(
            id="satisfy.obligation",
            tick=20,
            effects=(
                SatisfyObligationEffect(obligation_id="handoff"),
                SetStateEffect(cell_id="machine.mode", value="ready"),
            ),
        ),
    )


def test_g04_same_tick_event_order_does_not_change_stable_state() -> None:
    problem = _same_tick_problem()
    expected_hash: str | None = None

    for event_order in permutations(_same_tick_events()):
        schedule = ScheduleV1(events=event_order)
        result = ReferenceKernel.execute(problem, schedule)
        report = ReferenceValidator.validate(problem, schedule)

        assert report.ok
        assert report.final_snapshot is not None
        assert report.final_snapshot.state_hash == result.final_snapshot.state_hash
        if expected_hash is None:
            expected_hash = result.final_snapshot.state_hash
        assert result.final_snapshot.state_hash == expected_hash
        assert result.final_snapshot.active_obligations == ()
        assert [lease.owner_id for lease in result.final_snapshot.active_leases] == [
            "wafer.B"
        ]


@pytest.mark.parametrize("reverse", [False, True])
def test_g04_conflicting_same_tick_state_effects_are_rejected(
    reverse: bool,
) -> None:
    problem = _same_tick_problem()
    events = [
        EventSpec(
            id="set.idle",
            tick=20,
            effects=(SetStateEffect(cell_id="machine.mode", value="idle"),),
        ),
        EventSpec(
            id="set.ready",
            tick=20,
            effects=(SetStateEffect(cell_id="machine.mode", value="ready"),),
        ),
    ]
    if reverse:
        events.reverse()
    schedule = ScheduleV1(events=tuple(events))

    with pytest.raises(SemanticError) as exc_info:
        ReferenceKernel.execute(problem, schedule)
    report = ReferenceValidator.validate(problem, schedule)

    assert exc_info.value.code is DiagnosticCode.CONFLICTING_EFFECTS
    assert [issue.code for issue in report.issues] == [
        DiagnosticCode.CONFLICTING_EFFECTS
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_same_tick_type_errors_do_not_depend_on_python_scalar_equality(
    reverse: bool,
) -> None:
    problem = _problem(
        state_cells=(StateCellSpec(id="flag", value_type="bool"),),
        initial_state=InitialStateSpec(
            state_values=(StateAssignment(cell_id="flag", value=False),),
        ),
    )
    events = [
        EventSpec(
            id="set.bool",
            tick=1,
            effects=(SetStateEffect(cell_id="flag", value=True),),
        ),
        EventSpec(
            id="set.int",
            tick=1,
            effects=(SetStateEffect(cell_id="flag", value=1),),
        ),
    ]
    if reverse:
        events.reverse()
    schedule = ScheduleV1(events=tuple(events))

    with pytest.raises(SemanticError) as exc_info:
        ReferenceKernel.execute(problem, schedule)
    report = ReferenceValidator.validate(problem, schedule)

    assert exc_info.value.code is DiagnosticCode.TYPE_MISMATCH
    assert [issue.code for issue in report.issues] == [DiagnosticCode.TYPE_MISMATCH]


def _parameter(name: str, kind: str) -> ParameterSpec:
    return ParameterSpec(name=name, kind=kind)


def _parameter_ref(name: str) -> ParameterIdRef:
    return ParameterIdRef(parameter=name)


def _g05_problem(*, include_conflicting_intent: bool = False) -> ConstraintIRV1:
    transport = OperatorTemplateSpec(
        id="transport",
        origin="selectable",
        parameters=(
            _parameter("wafer", "owner"),
            _parameter("robot_resource", "resource"),
            _parameter("hand_resource", "resource"),
            _parameter("source_resource", "resource"),
            _parameter("target_resource", "resource"),
            _parameter("process_resource", "resource"),
        ),
        intervals=(
            IntervalTemplateSpec(
                id="move_to_source",
                start_offset=0,
                duration=2,
                audit_kind="MoveToSource",
                resource_uses=(
                    ResourceUseTemplate(resource=_parameter_ref("robot_resource")),
                ),
            ),
            IntervalTemplateSpec(
                id="pick",
                start_offset=2,
                duration=1,
                audit_kind="Pick",
                resource_uses=(
                    ResourceUseTemplate(resource=_parameter_ref("robot_resource")),
                ),
                start_effects=(
                    AcquireLeaseTemplateEffect(
                        resource=_parameter_ref("hand_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
                end_effects=(
                    ReleaseLeaseTemplateEffect(
                        resource=_parameter_ref("source_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
            ),
            IntervalTemplateSpec(
                id="move_to_target",
                start_offset=3,
                duration=4,
                audit_kind="MoveToTarget",
                resource_uses=(
                    ResourceUseTemplate(resource=_parameter_ref("robot_resource")),
                ),
            ),
            IntervalTemplateSpec(
                id="place",
                start_offset=7,
                duration=1,
                audit_kind="Place",
                resource_uses=(
                    ResourceUseTemplate(resource=_parameter_ref("robot_resource")),
                ),
                start_effects=(
                    AcquireLeaseTemplateEffect(
                        resource=_parameter_ref("target_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
                end_effects=(
                    ReleaseLeaseTemplateEffect(
                        resource=_parameter_ref("hand_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
            ),
        ),
    )
    process = OperatorTemplateSpec(
        id="process",
        origin="automatic",
        parameters=(
            _parameter("process_resource", "resource"),
        ),
        intervals=(
            IntervalTemplateSpec(
                id="process",
                start_offset=0,
                duration=10,
                audit_kind="Process",
                resource_uses=(
                    ResourceUseTemplate(resource=_parameter_ref("process_resource")),
                ),
            ),
        ),
    )
    resources = [
        ResourceSpec(id="robot.motion", capacity=1),
        ResourceSpec(id="robot.hand", capacity=1),
        ResourceSpec(id="source.A", capacity=1),
        ResourceSpec(id="target.A", capacity=1),
        ResourceSpec(id="pm.operation.A", capacity=1),
    ]
    leases = [LeaseSpec(resource_id="source.A", owner_id="wafer.A")]
    intents = [
        IntentSeedSpec(
            id="transport.A",
            operator_template_id="transport",
            bindings=(
                BindingAssignment(parameter="wafer", value="wafer.A"),
                BindingAssignment(parameter="robot_resource", value="robot.motion"),
                BindingAssignment(parameter="hand_resource", value="robot.hand"),
                BindingAssignment(parameter="source_resource", value="source.A"),
                BindingAssignment(parameter="target_resource", value="target.A"),
                BindingAssignment(parameter="process_resource", value="pm.operation.A"),
            ),
        )
    ]
    if include_conflicting_intent:
        resources.extend(
            [
                ResourceSpec(id="source.B", capacity=1),
                ResourceSpec(id="target.B", capacity=1),
                ResourceSpec(id="pm.operation.B", capacity=1),
            ]
        )
        leases.append(LeaseSpec(resource_id="source.B", owner_id="wafer.B"))
        intents.append(
            IntentSeedSpec(
                id="transport.B",
                operator_template_id="transport",
                bindings=(
                    BindingAssignment(parameter="wafer", value="wafer.B"),
                    BindingAssignment(parameter="robot_resource", value="robot.motion"),
                    BindingAssignment(parameter="hand_resource", value="robot.hand"),
                    BindingAssignment(parameter="source_resource", value="source.B"),
                    BindingAssignment(parameter="target_resource", value="target.B"),
                    BindingAssignment(
                        parameter="process_resource",
                        value="pm.operation.B",
                    ),
                ),
            )
        )
    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=tuple(resources),
        initial_state=InitialStateSpec(leases=tuple(leases)),
        operator_templates=(transport, process),
        intent_seeds=tuple(intents),
        automatic_rules=(
            AutomaticRuleSpec(
                id="process_after_place",
                trigger_operator_template_id="transport",
                trigger_boundary_id="place.end",
                emit_operator_template_id="process",
                binding_forwards=(
                    BindingForwardSpec(
                        source_parameter="process_resource",
                        target_parameter="process_resource",
                    ),
                ),
            ),
        ),
    )


def test_g05_transport_intent_expands_bundle_and_automatic_process() -> None:
    problem = _g05_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()

    assert [intent.id for intent in frame.intents] == ["transport.A"]
    result = session.commit(frame.frame_token, ("transport.A",))
    interval_kinds = {
        interval.audit_kind: (interval.start_tick, interval.end_tick)
        for interval in result.schedule.intervals
    }

    assert interval_kinds == {
        "MoveToSource": (0, 2),
        "Pick": (2, 3),
        "MoveToTarget": (3, 7),
        "Place": (7, 8),
        "Process": (8, 18),
    }
    assert all(intent.operator_template_id != "process" for intent in frame.intents)
    assert ReferenceValidator.validate(problem, result.schedule).ok

    reservations = {
        (item.resource_id, item.start_tick, item.end_tick)
        for item in result.reservations
    }
    assert ("robot.motion", 0, 8) in reservations
    assert ("target.A", 7, None) in reservations
    assert ("pm.operation.A", 8, 18) in reservations
    execution = session.advance_to(18)
    assert [
        (lease.resource_id, lease.owner_id)
        for lease in execution.final_snapshot.active_leases
    ] == [("target.A", "wafer.A")]


def test_g05_batch_commit_rejects_two_intents_using_the_same_robot() -> None:
    problem = _g05_problem(include_conflicting_intent=True)
    session = ReferenceKernel.start(problem)
    frame = session.frame()

    assert {intent.id for intent in frame.intents} == {
        "transport.A",
        "transport.B",
    }
    with pytest.raises(SemanticError) as exc_info:
        session.commit(
            frame.frame_token,
            ("transport.A", "transport.B"),
        )

    assert exc_info.value.code is DiagnosticCode.RESOURCE_OVER_CAPACITY


def test_g05_validator_rejects_missing_automatic_process_boundary() -> None:
    problem = _g05_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    result = session.commit(frame.frame_token, ("transport.A",))
    corrupted = result.schedule.model_copy(
        update={
            "events": tuple(
                event
                for event in result.schedule.events
                if event.audit_kind != "Process.start"
            )
        }
    )

    report = ReferenceValidator.validate(problem, corrupted)

    assert not report.ok
    assert [issue.code for issue in report.issues] == [
        DiagnosticCode.MISSING_AUTOMATIC_EVENT
    ]


def _g06_problem(
    *, forbid_overlap: bool = False, activity_label: str = "Cooling"
) -> ConstraintIRV1:
    place_parameters = [
        _parameter("interface_resource", "resource"),
        _parameter("cooling_resource", "resource"),
        _parameter("pressure_resource", "resource"),
        _parameter("pressure_level_cell", "state_cell"),
    ]
    cooling_parameters = [
        _parameter("cooling_resource", "resource"),
    ]
    pump_parameters = [
        _parameter("pressure_resource", "resource"),
        _parameter("pressure_level_cell", "state_cell"),
    ]
    cooling_uses = [
        ResourceUseTemplate(resource=_parameter_ref("cooling_resource")),
    ]
    pump_uses = [
        ResourceUseTemplate(resource=_parameter_ref("pressure_resource")),
    ]
    resources = [
        ResourceSpec(id="ll.interface", capacity=1),
        ResourceSpec(id="ll.cooling", capacity=1),
        ResourceSpec(id="ll.pressure", capacity=1),
    ]
    if forbid_overlap:
        place_parameters.append(_parameter("exclusion_resource", "resource"))
        cooling_parameters.append(_parameter("exclusion_resource", "resource"))
        pump_parameters.append(_parameter("exclusion_resource", "resource"))
        cooling_uses.append(
            ResourceUseTemplate(resource=_parameter_ref("exclusion_resource"))
        )
        pump_uses.append(
            ResourceUseTemplate(resource=_parameter_ref("exclusion_resource"))
        )
        resources.append(ResourceSpec(id="ll.exclusion", capacity=1))

    place = OperatorTemplateSpec(
        id="place_in_ll",
        origin="selectable",
        parameters=tuple(place_parameters),
        intervals=(
            IntervalTemplateSpec(
                id="place",
                start_offset=0,
                duration=1,
                audit_kind="Place",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("interface_resource")
                    ),
                ),
            ),
        ),
    )
    cooling = OperatorTemplateSpec(
        id="cooling",
        origin="automatic",
        parameters=tuple(cooling_parameters),
        intervals=(
            IntervalTemplateSpec(
                id="cooling",
                start_offset=0,
                duration=3,
                audit_kind=activity_label,
                resource_uses=tuple(cooling_uses),
            ),
        ),
    )
    pump = OperatorTemplateSpec(
        id="pump",
        origin="automatic",
        parameters=tuple(pump_parameters),
        intervals=(
            IntervalTemplateSpec(
                id="pump",
                start_offset=1,
                duration=4,
                audit_kind="Pump",
                resource_uses=tuple(pump_uses),
                end_effects=(
                    SetStateTemplateEffect(
                        cell=_parameter_ref("pressure_level_cell"),
                        value="vacuum",
                    ),
                ),
            ),
        ),
    )
    bindings = [
        BindingAssignment(parameter="interface_resource", value="ll.interface"),
        BindingAssignment(parameter="cooling_resource", value="ll.cooling"),
        BindingAssignment(parameter="pressure_resource", value="ll.pressure"),
        BindingAssignment(
            parameter="pressure_level_cell",
            value="ll.pressure_level",
        ),
    ]
    if forbid_overlap:
        bindings.append(
            BindingAssignment(
                parameter="exclusion_resource",
                value="ll.exclusion",
            )
        )

    def forwards(parameter_names: tuple[str, ...]) -> tuple[BindingForwardSpec, ...]:
        return tuple(
            BindingForwardSpec(
                source_parameter=name,
                target_parameter=name,
            )
            for name in parameter_names
        )

    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=tuple(resources),
        state_cells=(
            StateCellSpec(
                id="ll.pressure_level",
                value_type="enum",
                enum_values=("atmosphere", "vacuum"),
            ),
        ),
        initial_state=InitialStateSpec(
            state_values=(
                StateAssignment(
                    cell_id="ll.pressure_level",
                    value="atmosphere",
                ),
            ),
        ),
        operator_templates=(place, cooling, pump),
        intent_seeds=(
            IntentSeedSpec(
                id="place.wafer.A.in.LL",
                operator_template_id="place_in_ll",
                bindings=tuple(bindings),
            ),
        ),
        automatic_rules=(
            AutomaticRuleSpec(
                id="cool_after_place",
                trigger_operator_template_id="place_in_ll",
                trigger_boundary_id="place.end",
                emit_operator_template_id="cooling",
                binding_forwards=forwards(
                    tuple(parameter.name for parameter in cooling_parameters)
                ),
            ),
            AutomaticRuleSpec(
                id="pump_after_place",
                trigger_operator_template_id="place_in_ll",
                trigger_boundary_id="place.end",
                emit_operator_template_id="pump",
                binding_forwards=forwards(
                    tuple(parameter.name for parameter in pump_parameters)
                ),
            ),
        ),
    )


def _snapshot_values(snapshot) -> dict[str, object]:
    return {item.cell_id: item.value for item in snapshot.state_values}


@pytest.mark.parametrize("activity_label", ["Cooling", "Process", "Activity"])
def test_g06_pressure_transition_and_activity_advance_independently(
    activity_label: str,
) -> None:
    problem = _g06_problem(activity_label=activity_label)
    session = ReferenceKernel.start(problem)
    frame = session.frame()

    assert [intent.id for intent in frame.intents] == ["place.wafer.A.in.LL"]
    result = session.commit(frame.frame_token, (frame.intents[0].id,))
    execution = session.advance_to(6)
    snapshots = {snapshot.tick: snapshot for snapshot in execution.snapshots}

    intervals = {item.audit_kind: item for item in result.schedule.intervals}
    activity_id = intervals[activity_label].id
    pump_id = intervals["Pump"].id
    for tick, active_ids in (
        (1, {activity_id}),
        (2, {activity_id, pump_id}),
        (4, {pump_id}),
        (6, set()),
    ):
        assert set(snapshots[tick].active_interval_ids) == active_ids
        assert _snapshot_values(snapshots[tick]) == {
            "ll.pressure_level": "vacuum" if tick == 6 else "atmosphere",
        }
    # Completion is derived from the recorded end boundary, not a thermal flag.
    assert intervals[activity_label].end_tick == 4
    report = ReferenceValidator.validate(problem, result.schedule)
    assert report.ok
    assert report.final_snapshot == execution.final_snapshot


def test_g06_shared_resource_makes_overlap_non_committable() -> None:
    problem = _g06_problem(forbid_overlap=True)
    session = ReferenceKernel.start(problem)

    frame = session.frame()

    assert frame.intents == ()
    with pytest.raises(SemanticError) as exc_info:
        session.commit(frame.frame_token, ("place.wafer.A.in.LL",))
    assert exc_info.value.code is DiagnosticCode.INTENT_NOT_COMMITTABLE


def test_g06_validator_rejects_missing_pressure_effect() -> None:
    problem = _g06_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    result = session.commit(frame.frame_token, (frame.intents[0].id,))
    corrupted_events = []
    for event in result.schedule.events:
        if event.audit_kind == "Pump.end":
            effects = tuple(
                effect
                for effect in event.effects
                if not (
                    isinstance(effect, SetStateEffect)
                    and effect.cell_id == "ll.pressure_level"
                )
            )
            corrupted_events.append(
                event.model_copy(
                    update={
                        "effects": effects,
                        "effect_digest": canonical_effect_digest(effects),
                    }
                )
            )
        else:
            corrupted_events.append(event)
    corrupted = result.schedule.model_copy(update={"events": tuple(corrupted_events)})

    report = ReferenceValidator.validate(problem, corrupted)

    assert [issue.code for issue in report.issues] == [
        DiagnosticCode.MISSING_AUTOMATIC_EVENT
    ]


def _g07_problem() -> ConstraintIRV1:
    process = OperatorTemplateSpec(
        id="process.B",
        origin="selectable",
        parameters=(
            _parameter("operation_resource", "resource"),
            _parameter("wafer_count_cell", "state_cell"),
            _parameter("process_type_cell", "state_cell"),
            _parameter("last_clean_tick_cell", "state_cell"),
            _parameter("occupancy_cell", "state_cell"),
        ),
        intervals=(
            IntervalTemplateSpec(
                id="process",
                start_offset=0,
                duration=2,
                audit_kind="Process",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("operation_resource")
                    ),
                ),
                end_effects=(
                    IncrementStateTemplateEffect(
                        cell=_parameter_ref("wafer_count_cell"),
                        delta=1,
                    ),
                    SetStateTemplateEffect(
                        cell=_parameter_ref("process_type_cell"),
                        value="B",
                    ),
                    SetStateTemplateEffect(
                        cell=_parameter_ref("occupancy_cell"),
                        value=0,
                    ),
                    CreateObligationTemplateEffect(
                        obligation_id="clean.wafer_count",
                        deadline_offset=20,
                        condition=StateConditionTemplate(
                            cell=_parameter_ref("wafer_count_cell"),
                            operator="greater_equal",
                            value=2,
                            view="after",
                        ),
                        coalesce_key="clean.PM1",
                        priority=10,
                    ),
                    CreateObligationTemplateEffect(
                        obligation_id="clean.idle",
                        deadline_offset=20,
                        condition=StateConditionTemplate(
                            cell=_parameter_ref("last_clean_tick_cell"),
                            operator="elapsed_at_least",
                            value=2,
                            view="after",
                        ),
                        coalesce_key="clean.PM1",
                        priority=20,
                    ),
                    CreateObligationTemplateEffect(
                        obligation_id="clean.process_switch",
                        deadline_offset=20,
                        condition=StateConditionTemplate(
                            cell=_parameter_ref("process_type_cell"),
                            operator="not_equal",
                            value="B",
                            view="before",
                        ),
                        coalesce_key="clean.PM1",
                        priority=30,
                    ),
                ),
            ),
        ),
    )
    clean = OperatorTemplateSpec(
        id="clean.PM1",
        origin="selectable",
        parameters=(
            _parameter("operation_resource", "resource"),
            _parameter("wafer_count_cell", "state_cell"),
            _parameter("process_type_cell", "state_cell"),
            _parameter("last_clean_tick_cell", "state_cell"),
        ),
        intervals=(
            IntervalTemplateSpec(
                id="clean",
                start_offset=0,
                duration=5,
                audit_kind="Clean",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("operation_resource")
                    ),
                ),
                end_effects=(
                    SetStateTemplateEffect(
                        cell=_parameter_ref("wafer_count_cell"),
                        value=0,
                    ),
                    SetStateTemplateEffect(
                        cell=_parameter_ref("process_type_cell"),
                        value="none",
                    ),
                    SetCurrentTickTemplateEffect(
                        cell=_parameter_ref("last_clean_tick_cell")
                    ),
                    SatisfyObligationTemplateEffect(
                        obligation_id="clean.process_switch"
                    ),
                ),
            ),
        ),
    )
    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(ResourceSpec(id="pm.operation", capacity=1),),
        state_cells=(
            StateCellSpec(
                id="pm.wafer_count",
                value_type="int",
                minimum=0,
                maximum=100,
            ),
            StateCellSpec(
                id="pm.last_process_type",
                value_type="enum",
                enum_values=("none", "A", "B"),
            ),
            StateCellSpec(
                id="pm.last_clean_tick",
                value_type="int",
                minimum=0,
            ),
            StateCellSpec(
                id="pm.occupancy",
                value_type="int",
                minimum=0,
                maximum=1,
            ),
        ),
        initial_state=InitialStateSpec(
            state_values=(
                StateAssignment(cell_id="pm.wafer_count", value=1),
                StateAssignment(cell_id="pm.last_process_type", value="A"),
                StateAssignment(cell_id="pm.last_clean_tick", value=0),
                StateAssignment(cell_id="pm.occupancy", value=1),
            ),
        ),
        operator_templates=(process, clean),
        intent_seeds=(
            IntentSeedSpec(
                id="process.B",
                operator_template_id="process.B",
                bindings=(
                    BindingAssignment(
                        parameter="operation_resource",
                        value="pm.operation",
                    ),
                    BindingAssignment(
                        parameter="wafer_count_cell",
                        value="pm.wafer_count",
                    ),
                    BindingAssignment(
                        parameter="process_type_cell",
                        value="pm.last_process_type",
                    ),
                    BindingAssignment(
                        parameter="last_clean_tick_cell",
                        value="pm.last_clean_tick",
                    ),
                    BindingAssignment(
                        parameter="occupancy_cell",
                        value="pm.occupancy",
                    ),
                ),
            ),
            IntentSeedSpec(
                id="clean.PM1",
                operator_template_id="clean.PM1",
                required_obligation_ids=("clean.process_switch",),
                guards=(
                    StateCondition(
                        cell_id="pm.occupancy",
                        operator="equal",
                        value=0,
                    ),
                ),
                bindings=(
                    BindingAssignment(
                        parameter="operation_resource",
                        value="pm.operation",
                    ),
                    BindingAssignment(
                        parameter="wafer_count_cell",
                        value="pm.wafer_count",
                    ),
                    BindingAssignment(
                        parameter="process_type_cell",
                        value="pm.last_process_type",
                    ),
                    BindingAssignment(
                        parameter="last_clean_tick_cell",
                        value="pm.last_clean_tick",
                    ),
                ),
            ),
        ),
    )


def test_g07_conditions_coalesce_into_one_explicit_clean_intent() -> None:
    problem = _g07_problem()
    session = ReferenceKernel.start(problem)

    process_frame = session.frame()
    assert process_frame.tick == 0
    assert [intent.id for intent in process_frame.intents] == ["process.B"]

    process_result = session.commit(process_frame.frame_token, ("process.B",))
    assert process_result.execution.final_snapshot.tick == 0
    process_execution = session.advance_next()
    assert process_execution is not None
    assert process_execution.final_snapshot.tick == 2
    assert [
        obligation.id
        for obligation in process_execution.final_snapshot.active_obligations
    ] == ["clean.process_switch"]
    assert _snapshot_values(process_execution.final_snapshot) == {
        "pm.last_clean_tick": 0,
        "pm.last_process_type": "B",
        "pm.occupancy": 0,
        "pm.wafer_count": 2,
    }

    clean_frame = session.frame()
    assert clean_frame.tick == 2
    assert [intent.id for intent in clean_frame.intents] == ["clean.PM1"]

    clean_result = session.commit(clean_frame.frame_token, ("clean.PM1",))
    clean_interval = next(
        interval
        for interval in clean_result.schedule.intervals
        if interval.audit_kind == "Clean"
    )
    assert (clean_interval.start_tick, clean_interval.end_tick) == (2, 7)
    assert clean_interval.resource_uses == (
        ResourceUseSpec(resource_id="pm.operation"),
    )
    clean_execution = session.advance_next()
    assert clean_execution is not None
    assert clean_execution.final_snapshot.active_obligations == ()
    assert _snapshot_values(clean_execution.final_snapshot) == {
        "pm.last_clean_tick": 7,
        "pm.last_process_type": "none",
        "pm.occupancy": 0,
        "pm.wafer_count": 0,
    }

    report = ReferenceValidator.validate(problem, clean_result.schedule)
    assert report.ok
    assert report.final_snapshot is not None
    assert report.final_snapshot.tick == 7


def test_g07_ambiguous_coalesce_priority_is_rejected_statically() -> None:
    with pytest.raises(ValidationError, match="UNDER_SPECIFIED_PRIORITY"):
        OperatorTemplateSpec(
            id="ambiguous.clean.trigger",
            origin="selectable",
            parameters=(_parameter("resource", "resource"),),
            intervals=(
                IntervalTemplateSpec(
                    id="process",
                    start_offset=0,
                    duration=1,
                    audit_kind="Process",
                    resource_uses=(
                        ResourceUseTemplate(resource=_parameter_ref("resource")),
                    ),
                    end_effects=(
                        CreateObligationTemplateEffect(
                            obligation_id="clean.by_count",
                            deadline_offset=10,
                            coalesce_key="clean.PM1",
                            priority=30,
                        ),
                        CreateObligationTemplateEffect(
                            obligation_id="clean.by_switch",
                            deadline_offset=10,
                            coalesce_key="clean.PM1",
                            priority=30,
                        ),
                    ),
                ),
            ),
        )


def test_g07_validator_rejects_clean_without_obligation_satisfaction() -> None:
    problem = _g07_problem()
    session = ReferenceKernel.start(problem)
    process_frame = session.frame()
    session.commit(process_frame.frame_token, ("process.B",))
    session.advance_next()
    clean_frame = session.frame()
    result = session.commit(clean_frame.frame_token, ("clean.PM1",))
    corrupted_events = []
    for event in result.schedule.events:
        if event.audit_kind == "Clean.end":
            effects = tuple(
                effect
                for effect in event.effects
                if not isinstance(effect, SatisfyObligationEffect)
            )
            corrupted_events.append(
                event.model_copy(
                    update={
                        "effects": effects,
                        "effect_digest": canonical_effect_digest(effects),
                    }
                )
            )
        else:
            corrupted_events.append(event)
    corrupted = result.schedule.model_copy(update={"events": tuple(corrupted_events)})

    report = ReferenceValidator.validate(problem, corrupted)

    assert [issue.code for issue in report.issues] == [
        DiagnosticCode.DEADLINE_MISSED
    ]
    assert report.issues[0].tick == 22


def _g08_problem(
    *,
    motion_capacity: int,
    geometry_compatible: bool,
) -> ConstraintIRV1:
    parameters = [
        _parameter("wafer", "owner"),
        _parameter("motion_resource", "resource"),
        _parameter("hand_resource", "resource"),
        _parameter("source_resource", "resource"),
    ]
    resource_uses = [
        ResourceUseTemplate(resource=_parameter_ref("motion_resource")),
    ]
    resources = [
        ResourceSpec(id="robot.motion", capacity=motion_capacity),
        ResourceSpec(id="robot.arm0", capacity=1),
        ResourceSpec(id="robot.arm1", capacity=1),
        ResourceSpec(id="source.A", capacity=1),
        ResourceSpec(id="source.B", capacity=1),
    ]
    if not geometry_compatible:
        parameters.append(_parameter("geometry_resource", "resource"))
        resource_uses.append(
            ResourceUseTemplate(resource=_parameter_ref("geometry_resource"))
        )
        resources.append(
            ResourceSpec(id="robot.geometry.exclusion", capacity=1)
        )

    pick = OperatorTemplateSpec(
        id="pick",
        origin="selectable",
        parameters=tuple(parameters),
        intervals=(
            IntervalTemplateSpec(
                id="pick",
                start_offset=0,
                duration=2,
                audit_kind="Pick",
                resource_uses=tuple(resource_uses),
                start_effects=(
                    AcquireLeaseTemplateEffect(
                        resource=_parameter_ref("hand_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
                end_effects=(
                    ReleaseLeaseTemplateEffect(
                        resource=_parameter_ref("source_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
            ),
        ),
    )

    def pick_seed(
        intent_id: str,
        wafer: str,
        hand: str,
        source: str,
    ) -> IntentSeedSpec:
        bindings = [
            BindingAssignment(parameter="wafer", value=wafer),
            BindingAssignment(
                parameter="motion_resource",
                value="robot.motion",
            ),
            BindingAssignment(parameter="hand_resource", value=hand),
            BindingAssignment(parameter="source_resource", value=source),
        ]
        if not geometry_compatible:
            bindings.append(
                BindingAssignment(
                    parameter="geometry_resource",
                    value="robot.geometry.exclusion",
                )
            )
        return IntentSeedSpec(
            id=intent_id,
            operator_template_id="pick",
            bindings=tuple(bindings),
        )

    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=tuple(resources),
        initial_state=InitialStateSpec(
            leases=(
                LeaseSpec(resource_id="source.A", owner_id="wafer.A"),
                LeaseSpec(resource_id="source.B", owner_id="wafer.B"),
            ),
        ),
        operator_templates=(pick,),
        intent_seeds=(
            pick_seed(
                "pick.A.arm0",
                "wafer.A",
                "robot.arm0",
                "source.A",
            ),
            pick_seed(
                "pick.B.arm1",
                "wafer.B",
                "robot.arm1",
                "source.B",
            ),
        ),
    )


def test_g08_distinct_hand_leases_coexist_after_serial_robot_motion() -> None:
    problem = _g08_problem(
        motion_capacity=1,
        geometry_compatible=True,
    )
    session = ReferenceKernel.start(problem)
    first_frame = session.frame()

    session.commit(first_frame.frame_token, ("pick.A.arm0",))
    first_execution = session.advance_next()
    assert first_execution is not None
    assert [
        (lease.resource_id, lease.owner_id)
        for lease in first_execution.final_snapshot.active_leases
    ] == [
        ("robot.arm0", "wafer.A"),
        ("source.B", "wafer.B"),
    ]

    second_frame = session.frame()
    assert second_frame.tick == 2
    assert [intent.id for intent in second_frame.intents] == ["pick.B.arm1"]
    result = session.commit(second_frame.frame_token, ("pick.B.arm1",))
    execution = session.advance_next()
    assert execution is not None

    assert [
        (lease.resource_id, lease.owner_id)
        for lease in execution.final_snapshot.active_leases
    ] == [
        ("robot.arm0", "wafer.A"),
        ("robot.arm1", "wafer.B"),
    ]
    assert ReferenceValidator.validate(problem, result.schedule).ok


def test_g08_distinct_hands_do_not_bypass_robot_motion_capacity() -> None:
    problem = _g08_problem(
        motion_capacity=1,
        geometry_compatible=True,
    )
    session = ReferenceKernel.start(problem)
    frame = session.frame()

    assert {intent.id for intent in frame.intents} == {
        "pick.A.arm0",
        "pick.B.arm1",
    }
    with pytest.raises(SemanticError) as exc_info:
        session.commit(
            frame.frame_token,
            ("pick.A.arm0", "pick.B.arm1"),
        )

    assert exc_info.value.code is DiagnosticCode.RESOURCE_OVER_CAPACITY


def test_g08_capacity_two_and_compatible_geometry_allow_parallel_picks() -> None:
    problem = _g08_problem(
        motion_capacity=2,
        geometry_compatible=True,
    )
    session = ReferenceKernel.start(problem)
    frame = session.frame()

    result = session.commit(
        frame.frame_token,
        ("pick.A.arm0", "pick.B.arm1"),
    )

    assert {
        (interval.origin_intent_id, interval.start_tick, interval.end_tick)
        for interval in result.schedule.intervals
    } == {
        ("pick.A.arm0", 0, 2),
        ("pick.B.arm1", 0, 2),
    }
    execution = session.advance_next()
    assert execution is not None
    assert [
        (lease.resource_id, lease.owner_id)
        for lease in execution.final_snapshot.active_leases
    ] == [
        ("robot.arm0", "wafer.A"),
        ("robot.arm1", "wafer.B"),
    ]
    assert ReferenceValidator.validate(problem, result.schedule).ok

    capacity_one_report = ReferenceValidator.validate(
        _g08_problem(motion_capacity=1, geometry_compatible=True),
        result.schedule,
    )
    assert [issue.code for issue in capacity_one_report.issues] == [
        DiagnosticCode.RESOURCE_OVER_CAPACITY
    ]


def test_g08_geometry_exclusion_blocks_parallel_motion_and_is_audited() -> None:
    problem = _g08_problem(
        motion_capacity=2,
        geometry_compatible=False,
    )
    session = ReferenceKernel.start(problem)
    frame = session.frame()

    with pytest.raises(SemanticError) as exc_info:
        session.commit(
            frame.frame_token,
            ("pick.A.arm0", "pick.B.arm1"),
        )
    assert exc_info.value.code is DiagnosticCode.RESOURCE_OVER_CAPACITY

    result = session.commit(frame.frame_token, ("pick.A.arm0",))
    corrupted_intervals = tuple(
        interval.model_copy(
            update={
                "resource_uses": tuple(
                    use
                    for use in interval.resource_uses
                    if use.resource_id != "robot.geometry.exclusion"
                )
            }
        )
        for interval in result.schedule.intervals
    )
    corrupted = result.schedule.model_copy(
        update={"intervals": corrupted_intervals}
    )

    report = ReferenceValidator.validate(problem, corrupted)

    assert [issue.code for issue in report.issues] == [
        DiagnosticCode.OPERATOR_CONFORMANCE_MISMATCH
    ]


def _g09_problem() -> ConstraintIRV1:
    transport = OperatorTemplateSpec(
        id="transport.to.target",
        origin="selectable",
        parameters=(
            _parameter("wafer", "owner"),
            _parameter("motion_resource", "resource"),
            _parameter("target_resource", "resource"),
        ),
        intervals=(
            IntervalTemplateSpec(
                id="move",
                start_offset=0,
                duration=2,
                audit_kind="MoveToTarget",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("motion_resource")
                    ),
                ),
            ),
            IntervalTemplateSpec(
                id="place",
                start_offset=2,
                duration=1,
                audit_kind="Place",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("motion_resource")
                    ),
                ),
                start_effects=(
                    AcquireLeaseTemplateEffect(
                        resource=_parameter_ref("target_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
            ),
        ),
    )

    def transport_seed(
        intent_id: str,
        wafer: str,
        target: str,
        group_id: str,
        earliest_start: int,
        latest_start: int | None,
    ) -> IntentSeedSpec:
        return IntentSeedSpec(
            id=intent_id,
            operator_template_id="transport.to.target",
            alternative_group_id=group_id,
            earliest_start_offset=earliest_start,
            latest_start_offset=latest_start,
            bindings=(
                BindingAssignment(parameter="wafer", value=wafer),
                BindingAssignment(
                    parameter="motion_resource",
                    value="robot.motion",
                ),
                BindingAssignment(parameter="target_resource", value=target),
            ),
        )

    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(
            ResourceSpec(id="robot.motion", capacity=2),
            ResourceSpec(id="pm.PM1.slot", capacity=1),
            ResourceSpec(id="pm.PM2.slot", capacity=1),
        ),
        operator_templates=(transport,),
        intent_seeds=(
            transport_seed(
                "route.A.PM1",
                "wafer.A",
                "pm.PM1.slot",
                "route.visit.A",
                1,
                4,
            ),
            transport_seed(
                "route.A.PM2",
                "wafer.A",
                "pm.PM2.slot",
                "route.visit.A",
                4,
                8,
            ),
            transport_seed(
                "route.B.PM1",
                "wafer.B",
                "pm.PM1.slot",
                "route.visit.B",
                1,
                None,
            ),
        ),
    )


def test_g09_candidates_share_structure_and_expose_generic_footprints() -> None:
    problem = _g09_problem()
    frame = ReferenceKernel.start(problem).frame()
    candidates = {intent.id: intent for intent in frame.intents}

    pm1 = candidates["route.A.PM1"]
    pm2 = candidates["route.A.PM2"]
    assert pm1.operator_template_id == pm2.operator_template_id
    assert pm1.alternative_group_id == pm2.alternative_group_id == "route.visit.A"
    assert (pm1.earliest_start_tick, pm1.latest_start_tick) == (1, 4)
    assert (pm2.earliest_start_tick, pm2.latest_start_tick) == (4, 8)
    assert candidates["route.B.PM1"].latest_start_tick is None
    assert pm1.duration_ticks == pm2.duration_ticks == 3
    assert {
        (
            item.resource_id,
            item.start_tick,
            item.end_tick,
        )
        for item in pm1.resource_footprint
    } == {
        ("robot.motion", 1, 4),
        ("pm.PM1.slot", 3, None),
    }
    assert {
        (
            item.resource_id,
            item.start_tick,
            item.end_tick,
        )
        for item in pm2.resource_footprint
    } == {
        ("robot.motion", 4, 7),
        ("pm.PM2.slot", 6, None),
    }


def test_g09_alternative_group_allows_only_one_binding_and_reserves_target() -> None:
    problem = _g09_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()

    with pytest.raises(SemanticError) as exc_info:
        session.commit(
            frame.frame_token,
            ("route.A.PM1", "route.A.PM2"),
        )
    assert exc_info.value.code is DiagnosticCode.ALTERNATIVE_GROUP_CONFLICT

    result = session.commit(frame.frame_token, ("route.A.PM1",))
    assert (
        "route.A.PM1",
        "pm.PM1.slot",
        3,
        None,
    ) in {
        (
            item.intent_id,
            item.resource_id,
            item.start_tick,
            item.end_tick,
        )
        for item in result.reservations
    }
    next_frame = session.frame()
    assert "route.A.PM2" not in {intent.id for intent in next_frame.intents}
    assert ReferenceValidator.validate(problem, result.schedule).ok


def test_g09_batch_rejects_two_wafers_reserving_the_same_target() -> None:
    problem = _g09_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()

    with pytest.raises(SemanticError) as exc_info:
        session.commit(
            frame.frame_token,
            ("route.A.PM1", "route.B.PM1"),
        )

    assert exc_info.value.code is DiagnosticCode.RESOURCE_OVER_CAPACITY


def test_g09_validator_rejects_schedule_containing_two_alternatives() -> None:
    problem = _g09_problem()
    pm1_session = ReferenceKernel.start(problem)
    pm1_frame = pm1_session.frame()
    pm1 = pm1_session.commit(pm1_frame.frame_token, ("route.A.PM1",))
    pm2_session = ReferenceKernel.start(problem)
    pm2_frame = pm2_session.frame()
    pm2 = pm2_session.commit(pm2_frame.frame_token, ("route.A.PM2",))
    combined = ScheduleV1(
        events=pm1.schedule.events + pm2.schedule.events,
        intervals=pm1.schedule.intervals + pm2.schedule.intervals,
    )

    report = ReferenceValidator.validate(problem, combined)

    assert [issue.code for issue in report.issues] == [
        DiagnosticCode.ALTERNATIVE_GROUP_CONFLICT
    ]


def test_g09_rejects_inverted_start_window_statically() -> None:
    with pytest.raises(ValidationError, match="latest_start_offset"):
        IntentSeedSpec(
            id="invalid.window",
            operator_template_id="transport",
            earliest_start_offset=5,
            latest_start_offset=4,
            bindings=(),
        )


def _g10_problem() -> ConstraintIRV1:
    base = _g05_problem()
    unload = OperatorTemplateSpec(
        id="unload.target",
        origin="selectable",
        parameters=(
            _parameter("wafer", "owner"),
            _parameter("target_resource", "resource"),
            _parameter("interface_resource", "resource"),
        ),
        intervals=(
            IntervalTemplateSpec(
                id="unload",
                start_offset=0,
                duration=1,
                audit_kind="Unload",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("interface_resource")
                    ),
                ),
                end_effects=(
                    ReleaseLeaseTemplateEffect(
                        resource=_parameter_ref("target_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
            ),
        ),
    )
    inspect = OperatorTemplateSpec(
        id="inspect.auxiliary",
        origin="selectable",
        parameters=(_parameter("resource", "resource"),),
        intervals=(
            IntervalTemplateSpec(
                id="inspect",
                start_offset=0,
                duration=2,
                audit_kind="Inspect",
                resource_uses=(
                    ResourceUseTemplate(resource=_parameter_ref("resource")),
                ),
            ),
        ),
    )
    return ConstraintIRV1(
        time_domain=base.time_domain,
        resources=base.resources
        + (
            ResourceSpec(id="target.unload.interface", capacity=1),
            ResourceSpec(id="aux.inspect", capacity=1),
        ),
        state_cells=base.state_cells,
        initial_state=base.initial_state,
        operator_templates=base.operator_templates + (unload, inspect),
        intent_seeds=base.intent_seeds
        + (
            IntentSeedSpec(
                id="unload.A",
                operator_template_id="unload.target",
                bindings=(
                    BindingAssignment(parameter="wafer", value="wafer.A"),
                    BindingAssignment(
                        parameter="target_resource",
                        value="target.A",
                    ),
                    BindingAssignment(
                        parameter="interface_resource",
                        value="target.unload.interface",
                    ),
                ),
            ),
            IntentSeedSpec(
                id="inspect.aux",
                operator_template_id="inspect.auxiliary",
                bindings=(
                    BindingAssignment(
                        parameter="resource",
                        value="aux.inspect",
                    ),
                ),
            ),
        ),
        automatic_rules=base.automatic_rules,
    )


def test_g10_snapshot_round_trip_restores_frame_and_future_schedule() -> None:
    problem = _g10_problem()
    uninterrupted = ReferenceKernel.start(problem)
    initial_frame = uninterrupted.frame()
    initial_token = initial_frame.frame_token
    uninterrupted.commit(initial_token, ("transport.A",))

    checkpoint = uninterrupted.snapshot(3)
    payload = checkpoint.canonical_json()
    decoded = SessionSnapshot.model_validate_json(payload)
    assert decoded.canonical_json() == payload
    assert decoded.snapshot_hash == checkpoint.snapshot_hash
    assert checkpoint.problem_hash == problem.problem_hash
    assert checkpoint.schedule_hash == checkpoint.schedule.schedule_hash
    assert checkpoint.kernel_state_hash == checkpoint.kernel_snapshot.state_hash
    assert checkpoint.kernel_snapshot.active_interval_ids == (
        "intent.transport.A.move_to_target",
    )

    restored = ReferenceKernel.restore(problem, payload)
    independently_restored = ReferenceKernel.restore(problem, checkpoint)
    restored_frame = restored.frame()
    assert restored_frame == independently_restored.frame()
    assert restored_frame.tick == 3
    assert restored_frame.frame_token == checkpoint.frame_token
    assert [intent.id for intent in restored_frame.intents] == ["inspect.aux"]

    before_stale_commit = restored.snapshot().snapshot_hash
    with pytest.raises(SemanticError) as exc_info:
        restored.commit(initial_token, ("inspect.aux",))
    assert exc_info.value.code is DiagnosticCode.STALE_FRAME
    assert restored.snapshot().snapshot_hash == before_stale_commit

    restored.advance_to(18)
    restored_unload_frame = restored.frame()
    assert "unload.A" in {
        intent.id for intent in restored_unload_frame.intents
    }
    restored_result = restored.commit(
        restored_unload_frame.frame_token,
        ("unload.A",),
    )

    uninterrupted.advance_to(18)
    uninterrupted_unload_frame = uninterrupted.frame()
    uninterrupted_result = uninterrupted.commit(
        uninterrupted_unload_frame.frame_token,
        ("unload.A",),
    )
    assert (
        restored_result.schedule.canonical_json()
        == uninterrupted_result.schedule.canonical_json()
    )
    assert (
        restored_result.execution.final_snapshot.state_hash
        == uninterrupted_result.execution.final_snapshot.state_hash
    )

    report = ReferenceValidator.validate(
        problem,
        restored_result.schedule,
        require_terminal=True,
    )
    assert report.ok
    assert report.final_snapshot is not None
    assert report.final_snapshot.active_leases == ()
    assert report.final_snapshot.active_obligations == ()


def test_g10_canonical_hashes_ignore_declaration_and_schedule_order() -> None:
    problem = _g10_problem()
    reordered_problem = problem.model_copy(
        update={
            "resources": tuple(reversed(problem.resources)),
            "operator_templates": tuple(reversed(problem.operator_templates)),
            "intent_seeds": tuple(reversed(problem.intent_seeds)),
        }
    )
    assert reordered_problem.problem_hash == problem.problem_hash
    assert reordered_problem.canonical_json() == problem.canonical_json()

    session = ReferenceKernel.start(problem)
    frame = session.frame()
    result = session.commit(frame.frame_token, ("transport.A",))
    reordered_schedule = result.schedule.model_copy(
        update={
            "events": tuple(reversed(result.schedule.events)),
            "intervals": tuple(reversed(result.schedule.intervals)),
        }
    )
    assert reordered_schedule.schedule_hash == result.schedule.schedule_hash
    assert reordered_schedule.canonical_json() == result.schedule.canonical_json()


def test_g10_restore_rejects_wrong_problem_and_tampered_state_hash() -> None:
    problem = _g10_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    session.commit(frame.frame_token, ("transport.A",))
    checkpoint = session.snapshot(3)

    with pytest.raises(SemanticError) as wrong_problem:
        ReferenceKernel.restore(_g05_problem(), checkpoint)
    assert wrong_problem.value.code is DiagnosticCode.SNAPSHOT_PROBLEM_MISMATCH

    corrupted = checkpoint.model_copy(
        update={"kernel_state_hash": "0" * 64}
    )
    with pytest.raises(SemanticError) as wrong_state:
        ReferenceKernel.restore(problem, corrupted)
    assert wrong_state.value.code is DiagnosticCode.SNAPSHOT_STATE_MISMATCH


def test_g10_validator_rejects_effect_digest_and_boundary_tampering() -> None:
    problem = _g10_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    result = session.commit(frame.frame_token, ("transport.A",))

    digest_events = tuple(
        event.model_copy(update={"effect_digest": "0" * 64})
        if event.audit_kind == "Process.start"
        else event
        for event in result.schedule.events
    )
    digest_report = ReferenceValidator.validate(
        problem,
        result.schedule.model_copy(update={"events": digest_events}),
    )
    assert [issue.code for issue in digest_report.issues] == [
        DiagnosticCode.EFFECT_DIGEST_MISMATCH
    ]

    binding_events = tuple(
        event.model_copy(
            update={
                "bindings": (
                    BindingAssignment(
                        parameter="process_resource",
                        value="robot.motion",
                    ),
                )
            }
        )
        if event.audit_kind == "Process.start"
        else event
        for event in result.schedule.events
    )
    binding_report = ReferenceValidator.validate(
        problem,
        result.schedule.model_copy(update={"events": binding_events}),
    )
    assert [issue.code for issue in binding_report.issues] == [
        DiagnosticCode.MISSING_AUTOMATIC_EVENT
    ]


def test_g10_terminal_validation_rejects_open_leases() -> None:
    problem = _g10_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    result = session.commit(frame.frame_token, ("transport.A",))

    report = ReferenceValidator.validate(
        problem,
        result.schedule,
        require_terminal=True,
    )

    assert [issue.code for issue in report.issues] == [
        DiagnosticCode.NON_TERMINAL_STATE
    ]


def _composite_exchange_problem(
    *,
    process_ready: bool = True,
    place_blocked: bool = False,
    outgoing_hand_blocked: bool = False,
) -> ConstraintIRV1:
    exchange = OperatorTemplateSpec(
        id="exchange",
        origin="selectable",
        parameters=(
            _parameter("outgoing_wafer", "owner"),
            _parameter("incoming_wafer", "owner"),
            _parameter("motion_resource", "resource"),
            _parameter("outgoing_hand_resource", "resource"),
            _parameter("incoming_hand_resource", "resource"),
            _parameter("chamber_holder_resource", "resource"),
            _parameter("place_gate_resource", "resource"),
            _parameter("process_state_cell", "state_cell"),
        ),
        intervals=(
            IntervalTemplateSpec(
                id="pick_out",
                start_offset=0,
                duration=1,
                audit_kind="PickOut",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("motion_resource")
                    ),
                ),
                start_effects=(
                    AcquireLeaseTemplateEffect(
                        resource=_parameter_ref("outgoing_hand_resource"),
                        owner=_parameter_ref("outgoing_wafer"),
                    ),
                ),
                end_effects=(
                    ReleaseLeaseTemplateEffect(
                        resource=_parameter_ref("chamber_holder_resource"),
                        owner=_parameter_ref("outgoing_wafer"),
                    ),
                ),
            ),
            IntervalTemplateSpec(
                id="place_in",
                start_offset=1,
                duration=1,
                audit_kind="PlaceIn",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("motion_resource")
                    ),
                    ResourceUseTemplate(
                        resource=_parameter_ref("place_gate_resource")
                    ),
                ),
                start_effects=(
                    AcquireLeaseTemplateEffect(
                        resource=_parameter_ref("chamber_holder_resource"),
                        owner=_parameter_ref("incoming_wafer"),
                    ),
                ),
                end_effects=(
                    ReleaseLeaseTemplateEffect(
                        resource=_parameter_ref("incoming_hand_resource"),
                        owner=_parameter_ref("incoming_wafer"),
                    ),
                    SetStateTemplateEffect(
                        cell=_parameter_ref("process_state_cell"),
                        value="processing",
                    ),
                ),
            ),
        ),
        step_dependencies=(
            StepDependencySpec(
                predecessor_step_id="pick_out",
                predecessor_boundary="end",
                successor_step_id="place_in",
                successor_boundary="start",
            ),
        ),
    )
    pick_out = OperatorTemplateSpec(
        id="pick_out",
        origin="selectable",
        parameters=(
            _parameter("outgoing_wafer", "owner"),
            _parameter("motion_resource", "resource"),
            _parameter("outgoing_hand_resource", "resource"),
            _parameter("chamber_holder_resource", "resource"),
        ),
        intervals=(exchange.intervals[0],),
    )
    exchange_bindings = (
        BindingAssignment(parameter="outgoing_wafer", value="wafer1"),
        BindingAssignment(parameter="incoming_wafer", value="wafer2"),
        BindingAssignment(parameter="motion_resource", value="robot.motion"),
        BindingAssignment(
            parameter="outgoing_hand_resource",
            value="robot.arm1",
        ),
        BindingAssignment(
            parameter="incoming_hand_resource",
            value="robot.arm0",
        ),
        BindingAssignment(
            parameter="chamber_holder_resource",
            value="pm1.holder",
        ),
        BindingAssignment(
            parameter="place_gate_resource",
            value="pm1.place_gate",
        ),
        BindingAssignment(
            parameter="process_state_cell",
            value="pm1.process",
        ),
    )
    initial_leases = [
        LeaseSpec(resource_id="pm1.holder", owner_id="wafer1"),
        LeaseSpec(resource_id="robot.arm0", owner_id="wafer2"),
    ]
    if place_blocked:
        initial_leases.append(
            LeaseSpec(resource_id="pm1.place_gate", owner_id="cleaning")
        )
    if outgoing_hand_blocked:
        initial_leases.append(
            LeaseSpec(resource_id="robot.arm1", owner_id="blocking_wafer")
        )
    guard = StateCondition(
        cell_id="pm1.process",
        operator="equal",
        value="completed",
    )
    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(
            ResourceSpec(id="robot.motion", capacity=1),
            ResourceSpec(id="robot.arm0", capacity=1),
            ResourceSpec(id="robot.arm1", capacity=1),
            ResourceSpec(id="pm1.holder", capacity=1),
            ResourceSpec(id="pm1.place_gate", capacity=1),
        ),
        state_cells=(
            StateCellSpec(
                id="pm1.process",
                value_type="enum",
                enum_values=("processing", "completed"),
            ),
        ),
        initial_state=InitialStateSpec(
            state_values=(
                StateAssignment(
                    cell_id="pm1.process",
                    value="completed" if process_ready else "processing",
                ),
            ),
            leases=tuple(initial_leases),
        ),
        operator_templates=(exchange, pick_out),
        intent_seeds=(
            IntentSeedSpec(
                id="exchange.wafer1.wafer2.pm1",
                operator_template_id="exchange",
                bindings=exchange_bindings,
                guards=(
                    guard,
                    LeaseCondition(resource_id="pm1.holder", owner_id="wafer1"),
                    LeaseCondition(resource_id="robot.arm0", owner_id="wafer2"),
                ),
                choice_scope_claims=(
                    ChoiceScopeClaimSpec(
                        scope_key="holder-stage/wafer1/extract",
                        release_boundary_id="pick_out.end",
                    ),
                    ChoiceScopeClaimSpec(
                        scope_key="route-visit/wafer2/place",
                        release_boundary_id="place_in.end",
                    ),
                ),
            ),
            IntentSeedSpec(
                id="pick.wafer1.pm1",
                operator_template_id="pick_out",
                bindings=tuple(
                    binding
                    for binding in exchange_bindings
                    if binding.parameter
                    in {
                        "outgoing_wafer",
                        "motion_resource",
                        "outgoing_hand_resource",
                        "chamber_holder_resource",
                    }
                ),
                guards=(guard, LeaseCondition(resource_id="pm1.holder", owner_id="wafer1")),
            ),
        ),
    )


def test_composite_exchange_is_one_complete_bundle_with_predicted_delta() -> None:
    problem = _composite_exchange_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    candidates = {intent.id: intent for intent in frame.intents}

    assert set(candidates) == {
        "exchange.wafer1.wafer2.pm1",
        "pick.wafer1.pm1",
    }
    exchange = candidates["exchange.wafer1.wafer2.pm1"]
    assert exchange.duration_ticks == 2
    assert exchange.state_delta.completion_tick == 2
    assert [
        (item.cell_id, item.before, item.after)
        for item in exchange.state_delta.state_values
    ] == [("pm1.process", "completed", "processing")]
    assert {
        (
            item.resource_id,
            item.owner_id,
            item.before_amount,
            item.after_amount,
        )
        for item in exchange.state_delta.leases
    } == {
        ("pm1.holder", "wafer1", 1, 0),
        ("pm1.holder", "wafer2", 0, 1),
        ("robot.arm0", "wafer2", 1, 0),
        ("robot.arm1", "wafer1", 0, 1),
    }
    assert {
        "pm1.holder",
        "pm1.process",
        "robot.arm0",
        "robot.arm1",
        "wafer1",
        "wafer2",
    }.issubset(exchange.involved_entity_ids)

    result = session.commit(frame.frame_token, (exchange.id,))

    assert [interval.audit_kind for interval in result.schedule.intervals] == [
        "PickOut",
        "PlaceIn",
    ]
    assert result.execution.final_snapshot.tick == 0
    assert session.advance_next().final_snapshot.tick == 1
    assert session.advance_next().final_snapshot.tick == 2
    assert session.snapshot().active_choice_scope_keys == ()
    assert ReferenceValidator.validate(problem, result.schedule).ok


def test_composite_bundle_is_hidden_when_only_pick_step_is_feasible() -> None:
    problem = _composite_exchange_problem(place_blocked=True)
    session = ReferenceKernel.start(problem)
    before = session.snapshot()
    frame = session.frame()

    assert [intent.id for intent in frame.intents] == ["pick.wafer1.pm1"]
    with pytest.raises(SemanticError) as exc_info:
        session.commit(frame.frame_token, ("exchange.wafer1.wafer2.pm1",))

    assert exc_info.value.code is DiagnosticCode.INTENT_NOT_COMMITTABLE
    assert session.schedule == ScheduleV1()
    assert session.snapshot().snapshot_hash == before.snapshot_hash


@pytest.mark.parametrize(
    "problem",
    [
        pytest.param(
            _composite_exchange_problem(process_ready=False),
            id="process-active",
        ),
        pytest.param(
            _composite_exchange_problem(outgoing_hand_blocked=True),
            id="outgoing-hand-full",
        ),
    ],
)
def test_composite_exchange_guard_or_lease_failure_hides_candidate(
    problem: ConstraintIRV1,
) -> None:
    assert "exchange.wafer1.wafer2.pm1" not in {
        intent.id for intent in ReferenceKernel.start(problem).frame().intents
    }


def test_composite_step_dependencies_support_parallel_activity() -> None:
    template = OperatorTemplateSpec(
        id="ll.concurrent",
        origin="selectable",
        parameters=(
            _parameter("thermal_resource", "resource"),
            _parameter("pressure_resource", "resource"),
        ),
        intervals=(
            IntervalTemplateSpec(
                id="cooling",
                start_offset=0,
                duration=5,
                audit_kind="Cooling",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("thermal_resource")
                    ),
                ),
            ),
            IntervalTemplateSpec(
                id="transitioning",
                start_offset=0,
                duration=3,
                audit_kind="PressureTransition",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("pressure_resource")
                    ),
                ),
            ),
        ),
        step_dependencies=(
            StepDependencySpec(
                predecessor_step_id="cooling",
                predecessor_boundary="start",
                successor_step_id="transitioning",
                successor_boundary="start",
            ),
        ),
    )
    problem = ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(
            ResourceSpec(id="ll.thermal", capacity=1),
            ResourceSpec(id="ll.pressure", capacity=1),
        ),
        operator_templates=(template,),
        intent_seeds=(
            IntentSeedSpec(
                id="ll.concurrent.wafer1",
                operator_template_id="ll.concurrent",
                bindings=(
                    BindingAssignment(
                        parameter="thermal_resource",
                        value="ll.thermal",
                    ),
                    BindingAssignment(
                        parameter="pressure_resource",
                        value="ll.pressure",
                    ),
                ),
            ),
        ),
    )

    session = ReferenceKernel.start(problem)
    frame = session.frame()
    result = session.commit(frame.frame_token, (frame.intents[0].id,))
    at_start = next(
        snapshot for snapshot in result.execution.snapshots if snapshot.tick == 0
    )

    assert set(at_start.active_interval_ids) == {
        "intent.ll.concurrent.wafer1.cooling",
        "intent.ll.concurrent.wafer1.transitioning",
    }


def test_composite_rejects_offsets_that_violate_step_dependency() -> None:
    with pytest.raises(ValidationError, match="step dependency is not satisfied"):
        OperatorTemplateSpec(
            id="broken.exchange",
            origin="selectable",
            parameters=(_parameter("resource", "resource"),),
            intervals=(
                IntervalTemplateSpec(
                    id="pick",
                    start_offset=0,
                    duration=2,
                    audit_kind="Pick",
                    resource_uses=(
                        ResourceUseTemplate(resource=_parameter_ref("resource")),
                    ),
                ),
                IntervalTemplateSpec(
                    id="place",
                    start_offset=1,
                    duration=1,
                    audit_kind="Place",
                    resource_uses=(
                        ResourceUseTemplate(resource=_parameter_ref("resource")),
                    ),
                ),
            ),
            step_dependencies=(
                StepDependencySpec(
                    predecessor_step_id="pick",
                    successor_step_id="place",
                ),
            ),
        )


def test_candidate_identity_is_canonical_and_commit_accepts_key() -> None:
    problem = _g09_problem()
    reordered = problem.model_copy(
        update={"intent_seeds": tuple(reversed(problem.intent_seeds))}
    )
    first = ReferenceKernel.start(problem).frame()
    second = ReferenceKernel.start(reordered).frame()

    assert [item.candidate_key for item in first.intents] == [
        item.candidate_key for item in second.intents
    ]
    assert [item.candidate_digest for item in first.intents] == [
        item.candidate_digest for item in second.intents
    ]

    session = ReferenceKernel.start(problem)
    frame = session.frame()
    selected = next(item for item in frame.intents if item.id == "route.A.PM1")
    result = session.commit(frame.frame_token, (selected.candidate_key,))

    assert result.commit_record is not None
    assert result.commit_record.selections[0].candidate_key == selected.candidate_key
    assert session.commit_log == (result.commit_record,)


def test_choice_scope_conflict_is_independent_of_alternative_groups() -> None:
    problem = _g08_problem(motion_capacity=2, geometry_compatible=True)
    shared_scope = (
        ChoiceScopeClaimSpec(scope_key="holder-stage/shared"),
    )
    scoped = problem.model_copy(
        update={
            "intent_seeds": tuple(
                seed.model_copy(update={"choice_scope_claims": shared_scope})
                for seed in problem.intent_seeds
            )
        }
    )
    session = ReferenceKernel.start(scoped)
    frame = session.frame()

    with pytest.raises(SemanticError) as exc_info:
        session.commit(frame.frame_token, tuple(item.id for item in frame.intents))

    assert exc_info.value.code is DiagnosticCode.CHOICE_SCOPE_CONFLICT
    assert session.schedule == ScheduleV1()
    assert session.commit_log == ()


def test_commit_log_round_trip_and_tamper_detection() -> None:
    problem = _g05_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    session.commit(frame.frame_token, (frame.intents[0].candidate_key,))
    snapshot = session.snapshot()

    restored = ReferenceKernel.restore(problem, snapshot.canonical_json())

    assert restored.commit_log == session.commit_log
    assert restored.frame().frame_token == snapshot.frame_token

    record = snapshot.commit_log[0]
    selection = record.selections[0].model_copy(
        update={"candidate_digest": "0" * 64}
    )
    corrupted = snapshot.model_copy(
        update={
            "commit_log": (
                record.model_copy(update={"selections": (selection,)}),
            )
        }
    )
    with pytest.raises(SemanticError) as exc_info:
        ReferenceKernel.restore(problem, corrupted)

    assert exc_info.value.code is DiagnosticCode.COMMIT_LOG_MISMATCH


def test_reference_kernel_uses_injected_candidate_generator() -> None:
    class RecordingGenerator:
        def __init__(self) -> None:
            self.calls = 0
            self.delegate = LegacyIntentSeedCandidateGenerator()

        def generate(self, context):
            self.calls += 1
            return self.delegate.generate(context)

    generator = RecordingGenerator()
    session = ReferenceKernel.start(_g05_problem(), generator)

    assert session.frame().intents
    assert generator.calls == 1


def _dynamic_pick_problem(*, second_ready: bool = True) -> ConstraintIRV1:
    parameters = (
        _parameter("wafer", "owner"),
        _parameter("motion_resource", "resource"),
        _parameter("hand_resource", "resource"),
        _parameter("source_resource", "resource"),
        _parameter("ready_cell", "state_cell"),
    )
    pick = OperatorTemplateSpec(
        id="dynamic.pick",
        origin="selectable",
        parameters=parameters,
        intervals=(
            IntervalTemplateSpec(
                id="pick",
                start_offset=0,
                duration=1,
                audit_kind="Pick",
                resource_uses=(
                    ResourceUseTemplate(
                        resource=_parameter_ref("motion_resource")
                    ),
                ),
                start_effects=(
                    AcquireLeaseTemplateEffect(
                        resource=_parameter_ref("hand_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
                end_effects=(
                    ReleaseLeaseTemplateEffect(
                        resource=_parameter_ref("source_resource"),
                        owner=_parameter_ref("wafer"),
                    ),
                ),
            ),
        ),
    )
    domain = BindingDomainSpec(
        id="pick.bindings",
        parameters=parameters,
        rows=(
            BindingRowSpec(
                values=(
                    "wafer.A",
                    "robot.motion",
                    "robot.arm0",
                    "source.A",
                    "wafer.A.ready",
                )
            ),
            BindingRowSpec(
                values=(
                    "wafer.B",
                    "robot.motion",
                    "robot.arm1",
                    "source.B",
                    "wafer.B.ready",
                )
            ),
        ),
    )
    dynamic_pick = DynamicIntentSpec(
        id="pick.available.wafer",
        operator_template_id="dynamic.pick",
        binding_domain_id="pick.bindings",
        choice_scope_templates=(
            ChoiceScopeTemplateSpec(
                scope_prefix="holder-stage/pick",
                identity_parameters=("wafer", "source_resource"),
            ),
        ),
        guards=(
            LeaseConditionTemplate(
                resource=_parameter_ref("source_resource"),
                owner=_parameter_ref("wafer"),
            ),
            StateConditionTemplate(
                cell=_parameter_ref("ready_cell"),
                operator="equal",
                value="ready",
            ),
        ),
    )
    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(
            ResourceSpec(id="robot.motion", capacity=2),
            ResourceSpec(id="robot.arm0", capacity=1),
            ResourceSpec(id="robot.arm1", capacity=1),
            ResourceSpec(id="source.A", capacity=1),
            ResourceSpec(id="source.B", capacity=1),
        ),
        state_cells=(
            StateCellSpec(
                id="wafer.A.ready",
                value_type="enum",
                enum_values=("blocked", "ready"),
            ),
            StateCellSpec(
                id="wafer.B.ready",
                value_type="enum",
                enum_values=("blocked", "ready"),
            ),
        ),
        initial_state=InitialStateSpec(
            state_values=(
                StateAssignment(cell_id="wafer.A.ready", value="ready"),
                StateAssignment(
                    cell_id="wafer.B.ready",
                    value="ready" if second_ready else "blocked",
                ),
            ),
            leases=(
                LeaseSpec(resource_id="source.A", owner_id="wafer.A"),
                LeaseSpec(resource_id="source.B", owner_id="wafer.B"),
            ),
        ),
        operator_templates=(pick,),
        binding_domains=(domain,),
        dynamic_intents=(dynamic_pick,),
    )


def test_exhaustive_generator_filters_typed_rows_by_current_state() -> None:
    frame = ReferenceKernel.start(
        _dynamic_pick_problem(second_ready=False)
    ).frame()

    assert len(frame.intents) == 1
    assert dict(
        (item.parameter, item.value) for item in frame.intents[0].bindings
    )["wafer"] == "wafer.A"


def test_one_template_creates_multiple_dynamic_operator_instances() -> None:
    problem = _dynamic_pick_problem()
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    reversed_domain = problem.binding_domains[0].model_copy(
        update={"rows": tuple(reversed(problem.binding_domains[0].rows))}
    )
    reordered = problem.model_copy(
        update={"binding_domains": (reversed_domain,)}
    )

    assert len(frame.intents) == 2
    assert [item.candidate_key for item in frame.intents] == [
        item.candidate_key
        for item in ReferenceKernel.start(reordered).frame().intents
    ]
    assert {item.operator_template_id for item in frame.intents} == {
        "dynamic.pick"
    }
    result = session.commit(
        frame.frame_token,
        tuple(item.candidate_key for item in frame.intents),
    )
    instance_ids = {
        interval.operator_instance_id for interval in result.schedule.intervals
    }

    assert len(instance_ids) == 2
    assert session.snapshot().committed_intent_ids == ()
    assert len(session.snapshot().active_choice_scope_keys) == 2
    assert session.frame().intents == ()
    restored = ReferenceKernel.restore(problem, session.snapshot())
    assert restored.commit_log == session.commit_log


def test_binding_domain_rejects_unknown_typed_resource() -> None:
    problem = _dynamic_pick_problem()
    bad_domain = problem.binding_domains[0].model_copy(
        update={
            "rows": (
                BindingRowSpec(
                    values=(
                        "wafer.A",
                        "missing.motion",
                        "robot.arm0",
                        "source.A",
                        "wafer.A.ready",
                    )
                ),
            )
        }
    )

    with pytest.raises(ValidationError, match="unknown resource"):
        payload = problem.model_dump(mode="json")
        payload["binding_domains"] = [bad_domain.model_dump(mode="json")]
        ConstraintIRV1.model_validate(payload)
