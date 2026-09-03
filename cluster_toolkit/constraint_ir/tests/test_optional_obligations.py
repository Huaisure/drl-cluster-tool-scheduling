from itertools import permutations

import pytest
from pydantic import ValidationError

from cluster_toolkit.constraint_ir import (
    ConstraintIRV1,
    CreateObligationEffect,
    CreateObligationTemplateEffect,
    DiagnosticCode,
    EventSpec,
    InitialStateSpec,
    IntentSeedSpec,
    IntervalTemplateSpec,
    LiteralIdRef,
    ObligationInstanceSpec,
    OperatorTemplateSpec,
    ReferenceKernel,
    ReferenceValidator,
    ResourceSpec,
    ResourceUseTemplate,
    SatisfyObligationEffect,
    SatisfyObligationTemplateEffect,
    ScheduleV1,
    SemanticError,
    TimeDomain,
)


def _problem(**kwargs: object) -> ConstraintIRV1:
    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1), **kwargs
    )


def test_unbounded_initial_obligation_persists_without_time_events() -> None:
    problem = _problem(
        initial_state=InitialStateSpec(
            obligations=(ObligationInstanceSpec(id="work"),),
        ),
    )
    schedule = ScheduleV1()
    result = ReferenceKernel.execute(problem, schedule)
    report = ReferenceValidator.validate(problem, schedule)

    assert report.ok
    assert report.final_snapshot == result.final_snapshot
    assert result.final_snapshot.tick == 0
    assert result.final_snapshot.active_obligations[0].deadline_tick is None
    later = ReferenceKernel.execute_until(problem, schedule, 1_000_000)
    assert later.final_snapshot.active_obligations == (
        result.final_snapshot.active_obligations
    )
    terminal = ReferenceValidator.validate(problem, schedule, require_terminal=True)
    assert [issue.code for issue in terminal.issues] == [
        DiagnosticCode.NON_TERMINAL_STATE
    ]


@pytest.mark.parametrize("satisfy_tick", [10, 1_000_000])
def test_unbounded_event_obligation_can_be_satisfied_without_deadline(
    satisfy_tick: int,
) -> None:
    problem = _problem()
    schedule = ScheduleV1(
        events=(
            EventSpec(
                id="create", tick=10,
                effects=(CreateObligationEffect(obligation_id="work"),),
            ),
            EventSpec(
                id="satisfy", tick=satisfy_tick,
                effects=(SatisfyObligationEffect(obligation_id="work"),),
            ),
        ),
    )
    result = ReferenceKernel.execute(problem, schedule)
    report = ReferenceValidator.validate(problem, schedule, require_terminal=True)
    assert report.ok
    assert report.final_snapshot == result.final_snapshot
    assert result.final_snapshot.active_obligations == ()
    assert {snapshot.tick for snapshot in result.snapshots} == {10, satisfy_tick}


@pytest.mark.parametrize("satisfy_tick", [3, 4])
def test_unbounded_obligation_does_not_disable_finite_deadline_checks(
    satisfy_tick: int,
) -> None:
    problem = _problem(
        initial_state=InitialStateSpec(obligations=(
            ObligationInstanceSpec(id="unbounded"),
            ObligationInstanceSpec(id="bounded", deadline_tick=3),
        )),
    )
    schedule = ScheduleV1(events=(EventSpec(
        id="satisfy", tick=satisfy_tick,
        effects=(SatisfyObligationEffect(obligation_id="bounded"),),
    ),))
    report = ReferenceValidator.validate(problem, schedule)
    if satisfy_tick == 3:
        result = ReferenceKernel.execute(problem, schedule)
        assert report.ok
        assert report.final_snapshot == result.final_snapshot
        assert [item.id for item in result.final_snapshot.active_obligations] == [
            "unbounded"
        ]
    else:
        with pytest.raises(SemanticError) as exc_info:
            ReferenceKernel.execute(problem, schedule)
        assert exc_info.value.code is DiagnosticCode.DEADLINE_MISSED
        assert [issue.code for issue in report.issues] == [
            DiagnosticCode.DEADLINE_MISSED
        ]
        assert report.issues[0].tick == 3


@pytest.mark.parametrize("deadlines, expected", [
    ((None, None), None),
    ((None, 20), 20),
    ((30, None, 20), 20),
])
def test_coalescing_uses_earliest_finite_winning_deadline(
    deadlines: tuple[int | None, ...], expected: int | None,
) -> None:
    requests = tuple(
        CreateObligationEffect(
            obligation_id="work", deadline_tick=deadline,
            coalesce_key="scope", priority=1,
        )
        for deadline in deadlines
    ) + (CreateObligationEffect(
        obligation_id="superseded", deadline_tick=0,
        coalesce_key="scope", priority=0,
    ),)
    problem = _problem()
    for effects in permutations(requests):
        prefix = ScheduleV1(events=(EventSpec(id="create", tick=5, effects=effects),))
        state = ReferenceKernel.execute_until(problem, prefix, 5).final_snapshot
        assert len(state.active_obligations) == 1
        assert state.active_obligations[0].id == "work"
        assert state.active_obligations[0].deadline_tick == expected
        prefix_report = ReferenceValidator.validate(problem, prefix)
        if expected is None:
            assert prefix_report.ok
            assert prefix_report.final_snapshot == state
        else:
            assert [issue.code for issue in prefix_report.issues] == [
                DiagnosticCode.DEADLINE_MISSED
            ]
            assert prefix_report.issues[0].tick == expected
        schedule = ScheduleV1(events=prefix.events + (EventSpec(
            id="satisfy", tick=expected if expected is not None else 1_000_000,
            effects=(SatisfyObligationEffect(obligation_id="work"),),
        ),))
        result = ReferenceKernel.execute(problem, schedule)
        report = ReferenceValidator.validate(problem, schedule, require_terminal=True)
        assert report.ok
        assert report.final_snapshot == result.final_snapshot


def test_unbounded_template_obligation_gates_candidates_and_survives_restore() -> None:
    resource_uses = (ResourceUseTemplate(resource=LiteralIdRef(value="machine")),)
    problem = _problem(
        resources=(ResourceSpec(id="machine", capacity=1),),
        operator_templates=(
            OperatorTemplateSpec(
                id="issue", origin="selectable", parameters=(),
                intervals=(IntervalTemplateSpec(
                    id="step", start_offset=0, duration=1, audit_kind="Activity",
                    resource_uses=resource_uses,
                    end_effects=(CreateObligationTemplateEffect(obligation_id="work"),),
                ),),
            ),
            OperatorTemplateSpec(
                id="finish", origin="selectable", parameters=(),
                intervals=(IntervalTemplateSpec(
                    id="step", start_offset=0, duration=2, audit_kind="Activity",
                    resource_uses=resource_uses,
                    end_effects=(SatisfyObligationTemplateEffect(obligation_id="work"),),
                ),),
            ),
        ),
        intent_seeds=(
            IntentSeedSpec(id="issue", operator_template_id="issue", bindings=()),
            IntentSeedSpec(
                id="finish", operator_template_id="finish", bindings=(),
                required_obligation_ids=("work",),
            ),
        ),
    )
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    assert [candidate.id for candidate in frame.intents] == ["issue"]
    issued = session.commit(frame.frame_token, ("issue",))
    assert ReferenceValidator.validate(problem, issued.schedule).ok
    session.advance_next()
    active = session.snapshot().kernel_snapshot.active_obligations
    assert len(active) == 1 and active[0].deadline_tick is None
    snapshot = session.snapshot()
    restored = ReferenceKernel.restore(problem, snapshot.canonical_json())
    assert restored.snapshot().snapshot_hash == snapshot.snapshot_hash
    for current in (session, restored):
        frame = current.frame()
        assert [candidate.id for candidate in frame.intents] == ["finish"]
        result = current.commit(frame.frame_token, ("finish",))
        execution = current.advance_next()
        assert execution is not None
        assert execution.final_snapshot.active_obligations == ()
        assert ReferenceValidator.validate(
            problem, result.schedule, require_terminal=True,
        ).ok
    assert session.snapshot().snapshot_hash == restored.snapshot().snapshot_hash


@pytest.mark.parametrize("invalid", [-1, 0.5, True])
def test_optional_deadlines_still_require_nonnegative_integer_ticks(
    invalid: object,
) -> None:
    for model, values, field in (
        (ObligationInstanceSpec, {"id": "work"}, "deadline_tick"),
        (CreateObligationEffect, {"obligation_id": "work"}, "deadline_tick"),
        (CreateObligationTemplateEffect, {"obligation_id": "work"}, "deadline_offset"),
    ):
        assert getattr(model(**values, **{field: None}), field) is None
        with pytest.raises(ValidationError):
            model(**values, **{field: invalid})
