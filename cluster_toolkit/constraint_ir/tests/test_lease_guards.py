"""Lease predicates read current facts; no additional location state is stored."""

from itertools import permutations
from typing import Literal

import pytest
from pydantic import ValidationError

from cluster_toolkit.constraint_ir import (
    AcquireLeaseTemplateEffect,
    BindingAssignment,
    BindingDomainSpec,
    BindingRowSpec,
    ChoiceScopeTemplateSpec,
    ConstraintIRV1,
    DiagnosticCode,
    DynamicIntentSpec,
    InitialStateSpec,
    IntentSeedSpec,
    LeaseCondition,
    LeaseConditionTemplate,
    LeaseSpec,
    LiteralIdRef,
    ParameterIdRef,
    ParameterSpec,
    ReferenceKernel,
    ReferenceSession,
    ReferenceValidator,
    ReleaseLeaseTemplateEffect,
    ResourceSpec,
    SemanticError,
    StateAssignment,
    StateCellSpec,
    StateCondition,
    StateConditionTemplate,
    TimeDomain,
)
from cluster_toolkit.constraint_ir.tests.test_commit_audit import _retarget_first_commit
from cluster_toolkit.constraint_ir.tests.test_continuous_session import _activity
from cluster_toolkit.constraint_ir.tests.test_golden_cases import _composite_exchange_problem


def _problem(
    *, operator: Literal["present", "absent"] = "present", legacy: bool = False,
) -> ConstraintIRV1:
    parameters = (ParameterSpec(name="holder", kind="resource"), ParameterSpec(name="item", kind="owner"))
    rows = (BindingRowSpec(values=("hand0", "A")), BindingRowSpec(values=("hand1", "A")),
            BindingRowSpec(values=("hand0", "B")))
    template = _activity("work", 2, "motion").model_copy(update={"parameters": parameters})
    rule = DynamicIntentSpec(
        id="work", operator_template_id="work", binding_domain_id="holders",
        choice_scope_templates=(ChoiceScopeTemplateSpec(
            scope_prefix="work", identity_parameters=("holder", "item"),
            release_boundary_id="step.end",
        ),),
        guards=(
            LeaseConditionTemplate(resource=ParameterIdRef(parameter="holder"),
                                   owner=ParameterIdRef(parameter="item"), operator=operator),
            StateConditionTemplate(cell=LiteralIdRef(value="ready"), operator="equal", value=True),
        ),
    )
    seeds = tuple(IntentSeedSpec(
        id=f"work.{holder}.{item}", operator_template_id="work",
        bindings=(BindingAssignment(parameter="holder", value=holder),
                  BindingAssignment(parameter="item", value=item)),
        guards=(LeaseCondition(resource_id=holder, owner_id=item, operator=operator),
                StateCondition(cell_id="ready", operator="equal", value=True)),
    ) for holder, item in (row.values for row in rows))
    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=tuple(ResourceSpec(id=name, capacity=1) for name in ("hand0", "hand1", "motion")),
        state_cells=(StateCellSpec(id="ready", value_type="bool"),),
        initial_state=InitialStateSpec(
            state_values=(StateAssignment(cell_id="ready", value=True),),
            leases=(LeaseSpec(resource_id="hand0", owner_id="A"),),
        ),
        operator_templates=(template,),
        binding_domains=(BindingDomainSpec(id="holders", parameters=parameters, rows=rows),),
        dynamic_intents=() if legacy else (rule,),
        intent_seeds=seeds if legacy else (),
    )


def _pairs(session: ReferenceSession) -> set[tuple[str, str]]:
    values = [dict((binding.parameter, binding.value) for binding in candidate.bindings)
              for candidate in session.frame().intents if candidate.operator_template_id == "work"]
    return {(item["holder"], item["item"]) for item in values}


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("operator", ["present", "absent"])
def test_current_lease_filters_both_paths_and_combines_with_state(
    legacy: bool, operator: Literal["present", "absent"],
) -> None:
    problem = _problem(legacy=legacy, operator=operator)
    session = ReferenceKernel.start(problem)
    assert _pairs(session) == ({("hand0", "A")} if operator == "present"
                               else {("hand1", "A"), ("hand0", "B")})
    frame = session.frame()
    session.commit(frame.frame_token, (frame.intents[0].id,))
    session.advance_next()
    report = ReferenceValidator.validate_session(problem, session.snapshot().canonical_json())
    assert report.ok, report.issues
    assert ReferenceKernel.restore(problem, session.snapshot().canonical_json()).frame() == session.frame()
    blocked = problem.model_copy(update={"initial_state": problem.initial_state.model_copy(update={
        "state_values": (StateAssignment(cell_id="ready", value=False),),
    })})
    assert ReferenceKernel.start(blocked).frame().intents == ()


def test_absent_pair_is_not_empty_resource_or_capacity_permission() -> None:
    problem = _problem(operator="absent")
    assert ("hand0", "B") in _pairs(ReferenceKernel.start(problem))  # hand0 still holds A
    template = problem.operator_templates[0]
    acquiring = template.model_copy(update={"intervals": (template.intervals[0].model_copy(update={
        "start_effects": (AcquireLeaseTemplateEffect(
            resource=ParameterIdRef(parameter="holder"), owner=ParameterIdRef(parameter="item"),
        ),),
    }),)})
    constrained = problem.model_copy(update={"operator_templates": (acquiring,)})
    assert _pairs(ReferenceKernel.start(constrained)) == {("hand1", "A")}


def test_present_is_membership_not_exclusive_ownership_or_exact_amount() -> None:
    base = _problem()
    problem = base.model_copy(update={
        "resources": tuple(item.model_copy(update={"capacity": 3}) if item.id == "hand0" else item
                           for item in base.resources),
        "initial_state": base.initial_state.model_copy(update={"leases": (
            LeaseSpec(resource_id="hand0", owner_id="A", amount=2),
            LeaseSpec(resource_id="hand0", owner_id="B"),
        )}),
    })
    assert _pairs(ReferenceKernel.start(problem)) == {("hand0", "A"), ("hand0", "B")}


def test_lease_change_refreshes_candidates_and_invalidates_old_frame() -> None:
    base = _problem()
    transfer = _activity("transfer", 1, "motion", end_effects=(
        ReleaseLeaseTemplateEffect(resource=LiteralIdRef(value="hand0"), owner=LiteralIdRef(value="A")),
        AcquireLeaseTemplateEffect(resource=LiteralIdRef(value="hand1"), owner=LiteralIdRef(value="A")),
    ))
    problem = base.model_copy(update={
        "operator_templates": base.operator_templates + (transfer,),
        "intent_seeds": (IntentSeedSpec(id="transfer", operator_template_id="transfer", bindings=()),),
    })
    session = ReferenceKernel.start(problem)
    old_frame = session.frame()
    old = next(item for item in old_frame.intents if item.operator_template_id == "work")
    session.commit(old_frame.frame_token, ("transfer",))
    session.advance_next()
    assert _pairs(session) == {("hand1", "A")}
    before = session.snapshot().snapshot_hash
    with pytest.raises(SemanticError) as error:
        session.commit(old_frame.frame_token, (old.id,))
    assert error.value.code is DiagnosticCode.STALE_FRAME
    with pytest.raises(SemanticError) as error:
        session.commit(session.frame().frame_token, (old.id,))
    assert error.value.code is DiagnosticCode.INTENT_NOT_COMMITTABLE
    assert session.snapshot().snapshot_hash == before
    for _ in range(2):
        frame = session.frame()
        session.commit(frame.frame_token, (frame.intents[0].id,))
        session.advance_next()
    assert ReferenceValidator.validate_session(problem, session.snapshot()).ok


def test_candidate_cannot_use_its_own_effect_to_make_admission_guard_true() -> None:
    base = _problem()
    template = base.operator_templates[0]
    problem = base.model_copy(update={
        "initial_state": base.initial_state.model_copy(update={"leases": ()}),
        "operator_templates": (template.model_copy(update={"intervals": (
            template.intervals[0].model_copy(update={"start_effects": (AcquireLeaseTemplateEffect(
                resource=ParameterIdRef(parameter="holder"), owner=ParameterIdRef(parameter="item"),
            ),)}),
        )}),),
    })
    assert ReferenceKernel.start(problem).frame().intents == ()


@pytest.mark.parametrize("legacy", [False, True])
def test_independent_audit_rejects_forged_admission_even_with_consistent_hashes(
    legacy: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cluster_toolkit.constraint_ir import reference_session

    problem = _problem(legacy=legacy)
    session = ReferenceKernel.start(problem)
    frame = session.frame()
    session.commit(frame.frame_token, (frame.intents[0].id,))
    snapshot = session.snapshot()
    # Flip only the predicate. Timings/effects/leases remain physically valid,
    # and every stored digest/frame/instance identity is recomputed.
    if legacy:
        changed = problem.model_copy(update={"intent_seeds": tuple(seed.model_copy(update={
            "guards": (seed.guards[0].model_copy(update={"operator": "absent"}), *seed.guards[1:]),
        }) for seed in problem.intent_seeds)})
    else:
        rule = problem.dynamic_intents[0]
        changed = problem.model_copy(update={"dynamic_intents": (rule.model_copy(update={
            "guards": (rule.guards[0].model_copy(update={"operator": "absent"}), *rule.guards[1:]),
        }),)})
    forged = _retarget_first_commit(snapshot, changed)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("audit must not use Session/Kernel guard evaluation")

    monkeypatch.setattr(reference_session, "_plan_seed", forbidden)
    monkeypatch.setattr(reference_session, "_resolve_guard", forbidden)
    monkeypatch.setattr(ReferenceKernel, "execute_until", forbidden)
    assert ReferenceValidator.validate_session(problem, snapshot).ok
    report = ReferenceValidator.validate_session(changed, forged)
    assert not report.ok
    assert report.issues[0].code is DiagnosticCode.INTENT_NOT_COMMITTABLE
    with pytest.raises(SemanticError) as error:
        ReferenceKernel.restore(changed, forged)
    assert error.value.code is DiagnosticCode.INTENT_NOT_COMMITTABLE


@pytest.mark.parametrize(("field", "reference", "message"), [
    ("resource", LiteralIdRef(value="unknown"), "unknown resource"),
    ("resource", ParameterIdRef(parameter="unknown"), "unknown parameter"),
    ("resource", ParameterIdRef(parameter="item"), "must be a resource"),
    ("owner", ParameterIdRef(parameter="holder"), "must be an owner/id"),
    ("owner", ParameterIdRef(parameter="unknown"), "unknown parameter"),
])
def test_template_lease_guard_rejects_bad_references(
    field: str, reference: LiteralIdRef | ParameterIdRef, message: str,
) -> None:
    base = _problem()
    rule = base.dynamic_intents[0]
    bad = base.model_copy(update={"dynamic_intents": (rule.model_copy(update={
        "guards": (rule.guards[0].model_copy(update={field: reference}),),
    }),)})
    with pytest.raises(ValidationError, match=message):
        ConstraintIRV1.model_validate(bad.model_dump())


@pytest.mark.parametrize("changes", [
    {"resource_id": ""}, {"owner_id": ""}, {"owner_id": 1}, {"operator": "empty"}, {"view": "before"},
])
def test_lease_guard_is_closed_and_strict(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LeaseCondition.model_validate({"resource_id": "hand0", "owner_id": "A", **changes})


def test_legacy_lease_guard_rejects_unknown_resource() -> None:
    base = _problem(legacy=True)
    seed = base.intent_seeds[0]
    bad = base.model_copy(update={"intent_seeds": (seed.model_copy(update={
        "guards": (LeaseCondition(resource_id="unknown", owner_id="A"),),
    }),)})
    with pytest.raises(ValidationError, match="unknown resource"):
        ConstraintIRV1.model_validate(bad.model_dump())


def test_mixed_guard_order_and_plain_json_round_trip_are_deterministic() -> None:
    problem = _problem()
    expected = ReferenceKernel.start(problem).frame()
    rule = problem.dynamic_intents[0]
    for guards in permutations(rule.guards):
        changed = problem.model_copy(update={"dynamic_intents": (rule.model_copy(update={"guards": guards}),)})
        parsed = ConstraintIRV1.model_validate_json(changed.model_dump_json())
        assert parsed.problem_hash == problem.problem_hash
        assert ReferenceKernel.start(parsed).frame() == expected


def test_exchange_requires_incoming_wafer_in_declared_hand_but_keeps_pick_out() -> None:
    base = _composite_exchange_problem()
    problem = base.model_copy(update={"initial_state": base.initial_state.model_copy(update={
        "leases": tuple(item.model_copy(update={"owner_id": "other"})
                        if item.resource_id == "robot.arm0" else item
                        for item in base.initial_state.leases),
    })})
    session = ReferenceKernel.start(problem)
    assert [item.id for item in session.frame().intents] == ["pick.wafer1.pm1"]
    before = session.snapshot().snapshot_hash
    with pytest.raises(SemanticError) as error:
        session.commit(session.frame().frame_token, ("exchange.wafer1.wafer2.pm1",))
    assert error.value.code is DiagnosticCode.INTENT_NOT_COMMITTABLE
    assert session.snapshot().snapshot_hash == before


def test_future_lease_is_not_current_possession_for_admission() -> None:
    base = _problem()
    arriving = _activity("arrive", 1, "motion", end_effects=(AcquireLeaseTemplateEffect(
        resource=LiteralIdRef(value="hand1"), owner=LiteralIdRef(value="A"),
    ),))
    problem = base.model_copy(update={
        "initial_state": base.initial_state.model_copy(update={"leases": ()}),
        "operator_templates": base.operator_templates + (arriving,),
        "intent_seeds": (IntentSeedSpec(id="arrive", operator_template_id="arrive", bindings=(),
                                       earliest_start_offset=3),),
    })
    session = ReferenceKernel.start(problem)
    result = session.commit(session.frame().frame_token, ("arrive",))
    assert any(item.resource_id == "hand1" and item.start_tick == 4 for item in result.reservations)
    assert _pairs(session) == set()
    session.advance_to(4)
    assert _pairs(session) == {("hand1", "A")}
