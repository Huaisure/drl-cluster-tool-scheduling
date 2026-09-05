from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from .diagnostics import DiagnosticCode, SemanticError
from .reference_kernel import ExecutionResult, ReferenceKernel
from .reference_validator import ReferenceValidator
from .schema import (
    AcquireLeaseEffect,
    AcquireLeaseTemplateEffect,
    AutomaticRuleSpec,
    BindingAssignment,
    ChoiceScopeClaimSpec,
    CommitRecordSpec,
    CommittedIntentSpec,
    ConstraintIRV1,
    CreateObligationEffect,
    CreateObligationTemplateEffect,
    DynamicIntentSpec,
    EventSpec,
    IntentSeedSpec,
    IncrementStateEffect,
    IncrementStateTemplateEffect,
    IntervalSpec,
    KernelSnapshot,
    LeaseCondition,
    LeaseConditionTemplate,
    LiteralIdRef,
    OperatorTemplateSpec,
    ParameterIdRef,
    ReleaseLeaseEffect,
    ReleaseLeaseTemplateEffect,
    ResourceUseSpec,
    ScheduleV1,
    SessionSnapshot,
    SatisfyObligationEffect,
    SatisfyObligationTemplateEffect,
    SetStateEffect,
    SetCurrentTickTemplateEffect,
    SetStateTemplateEffect,
    StateCondition,
    StateConditionTemplate,
    canonical_digest,
    canonical_effect_digest,
)


@dataclass(frozen=True, slots=True)
class Reservation:
    intent_id: str
    resource_id: str
    amount: int
    start_tick: int
    end_tick: int | None


@dataclass(frozen=True, slots=True)
class StateValueDelta:
    cell_id: str
    before: bool | int | str
    after: bool | int | str


@dataclass(frozen=True, slots=True)
class LeaseDelta:
    resource_id: str
    owner_id: str
    before_amount: int
    after_amount: int


@dataclass(frozen=True, slots=True)
class IntentStateDelta:
    completion_tick: int
    state_values: tuple[StateValueDelta, ...]
    leases: tuple[LeaseDelta, ...]


@dataclass(frozen=True, slots=True)
class IntentCandidate:
    id: str
    candidate_key: str
    candidate_digest: str
    operator_template_id: str
    bindings: tuple[BindingAssignment, ...]
    choice_scope_claims: tuple[ChoiceScopeClaimSpec, ...]
    alternative_group_id: str | None
    earliest_start_tick: int
    latest_start_tick: int | None
    duration_ticks: int
    resource_footprint: tuple[Reservation, ...]
    involved_entity_ids: tuple[str, ...]
    state_delta: IntentStateDelta


@dataclass(frozen=True, slots=True)
class DecisionFrame:
    frame_token: str
    tick: int
    intents: tuple[IntentCandidate, ...]


@dataclass(frozen=True, slots=True)
class CommitResult:
    schedule: ScheduleV1
    execution: ExecutionResult
    reservations: tuple[Reservation, ...]
    commit_record: CommitRecordSpec | None = None


@dataclass(frozen=True, slots=True)
class CandidateGenerationContext:
    problem: ConstraintIRV1
    tick: int
    schedule: ScheduleV1
    current_snapshot: KernelSnapshot
    active_obligation_ids: frozenset[str]
    committed_intent_ids: frozenset[str]
    committed_alternative_group_ids: frozenset[str]
    active_choice_scope_keys: frozenset[str]
    commit_log: tuple[CommitRecordSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    candidate: IntentCandidate
    seed: IntentSeedSpec
    consume_source: bool
    addition: ScheduleV1


class CandidateGenerator(Protocol):
    def generate(
        self,
        context: CandidateGenerationContext,
    ) -> tuple[CandidatePlan, ...]: ...


class LegacyIntentSeedCandidateGenerator:
    """Compatibility Adapter from finite one-shot seeds to candidates."""

    def generate(
        self,
        context: CandidateGenerationContext,
    ) -> tuple[CandidatePlan, ...]:
        plans: list[CandidatePlan] = []
        for seed in context.problem.intent_seeds:
            if seed.id in context.committed_intent_ids:
                continue
            if (
                seed.alternative_group_id
                in context.committed_alternative_group_ids
            ):
                continue
            scopes = _choice_scope_claims(seed)
            plan = _plan_seed(context, seed, scopes, consume_source=True)
            if plan is not None:
                plans.append(plan)
        return tuple(plans)


class ExhaustiveReferenceCandidateGenerator:
    """Enumerate every row in finite typed Binding Domains."""

    def __init__(self, *, include_legacy: bool = True) -> None:
        self._include_legacy = include_legacy
        self._legacy_generator = LegacyIntentSeedCandidateGenerator()
        self._compiled_rows: dict[
            tuple[int, str],
            tuple[
                tuple[
                    tuple[BindingAssignment, ...],
                    tuple[ChoiceScopeClaimSpec, ...],
                    tuple[StateCondition, ...],
                    str,
                ],
                ...,
            ],
        ] = {}

    def _rows(
        self,
        problem: ConstraintIRV1,
        rule: DynamicIntentSpec,
    ) -> tuple[
        tuple[
            tuple[BindingAssignment, ...],
            tuple[ChoiceScopeClaimSpec, ...],
            tuple[StateCondition, ...],
            str,
        ],
        ...,
    ]:
        # ConstraintIRV1 is immutable for a session.  Its problem_hash property
        # canonicalizes every binding row, so using the object identity here
        # avoids recomputing the entire problem merely to look up this cache.
        key = (id(problem), rule.id)
        cached = self._compiled_rows.get(key)
        if cached is not None:
            return cached
        domain = next(
            item for item in problem.binding_domains
            if item.id == rule.binding_domain_id
        )
        parameter_names = tuple(parameter.name for parameter in domain.parameters)
        compiled = []
        for row in domain.rows:
            values = dict(zip(parameter_names, row.values))
            bindings = tuple(
                BindingAssignment(parameter=name, value=value)
                for name, value in sorted(values.items())
            )
            compiled.append((
                bindings,
                _render_choice_scopes(rule, values),
                tuple(_resolve_guard(condition, values) for condition in rule.guards),
                canonical_digest([
                    item.model_dump(mode="json") for item in bindings
                ]),
            ))
        result = tuple(compiled)
        self._compiled_rows[key] = result
        return result

    def generate(
        self,
        context: CandidateGenerationContext,
    ) -> tuple[CandidatePlan, ...]:
        plans = list(
            self._legacy_generator.generate(context)
            if self._include_legacy
            else ()
        )
        occurrences = Counter(
            (
                selection.operator_template_id,
                tuple(sorted(selection.bindings, key=lambda item: item.parameter)),
            )
            for record in context.commit_log
            for selection in record.selections
        )
        for rule in context.problem.dynamic_intents:
            for bindings, scopes, guards, binding_digest in self._rows(
                context.problem,
                rule,
            ):
                occurrence = occurrences[(rule.operator_template_id, bindings)]
                seed = IntentSeedSpec(
                    id=(
                        f"dynamic/{rule.id}/"
                        f"{binding_digest}"
                        f"/{occurrence}"
                    ),
                    operator_template_id=rule.operator_template_id,
                    bindings=bindings,
                    earliest_start_offset=rule.earliest_start_offset,
                    latest_start_offset=rule.latest_start_offset,
                    required_obligation_ids=rule.required_obligation_ids,
                    guards=guards,
                    choice_scope_claims=scopes,
                )
                plan = _plan_seed(
                    context,
                    seed,
                    scopes,
                    consume_source=False,
                )
                if plan is not None:
                    plans.append(plan)
        return tuple(plans)


class ReferenceSession:
    """Reference multi-epoch session for Intent expansion and reservations."""

    def __init__(
        self,
        problem: ConstraintIRV1,
        candidate_generator: CandidateGenerator | None = None,
    ) -> None:
        self.problem = problem
        self._problem_hash = problem.problem_hash
        self._candidate_generator = (
            candidate_generator or ExhaustiveReferenceCandidateGenerator()
        )
        self._revision = 0
        self._tick = 0
        self._committed_ids: set[str] = set()
        self._committed_alternative_groups: set[str] = set()
        self._commit_log: tuple[CommitRecordSpec, ...] = ()
        self._active_choice_scope_keys: set[str] = set()
        self._schedule = ScheduleV1()
        self._last_result: CommitResult | None = None
        self._cached_frame_token: str | None = None
        self._cached_plans: tuple[CandidatePlan, ...] | None = None

    @classmethod
    def from_snapshot(
        cls,
        problem: ConstraintIRV1,
        snapshot: SessionSnapshot,
        candidate_generator: CandidateGenerator | None = None,
    ) -> "ReferenceSession":
        report = ReferenceValidator.validate_session(problem, snapshot)
        if not report.ok:
            issue = report.issues[0]
            raise SemanticError(issue.code, issue.message)

        # Canonical wire arrays may be reordered. The independently validated
        # chain, not the array position, defines subsequent occurrence history.
        children = {record.previous_commit_id: record for record in snapshot.commit_log}
        records: list[CommitRecordSpec] = []
        previous = None
        while previous in children:
            record = children[previous]
            records.append(record)
            previous = record.commit_id
        execution = ReferenceKernel.execute_until(
            problem,
            snapshot.schedule,
            snapshot.tick,
        )
        if execution.final_snapshot.state_hash != snapshot.kernel_state_hash:
            raise SemanticError(
                DiagnosticCode.SNAPSHOT_STATE_MISMATCH,
                "snapshot state cannot be reconstructed from its Schedule",
            )
        session = cls(problem, candidate_generator)
        session._revision = snapshot.revision
        session._tick = snapshot.tick
        session._committed_ids = set(snapshot.committed_intent_ids)
        session._committed_alternative_groups = set(
            snapshot.committed_alternative_group_ids
        )
        session._commit_log = tuple(records)
        session._active_choice_scope_keys = set(snapshot.active_choice_scope_keys)
        session._schedule = snapshot.schedule
        session._last_result = CommitResult(
            schedule=snapshot.schedule,
            execution=execution,
            reservations=_derive_reservations(snapshot.schedule),
        )
        return session

    def frame(self) -> DecisionFrame:
        token = self._frame_token
        if self._cached_frame_token == token and self._cached_plans is not None:
            plans = self._cached_plans
        else:
            plans = self._candidate_plans()
            self._cached_frame_token = token
            self._cached_plans = plans
        return DecisionFrame(
            frame_token=token,
            tick=self._tick,
            intents=tuple(plan.candidate for plan in plans),
        )

    def commit(
        self,
        frame_token: str,
        intent_ids: tuple[str, ...],
    ) -> CommitResult:
        """Reserve the full bundle and settle this tick, without advancing time."""
        if frame_token != self._frame_token:
            raise SemanticError(
                DiagnosticCode.STALE_FRAME,
                "the decision frame is no longer current",
            )
        if not intent_ids or len(set(intent_ids)) != len(intent_ids):
            raise SemanticError(
                DiagnosticCode.INVALID_SCHEDULE,
                "commit requires a non-empty set of distinct intent ids",
            )
        plans = (
            self._cached_plans
            if self._cached_frame_token == frame_token and self._cached_plans is not None
            else self._candidate_plans()
        )
        plans_by_reference = {
            reference: plan
            for plan in plans
            for reference in (plan.candidate.id, plan.candidate.candidate_key)
        }
        rejected_ids = sorted(
            reference
            for reference in intent_ids
            if reference not in plans_by_reference
        )
        if rejected_ids:
            raise SemanticError(
                DiagnosticCode.INTENT_NOT_COMMITTABLE,
                f"intents are not committable in the current frame: {rejected_ids}",
            )
        selected_plans = tuple(plans_by_reference[item] for item in intent_ids)
        if len({plan.candidate.candidate_key for plan in selected_plans}) != len(
            selected_plans
        ):
            raise SemanticError(
                DiagnosticCode.INVALID_SCHEDULE,
                "commit selects the same candidate more than once",
            )
        selected = tuple(plan.seed for plan in selected_plans)

        _validate_alternative_selection(selected)
        _validate_choice_scope_selection(selected_plans)

        addition = (
            selected_plans[0].addition
            if len(selected_plans) == 1
            else _expand_intents(
                self.problem,
                selected,
                anchor_tick=self._tick,
                decision_round=_next_decision_round(self._schedule, self._tick),
            )
        )
        schedule = _merge_schedules(self._schedule, addition)
        horizon = _schedule_horizon(schedule, self._tick)
        current_snapshot = self._current_kernel_snapshot
        ReferenceKernel.preview_addition(
            self.problem,
            schedule,
            addition,
            current_snapshot,
            horizon,
            allow_open_obligations=True,
        )
        execution = ReferenceKernel.preview_addition(
            self.problem,
            schedule,
            addition,
            current_snapshot,
            self._tick,
        )
        commit_record = _make_commit_record(
            previous_commit_id=(
                None if not self._commit_log else self._commit_log[-1].commit_id
            ),
            frame_token=frame_token,
            tick=self._tick,
            plans=selected_plans,
            addition=addition,
        )
        result = CommitResult(
            schedule=schedule,
            execution=execution,
            reservations=_derive_reservations(schedule),
            commit_record=commit_record,
        )
        self._schedule = schedule
        self._committed_ids.update(
            plan.seed.id for plan in selected_plans if plan.consume_source
        )
        self._committed_alternative_groups.update(
            plan.seed.alternative_group_id
            for plan in selected_plans
            if plan.consume_source
            and plan.seed.alternative_group_id is not None
        )
        self._commit_log += (commit_record,)
        self._active_choice_scope_keys = _active_scope_keys_at_tick(
            self._commit_log,
            self._schedule,
            self._tick,
        )
        self._revision += 1
        self._last_result = result
        self._cached_frame_token = None
        self._cached_plans = None
        return result

    def snapshot(self, tick: int | None = None) -> SessionSnapshot:
        snapshot_tick = self._tick if tick is None else tick
        horizon = _schedule_horizon(self._schedule, self._tick)
        if snapshot_tick < 0 or snapshot_tick > horizon:
            raise SemanticError(
                DiagnosticCode.INVALID_TIME_VALUE,
                f"snapshot tick must be between 0 and {horizon}",
            )
        if self._commit_log and snapshot_tick < self._commit_log[-1].tick:
            raise SemanticError(
                DiagnosticCode.INVALID_TIME_VALUE,
                "snapshot must not precede the latest committed decision",
            )
        kernel_snapshot = (
            self._current_kernel_snapshot
            if snapshot_tick == self._tick
            else ReferenceKernel.execute_until(
                self.problem,
                self._schedule,
                snapshot_tick,
            ).final_snapshot
        )
        return SessionSnapshot(
            problem_hash=self._problem_hash,
            revision=self._revision,
            tick=snapshot_tick,
            committed_intent_ids=tuple(sorted(self._committed_ids)),
            committed_alternative_group_ids=tuple(
                sorted(self._committed_alternative_groups)
            ),
            schedule=self._schedule,
            schedule_hash=self._schedule.schedule_hash,
            kernel_snapshot=kernel_snapshot,
            kernel_state_hash=kernel_snapshot.state_hash,
            commit_log=self._commit_log,
            active_choice_scope_keys=tuple(
                sorted(
                    _active_scope_keys_at_tick(
                        self._commit_log,
                        self._schedule,
                        snapshot_tick,
                    )
                )
            ),
        )

    def fork(self) -> "ReferenceSession":
        """Cheap in-process clone for speculative planning from trusted state."""
        clone = object.__new__(type(self))
        clone.problem = self.problem
        clone._problem_hash = self._problem_hash
        clone._candidate_generator = self._candidate_generator
        clone._revision = self._revision
        clone._tick = self._tick
        clone._committed_ids = set(self._committed_ids)
        clone._committed_alternative_groups = set(
            self._committed_alternative_groups
        )
        clone._commit_log = self._commit_log
        clone._active_choice_scope_keys = set(self._active_choice_scope_keys)
        clone._schedule = self._schedule
        clone._last_result = self._last_result
        clone._cached_frame_token = self._cached_frame_token
        clone._cached_plans = self._cached_plans
        return clone

    def advance_to(self, tick: int) -> ExecutionResult:
        """Explicit replay/diagnostic fast-forward; policy loops use advance_next."""
        if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
            raise SemanticError(
                DiagnosticCode.INVALID_TIME_VALUE,
                "advance target must be a nonnegative integer tick",
            )
        if tick < self._tick:
            raise SemanticError(
                DiagnosticCode.INVALID_TIME_VALUE,
                "advance target must not precede the current tick",
            )
        execution = ReferenceKernel.preview_addition(
            self.problem,
            self._schedule,
            ScheduleV1(),
            self._current_kernel_snapshot,
            tick,
        )
        self._tick = tick
        self._active_choice_scope_keys = _active_scope_keys_at_tick(
            self._commit_log,
            self._schedule,
            self._tick,
        )
        self._last_result = CommitResult(
            schedule=self._schedule,
            execution=execution,
            reservations=_derive_reservations(self._schedule),
        )
        self._cached_frame_token = None
        self._cached_plans = None
        return execution

    def advance_next(self) -> ExecutionResult | None:
        """Advance to the next scheduled boundary; None does not imply terminal."""
        boundaries = {event.tick for event in self._schedule.events}
        boundaries.update(
            tick
            for interval in self._schedule.intervals
            for tick in (interval.start_tick, interval.end_tick)
        )
        boundaries.update(
            obligation.deadline_tick
            for obligation in self._current_kernel_snapshot.active_obligations
            if obligation.deadline_tick is not None
        )
        next_tick = min(
            (tick for tick in boundaries if tick > self._tick),
            default=None,
        )
        return None if next_tick is None else self.advance_to(next_tick)

    @property
    def schedule(self) -> ScheduleV1:
        return self._schedule

    @property
    def commit_log(self) -> tuple[CommitRecordSpec, ...]:
        return self._commit_log

    def _candidate_plans(self) -> tuple[CandidatePlan, ...]:
        plans = self._candidate_generator.generate(
            CandidateGenerationContext(
                problem=self.problem,
                tick=self._tick,
                schedule=self._schedule,
                current_snapshot=self._current_kernel_snapshot,
                active_obligation_ids=frozenset(self._active_obligation_ids),
                committed_intent_ids=frozenset(self._committed_ids),
                committed_alternative_group_ids=frozenset(
                    self._committed_alternative_groups
                ),
                active_choice_scope_keys=frozenset(
                    self._active_choice_scope_keys
                ),
                commit_log=self._commit_log,
            )
        )
        by_key: dict[str, str] = {}
        for plan in plans:
            previous_digest = by_key.get(plan.candidate.candidate_key)
            if (
                previous_digest is not None
                and previous_digest != plan.candidate.candidate_digest
            ):
                raise SemanticError(
                    DiagnosticCode.CANDIDATE_DIGEST_MISMATCH,
                    "CandidateGenerator returned one key with different content",
                )
            if previous_digest is not None:
                raise SemanticError(
                    DiagnosticCode.INVALID_SCHEDULE,
                    "CandidateGenerator returned a duplicate candidate",
                )
            by_key[plan.candidate.candidate_key] = plan.candidate.candidate_digest
        return tuple(sorted(plans, key=_candidate_plan_sort_key))

    @property
    def _frame_token(self) -> str:
        return canonical_digest(
            {
                "problem_hash": self._problem_hash,
                "revision": self._revision,
                "state_hash": self._current_kernel_snapshot.state_hash,
                "commitment_hash": self._commitment_hash,
            }
        )

    @property
    def _commitment_hash(self) -> str:
        return canonical_digest(
            {
                "commit_ids": [record.commit_id for record in self._commit_log],
                "active_choice_scope_keys": sorted(
                    self._active_choice_scope_keys
                ),
            }
        )

    @property
    def _current_kernel_snapshot(self) -> KernelSnapshot:
        if self._last_result is not None:
            return self._last_result.execution.final_snapshot
        return ReferenceKernel.execute_until(
            self.problem,
            self._schedule,
            self._tick,
        ).final_snapshot

    @property
    def _active_obligation_ids(self) -> set[str]:
        if self._last_result is None:
            return {item.id for item in self.problem.initial_state.obligations}
        return {
            item.id
            for item in self._last_result.execution.final_snapshot.active_obligations
        }

def _plan_seed(
    context: CandidateGenerationContext,
    seed: IntentSeedSpec,
    scopes: tuple[ChoiceScopeClaimSpec, ...],
    *,
    consume_source: bool,
) -> CandidatePlan | None:
    state_values = {
        assignment.cell_id: assignment.value
        for assignment in context.current_snapshot.state_values
    }
    if not set(seed.required_obligation_ids).issubset(
        context.active_obligation_ids
    ):
        return None
    leases = {(item.resource_id, item.owner_id) for item in context.current_snapshot.active_leases}
    for condition in seed.guards:
        if isinstance(condition, LeaseCondition):
            present = (condition.resource_id, condition.owner_id) in leases
            if present != (condition.operator == "present"):
                return None
        elif not _condition_holds(condition, state_values, context.tick):
            return None
    if any(
        claim.scope_key in context.active_choice_scope_keys
        for claim in scopes
    ):
        return None
    try:
        addition = _expand_intents(
            context.problem,
            (seed,),
            anchor_tick=context.tick,
            decision_round=_next_decision_round(context.schedule, context.tick),
        )
        schedule = _merge_schedules(context.schedule, addition)
        completion_tick = _schedule_horizon(
            addition,
            context.tick + seed.earliest_start_offset,
        )
        # Current-tick effects must be checked with hard deadlines enabled;
        # the longer preview below may defer only future open obligations.
        ReferenceKernel.preview_addition(
            context.problem, schedule, addition, context.current_snapshot,
            context.tick,
        )
        validated = ReferenceKernel.preview_addition(
            context.problem,
            schedule,
            addition,
            context.current_snapshot,
            _schedule_horizon(schedule, context.tick),
            allow_open_obligations=True,
        )
        predicted = (
            validated
            if validated.final_snapshot.tick == completion_tick
            else ReferenceKernel.preview_addition(
                context.problem,
                schedule,
                addition,
                context.current_snapshot,
                completion_tick,
                allow_open_obligations=True,
            )
        )
    except SemanticError:
        return None
    footprint = _derive_reservations(addition)
    state_delta = _derive_state_delta(
        context.current_snapshot,
        predicted.final_snapshot,
    )
    return CandidatePlan(
        candidate=_make_candidate(
            context.problem,
            seed,
            scopes,
            footprint,
            state_delta,
            context.tick,
            completion_tick,
        ),
        seed=seed,
        consume_source=consume_source,
        addition=addition,
    )


def _render_choice_scopes(
    rule: DynamicIntentSpec,
    values: dict[str, str],
) -> tuple[ChoiceScopeClaimSpec, ...]:
    return tuple(
        sorted(
            (
                ChoiceScopeClaimSpec(
                    scope_key=(
                        f"{scope.scope_prefix}/"
                        f"{canonical_digest([values[name] for name in scope.identity_parameters])}"
                    ),
                    release_boundary_id=scope.release_boundary_id,
                )
                for scope in rule.choice_scope_templates
            ),
            key=lambda claim: claim.scope_key,
        )
    )


def _resolve_guard(
    condition: StateConditionTemplate | LeaseConditionTemplate,
    values: dict[str, str],
) -> StateCondition | LeaseCondition:
    if isinstance(condition, LeaseConditionTemplate):
        return LeaseCondition(
            resource_id=_resolve_id(condition.resource, values),
            owner_id=_resolve_id(condition.owner, values),
            operator=condition.operator,
        )
    return StateCondition(
        cell_id=_resolve_id(condition.cell, values),
        operator=condition.operator,
        value=condition.value,
        view=condition.view,
    )


def _choice_scope_claims(
    seed: IntentSeedSpec,
) -> tuple[ChoiceScopeClaimSpec, ...]:
    claims = list(seed.choice_scope_claims)
    if not claims:
        claims.append(
            ChoiceScopeClaimSpec(scope_key=f"legacy-seed/{seed.id}")
        )
    if seed.alternative_group_id is not None:
        claims.append(
            ChoiceScopeClaimSpec(
                scope_key=f"legacy-alternative/{seed.alternative_group_id}"
            )
        )
    return tuple(sorted(claims, key=lambda claim: claim.scope_key))


def _candidate_plan_sort_key(plan: CandidatePlan) -> tuple[object, ...]:
    candidate = plan.candidate
    return (
        candidate.operator_template_id,
        tuple(sorted((item.parameter, item.value) for item in candidate.bindings)),
        tuple(
            (item.scope_key, item.release_boundary_id or "")
            for item in candidate.choice_scope_claims
        ),
        candidate.earliest_start_tick,
        (
            float("inf")
            if candidate.latest_start_tick is None
            else candidate.latest_start_tick
        ),
        candidate.candidate_key,
    )


def _make_candidate(
    problem: ConstraintIRV1,
    seed: IntentSeedSpec,
    scopes: tuple[ChoiceScopeClaimSpec, ...],
    footprint: tuple[Reservation, ...],
    state_delta: IntentStateDelta,
    tick: int,
    completion_tick: int,
) -> IntentCandidate:
    key_payload = {
        "semantic_version": problem.semantic_version,
        "source_intent_id": seed.id,
        "operator_template_id": seed.operator_template_id,
        "bindings": [item.model_dump(mode="json") for item in seed.bindings],
        "choice_scope_claims": [
            item.model_dump(mode="json") for item in scopes
        ],
        "temporal_variant": {
            "earliest_start_offset": seed.earliest_start_offset,
            "latest_start_offset": seed.latest_start_offset,
        },
    }
    candidate_key = canonical_digest(key_payload)
    candidate_payload = {
        **key_payload,
        "candidate_key": candidate_key,
        "tick": tick,
        "completion_tick": state_delta.completion_tick,
        "resource_footprint": [
            {
                "resource_id": item.resource_id,
                "amount": item.amount,
                "start_tick": item.start_tick,
                "end_tick": item.end_tick,
            }
            for item in footprint
        ],
        "state_values": [
            {
                "cell_id": item.cell_id,
                "before": item.before,
                "after": item.after,
            }
            for item in state_delta.state_values
        ],
        "leases": [
            {
                "resource_id": item.resource_id,
                "owner_id": item.owner_id,
                "before_amount": item.before_amount,
                "after_amount": item.after_amount,
            }
            for item in state_delta.leases
        ],
    }
    return IntentCandidate(
        id=seed.id,
        candidate_key=candidate_key,
        candidate_digest=canonical_digest(candidate_payload),
        operator_template_id=seed.operator_template_id,
        bindings=seed.bindings,
        choice_scope_claims=scopes,
        alternative_group_id=seed.alternative_group_id,
        earliest_start_tick=tick + seed.earliest_start_offset,
        latest_start_tick=(
            None
            if seed.latest_start_offset is None
            else tick + seed.latest_start_offset
        ),
        duration_ticks=(
            completion_tick - tick - seed.earliest_start_offset
        ),
        resource_footprint=footprint,
        involved_entity_ids=_derive_involved_entity_ids(
            seed,
            footprint,
            state_delta,
        ),
        state_delta=state_delta,
    )


def _validate_choice_scope_selection(
    plans: tuple[CandidatePlan, ...],
) -> None:
    owners: dict[str, str] = {}
    for plan in plans:
        for claim in plan.candidate.choice_scope_claims:
            previous = owners.get(claim.scope_key)
            if previous is not None:
                raise SemanticError(
                    DiagnosticCode.CHOICE_SCOPE_CONFLICT,
                    "commit selects candidates claiming the same choice scope",
                    details={
                        "scope_key": claim.scope_key,
                        "candidate_keys": [
                            previous,
                            plan.candidate.candidate_key,
                        ],
                    },
                )
            owners[claim.scope_key] = plan.candidate.candidate_key


def _make_commit_record(
    *,
    previous_commit_id: str | None,
    frame_token: str,
    tick: int,
    plans: tuple[CandidatePlan, ...],
    addition: ScheduleV1,
) -> CommitRecordSpec:
    selections = tuple(
        sorted(
            (
                CommittedIntentSpec(
                    source_intent_id=plan.seed.id,
                    operator_template_id=plan.seed.operator_template_id,
                    bindings=tuple(
                        sorted(
                            plan.seed.bindings,
                            key=lambda item: (item.parameter, item.value),
                        )
                    ),
                    earliest_start_offset=plan.seed.earliest_start_offset,
                    candidate_key=plan.candidate.candidate_key,
                    candidate_digest=plan.candidate.candidate_digest,
                    intent_instance_id=canonical_digest(
                        {
                            "frame_token": frame_token,
                            "candidate_key": plan.candidate.candidate_key,
                        }
                    ),
                    operator_instance_ids=tuple(
                        sorted(
                            {
                                item.operator_instance_id
                                for item in addition.intervals + addition.events
                                if item.origin_intent_id == plan.seed.id
                                and item.operator_instance_id is not None
                            }
                        )
                    ),
                    choice_scope_claims=plan.candidate.choice_scope_claims,
                )
                for plan in plans
            ),
            key=lambda item: item.candidate_key,
        )
    )
    payload = _commit_record_payload(
        previous_commit_id=previous_commit_id,
        frame_token=frame_token,
        tick=tick,
        selections=selections,
        expanded_schedule_digest=addition.schedule_hash,
    )
    return CommitRecordSpec(
        commit_id=canonical_digest(payload),
        **payload,
    )


def _commit_record_payload(
    *,
    previous_commit_id: str | None,
    frame_token: str,
    tick: int,
    selections: tuple[CommittedIntentSpec, ...],
    expanded_schedule_digest: str,
) -> dict[str, object]:
    return {
        "previous_commit_id": previous_commit_id,
        "frame_token": frame_token,
        "tick": tick,
        "selections": [
            item.model_dump(mode="json") for item in selections
        ],
        "expanded_schedule_digest": expanded_schedule_digest,
    }


def _active_scope_keys_at_tick(
    records: tuple[CommitRecordSpec, ...],
    schedule: ScheduleV1,
    tick: int,
) -> set[str]:
    active: set[str] = set()
    for record in records:
        if record.tick > tick:
            continue
        for selection in record.selections:
            for claim in selection.choice_scope_claims:
                if claim.release_boundary_id is None:
                    active.add(claim.scope_key)
                    continue
                releases = [
                    event.tick
                    for event in schedule.events
                    if event.operator_instance_id
                    in selection.operator_instance_ids
                    and event.origin_rule_id is None
                    and event.boundary_id == claim.release_boundary_id
                ]
                if not releases:
                    raise SemanticError(
                        DiagnosticCode.COMMIT_LOG_MISMATCH,
                        "choice scope release boundary is missing from Schedule",
                        details={"scope_key": claim.scope_key},
                    )
                if tick < min(releases):
                    active.add(claim.scope_key)
    return active


def _expand_intents(
    problem: ConstraintIRV1,
    seeds: tuple[IntentSeedSpec, ...],
    *,
    anchor_tick: int,
    decision_round: int = 0,
) -> ScheduleV1:
    templates = {template.id: template for template in problem.operator_templates}
    events: list[EventSpec] = []
    intervals: list[IntervalSpec] = []
    for seed in seeds:
        seed_anchor_tick = anchor_tick + seed.earliest_start_offset
        new_events, new_intervals = _instantiate_operator(
            templates[seed.operator_template_id],
            seed.bindings,
            anchor_tick=seed_anchor_tick,
            instance_id=f"intent.{seed.id}",
            origin_intent_id=seed.id,
        )
        events.extend(new_events)
        intervals.extend(new_intervals)

    rules = tuple(problem.automatic_rules)
    queue = list(events)
    emitted: set[tuple[str, str]] = set()
    while queue:
        trigger_event = queue.pop(0)
        for rule in rules:
            if not _matches(rule, trigger_event):
                continue
            emission_key = (rule.id, trigger_event.id)
            if emission_key in emitted:
                continue
            emitted.add(emission_key)
            if len(emitted) > 1000:
                raise SemanticError(
                    DiagnosticCode.INVALID_SCHEDULE,
                    "automatic expansion exceeded the reference safety limit",
                )
            trigger_bindings = _binding_map(trigger_event.bindings)
            forwarded = tuple(
                BindingAssignment(
                    parameter=item.target_parameter,
                    value=trigger_bindings[item.source_parameter],
                )
                for item in rule.binding_forwards
            )
            new_events, new_intervals = _instantiate_operator(
                templates[rule.emit_operator_template_id],
                forwarded,
                anchor_tick=trigger_event.tick,
                instance_id=f"auto.{rule.id}.{trigger_event.id}",
                origin_intent_id=trigger_event.origin_intent_id,
                origin_rule_id=rule.id,
                trigger_event_id=trigger_event.id,
            )
            events.extend(new_events)
            intervals.extend(new_intervals)
            queue.extend(new_events)

    # Future precommitted boundaries settle in round 0. Only effects introduced
    # at this decision tick follow the stable state on which the choice was made.
    return ScheduleV1(
        events=tuple(
            event.model_copy(update={"decision_round": decision_round})
            if event.tick == anchor_tick else event
            for event in events
        ),
        intervals=tuple(intervals),
    )


def _next_decision_round(schedule: ScheduleV1, tick: int) -> int:
    return 1 + max(
        (event.decision_round for event in schedule.events if event.tick == tick),
        default=0,
    )


def _instantiate_operator(
    template: OperatorTemplateSpec,
    bindings: tuple[BindingAssignment, ...],
    *,
    anchor_tick: int,
    instance_id: str,
    origin_intent_id: str | None,
    origin_rule_id: str | None = None,
    trigger_event_id: str | None = None,
) -> tuple[list[EventSpec], list[IntervalSpec]]:
    values = _binding_map(bindings)
    events: list[EventSpec] = []
    intervals: list[IntervalSpec] = []
    for interval in template.intervals:
        start_tick = anchor_tick + interval.start_offset
        end_tick = start_tick + interval.duration
        start_event_id = f"{instance_id}.{interval.id}.start"
        end_event_id = f"{instance_id}.{interval.id}.end"
        common_event = {
            "operator_instance_id": instance_id,
            "operator_template_id": template.id,
            "origin_intent_id": origin_intent_id,
            "origin_rule_id": origin_rule_id,
            "trigger_event_id": trigger_event_id,
            "bindings": bindings,
        }
        start_effects = tuple(
            _resolve_effect(effect, values, start_tick)
            for effect in interval.start_effects
        )
        end_effects = tuple(
            _resolve_effect(effect, values, end_tick)
            for effect in interval.end_effects
        )
        events.append(
            EventSpec(
                id=start_event_id,
                tick=start_tick,
                effects=start_effects,
                effect_digest=canonical_effect_digest(start_effects),
                audit_kind=f"{interval.audit_kind}.start",
                boundary_id=f"{interval.id}.start",
                **common_event,
            )
        )
        events.append(
            EventSpec(
                id=end_event_id,
                tick=end_tick,
                effects=end_effects,
                effect_digest=canonical_effect_digest(end_effects),
                audit_kind=f"{interval.audit_kind}.end",
                boundary_id=f"{interval.id}.end",
                **common_event,
            )
        )
        intervals.append(
            IntervalSpec(
                id=f"{instance_id}.{interval.id}",
                start_tick=start_tick,
                end_tick=end_tick,
                resource_uses=tuple(
                    ResourceUseSpec(
                        resource_id=_resolve_id(use.resource, values),
                        amount=use.amount,
                    )
                    for use in interval.resource_uses
                ),
                audit_kind=interval.audit_kind,
                operator_instance_id=instance_id,
                operator_template_id=template.id,
                template_interval_id=interval.id,
                origin_intent_id=origin_intent_id,
                origin_rule_id=origin_rule_id,
                trigger_event_id=trigger_event_id,
                bindings=bindings,
            )
        )
    return events, intervals


def _resolve_effect(
    effect: (
        AcquireLeaseTemplateEffect
        | ReleaseLeaseTemplateEffect
        | SetStateTemplateEffect
        | IncrementStateTemplateEffect
        | SetCurrentTickTemplateEffect
        | CreateObligationTemplateEffect
        | SatisfyObligationTemplateEffect
    ),
    bindings: dict[str, str],
    boundary_tick: int,
) -> (
    AcquireLeaseEffect
    | ReleaseLeaseEffect
    | SetStateEffect
    | IncrementStateEffect
    | CreateObligationEffect
    | SatisfyObligationEffect
):
    if isinstance(effect, AcquireLeaseTemplateEffect):
        return AcquireLeaseEffect(
            resource_id=_resolve_id(effect.resource, bindings),
            owner_id=_resolve_id(effect.owner, bindings),
            amount=effect.amount,
        )
    if isinstance(effect, ReleaseLeaseTemplateEffect):
        return ReleaseLeaseEffect(
            resource_id=_resolve_id(effect.resource, bindings),
            owner_id=_resolve_id(effect.owner, bindings),
        )
    if isinstance(effect, SetStateTemplateEffect):
        return SetStateEffect(
            cell_id=_resolve_id(effect.cell, bindings),
            value=effect.value,
        )
    if isinstance(effect, IncrementStateTemplateEffect):
        return IncrementStateEffect(
            cell_id=_resolve_id(effect.cell, bindings),
            delta=effect.delta,
        )
    if isinstance(effect, SetCurrentTickTemplateEffect):
        return SetStateEffect(
            cell_id=_resolve_id(effect.cell, bindings),
            value=boundary_tick,
        )
    if isinstance(effect, CreateObligationTemplateEffect):
        condition = None
        if effect.condition is not None:
            condition = StateCondition(
                cell_id=_resolve_id(effect.condition.cell, bindings),
                operator=effect.condition.operator,
                value=effect.condition.value,
                view=effect.condition.view,
            )
        return CreateObligationEffect(
            obligation_id=effect.obligation_id,
            deadline_tick=(
                None
                if effect.deadline_offset is None
                else boundary_tick + effect.deadline_offset
            ),
            condition=condition,
            coalesce_key=effect.coalesce_key,
            priority=effect.priority,
        )
    return SatisfyObligationEffect(obligation_id=effect.obligation_id)


def _resolve_id(
    reference: LiteralIdRef | ParameterIdRef,
    bindings: dict[str, str],
) -> str:
    if isinstance(reference, LiteralIdRef):
        return reference.value
    return bindings[reference.parameter]


def _binding_map(bindings: tuple[BindingAssignment, ...]) -> dict[str, str]:
    return {binding.parameter: binding.value for binding in bindings}


def _matches(rule: AutomaticRuleSpec, event: EventSpec) -> bool:
    return (
        event.operator_template_id == rule.trigger_operator_template_id
        and event.boundary_id == rule.trigger_boundary_id
    )


def _validate_alternative_selection(
    seeds: tuple[IntentSeedSpec, ...],
) -> None:
    groups: dict[str, list[str]] = {}
    for seed in seeds:
        if seed.alternative_group_id is None:
            continue
        groups.setdefault(seed.alternative_group_id, []).append(seed.id)
    conflicts = {
        group_id: intent_ids
        for group_id, intent_ids in groups.items()
        if len(intent_ids) > 1
    }
    if conflicts:
        raise SemanticError(
            DiagnosticCode.ALTERNATIVE_GROUP_CONFLICT,
            f"commit selects multiple intents from alternative groups: {conflicts}",
        )


def _merge_schedules(left: ScheduleV1, right: ScheduleV1) -> ScheduleV1:
    return ScheduleV1(
        events=left.events + right.events,
        intervals=left.intervals + right.intervals,
    )


def _schedule_horizon(schedule: ScheduleV1, fallback: int) -> int:
    return max(
        [fallback]
        + [event.tick for event in schedule.events]
        + [interval.end_tick for interval in schedule.intervals]
    )


def _condition_holds(
    condition: StateCondition,
    values: dict[str, object],
    tick: int,
) -> bool:
    actual = values.get(condition.cell_id)
    if condition.operator == "equal":
        return actual == condition.value and type(actual) is type(condition.value)
    if condition.operator == "not_equal":
        return actual != condition.value or type(actual) is not type(condition.value)
    if not isinstance(actual, int) or isinstance(actual, bool):
        return False
    assert isinstance(condition.value, int) and not isinstance(condition.value, bool)
    if condition.operator == "greater_equal":
        return actual >= condition.value
    return tick - actual >= condition.value


def _derive_reservations(schedule: ScheduleV1) -> tuple[Reservation, ...]:
    raw: list[Reservation] = []
    for interval in schedule.intervals:
        if interval.origin_intent_id is None:
            continue
        for use in interval.resource_uses:
            raw.append(
                Reservation(
                    intent_id=interval.origin_intent_id,
                    resource_id=use.resource_id,
                    amount=use.amount,
                    start_tick=interval.start_tick,
                    end_tick=interval.end_tick,
                )
            )

    ordered_events = sorted(
        schedule.events,
        key=lambda event: (event.tick, event.decision_round, event.id),
    )
    for event in ordered_events:
        for effect in event.effects:
            if not isinstance(effect, AcquireLeaseEffect):
                continue
            release_tick: int | None = None
            for later in ordered_events:
                if (later.tick, later.decision_round) <= (
                    event.tick, event.decision_round,
                ):
                    continue
                if any(
                    isinstance(candidate, ReleaseLeaseEffect)
                    and candidate.resource_id == effect.resource_id
                    and candidate.owner_id == effect.owner_id
                    for candidate in later.effects
                ):
                    release_tick = later.tick
                    break
            if event.origin_intent_id is not None and (
                release_tick is None or release_tick > event.tick
            ):
                raw.append(
                    Reservation(
                        intent_id=event.origin_intent_id,
                        resource_id=effect.resource_id,
                        amount=effect.amount,
                        start_tick=event.tick,
                        end_tick=release_tick,
                    )
                )
    return _coalesce_reservations(raw)


def _derive_state_delta(
    before: KernelSnapshot,
    after: KernelSnapshot,
) -> IntentStateDelta:
    before_values = {
        assignment.cell_id: assignment.value
        for assignment in before.state_values
    }
    after_values = {
        assignment.cell_id: assignment.value
        for assignment in after.state_values
    }
    value_changes = tuple(
        StateValueDelta(
            cell_id=cell_id,
            before=before_values[cell_id],
            after=after_values[cell_id],
        )
        for cell_id in sorted(before_values.keys() | after_values.keys())
        if before_values.get(cell_id) != after_values.get(cell_id)
        or type(before_values.get(cell_id)) is not type(after_values.get(cell_id))
    )

    def lease_amounts(snapshot: KernelSnapshot) -> dict[tuple[str, str], int]:
        return {
            (lease.resource_id, lease.owner_id): lease.amount
            for lease in snapshot.active_leases
        }

    before_leases = lease_amounts(before)
    after_leases = lease_amounts(after)
    lease_changes = tuple(
        LeaseDelta(
            resource_id=resource_id,
            owner_id=owner_id,
            before_amount=before_leases.get((resource_id, owner_id), 0),
            after_amount=after_leases.get((resource_id, owner_id), 0),
        )
        for resource_id, owner_id in sorted(
            before_leases.keys() | after_leases.keys()
        )
        if before_leases.get((resource_id, owner_id), 0)
        != after_leases.get((resource_id, owner_id), 0)
    )
    return IntentStateDelta(
        completion_tick=after.tick,
        state_values=value_changes,
        leases=lease_changes,
    )


def _derive_involved_entity_ids(
    seed: IntentSeedSpec,
    reservations: tuple[Reservation, ...],
    state_delta: IntentStateDelta,
) -> tuple[str, ...]:
    entity_ids = {binding.value for binding in seed.bindings}
    entity_ids.update(item.resource_id for item in reservations)
    entity_ids.update(item.cell_id for item in state_delta.state_values)
    for item in state_delta.leases:
        entity_ids.add(item.resource_id)
        entity_ids.add(item.owner_id)
    return tuple(sorted(entity_ids))


def _coalesce_reservations(items: list[Reservation]) -> tuple[Reservation, ...]:
    ordered = sorted(
        items,
        key=lambda item: (
            item.intent_id,
            item.resource_id,
            item.amount,
            item.start_tick,
            float("inf") if item.end_tick is None else item.end_tick,
        ),
    )
    merged: list[Reservation] = []
    for item in ordered:
        if merged:
            previous = merged[-1]
            same_key = (
                previous.intent_id == item.intent_id
                and previous.resource_id == item.resource_id
                and previous.amount == item.amount
            )
            overlaps = previous.end_tick is None or item.start_tick <= previous.end_tick
            if same_key and overlaps:
                if previous.end_tick is None or item.end_tick is None:
                    merged_end = None
                else:
                    merged_end = max(previous.end_tick, item.end_tick)
                merged[-1] = Reservation(
                    intent_id=previous.intent_id,
                    resource_id=previous.resource_id,
                    amount=previous.amount,
                    start_tick=previous.start_tick,
                    end_tick=merged_end,
                )
                continue
        merged.append(item)
    return tuple(merged)
