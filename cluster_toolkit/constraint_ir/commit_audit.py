"""Independent audit of the declared reference Session protocol.

This module deliberately does not import the Session, its generator/expander,
or the Kernel. Only immutable schema and the independent Validator replay are
shared. It proves selected-candidate legality, not candidate-set completeness
or global schedulability.
"""

from __future__ import annotations

from collections import defaultdict, deque

from pydantic import ValidationError

from .diagnostics import DiagnosticCode
from .reference_validator import (
    ValidationIssue, ValidationReport, _IndependentReplay, _terminal_state_matches,
)
from .schema import (
    AcquireLeaseEffect,
    BindingAssignment,
    ChoiceScopeClaimSpec,
    CommitRecordSpec,
    CommittedIntentSpec,
    ConstraintIRV1,
    EventSpec,
    IntentSeedSpec,
    IntervalSpec,
    KernelSnapshot,
    LeaseCondition,
    LeaseConditionTemplate,
    LiteralIdRef,
    ParameterIdRef,
    ReleaseLeaseEffect,
    ResourceUseSpec,
    ScheduleV1,
    SessionSnapshot,
    StateCondition,
    StateConditionTemplate,
    canonical_digest,
    canonical_effect_digest,
)


class _Rejected(Exception):
    def __init__(self, report: ValidationReport) -> None:
        self.report = report


def _reject(code: DiagnosticCode, message: str, tick: int | None = None) -> None:
    raise _Rejected(ValidationReport(issues=(ValidationIssue(code, message, tick),)))


def validate_session(
    problem: ConstraintIRV1,
    snapshot: SessionSnapshot | str,
    *,
    require_terminal: bool,
) -> ValidationReport:
    try:
        # Revalidate model_copy inputs too; a caller may have bypassed Pydantic.
        problem = ConstraintIRV1.model_validate(problem.model_dump(mode="python"))
        snapshot = (
            SessionSnapshot.model_validate_json(snapshot)
            if isinstance(snapshot, str)
            else SessionSnapshot.model_validate(snapshot.model_dump(mode="python"))
        )
        return _CommitAudit(problem, snapshot).run(require_terminal=require_terminal)
    except _Rejected as rejected:
        return rejected.report
    except ValidationError as error:
        return ValidationReport(issues=(ValidationIssue(
            DiagnosticCode.INVALID_SCHEDULE, f"invalid audit input: {error}",
        ),))


def _ordered_records(records: tuple[CommitRecordSpec, ...]) -> tuple[CommitRecordSpec, ...]:
    """Wire arrays may be sorted; previous_commit_id carries causal order."""
    children: dict[str | None, CommitRecordSpec] = {}
    ids: set[str] = set()
    for record in records:
        if record.commit_id in ids or record.previous_commit_id in children:
            _reject(DiagnosticCode.COMMIT_LOG_MISMATCH, "duplicate or branched CommitLog")
        if canonical_digest(record.model_dump(exclude={"commit_id"})) != record.commit_id:
            _reject(DiagnosticCode.COMMIT_LOG_MISMATCH, "commit digest mismatch", record.tick)
        children[record.previous_commit_id] = record
        ids.add(record.commit_id)
    ordered: list[CommitRecordSpec] = []
    previous = None
    while previous in children and len(ordered) < len(records):
        record = children[previous]
        ordered.append(record)
        previous = record.commit_id
    if len(ordered) != len(records):
        _reject(DiagnosticCode.COMMIT_LOG_MISMATCH, "disconnected or cyclic CommitLog")
    return tuple(ordered)


def _combine(*schedules: ScheduleV1) -> ScheduleV1:
    return ScheduleV1(
        events=tuple(event for schedule in schedules for event in schedule.events),
        intervals=tuple(item for schedule in schedules for item in schedule.intervals),
    )


def _horizon(schedule: ScheduleV1, fallback: int) -> int:
    return max(
        [fallback] + [event.tick for event in schedule.events]
        + [interval.end_tick for interval in schedule.intervals]
    )


class _CommitAudit:
    def __init__(self, problem: ConstraintIRV1, snapshot: SessionSnapshot) -> None:
        self.problem = problem
        self.snapshot = snapshot
        self.templates = {item.id: item for item in problem.operator_templates}
        self.seeds = {item.id: item for item in problem.intent_seeds}
        self.domains = {item.id: item for item in problem.binding_domains}
        self.prefix = ScheduleV1()
        self.history: list[CommitRecordSpec] = []
        self.consumed: set[str] = set()
        self.groups: set[str] = set()
        self.scopes: dict[str, int | None] = {}
        self.resolver = _IndependentReplay(problem, ScheduleV1(), require_terminal=False)

    def run(self, *, require_terminal: bool) -> ValidationReport:
        snapshot = self.snapshot
        if snapshot.problem_hash != self.problem.problem_hash:
            _reject(DiagnosticCode.SNAPSHOT_PROBLEM_MISMATCH, "snapshot belongs to another Problem")
        if snapshot.schedule_hash != snapshot.schedule.schedule_hash:
            _reject(DiagnosticCode.SNAPSHOT_STATE_MISMATCH, "stored Schedule hash mismatch")
        if snapshot.kernel_state_hash != snapshot.kernel_snapshot.state_hash:
            _reject(DiagnosticCode.SNAPSHOT_STATE_MISMATCH, "stored KernelSnapshot hash mismatch")
        records = _ordered_records(snapshot.commit_log)
        if snapshot.revision != len(records):
            _reject(DiagnosticCode.COMMIT_LOG_MISMATCH, "revision is not the number of commits")

        previous_tick = 0
        for record in records:
            if not previous_tick <= record.tick <= snapshot.tick:
                _reject(DiagnosticCode.COMMIT_LOG_MISMATCH, "invalid decision tick order", record.tick)
            before = self._state(self.prefix, record.tick)
            self._expire_scopes(record.tick)
            expected_frame = canonical_digest({
                "problem_hash": self.problem.problem_hash,
                "revision": len(self.history),
                "state_hash": before.state_hash,
                "commitment_hash": canonical_digest({
                    "commit_ids": [item.commit_id for item in self.history],
                    "active_choice_scope_keys": sorted(self.scopes),
                }),
            })
            if record.frame_token != expected_frame:
                _reject(
                    DiagnosticCode.STALE_FRAME,
                    "commit was not made from its replayed state", record.tick,
                )
            self._audit_commit(record, before)
            self.history.append(record)
            previous_tick = record.tick

        if self.prefix.canonical_dict() != snapshot.schedule.canonical_dict():
            _reject(
                DiagnosticCode.OPERATOR_CONFORMANCE_MISMATCH,
                "Schedule is not exactly the expansion of the audited commits",
            )
        after = self._state(self.prefix, snapshot.tick)
        self._expire_scopes(snapshot.tick)
        if (
            after.canonical_dict() != snapshot.kernel_snapshot.canonical_dict()
            or sorted(self.consumed) != sorted(snapshot.committed_intent_ids)
            or sorted(self.groups) != sorted(snapshot.committed_alternative_group_ids)
            or sorted(self.scopes) != sorted(snapshot.active_choice_scope_keys)
        ):
            _reject(DiagnosticCode.SNAPSHOT_STATE_MISMATCH, "final state or commitments disagree with replay")
        if require_terminal and (
            not _terminal_state_matches(self.problem, after)
            or _horizon(self.prefix, snapshot.tick) > snapshot.tick
        ):
            _reject(
                DiagnosticCode.NON_TERMINAL_STATE,
                "snapshot does not satisfy terminal state or leaves open work or future boundaries", snapshot.tick,
            )
        return ValidationReport(issues=(), final_snapshot=after)

    def _state(
        self, schedule: ScheduleV1, tick: int, *, forecast: bool = False,
    ) -> KernelSnapshot:
        report = _IndependentReplay(self.problem, schedule, require_terminal=False).validate(
            until_tick=tick,
            allow_open_obligations=forecast,
            check_conformance=False,
        )
        if not report.ok:
            raise _Rejected(report)
        assert report.final_snapshot is not None
        return report.final_snapshot

    def _expire_scopes(self, tick: int) -> None:
        self.scopes = {
            key: release for key, release in self.scopes.items()
            if release is None or release > tick
        }

    def _audit_commit(self, record: CommitRecordSpec, before: KernelSnapshot) -> None:
        decision_round = max(
            (event.decision_round for event in self.prefix.events if event.tick == record.tick),
            default=0,
        ) + 1
        additions: list[ScheduleV1] = []
        pending_scopes: dict[str, int | None] = {}
        pending_consumed: set[str] = set()
        pending_groups: set[str] = set()
        for selection in record.selections:
            seed, legacy = self._definition(selection)
            if legacy and (seed.id in self.consumed or seed.id in pending_consumed):
                _reject(
                    DiagnosticCode.INTENT_NOT_COMMITTABLE,
                    "one-shot source was already consumed", record.tick,
                )
            if seed.alternative_group_id is not None:
                if seed.alternative_group_id in self.groups | pending_groups:
                    _reject(
                        DiagnosticCode.ALTERNATIVE_GROUP_CONFLICT,
                        "alternative group selected twice", record.tick,
                    )
                pending_groups.add(seed.alternative_group_id)
            if not self._guards_hold(seed, before):
                _reject(
                    DiagnosticCode.INTENT_NOT_COMMITTABLE,
                    "selected candidate's prerequisites were false", record.tick,
                )
            for scope in seed.choice_scope_claims:
                if scope.scope_key in self.scopes or scope.scope_key in pending_scopes:
                    _reject(
                        DiagnosticCode.CHOICE_SCOPE_CONFLICT,
                        "choice scope is already claimed", record.tick,
                    )
                # Mark before expansion so duplicate claims cannot disappear at a
                # start boundary inside the very same batch.
                pending_scopes[scope.scope_key] = None

            addition = self._expand(seed, record.tick, decision_round)
            combined = _combine(self.prefix, addition)
            self._state(combined, record.tick)
            self._state(combined, _horizon(combined, record.tick), forecast=True)
            predicted = self._state(combined, _horizon(addition, record.tick), forecast=True)
            key, digest = self._candidate_identity(seed, addition, before, predicted)
            if selection.candidate_key != key or selection.candidate_digest != digest:
                _reject(
                    DiagnosticCode.CANDIDATE_DIGEST_MISMATCH,
                    "candidate key or projected effects were changed", record.tick,
                )
            instance_ids = sorted({item.operator_instance_id for item in addition.intervals})
            if (
                selection.intent_instance_id != canonical_digest({
                    "frame_token": record.frame_token, "candidate_key": key,
                })
                or sorted(selection.operator_instance_ids) != instance_ids
            ):
                _reject(
                    DiagnosticCode.COMMIT_LOG_MISMATCH,
                    "incorrect occurrence or OperatorInstance identity", record.tick,
                )
            for scope in seed.choice_scope_claims:
                if scope.release_boundary_id is not None:
                    release = next(event.tick for event in addition.events if (
                        event.origin_rule_id is None
                        and event.boundary_id == scope.release_boundary_id
                    ))
                    pending_scopes[scope.scope_key] = release
            if legacy:
                pending_consumed.add(seed.id)
            additions.append(addition)

        addition = _combine(*additions)
        automatic_instances = {
            item.operator_instance_id for item in addition.intervals if item.origin_rule_id
        }
        if len(automatic_instances) > 1000:
            _reject(
                DiagnosticCode.INVALID_SCHEDULE,
                "automatic expansion exceeds reference limit", record.tick,
            )
        if addition.schedule_hash != record.expanded_schedule_digest:
            _reject(
                DiagnosticCode.COMMIT_LOG_MISMATCH,
                "commit expansion digest does not match its definitions", record.tick,
            )
        combined = _combine(self.prefix, addition)
        self._state(combined, record.tick)
        self._state(combined, _horizon(combined, record.tick), forecast=True)
        self.prefix = combined
        self.scopes.update(pending_scopes)
        self._expire_scopes(record.tick)
        self.consumed.update(pending_consumed)
        self.groups.update(pending_groups)

    def _definition(self, selection: CommittedIntentSpec) -> tuple[IntentSeedSpec, bool]:
        bindings = tuple(sorted(selection.bindings, key=lambda item: item.parameter))
        values = {item.parameter: item.value for item in bindings}
        if len(values) != len(bindings):
            _reject(DiagnosticCode.INTENT_NOT_COMMITTABLE, "duplicate binding parameters")
        matches: list[tuple[IntentSeedSpec, bool]] = []
        seed = self.seeds.get(selection.source_intent_id)
        if seed is not None:
            scopes = list(seed.choice_scope_claims) or [
                ChoiceScopeClaimSpec(scope_key=f"legacy-seed/{seed.id}")
            ]
            if seed.alternative_group_id is not None:
                scopes.append(ChoiceScopeClaimSpec(
                    scope_key=f"legacy-alternative/{seed.alternative_group_id}",
                ))
            matches.append((seed.model_copy(update={"choice_scope_claims": tuple(scopes)}), True))
        for rule in self.problem.dynamic_intents:
            if rule.operator_template_id != selection.operator_template_id:
                continue
            domain = self.domains[rule.binding_domain_id]
            if set(values) != {item.name for item in domain.parameters}:
                continue
            row = tuple(values[item.name] for item in domain.parameters)
            if not any(item.values == row for item in domain.rows):
                continue
            occurrence = sum(
                item.operator_template_id == rule.operator_template_id
                and tuple(sorted(item.bindings, key=lambda binding: binding.parameter)) == bindings
                for record in self.history for item in record.selections
            )
            source = f"dynamic/{rule.id}/{canonical_digest(bindings)}/{occurrence}"
            if source != selection.source_intent_id:
                continue
            scopes = tuple(ChoiceScopeClaimSpec(
                scope_key=(
                    f"{scope.scope_prefix}/"
                    f"{canonical_digest([values[name] for name in scope.identity_parameters])}"
                ),
                release_boundary_id=scope.release_boundary_id,
            ) for scope in rule.choice_scope_templates)
            matches.append((IntentSeedSpec(
                id=source, operator_template_id=rule.operator_template_id,
                bindings=bindings, choice_scope_claims=scopes,
                earliest_start_offset=rule.earliest_start_offset,
                latest_start_offset=rule.latest_start_offset,
                required_obligation_ids=rule.required_obligation_ids,
                guards=tuple(self._resolve_guard(condition, values) for condition in rule.guards),
            ), False))
        if len(matches) != 1:
            _reject(DiagnosticCode.INTENT_NOT_COMMITTABLE, "source is not one declared candidate occurrence")
        seed, legacy = matches[0]
        if (
            seed.operator_template_id != selection.operator_template_id
            or dict((item.parameter, item.value) for item in seed.bindings) != values
            or seed.earliest_start_offset != selection.earliest_start_offset
        ):
            _reject(
                DiagnosticCode.INTENT_NOT_COMMITTABLE,
                "candidate binding or temporal variant was changed",
            )
        if canonical_digest(seed.choice_scope_claims) != canonical_digest(selection.choice_scope_claims):
            _reject(DiagnosticCode.CHOICE_SCOPE_CONFLICT, "candidate scope declaration was changed")
        return seed, legacy

    @staticmethod
    def _guards_hold(seed: IntentSeedSpec, before: KernelSnapshot) -> bool:
        if not set(seed.required_obligation_ids) <= {item.id for item in before.active_obligations}:
            return False
        values = {item.cell_id: item.value for item in before.state_values}
        for condition in seed.guards:
            if isinstance(condition, LeaseCondition):
                present = any(
                    item.resource_id == condition.resource_id and item.owner_id == condition.owner_id
                    for item in before.active_leases
                )
                if present != (condition.operator == "present"):
                    return False
                continue
            actual = values[condition.cell_id]
            same = type(actual) is type(condition.value) and actual == condition.value
            if condition.operator == "equal":
                holds = same
            elif condition.operator == "not_equal":
                holds = not same
            elif type(actual) is not int:
                holds = False
            elif condition.operator == "greater_equal":
                holds = actual >= condition.value
            else:
                holds = before.tick - actual >= condition.value
            if not holds:
                return False
        return True

    def _resolve_guard(
        self, condition: StateConditionTemplate | LeaseConditionTemplate,
        values: dict[str, str],
    ) -> StateCondition | LeaseCondition:
        if isinstance(condition, LeaseConditionTemplate):
            return LeaseCondition(
                resource_id=self._resolve(condition.resource, values),
                owner_id=self._resolve(condition.owner, values),
                operator=condition.operator,
            )
        return StateCondition(
            cell_id=self._resolve(condition.cell, values), operator=condition.operator,
            value=condition.value, view=condition.view,
        )

    @staticmethod
    def _resolve(reference: LiteralIdRef | ParameterIdRef, values: dict[str, str]) -> str:
        return reference.value if isinstance(reference, LiteralIdRef) else values[reference.parameter]

    def _expand(self, seed: IntentSeedSpec, tick: int, decision_round: int) -> ScheduleV1:
        events: list[EventSpec] = []
        intervals: list[IntervalSpec] = []
        pending = deque([(
            self.templates[seed.operator_template_id], seed.bindings,
            tick + seed.earliest_start_offset, f"intent.{seed.id}", None, None,
        )])
        emissions = 0
        while pending:
            template, bindings, anchor, instance_id, rule_id, trigger_id = pending.popleft()
            values = {item.parameter: item.value for item in bindings}
            for step in template.intervals:
                start, end = anchor + step.start_offset, anchor + step.start_offset + step.duration
                common = {
                    "operator_instance_id": instance_id,
                    "operator_template_id": template.id,
                    "origin_intent_id": seed.id, "origin_rule_id": rule_id,
                    "trigger_event_id": trigger_id, "bindings": bindings,
                }
                intervals.append(IntervalSpec(
                    id=f"{instance_id}.{step.id}", start_tick=start, end_tick=end,
                    template_interval_id=step.id, audit_kind=step.audit_kind,
                    resource_uses=tuple(ResourceUseSpec(
                        resource_id=self._resolve(use.resource, values), amount=use.amount,
                    ) for use in step.resource_uses),
                    **common,
                ))
                for boundary, at_tick, templates in (
                    ("start", start, step.start_effects), ("end", end, step.end_effects),
                ):
                    effects = tuple(self.resolver._resolve_template_effect(effect, values, at_tick)
                                    for effect in templates)
                    event = EventSpec(
                        id=f"{instance_id}.{step.id}.{boundary}", tick=at_tick,
                        decision_round=decision_round if at_tick == tick else 0,
                        boundary_id=f"{step.id}.{boundary}",
                        audit_kind=f"{step.audit_kind}.{boundary}",
                        effects=effects, effect_digest=canonical_effect_digest(effects),
                        **common,
                    )
                    events.append(event)
                    for rule in self.problem.automatic_rules:
                        if (rule.trigger_operator_template_id, rule.trigger_boundary_id) != (
                            template.id, event.boundary_id,
                        ):
                            continue
                        emissions += 1
                        if emissions > 1000:
                            _reject(
                                DiagnosticCode.INVALID_SCHEDULE,
                                "automatic expansion exceeds reference limit", tick,
                            )
                        forwarded = tuple(BindingAssignment(
                            parameter=item.target_parameter, value=values[item.source_parameter],
                        ) for item in rule.binding_forwards)
                        pending.append((
                            self.templates[rule.emit_operator_template_id], forwarded,
                            at_tick, f"auto.{rule.id}.{event.id}", rule.id, event.id,
                        ))
        return ScheduleV1(events=tuple(events), intervals=tuple(intervals))

    def _candidate_identity(
        self, seed: IntentSeedSpec, addition: ScheduleV1,
        before: KernelSnapshot, predicted: KernelSnapshot,
    ) -> tuple[str, str]:
        key_data = {
            "semantic_version": self.problem.semantic_version,
            "source_intent_id": seed.id,
            "operator_template_id": seed.operator_template_id,
            "bindings": seed.bindings, "choice_scope_claims": seed.choice_scope_claims,
            "temporal_variant": {
                "earliest_start_offset": seed.earliest_start_offset,
                "latest_start_offset": seed.latest_start_offset,
            },
        }
        key = canonical_digest(key_data)
        old_values = {item.cell_id: item.value for item in before.state_values}
        state_delta = [{"cell_id": item.cell_id, "before": old_values[item.cell_id], "after": item.value}
                       for item in predicted.state_values
                       if type(item.value) is not type(old_values[item.cell_id])
                       or item.value != old_values[item.cell_id]]
        old_leases = {(item.resource_id, item.owner_id): item.amount for item in before.active_leases}
        new_leases = {(item.resource_id, item.owner_id): item.amount for item in predicted.active_leases}
        lease_delta = [{
            "resource_id": resource, "owner_id": owner,
            "before_amount": old_leases.get((resource, owner), 0),
            "after_amount": new_leases.get((resource, owner), 0),
        } for resource, owner in old_leases.keys() | new_leases.keys()
            if old_leases.get((resource, owner), 0) != new_leases.get((resource, owner), 0)]
        return key, canonical_digest({
            **key_data, "candidate_key": key, "tick": before.tick,
            "completion_tick": predicted.tick,
            "resource_footprint": _footprint(addition),
            "state_values": state_delta, "leases": lease_delta,
        })


def _footprint(schedule: ScheduleV1) -> list[dict[str, object]]:
    ranges: dict[tuple[str, int], list[tuple[int, int | None]]] = defaultdict(list)
    for interval in schedule.intervals:
        for use in interval.resource_uses:
            ranges[(use.resource_id, use.amount)].append((interval.start_tick, interval.end_tick))
    releases: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for event in schedule.events:
        for effect in event.effects:
            if isinstance(effect, ReleaseLeaseEffect):
                releases[(effect.resource_id, effect.owner_id)].append((event.tick, event.decision_round))
    for event in schedule.events:
        for effect in event.effects:
            if not isinstance(effect, AcquireLeaseEffect):
                continue
            following = [position for position in releases[(effect.resource_id, effect.owner_id)]
                         if position > (event.tick, event.decision_round)]
            end = min(following)[0] if following else None
            if end is None or end > event.tick:
                ranges[(effect.resource_id, effect.amount)].append((event.tick, end))
    footprint: list[dict[str, object]] = []
    for (resource, amount), spans in ranges.items():
        merged: list[tuple[int, int | None]] = []
        for start, end in sorted(spans, key=lambda item: (item[0], item[1] is None, item[1] or 0)):
            if merged and (merged[-1][1] is None or start <= merged[-1][1]):
                old_start, old_end = merged.pop()
                merged.append((old_start, None if old_end is None or end is None else max(old_end, end)))
            else:
                merged.append((start, end))
        footprint.extend({
            "resource_id": resource, "amount": amount, "start_tick": start, "end_tick": end,
        } for start, end in merged)
    return footprint
