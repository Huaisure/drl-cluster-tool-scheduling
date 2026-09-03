from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from .diagnostics import DiagnosticCode
from .schema import (
    AcquireLeaseEffect,
    AcquireLeaseTemplateEffect,
    ActiveObligation,
    AutomaticRuleSpec,
    ConstraintIRV1,
    CreateObligationEffect,
    CreateObligationTemplateEffect,
    EventSpec,
    IncrementStateEffect,
    IncrementStateTemplateEffect,
    IntervalSpec,
    IntervalTemplateSpec,
    KernelSnapshot,
    LeaseSpec,
    LiteralIdRef,
    OperatorTemplateSpec,
    ParameterIdRef,
    ReleaseLeaseEffect,
    ReleaseLeaseTemplateEffect,
    SatisfyObligationEffect,
    SatisfyObligationTemplateEffect,
    ScheduleV1,
    SessionSnapshot,
    SetStateEffect,
    SetCurrentTickTemplateEffect,
    SetStateTemplateEffect,
    StateCondition,
    StateAssignment,
    canonical_effect_digest,
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: DiagnosticCode
    message: str
    tick: int | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    final_snapshot: KernelSnapshot | None = None

    @property
    def ok(self) -> bool:
        return not self.issues


class ReferenceValidator:
    """Independent schedule and declared-session audit; never calls the Kernel."""

    @classmethod
    def validate(
        cls,
        problem: ConstraintIRV1,
        schedule: ScheduleV1,
        *,
        require_terminal: bool = False,
    ) -> ValidationReport:
        replay = _IndependentReplay(
            problem,
            schedule,
            require_terminal=require_terminal,
        )
        return replay.validate()

    @classmethod
    def validate_session(
        cls,
        problem: ConstraintIRV1,
        snapshot: SessionSnapshot | str,
        *,
        require_terminal: bool = False,
    ) -> ValidationReport:
        """Audit declared candidate choices and independently reconstruct their trace."""
        from .commit_audit import validate_session

        return validate_session(problem, snapshot, require_terminal=require_terminal)


def _terminal_state_matches(problem: ConstraintIRV1, snapshot: KernelSnapshot) -> bool:
    if snapshot.active_obligations or snapshot.active_interval_ids:
        return False
    goal = problem.terminal_state
    if goal is None:
        return not snapshot.active_leases
    values = {item.cell_id: item.value for item in snapshot.state_values}
    return (
        all(type(values[item.cell_id]) is type(item.value) and values[item.cell_id] == item.value
            for item in goal.state_values)
        and {(item.resource_id, item.owner_id, item.amount) for item in snapshot.active_leases}
        == {(item.resource_id, item.owner_id, item.amount) for item in goal.leases}
    )


class _IndependentReplay:
    def __init__(
        self,
        problem: ConstraintIRV1,
        schedule: ScheduleV1,
        *,
        require_terminal: bool,
    ) -> None:
        self.problem = problem
        self.schedule = schedule
        self.resources = {resource.id: resource.capacity for resource in problem.resources}
        self.cells = {cell.id: cell for cell in problem.state_cells}
        self.values = {
            assignment.cell_id: assignment.value
            for assignment in problem.initial_state.state_values
        }
        self.leases = {
            (lease.resource_id, lease.owner_id): lease.amount
            for lease in problem.initial_state.leases
        }
        self.obligations = {
            obligation.id: obligation.deadline_tick
            for obligation in problem.initial_state.obligations
        }
        self.issues: list[ValidationIssue] = []
        self.require_terminal = require_terminal
        self.allow_open_obligations = False
        self.start_rounds = {
            (event.operator_instance_id, event.boundary_id): event.decision_round
            for event in schedule.events
        }

    def validate(
        self,
        *,
        until_tick: int | None = None,
        allow_open_obligations: bool = False,
        check_conformance: bool = True,
    ) -> ValidationReport:
        # The audit path reconstructs bundles itself; it only needs this
        # independent state replay, including partial and hypothetical horizons.
        self.allow_open_obligations = allow_open_obligations
        if not self._references_are_valid():
            return ValidationReport(issues=tuple(self.issues))
        if not self._effect_digests_are_valid():
            return ValidationReport(issues=tuple(self.issues))
        if check_conformance and not self._alternative_groups_are_valid():
            return ValidationReport(issues=tuple(self.issues))
        if check_conformance and not self._automatic_rules_are_satisfied():
            return ValidationReport(issues=tuple(self.issues))

        events_by_tick: dict[int, dict[int, list[object]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for event in self.schedule.events:
            events_by_tick[event.tick][event.decision_round].extend(event.effects)

        fixed_ticks = set(events_by_tick)
        for interval in self.schedule.intervals:
            fixed_ticks.add(interval.start_tick)
            fixed_ticks.add(interval.end_tick)
        if until_tick is not None:
            fixed_ticks = {tick for tick in fixed_ticks if tick <= until_tick}
            fixed_ticks.add(until_tick)
        if not fixed_ticks and all(
            deadline is None for deadline in self.obligations.values()
        ):
            fixed_ticks.add(0)

        final_snapshot: KernelSnapshot | None = None
        processed: set[int] = set()
        while True:
            active_deadlines = {
                deadline
                for deadline in self.obligations.values()
                if not allow_open_obligations
                and deadline is not None
                and (until_tick is None or deadline <= until_tick)
            }
            remaining = sorted(
                tick
                for tick in fixed_ticks | active_deadlines
                if tick not in processed
            )
            if not remaining:
                break
            tick = remaining[0]
            processed.add(tick)
            for decision_round, effects in sorted(events_by_tick.get(tick, {}).items()):
                previous_values = dict(self.values)
                if not self._replay_state_effects(tick, effects):
                    break
                if not self._replay_leases(tick, effects):
                    break
                if not self._replay_obligations(tick, effects, previous_values):
                    break
                if not self._capacity_is_valid(tick, decision_round):
                    break
                if not self._deadlines_are_valid(tick):
                    break
            if self.issues:
                break
            if not self._capacity_is_valid(tick):
                break
            if not self._deadlines_are_valid(tick):
                break
            final_snapshot = self._snapshot(tick)

        if check_conformance and not self.issues and not self._selectable_operators_are_conformant():
            return ValidationReport(issues=tuple(self.issues))
        if self.require_terminal and not self.issues:
            if final_snapshot is None or not _terminal_state_matches(self.problem, final_snapshot):
                self._issue(
                    DiagnosticCode.NON_TERMINAL_STATE,
                    "Schedule does not satisfy terminal state or leaves open work",
                )
        return ValidationReport(
            issues=tuple(self.issues),
            final_snapshot=final_snapshot if not self.issues else None,
        )

    def _references_are_valid(self) -> bool:
        for interval in self.schedule.intervals:
            for use in interval.resource_uses:
                if use.resource_id not in self.resources:
                    self._issue(
                        DiagnosticCode.UNKNOWN_REFERENCE,
                        f"interval {interval.id} references unknown resource {use.resource_id}",
                    )
        for event in self.schedule.events:
            for effect in event.effects:
                if isinstance(effect, (SetStateEffect, IncrementStateEffect)):
                    if effect.cell_id not in self.cells:
                        self._issue(
                            DiagnosticCode.UNKNOWN_REFERENCE,
                            f"event {event.id} references unknown cell {effect.cell_id}",
                            event.tick,
                        )
                elif isinstance(effect, (AcquireLeaseEffect, ReleaseLeaseEffect)):
                    if effect.resource_id not in self.resources:
                        self._issue(
                            DiagnosticCode.UNKNOWN_REFERENCE,
                            f"event {event.id} references unknown resource {effect.resource_id}",
                            event.tick,
                        )
                elif isinstance(effect, CreateObligationEffect):
                    if (
                        effect.condition is not None
                        and effect.condition.cell_id not in self.cells
                    ):
                        self._issue(
                            DiagnosticCode.UNKNOWN_REFERENCE,
                            f"event {event.id} condition references unknown cell",
                            event.tick,
                        )
        return not self.issues

    def _effect_digests_are_valid(self) -> bool:
        for event in self.schedule.events:
            if event.operator_template_id is None:
                continue
            expected = canonical_effect_digest(event.effects)
            if event.effect_digest != expected:
                self._issue(
                    DiagnosticCode.EFFECT_DIGEST_MISMATCH,
                    f"event {event.id} effect digest does not match its effects",
                    event.tick,
                )
                return False
        return True

    def _alternative_groups_are_valid(self) -> bool:
        seeds = {seed.id: seed for seed in self.problem.intent_seeds}
        selected_intent_ids = {
            item.origin_intent_id
            for item in self.schedule.intervals + self.schedule.events
            if item.origin_rule_id is None and item.origin_intent_id in seeds
        }
        selected_by_group: dict[str, list[str]] = defaultdict(list)
        for intent_id in sorted(selected_intent_ids):
            group_id = seeds[intent_id].alternative_group_id
            if group_id is not None:
                selected_by_group[group_id].append(intent_id)
        conflicts = {
            group_id: intent_ids
            for group_id, intent_ids in selected_by_group.items()
            if len(intent_ids) > 1
        }
        if conflicts:
            self._issue(
                DiagnosticCode.ALTERNATIVE_GROUP_CONFLICT,
                f"schedule contains multiple alternatives: {conflicts}",
            )
            return False
        return True

    def _automatic_rules_are_satisfied(self) -> bool:
        templates = {
            template.id: template for template in self.problem.operator_templates
        }
        for rule in self.problem.automatic_rules:
            emitted_template = templates[rule.emit_operator_template_id]
            triggers = [
                event
                for event in self.schedule.events
                if event.operator_template_id == rule.trigger_operator_template_id
                and event.boundary_id == rule.trigger_boundary_id
            ]
            for trigger in triggers:
                if not self._check_one_automatic_emission(
                    rule,
                    emitted_template,
                    trigger,
                ):
                    return False
        return True

    def _selectable_operators_are_conformant(self) -> bool:
        templates = {
            template.id: template for template in self.problem.operator_templates
        }
        seeds = {seed.id: seed for seed in self.problem.intent_seeds}
        selectable_records = [
            item
            for item in self.schedule.intervals + self.schedule.events
            if item.origin_rule_id is None
            and (
                item.operator_template_id is not None
                or item.origin_intent_id is not None
            )
        ]
        for item in selectable_records:
            if item.origin_intent_id not in seeds:
                self._issue(
                    DiagnosticCode.OPERATOR_CONFORMANCE_MISMATCH,
                    f"selectable record {item.id} has no known origin intent",
                )
                return False

        for seed in self.problem.intent_seeds:
            intervals = [
                interval
                for interval in self.schedule.intervals
                if interval.origin_rule_id is None
                and interval.origin_intent_id == seed.id
            ]
            events = [
                event
                for event in self.schedule.events
                if event.origin_rule_id is None
                and event.origin_intent_id == seed.id
            ]
            if not intervals and not events:
                continue
            template = templates[seed.operator_template_id]
            if not self._selectable_instance_matches(
                seed.id,
                template,
                {binding.parameter: binding.value for binding in seed.bindings},
                intervals,
                events,
            ):
                return False
        return True

    def _selectable_instance_matches(
        self,
        intent_id: str,
        template: OperatorTemplateSpec,
        expected_bindings: dict[str, str],
        intervals: list[IntervalSpec],
        events: list[EventSpec],
    ) -> bool:
        expected_instance_id = f"intent.{intent_id}"
        if len(intervals) != len(template.intervals):
            return self._selectable_mismatch(
                intent_id,
                "has an incomplete interval bundle",
            )
        if len(events) != 2 * len(template.intervals):
            return self._selectable_mismatch(
                intent_id,
                "has an incomplete boundary bundle",
            )

        actual_by_template_id = {
            interval.template_interval_id: interval for interval in intervals
        }
        if len(actual_by_template_id) != len(intervals):
            return self._selectable_mismatch(
                intent_id,
                "contains duplicate template intervals",
            )
        first_template = template.intervals[0]
        first_actual = actual_by_template_id.get(first_template.id)
        if first_actual is None:
            return self._selectable_mismatch(
                intent_id,
                f"is missing interval {first_template.id}",
            )
        anchor_tick = first_actual.start_tick - first_template.start_offset
        if anchor_tick < 0:
            return self._selectable_mismatch(
                intent_id,
                "has an invalid anchor tick",
            )

        for interval_template in template.intervals:
            actual = actual_by_template_id.get(interval_template.id)
            if actual is None:
                return self._selectable_mismatch(
                    intent_id,
                    f"is missing interval {interval_template.id}",
                )
            expected_start = anchor_tick + interval_template.start_offset
            expected_end = expected_start + interval_template.duration
            expected_uses = sorted(
                (
                    self._resolve_template_id(use.resource, expected_bindings),
                    use.amount,
                )
                for use in interval_template.resource_uses
            )
            actual_uses = sorted(
                (use.resource_id, use.amount) for use in actual.resource_uses
            )
            actual_bindings = {
                binding.parameter: binding.value for binding in actual.bindings
            }
            if (
                actual.operator_instance_id != expected_instance_id
                or actual.operator_template_id != template.id
                or actual.template_interval_id != interval_template.id
                or actual.origin_intent_id != intent_id
                or actual.trigger_event_id is not None
                or actual_bindings != expected_bindings
                or actual.start_tick != expected_start
                or actual.end_tick != expected_end
                or actual.audit_kind != interval_template.audit_kind
                or actual_uses != expected_uses
            ):
                return self._selectable_mismatch(
                    intent_id,
                    f"interval {interval_template.id} does not match its template",
                )
            if not self._selectable_boundaries_match(
                intent_id,
                expected_instance_id,
                template,
                interval_template,
                expected_bindings,
                expected_start,
                expected_end,
                events,
            ):
                return False
        return True

    def _selectable_boundaries_match(
        self,
        intent_id: str,
        instance_id: str,
        template: OperatorTemplateSpec,
        interval_template: IntervalTemplateSpec,
        expected_bindings: dict[str, str],
        start_tick: int,
        end_tick: int,
        events: list[EventSpec],
    ) -> bool:
        expected_by_boundary = {
            f"{interval_template.id}.start": (
                start_tick,
                f"{interval_template.audit_kind}.start",
                tuple(
                    self._resolve_template_effect(
                        effect,
                        expected_bindings,
                        start_tick,
                    )
                    for effect in interval_template.start_effects
                ),
            ),
            f"{interval_template.id}.end": (
                end_tick,
                f"{interval_template.audit_kind}.end",
                tuple(
                    self._resolve_template_effect(
                        effect,
                        expected_bindings,
                        end_tick,
                    )
                    for effect in interval_template.end_effects
                ),
            ),
        }
        matching = [
            event
            for event in events
            if event.boundary_id in expected_by_boundary
        ]
        if len(matching) != 2:
            return self._selectable_mismatch(
                intent_id,
                f"interval {interval_template.id} has incomplete boundaries",
            )
        for event in matching:
            tick, audit_kind, effects = expected_by_boundary[event.boundary_id]
            actual_bindings = {
                binding.parameter: binding.value for binding in event.bindings
            }
            if (
                event.operator_instance_id != instance_id
                or event.operator_template_id != template.id
                or event.origin_intent_id != intent_id
                or event.trigger_event_id is not None
                or event.tick != tick
                or event.audit_kind != audit_kind
                or actual_bindings != expected_bindings
                or self._canonical_effects(event.effects)
                != self._canonical_effects(effects)
            ):
                return self._selectable_mismatch(
                    intent_id,
                    f"boundary {event.boundary_id} does not match its template",
                )
        return True

    def _selectable_mismatch(self, intent_id: str, reason: str) -> bool:
        self._issue(
            DiagnosticCode.OPERATOR_CONFORMANCE_MISMATCH,
            f"selectable intent {intent_id} {reason}",
        )
        return False

    def _check_one_automatic_emission(
        self,
        rule: AutomaticRuleSpec,
        template: OperatorTemplateSpec,
        trigger: EventSpec,
    ) -> bool:
        trigger_bindings = {
            binding.parameter: binding.value for binding in trigger.bindings
        }
        expected_bindings = {
            item.target_parameter: trigger_bindings[item.source_parameter]
            for item in rule.binding_forwards
        }
        emitted_intervals = [
            interval
            for interval in self.schedule.intervals
            if interval.origin_rule_id == rule.id
            and interval.trigger_event_id == trigger.id
            and interval.operator_template_id == template.id
        ]
        if len(emitted_intervals) != len(template.intervals):
            self._issue(
                DiagnosticCode.MISSING_AUTOMATIC_EVENT,
                f"trigger {trigger.id} must emit operator {template.id}",
                trigger.tick,
            )
            return False
        by_template_id = {
            interval.template_interval_id: interval for interval in emitted_intervals
        }
        for interval_template in template.intervals:
            actual = by_template_id.get(interval_template.id)
            if actual is None:
                self._issue(
                    DiagnosticCode.MISSING_AUTOMATIC_EVENT,
                    f"automatic operator {template.id} is missing interval {interval_template.id}",
                    trigger.tick,
                )
                return False
            actual_bindings = {
                binding.parameter: binding.value for binding in actual.bindings
            }
            expected_start = trigger.tick + interval_template.start_offset
            expected_end = expected_start + interval_template.duration
            expected_uses = sorted(
                (
                    self._resolve_template_id(use.resource, expected_bindings),
                    use.amount,
                )
                for use in interval_template.resource_uses
            )
            actual_uses = sorted(
                (use.resource_id, use.amount) for use in actual.resource_uses
            )
            if (
                actual_bindings != expected_bindings
                or actual.origin_intent_id != trigger.origin_intent_id
                or actual.start_tick != expected_start
                or actual.end_tick != expected_end
                or actual_uses != expected_uses
            ):
                self._issue(
                    DiagnosticCode.MISSING_AUTOMATIC_EVENT,
                    f"automatic interval {actual.id} does not match template {template.id}",
                    trigger.tick,
                )
                return False
            if not self._automatic_boundaries_exist(
                rule,
                trigger.id,
                template.id,
                interval_template.id,
                expected_start,
                expected_end,
                expected_bindings,
                trigger.origin_intent_id,
                tuple(
                    self._resolve_template_effect(
                        effect,
                        expected_bindings,
                        expected_start,
                    )
                    for effect in interval_template.start_effects
                ),
                tuple(
                    self._resolve_template_effect(
                        effect,
                        expected_bindings,
                        expected_end,
                    )
                    for effect in interval_template.end_effects
                ),
            ):
                return False
        return True

    def _automatic_boundaries_exist(
        self,
        rule: AutomaticRuleSpec,
        trigger_event_id: str,
        template_id: str,
        interval_id: str,
        start_tick: int,
        end_tick: int,
        expected_bindings: dict[str, str],
        origin_intent_id: str | None,
        expected_start_effects: tuple[object, ...],
        expected_end_effects: tuple[object, ...],
    ) -> bool:
        expected = sorted(
            [
                (f"{interval_id}.start", start_tick),
                (f"{interval_id}.end", end_tick),
            ]
        )
        matching_events = [
            event
            for event in self.schedule.events
            if event.origin_rule_id == rule.id
            and event.trigger_event_id == trigger_event_id
            and event.operator_template_id == template_id
            and event.boundary_id in {item[0] for item in expected}
        ]
        actual = sorted((event.boundary_id, event.tick) for event in matching_events)
        if actual != expected:
            self._issue(
                DiagnosticCode.MISSING_AUTOMATIC_EVENT,
                f"automatic interval {interval_id} is missing auditable boundaries",
                start_tick,
            )
            return False
        for event in matching_events:
            bindings = {
                binding.parameter: binding.value for binding in event.bindings
            }
            if bindings != expected_bindings or event.origin_intent_id != origin_intent_id:
                self._issue(
                    DiagnosticCode.MISSING_AUTOMATIC_EVENT,
                    f"automatic boundary {event.id} has incorrect bindings or origin",
                    event.tick,
                )
                return False
            expected_effects = (
                expected_start_effects
                if event.boundary_id == f"{interval_id}.start"
                else expected_end_effects
            )
            if self._canonical_effects(event.effects) != self._canonical_effects(
                expected_effects
            ):
                self._issue(
                    DiagnosticCode.MISSING_AUTOMATIC_EVENT,
                    f"automatic boundary {event.id} has incorrect effects",
                    event.tick,
                )
                return False
        return True

    @staticmethod
    def _resolve_template_id(
        reference: LiteralIdRef | ParameterIdRef,
        bindings: dict[str, str],
    ) -> str:
        if isinstance(reference, LiteralIdRef):
            return reference.value
        return bindings[reference.parameter]

    def _resolve_template_effect(
        self,
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
    ) -> object:
        if isinstance(effect, AcquireLeaseTemplateEffect):
            return AcquireLeaseEffect(
                resource_id=self._resolve_template_id(effect.resource, bindings),
                owner_id=self._resolve_template_id(effect.owner, bindings),
                amount=effect.amount,
            )
        if isinstance(effect, ReleaseLeaseTemplateEffect):
            return ReleaseLeaseEffect(
                resource_id=self._resolve_template_id(effect.resource, bindings),
                owner_id=self._resolve_template_id(effect.owner, bindings),
            )
        if isinstance(effect, SetStateTemplateEffect):
            return SetStateEffect(
                cell_id=self._resolve_template_id(effect.cell, bindings),
                value=effect.value,
            )
        if isinstance(effect, IncrementStateTemplateEffect):
            return IncrementStateEffect(
                cell_id=self._resolve_template_id(effect.cell, bindings),
                delta=effect.delta,
            )
        if isinstance(effect, SetCurrentTickTemplateEffect):
            return SetStateEffect(
                cell_id=self._resolve_template_id(effect.cell, bindings),
                value=boundary_tick,
            )
        if isinstance(effect, CreateObligationTemplateEffect):
            condition = None
            if effect.condition is not None:
                condition = StateCondition(
                    cell_id=self._resolve_template_id(
                        effect.condition.cell,
                        bindings,
                    ),
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

    @staticmethod
    def _canonical_effects(effects: tuple[object, ...]) -> tuple[str, ...]:
        return tuple(
            sorted(
                json.dumps(effect.model_dump(mode="json"), sort_keys=True)
                for effect in effects
            )
        )

    def _replay_state_effects(self, tick: int, effects: list[object]) -> bool:
        set_values: dict[str, list[object]] = defaultdict(list)
        increment_values: dict[str, list[int]] = defaultdict(list)
        for effect in effects:
            if isinstance(effect, SetStateEffect):
                if not self.cells[effect.cell_id].accepts(effect.value):
                    self._issue(
                        DiagnosticCode.TYPE_MISMATCH,
                        f"tick {tick} writes an out-of-domain value to {effect.cell_id}",
                        tick,
                    )
                    return False
                set_values[effect.cell_id].append(effect.value)
            elif isinstance(effect, IncrementStateEffect):
                if self.cells[effect.cell_id].value_type != "int":
                    self._issue(
                        DiagnosticCode.TYPE_MISMATCH,
                        f"tick {tick} cannot increment non-integer cell {effect.cell_id}",
                        tick,
                    )
                    return False
                increment_values[effect.cell_id].append(effect.delta)

        for cell_id in sorted(set(set_values) | set(increment_values)):
            if cell_id in set_values and cell_id in increment_values:
                self._issue(
                    DiagnosticCode.CONFLICTING_EFFECTS,
                    f"tick {tick} mixes set and increment for {cell_id}",
                    tick,
                )
                return False
            if cell_id in set_values:
                values = set(set_values[cell_id])
                if len(values) != 1:
                    self._issue(
                        DiagnosticCode.CONFLICTING_EFFECTS,
                        f"tick {tick} writes different values to {cell_id}",
                        tick,
                    )
                    return False
                next_value = next(iter(values))
            else:
                current = self.values.get(cell_id)
                if not isinstance(current, int) or isinstance(current, bool):
                    self._issue(
                        DiagnosticCode.TYPE_MISMATCH,
                        f"tick {tick} cannot increment non-integer cell {cell_id}",
                        tick,
                    )
                    return False
                next_value = current + sum(increment_values[cell_id])
            if not self.cells[cell_id].accepts(next_value):
                self._issue(
                    DiagnosticCode.TYPE_MISMATCH,
                    f"tick {tick} writes an out-of-domain value to {cell_id}",
                    tick,
                )
                return False
            self.values[cell_id] = next_value
        return True

    def _replay_leases(self, tick: int, effects: list[object]) -> bool:
        releases: list[tuple[str, str]] = []
        acquires: list[tuple[tuple[str, str], int]] = []
        for effect in effects:
            if isinstance(effect, ReleaseLeaseEffect):
                releases.append((effect.resource_id, effect.owner_id))
            elif isinstance(effect, AcquireLeaseEffect):
                acquires.append(
                    ((effect.resource_id, effect.owner_id), effect.amount)
                )
        release_keys = set(releases)
        acquire_keys = {key for key, _amount in acquires}
        if len(release_keys) != len(releases) or len(acquire_keys) != len(acquires):
            self._issue(
                DiagnosticCode.INVALID_LEASE,
                f"tick {tick} contains duplicate lease changes",
                tick,
            )
            return False
        if release_keys & acquire_keys:
            self._issue(
                DiagnosticCode.INVALID_LEASE,
                f"tick {tick} releases and acquires the same lease",
                tick,
            )
            return False
        if any(key not in self.leases for key in release_keys):
            self._issue(
                DiagnosticCode.INVALID_LEASE,
                f"tick {tick} releases an inactive lease",
                tick,
            )
            return False
        if any(key in self.leases for key in acquire_keys):
            self._issue(
                DiagnosticCode.INVALID_LEASE,
                f"tick {tick} acquires an active lease",
                tick,
            )
            return False
        for key in release_keys:
            del self.leases[key]
        for key, amount in acquires:
            self.leases[key] = amount
        return True

    def _replay_obligations(
        self,
        tick: int,
        effects: list[object],
        previous_values: dict[str, object],
    ) -> bool:
        requests: list[CreateObligationEffect] = []
        satisfies: set[str] = set()
        for effect in effects:
            if isinstance(effect, CreateObligationEffect):
                if effect.condition is None or self._condition_holds(
                    effect.condition,
                    tick,
                    previous_values,
                ):
                    requests.append(effect)
            elif isinstance(effect, SatisfyObligationEffect):
                satisfies.add(effect.obligation_id)
        creates = self._coalesce_obligation_requests(tick, requests)
        if creates is None:
            return False
        for obligation_id, deadline in creates.items():
            if obligation_id in self.obligations:
                self._issue(
                    DiagnosticCode.INVALID_SCHEDULE,
                    f"tick {tick} recreates active obligation {obligation_id}",
                    tick,
                )
                return False
            if deadline is not None and deadline < tick:
                self._issue(
                    DiagnosticCode.DEADLINE_MISSED,
                    f"tick {tick} creates already-expired obligation {obligation_id}",
                    tick,
                )
                return False
            self.obligations[obligation_id] = deadline
        for obligation_id in satisfies:
            if obligation_id not in self.obligations:
                self._issue(
                    DiagnosticCode.INVALID_SCHEDULE,
                    f"tick {tick} satisfies unknown obligation {obligation_id}",
                    tick,
                )
                return False
            deadline = self.obligations[obligation_id]
            if deadline is not None and tick > deadline:
                self._issue(
                    DiagnosticCode.DEADLINE_MISSED,
                    f"tick {tick} satisfies obligation {obligation_id} too late",
                    tick,
                )
                return False
            del self.obligations[obligation_id]
        return True

    def _coalesce_obligation_requests(
        self,
        tick: int,
        requests: list[CreateObligationEffect],
    ) -> dict[str, int | None] | None:
        creates: dict[str, int | None] = {}
        groups: dict[str, list[CreateObligationEffect]] = defaultdict(list)
        for request in requests:
            if request.coalesce_key is None:
                if request.obligation_id in creates:
                    self._issue(
                        DiagnosticCode.INVALID_SCHEDULE,
                        f"tick {tick} creates an obligation more than once",
                        tick,
                    )
                    return None
                creates[request.obligation_id] = request.deadline_tick
            else:
                groups[request.coalesce_key].append(request)
        for coalesce_key, group in groups.items():
            max_priority = max(request.priority for request in group)
            winners = [
                request for request in group if request.priority == max_priority
            ]
            winner_ids = {request.obligation_id for request in winners}
            if len(winner_ids) != 1:
                self._issue(
                    DiagnosticCode.UNDER_SPECIFIED_PRIORITY,
                    f"tick {tick} has ambiguous obligation priority for {coalesce_key}",
                    tick,
                )
                return None
            winner_id = next(iter(winner_ids))
            if winner_id in creates:
                self._issue(
                    DiagnosticCode.INVALID_SCHEDULE,
                    f"tick {tick} creates an obligation more than once",
                    tick,
                )
                return None
            creates[winner_id] = min(
                (
                    request.deadline_tick
                    for request in winners
                    if request.deadline_tick is not None
                ),
                default=None,
            )
        return creates

    def _condition_holds(
        self,
        condition: StateCondition,
        tick: int,
        previous_values: dict[str, object],
    ) -> bool:
        values = previous_values if condition.view == "before" else self.values
        actual = values.get(condition.cell_id)
        if condition.operator == "equal":
            return actual == condition.value and type(actual) is type(condition.value)
        if condition.operator == "not_equal":
            return actual != condition.value or type(actual) is not type(condition.value)
        if not isinstance(actual, int) or isinstance(actual, bool):
            self._issue(
                DiagnosticCode.TYPE_MISMATCH,
                f"condition on {condition.cell_id} requires an integer state",
                tick,
            )
            return False
        assert isinstance(condition.value, int) and not isinstance(condition.value, bool)
        if condition.operator == "greater_equal":
            return actual >= condition.value
        return tick - actual >= condition.value

    def _deadlines_are_valid(self, tick: int) -> bool:
        if self.allow_open_obligations:
            return True
        missed = sorted(
            obligation_id
            for obligation_id, deadline in self.obligations.items()
            if deadline is not None and deadline <= tick
        )
        if missed:
            self._issue(
                DiagnosticCode.DEADLINE_MISSED,
                f"hard obligations missed at tick {tick}: {missed}",
                tick,
            )
        return not missed

    def _capacity_is_valid(self, tick: int, decision_round: int | None = None) -> bool:
        usage: dict[str, int] = defaultdict(int)
        for (resource_id, _owner_id), amount in self.leases.items():
            usage[resource_id] += amount
        for interval in self.schedule.intervals:
            if (
                decision_round is not None
                and interval.start_tick == tick
                and self.start_rounds.get(
                    (interval.operator_instance_id, f"{interval.template_interval_id}.start"),
                    0,
                ) > decision_round
            ):
                continue
            if interval.start_tick <= tick < interval.end_tick:
                for use in interval.resource_uses:
                    usage[use.resource_id] += use.amount
        for resource_id, amount in usage.items():
            if amount > self.resources[resource_id]:
                self._issue(
                    DiagnosticCode.RESOURCE_OVER_CAPACITY,
                    (
                        f"resource {resource_id} uses {amount}/"
                        f"{self.resources[resource_id]} at tick {tick}"
                    ),
                    tick,
                )
                return False
        return True

    def _snapshot(self, tick: int) -> KernelSnapshot:
        return KernelSnapshot(
            tick=tick,
            state_values=tuple(
                StateAssignment(cell_id=cell_id, value=value)
                for cell_id, value in sorted(self.values.items())
            ),
            active_leases=tuple(
                LeaseSpec(resource_id=resource_id, owner_id=owner_id, amount=amount)
                for (resource_id, owner_id), amount in sorted(self.leases.items())
            ),
            active_obligations=tuple(
                ActiveObligation(id=obligation_id, deadline_tick=deadline)
                for obligation_id, deadline in sorted(self.obligations.items())
            ),
            active_interval_ids=tuple(
                sorted(
                    interval.id
                    for interval in self.schedule.intervals
                    if interval.start_tick <= tick < interval.end_tick
                )
            ),
        )

    def _issue(
        self,
        code: DiagnosticCode,
        message: str,
        tick: int | None = None,
    ) -> None:
        self.issues.append(ValidationIssue(code=code, message=message, tick=tick))
