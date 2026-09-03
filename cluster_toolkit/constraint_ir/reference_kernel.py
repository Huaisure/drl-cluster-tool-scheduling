from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .diagnostics import DiagnosticCode, SemanticError
from .schema import (
    AcquireLeaseEffect,
    ActiveObligation,
    ConstraintIRV1,
    CreateObligationEffect,
    EventSpec,
    IncrementStateEffect,
    KernelSnapshot,
    LeaseSpec,
    ReleaseLeaseEffect,
    SatisfyObligationEffect,
    ScheduleV1,
    SessionSnapshot,
    SetStateEffect,
    StateCondition,
    StateAssignment,
)

if TYPE_CHECKING:
    from .reference_session import CandidateGenerator, ReferenceSession


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    snapshots: tuple[KernelSnapshot, ...]

    @property
    def final_snapshot(self) -> KernelSnapshot:
        return self.snapshots[-1]


class ReferenceKernel:
    """Small executable oracle for the G01-G10 semantic foundation."""

    @classmethod
    def execute(
        cls,
        problem: ConstraintIRV1,
        schedule: ScheduleV1,
    ) -> ExecutionResult:
        runtime = _Runtime(problem, schedule)
        return runtime.execute()

    @classmethod
    def execute_until(
        cls,
        problem: ConstraintIRV1,
        schedule: ScheduleV1,
        until_tick: int,
        *,
        allow_open_obligations: bool = False,
    ) -> ExecutionResult:
        """Replay to a tick; planning may defer only as-yet-unscheduled duties."""
        runtime = _Runtime(problem, schedule)
        return runtime.execute(
            until_tick=until_tick,
            allow_open_obligations=allow_open_obligations,
        )

    @classmethod
    def start(
        cls,
        problem: ConstraintIRV1,
        candidate_generator: "CandidateGenerator | None" = None,
    ) -> "ReferenceSession":
        from .reference_session import ReferenceSession

        return ReferenceSession(problem, candidate_generator)

    @classmethod
    def restore(
        cls,
        problem: ConstraintIRV1,
        snapshot: SessionSnapshot | str,
        candidate_generator: "CandidateGenerator | None" = None,
    ) -> "ReferenceSession":
        from .reference_session import ReferenceSession

        parsed = (
            SessionSnapshot.model_validate_json(snapshot)
            if isinstance(snapshot, str)
            else snapshot
        )
        return ReferenceSession.from_snapshot(
            problem,
            parsed,
            candidate_generator,
        )


class _Runtime:
    def __init__(self, problem: ConstraintIRV1, schedule: ScheduleV1) -> None:
        self.problem = problem
        self.schedule = schedule
        self.resources = {resource.id: resource for resource in problem.resources}
        self.cells = {cell.id: cell for cell in problem.state_cells}
        self.state_values = {
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
        self.events_by_tick: dict[int, list[EventSpec]] = defaultdict(list)
        for event in schedule.events:
            self.events_by_tick[event.tick].append(event)
        self.start_rounds = {
            (event.operator_instance_id, event.boundary_id): event.decision_round
            for event in schedule.events
        }
        self.allow_open_obligations = False
        self._validate_schedule_references()

    def execute(
        self, until_tick: int | None = None, *, allow_open_obligations: bool = False,
    ) -> ExecutionResult:
        self.allow_open_obligations = allow_open_obligations
        fixed_ticks = set(self.events_by_tick)
        fixed_ticks.update(
            boundary
            for interval in self.schedule.intervals
            for boundary in (interval.start_tick, interval.end_tick)
        )
        if until_tick is not None:
            fixed_ticks = {tick for tick in fixed_ticks if tick <= until_tick}
            fixed_ticks.add(until_tick)
        if not fixed_ticks and all(
            deadline is None for deadline in self.obligations.values()
        ):
            fixed_ticks.add(0)

        snapshots: list[KernelSnapshot] = []
        processed: set[int] = set()
        while True:
            active_deadlines = {
                deadline
                for deadline in self.obligations.values()
                if not self.allow_open_obligations and deadline is not None
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
            self._apply_tick(tick, tuple(self.events_by_tick.get(tick, ())))
            self._check_resource_capacity(tick)
            self._check_deadlines(tick)
            snapshots.append(self._snapshot(tick))
        return ExecutionResult(snapshots=tuple(snapshots))

    def _validate_schedule_references(self) -> None:
        for interval in self.schedule.intervals:
            for use in interval.resource_uses:
                if use.resource_id not in self.resources:
                    self._fail(
                        DiagnosticCode.UNKNOWN_REFERENCE,
                        f"interval {interval.id} references unknown resource {use.resource_id}",
                    )
        for event in self.schedule.events:
            for effect in event.effects:
                if isinstance(effect, (SetStateEffect, IncrementStateEffect)):
                    if effect.cell_id not in self.cells:
                        self._fail(
                            DiagnosticCode.UNKNOWN_REFERENCE,
                            f"event {event.id} references unknown cell {effect.cell_id}",
                        )
                if isinstance(effect, (AcquireLeaseEffect, ReleaseLeaseEffect)):
                    if effect.resource_id not in self.resources:
                        self._fail(
                            DiagnosticCode.UNKNOWN_REFERENCE,
                            f"event {event.id} references unknown resource {effect.resource_id}",
                        )
                if isinstance(effect, CreateObligationEffect):
                    if (
                        effect.condition is not None
                        and effect.condition.cell_id not in self.cells
                    ):
                        self._fail(
                            DiagnosticCode.UNKNOWN_REFERENCE,
                            f"event {event.id} condition references unknown cell",
                        )

    def _apply_tick(self, tick: int, events: tuple[EventSpec, ...]) -> None:
        rounds: dict[int, list[object]] = defaultdict(list)
        for event in events:
            rounds[event.decision_round].extend(event.effects)
        for decision_round in sorted(rounds):
            effects = rounds[decision_round]
            previous_values = dict(self.state_values)
            self._apply_state_effects(tick, effects)
            self._apply_lease_effects(tick, effects)
            self._apply_obligation_effects(tick, effects, previous_values)
            self._check_resource_capacity(tick, decision_round)
            self._check_deadlines(tick)

    def _apply_state_effects(self, tick: int, effects: list[object]) -> None:
        sets: dict[str, set[object]] = defaultdict(set)
        increments: dict[str, int] = defaultdict(int)
        for effect in effects:
            if isinstance(effect, SetStateEffect):
                if not self.cells[effect.cell_id].accepts(effect.value):
                    self._fail(
                        DiagnosticCode.TYPE_MISMATCH,
                        f"tick {tick} writes an out-of-domain value to {effect.cell_id}",
                    )
                sets[effect.cell_id].add(effect.value)
            elif isinstance(effect, IncrementStateEffect):
                if self.cells[effect.cell_id].value_type != "int":
                    self._fail(
                        DiagnosticCode.TYPE_MISMATCH,
                        f"tick {tick} cannot increment non-integer cell {effect.cell_id}",
                    )
                increments[effect.cell_id] += effect.delta

        for cell_id in sets.keys() & increments.keys():
            self._fail(
                DiagnosticCode.CONFLICTING_EFFECTS,
                f"tick {tick} mixes set and increment for {cell_id}",
            )
        for cell_id, values in sets.items():
            if len(values) != 1:
                self._fail(
                    DiagnosticCode.CONFLICTING_EFFECTS,
                    f"tick {tick} writes different values to {cell_id}",
                )
            value = next(iter(values))
            self._set_state_value(cell_id, value, tick)
        for cell_id, delta in increments.items():
            current = self.state_values.get(cell_id)
            if not isinstance(current, int) or isinstance(current, bool):
                self._fail(
                    DiagnosticCode.TYPE_MISMATCH,
                    f"tick {tick} cannot increment non-integer cell {cell_id}",
                )
            self._set_state_value(cell_id, current + delta, tick)

    def _set_state_value(self, cell_id: str, value: object, tick: int) -> None:
        cell = self.cells[cell_id]
        if not cell.accepts(value):
            self._fail(
                DiagnosticCode.TYPE_MISMATCH,
                f"tick {tick} writes an out-of-domain value to {cell_id}",
            )
        self.state_values[cell_id] = value

    def _apply_lease_effects(self, tick: int, effects: list[object]) -> None:
        releases: set[tuple[str, str]] = set()
        acquires: dict[tuple[str, str], int] = {}
        for effect in effects:
            if isinstance(effect, ReleaseLeaseEffect):
                key = (effect.resource_id, effect.owner_id)
                if key in releases:
                    self._fail(
                        DiagnosticCode.INVALID_LEASE,
                        f"tick {tick} releases lease {key!r} more than once",
                    )
                releases.add(key)
            elif isinstance(effect, AcquireLeaseEffect):
                key = (effect.resource_id, effect.owner_id)
                if key in acquires:
                    self._fail(
                        DiagnosticCode.INVALID_LEASE,
                        f"tick {tick} acquires lease {key!r} more than once",
                    )
                acquires[key] = effect.amount

        if releases & acquires.keys():
            self._fail(
                DiagnosticCode.INVALID_LEASE,
                f"tick {tick} releases and acquires the same lease",
            )
        for key in releases:
            if key not in self.leases:
                self._fail(
                    DiagnosticCode.INVALID_LEASE,
                    f"tick {tick} releases inactive lease {key!r}",
                )
            del self.leases[key]
        for key, amount in acquires.items():
            if key in self.leases:
                self._fail(
                    DiagnosticCode.INVALID_LEASE,
                    f"tick {tick} acquires active lease {key!r}",
                )
            self.leases[key] = amount

    def _apply_obligation_effects(
        self,
        tick: int,
        effects: list[object],
        previous_values: dict[str, object],
    ) -> None:
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
        for obligation_id, deadline in creates.items():
            if obligation_id in self.obligations:
                self._fail(
                    DiagnosticCode.INVALID_SCHEDULE,
                    f"tick {tick} recreates active obligation {obligation_id}",
                )
            if deadline is not None and deadline < tick:
                self._fail(
                    DiagnosticCode.DEADLINE_MISSED,
                    f"tick {tick} creates already-expired obligation {obligation_id}",
                )
            self.obligations[obligation_id] = deadline
        for obligation_id in satisfies:
            if obligation_id not in self.obligations:
                self._fail(
                    DiagnosticCode.INVALID_SCHEDULE,
                    f"tick {tick} satisfies unknown obligation {obligation_id}",
                )
            deadline = self.obligations[obligation_id]
            if deadline is not None and tick > deadline:
                self._fail(
                    DiagnosticCode.DEADLINE_MISSED,
                    f"tick {tick} satisfies obligation {obligation_id} too late",
                )
            del self.obligations[obligation_id]

    def _coalesce_obligation_requests(
        self,
        tick: int,
        requests: list[CreateObligationEffect],
    ) -> dict[str, int | None]:
        creates: dict[str, int | None] = {}
        groups: dict[str, list[CreateObligationEffect]] = defaultdict(list)
        for request in requests:
            if request.coalesce_key is None:
                if request.obligation_id in creates:
                    self._fail(
                        DiagnosticCode.INVALID_SCHEDULE,
                        f"tick {tick} creates obligation {request.obligation_id} twice",
                    )
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
                self._fail(
                    DiagnosticCode.UNDER_SPECIFIED_PRIORITY,
                    f"tick {tick} has ambiguous obligation priority for {coalesce_key}",
                )
            winner_id = next(iter(winner_ids))
            if winner_id in creates:
                self._fail(
                    DiagnosticCode.INVALID_SCHEDULE,
                    f"tick {tick} creates obligation {winner_id} more than once",
                )
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
        values = previous_values if condition.view == "before" else self.state_values
        actual = values.get(condition.cell_id)
        if condition.operator == "equal":
            return actual == condition.value and type(actual) is type(condition.value)
        if condition.operator == "not_equal":
            return actual != condition.value or type(actual) is not type(condition.value)
        if not isinstance(actual, int) or isinstance(actual, bool):
            self._fail(
                DiagnosticCode.TYPE_MISMATCH,
                f"condition on {condition.cell_id} requires an integer state",
            )
        assert isinstance(condition.value, int) and not isinstance(condition.value, bool)
        if condition.operator == "greater_equal":
            return actual >= condition.value
        return tick - actual >= condition.value

    def _check_resource_capacity(
        self, tick: int, decision_round: int | None = None,
    ) -> None:
        totals: dict[str, int] = defaultdict(int)
        for (resource_id, _owner_id), amount in self.leases.items():
            totals[resource_id] += amount
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
                    totals[use.resource_id] += use.amount
        for resource_id, total in totals.items():
            capacity = self.resources[resource_id].capacity
            if total > capacity:
                self._fail(
                    DiagnosticCode.RESOURCE_OVER_CAPACITY,
                    f"resource {resource_id} uses {total}/{capacity} at tick {tick}",
                )

    def _check_deadlines(self, tick: int) -> None:
        if self.allow_open_obligations:
            return
        missed = sorted(
            obligation_id
            for obligation_id, deadline in self.obligations.items()
            if deadline is not None and deadline <= tick
        )
        if missed:
            self._fail(
                DiagnosticCode.DEADLINE_MISSED,
                f"hard obligations missed at tick {tick}: {missed}",
            )

    def _snapshot(self, tick: int) -> KernelSnapshot:
        active_intervals = tuple(
            interval.id
            for interval in self.schedule.intervals
            if interval.start_tick <= tick < interval.end_tick
        )
        return KernelSnapshot(
            tick=tick,
            state_values=tuple(
                StateAssignment(cell_id=cell_id, value=value)
                for cell_id, value in sorted(self.state_values.items())
            ),
            active_leases=tuple(
                LeaseSpec(resource_id=resource_id, owner_id=owner_id, amount=amount)
                for (resource_id, owner_id), amount in sorted(self.leases.items())
            ),
            active_obligations=tuple(
                ActiveObligation(id=obligation_id, deadline_tick=deadline)
                for obligation_id, deadline in sorted(self.obligations.items())
            ),
            active_interval_ids=tuple(sorted(active_intervals)),
        )

    @staticmethod
    def _fail(code: DiagnosticCode, message: str) -> None:
        raise SemanticError(code, message)
