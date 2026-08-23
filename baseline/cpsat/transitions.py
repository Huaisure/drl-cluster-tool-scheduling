"""Finite transition CP-SAT models around a fixed periodic boundary state.

The cyclic unroll supplies a known-feasible discrete schedule and therefore a
precise boundary state.  Startup retimes only the prefix while preserving the
steady-state suffix relative to a variable boundary.  Closedown preserves the
prefix and retimes only the drain.  Module choices, Robot choices, and resource
orders are fixed to the cyclic solution; action times remain CP-SAT variables.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from ortools.sat.python import cp_model

from cluster_toolkit.problem import ClusterProblem, ModuleType, TMArmType, WaferKey

from .direct import (
    FeasibilityConsistencyError,
    _as_integer_time,
    _best_integer_bound,
    _new_solver,
)


@dataclass(frozen=True, slots=True)
class TransitionSolveResult:
    status: str
    objective: int | None
    best_bound: int | None
    runtime_seconds: float
    actions: tuple[Mapping[str, object], ...]
    method: str = "cpsat_fixed_boundary_state"


@dataclass(frozen=True, slots=True)
class _TimedAction:
    index: int
    record: Mapping[str, object]
    start: cp_model.IntVar
    end: cp_model.IntVar

    @property
    def wafer_key(self) -> WaferKey:
        return str(self.record["route_id"]), int(self.record["wafer_index"])


def solve_startup(
    problem: ClusterProblem,
    reference_actions: Sequence[Mapping[str, object]],
    *,
    boundary_time: int,
    time_limit_seconds: float,
    random_seed: int,
    num_search_workers: int,
) -> TransitionSolveResult:
    """Solve empty-tool to fixed steady boundary, minimizing boundary time."""

    boundary_time = _as_integer_time(boundary_time, "startup boundary_time")
    model, actions, horizon = _build_retiming_model(problem, reference_actions)
    boundary = model.new_int_var(0, boundary_time, "startup_boundary")
    for action in actions:
        if str(action.record.get("periodic_phase")) == "startup":
            continue
        relative_start = int(action.record["start"]) - boundary_time
        relative_end = int(action.record["end"]) - boundary_time
        model.add(action.start == boundary + relative_start)
        model.add(action.end == boundary + relative_end)
    model.minimize(boundary)
    return _solve_transition_model(
        model,
        actions,
        objective=boundary,
        objective_offset=0,
        reference_objective=boundary_time,
        time_limit_seconds=time_limit_seconds,
        random_seed=random_seed,
        num_search_workers=num_search_workers,
        horizon=horizon,
    )


def solve_closedown(
    problem: ClusterProblem,
    reference_actions: Sequence[Mapping[str, object]],
    *,
    boundary_time: int,
    time_limit_seconds: float,
    random_seed: int,
    num_search_workers: int,
) -> TransitionSolveResult:
    """Solve fixed steady boundary to empty-tool, minimizing drain duration."""

    boundary_time = _as_integer_time(boundary_time, "closedown boundary_time")
    model, actions, horizon = _build_retiming_model(problem, reference_actions)
    for action in actions:
        if str(action.record.get("periodic_phase")) == "closedown":
            model.add(action.start >= boundary_time)
            continue
        model.add(action.start == int(action.record["start"]))
        model.add(action.end == int(action.record["end"]))
    makespan = model.new_int_var(boundary_time, horizon, "closedown_makespan")
    for action in actions:
        model.add(makespan >= action.end)
    model.minimize(makespan)
    reference_makespan = max(int(action.record["end"]) for action in actions)
    return _solve_transition_model(
        model,
        actions,
        objective=makespan,
        objective_offset=boundary_time,
        reference_objective=reference_makespan,
        time_limit_seconds=time_limit_seconds,
        random_seed=random_seed,
        num_search_workers=num_search_workers,
        horizon=horizon,
    )


def _build_retiming_model(
    problem: ClusterProblem,
    reference_actions: Sequence[Mapping[str, object]],
) -> tuple[cp_model.CpModel, list[_TimedAction], int]:
    if not reference_actions:
        raise ValueError("transition model requires reference actions")
    reference_makespan = max(int(action["end"]) for action in reference_actions)
    horizon = max(1, 2 * reference_makespan)
    model = cp_model.CpModel()
    actions: list[_TimedAction] = []
    for index, record in enumerate(reference_actions):
        start = model.new_int_var(0, horizon, f"transition_a{index}_start")
        end = model.new_int_var(0, horizon, f"transition_a{index}_end")
        duration = int(record["end"]) - int(record["start"])
        model.add(end == start + duration)
        actions.append(
            _TimedAction(
                index=index,
                record=record,
                start=start,
                end=end,
            )
        )

    by_key = {
        (
            action.wafer_key,
            str(action.record["action_type"]),
            int(action.record["step_index"]),
        ): action
        for action in actions
    }
    for wafer_key, initial in problem.initial_state.to_snapshot().wafers_by_key.items():
        route = problem.routes[initial.route_id]
        for edge_index in range(len(route.visits) + 1):
            pick = by_key[wafer_key, "pick", edge_index]
            place = by_key[wafer_key, "place", edge_index + 1]
            model.add(place.start >= pick.end)
        for step_index, visit in enumerate(route.visits, start=1):
            place = by_key[wafer_key, "place", step_index]
            pick = by_key[wafer_key, "pick", step_index]
            process_time = _as_integer_time(
                visit.process_time or 0,
                "Recipe process_time",
            )
            model.add(pick.start >= place.end + process_time)

    for robot_id, robot in problem.ClusterTool.items():
        robot_actions = sorted(
            (
                action
                for action in actions
                if str(action.record["tm_id"]) == robot_id
            ),
            key=lambda action: (
                int(action.record["start"]),
                int(action.record["end"]),
                action.index,
            ),
        )
        for left, right in zip(robot_actions, robot_actions[1:]):
            travel = (
                0
                if left.record["module_id"] == right.record["module_id"]
                else _as_integer_time(
                    robot.travel_times,
                    f"Robot {robot_id} travel_time",
                )
            )
            model.add(right.start >= left.end + travel)

        holding_intervals: list[cp_model.IntervalVar] = []
        for wafer_key, initial in problem.initial_state.to_snapshot().wafers_by_key.items():
            route = problem.routes[initial.route_id]
            for edge_index in range(len(route.visits) + 1):
                pick = by_key[wafer_key, "pick", edge_index]
                if str(pick.record["tm_id"]) != robot_id:
                    continue
                place = by_key[wafer_key, "place", edge_index + 1]
                size = model.new_int_var(
                    0,
                    horizon,
                    f"{robot_id}_{wafer_key}_{edge_index}_holding_size",
                )
                model.add(size == place.end - pick.start)
                holding_intervals.append(
                    model.new_interval_var(
                        pick.start,
                        size,
                        place.end,
                        f"{robot_id}_{wafer_key}_{edge_index}_holding",
                    )
                )
        capacity = 1 if robot.arm_type is TMArmType.SINGLE_ARM else 2
        model.add_cumulative(
            holding_intervals,
            [1] * len(holding_intervals),
            capacity,
        )

    occupancy_by_module: dict[str, list[cp_model.IntervalVar]] = {
        module_id: []
        for module_id, module in problem.Modules.items()
        if module.type not in {ModuleType.IO, ModuleType.LP}
    }
    for wafer_key, initial in problem.initial_state.to_snapshot().wafers_by_key.items():
        route = problem.routes[initial.route_id]
        for step_index in range(1, len(route.visits) + 1):
            place = by_key[wafer_key, "place", step_index]
            pick = by_key[wafer_key, "pick", step_index]
            module_id = str(place.record["module_id"])
            size = model.new_int_var(
                0,
                horizon,
                f"{module_id}_{wafer_key}_{step_index}_occupancy_size",
            )
            model.add(size == pick.end - place.start)
            occupancy_by_module[module_id].append(
                model.new_interval_var(
                    place.start,
                    size,
                    pick.end,
                    f"{module_id}_{wafer_key}_{step_index}_occupancy",
                )
            )
    for module_id, intervals in occupancy_by_module.items():
        capacity = problem.Modules[module_id].capacity
        model.add_cumulative(intervals, [1] * len(intervals), capacity)

    return model, actions, horizon


def _solve_transition_model(
    model: cp_model.CpModel,
    actions: list[_TimedAction],
    *,
    objective: cp_model.IntVar,
    objective_offset: int,
    reference_objective: int,
    time_limit_seconds: float,
    random_seed: int,
    num_search_workers: int,
    horizon: int,
) -> TransitionSolveResult:
    for action in actions:
        model.add_hint(action.start, int(action.record["start"]))
        model.add_hint(action.end, int(action.record["end"]))
    model.add_hint(objective, reference_objective)
    model_error = model.validate()
    if model_error:
        raise FeasibilityConsistencyError(
            f"invalid transition CP-SAT model: {model_error}"
        )

    started = time.monotonic()
    incumbent_budget = min(1.0, max(0.05, float(time_limit_seconds) * 0.1))
    incumbent_solver = _new_solver(
        time_limit_seconds=incumbent_budget,
        random_seed=random_seed,
        num_search_workers=1,
        fix_hints=True,
    )
    incumbent_status = incumbent_solver.solve(model)
    if incumbent_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if incumbent_status in (cp_model.MODEL_INVALID, cp_model.INFEASIBLE):
            raise FeasibilityConsistencyError(
                "transition model rejected its known feasible cyclic hint: "
                f"{incumbent_solver.status_name(incumbent_status)}"
            )
        return TransitionSolveResult(
            status=incumbent_solver.status_name(incumbent_status),
            objective=None,
            best_bound=None,
            runtime_seconds=time.monotonic() - started,
            actions=(),
        )

    remaining = max(0.001, float(time_limit_seconds) - (time.monotonic() - started))
    solver = _new_solver(
        time_limit_seconds=remaining,
        random_seed=random_seed,
        num_search_workers=num_search_workers,
        fix_hints=False,
    )
    status_code = solver.solve(model)
    runtime = time.monotonic() - started
    has_solution = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    solution_solver = solver if has_solution else incumbent_solver
    status = solver.status_name(status_code) if has_solution else "FEASIBLE"
    decoded = tuple(
        MappingProxyType(
            {
                **dict(action.record),
                "start": int(solution_solver.value(action.start)),
                "end": int(solution_solver.value(action.end)),
            }
        )
        for action in actions
    )
    objective_value = int(solution_solver.value(objective)) - objective_offset
    if status_code == cp_model.OPTIMAL:
        best_bound = objective_value
    elif has_solution and math.isfinite(solver.best_objective_bound):
        raw_bound = _best_integer_bound(solver.best_objective_bound)
        best_bound = (
            None if raw_bound is None else max(0, raw_bound - objective_offset)
        )
    else:
        best_bound = None
    return TransitionSolveResult(
        status=status,
        objective=objective_value,
        best_bound=best_bound,
        runtime_seconds=runtime,
        actions=decoded,
    )
