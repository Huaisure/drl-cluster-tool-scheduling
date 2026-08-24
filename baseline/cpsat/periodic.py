"""Fixed-ratio cyclic CP-SAT with finite-batch startup and closedown.

The cycle model schedules one token for every part of the normalized product
ratio.  Robot actions, Robot holding intervals, and PM occupancy intervals are
all periodic across the cycle boundary.  A finite workload is materialized by
introducing one ratio batch per period, beginning from an empty tool and then
stopping introductions after the requested number of repetitions.  Filtering
the infinite periodic schedule this way produces a legal startup, steady
section, and closedown without adding constraints that are outside the current
direct IO/PM problem scope.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ortools.sat.python import cp_model

from cluster_toolkit.cluster_generator.heuristic import build_safe_reference_schedule
from cluster_toolkit.cluster_generator.pipeline import PERIODIC_RATIOS
from cluster_toolkit.cluster_generator.pipeline_models import (
    SchedulingInstance,
    WorkloadItem,
)
from cluster_toolkit.cluster_generator.problem_adapter import to_cluster_problem
from cluster_toolkit.problem import ClusterProblem, ModuleType, TMArmType, WaferKey
from cluster_toolkit.validator import ValidatorSuite

from .direct import (
    FeasibilityConsistencyError,
    _as_integer_time,
    _best_integer_bound,
    _domain_values,
    _new_solver,
    _new_transfer_assignment,
    _require_non_negative_integer,
    _require_positive_integer,
    _require_positive_number,
    _require_supported_problem,
)
from .transitions import TransitionSolveResult, solve_closedown, solve_startup


SOLVER_NAME = "cpsat_periodic"
SOLVER_VERSION = "0.2.1"


@dataclass(frozen=True, slots=True)
class PeriodicComponentResult:
    status: str
    objective: int | None
    best_bound: int | None
    runtime_seconds: float
    action_count: int
    method: str


@dataclass(frozen=True, slots=True)
class PeriodicResult:
    """One periodic labeling attempt for a complete finite workload."""

    status: str
    ratio: tuple[int, ...] | None
    repeat_count: int | None
    period: int | None
    pipeline_depth_periods: int | None
    steady_cycle_count: int | None
    boundary_shift: int | None
    boundary_candidates_evaluated: int | None
    composition_runtime_seconds: float | None
    actions: tuple[Mapping[str, object], ...]
    cycle_actions: tuple[Mapping[str, object], ...]
    makespan: int | None
    validation_ok: bool | None
    cycle: PeriodicComponentResult | None
    startup: PeriodicComponentResult | None
    closedown: PeriodicComponentResult | None
    solver_name: str = SOLVER_NAME
    solver_version: str = SOLVER_VERSION


@dataclass(frozen=True, slots=True)
class _CycleAction:
    node_id: int
    action_type: str
    token_key: WaferKey
    edge_index: int
    step_index: int
    duration: cp_model.IntVar
    location: cp_model.IntVar
    assigned: Mapping[str, cp_model.IntVar]
    start: cp_model.IntVar
    end: cp_model.IntVar


@dataclass(frozen=True, slots=True)
class _CycleTransfer:
    token_key: WaferKey
    edge_index: int
    pick: _CycleAction
    place: _CycleAction
    wrap: cp_model.IntVar


@dataclass(frozen=True, slots=True)
class _CycleVisit:
    token_key: WaferKey
    step_index: int
    location: cp_model.IntVar
    selected: Mapping[str, cp_model.IntVar]
    incoming_place: _CycleAction
    outgoing_pick: _CycleAction
    wrap: cp_model.IntVar


@dataclass(frozen=True, slots=True)
class _SolvedCycle:
    status: str
    period: int
    best_bound: int | None
    runtime_seconds: float
    actions: tuple[Mapping[str, object], ...]
    transfer_wraps: Mapping[tuple[WaferKey, int], int]
    process_wraps: Mapping[tuple[WaferKey, int], int]


@dataclass(frozen=True, slots=True)
class _BoundaryCandidate:
    solved: _SolvedCycle
    actions: tuple[Mapping[str, object], ...]
    phase_counts: dict[str, int]
    pipeline_depth: int
    shift: int
    estimated_makespan: int


def periodic_ratio(instance: SchedulingInstance) -> tuple[int, ...] | None:
    """Return the supported normalized workload ratio, otherwise ``None``."""

    if instance.provenance.periodic_ratio is None:
        return None

    counts_by_recipe = {item.recipe_id: item.wafer_count for item in instance.workload}
    counts = tuple(counts_by_recipe[recipe.recipe_id] for recipe in instance.recipes)
    supported = PERIODIC_RATIOS.get(len(counts))
    if supported is None:
        return None
    divisor = math.gcd(*counts)
    normalized = tuple(count // divisor for count in counts)
    if normalized not in supported:
        return None
    domains = [
        frozenset(step.candidate_module_ids)
        for recipe in instance.recipes
        for step in recipe.steps
    ]
    if any(
        left.intersection(right) and left != right
        for index, left in enumerate(domains)
        for right in domains[index + 1 :]
    ):
        return None
    return normalized


def solve_periodic_instance(
    instance: SchedulingInstance,
    *,
    time_limit_seconds: float = 1800,
    random_seed: int = 0,
    num_search_workers: int = 1,
    startup_time_limit_seconds: float | None = None,
    closedown_time_limit_seconds: float | None = None,
) -> PeriodicResult:
    """Solve a supported fixed-ratio instance and materialize its finite batch."""

    _require_positive_number("time_limit_seconds", time_limit_seconds)
    _require_non_negative_integer("random_seed", random_seed)
    _require_positive_integer("num_search_workers", num_search_workers)
    ratio = periodic_ratio(instance)
    if ratio is None:
        return PeriodicResult(
            status="NOT_ELIGIBLE",
            ratio=None,
            repeat_count=None,
            period=None,
            pipeline_depth_periods=None,
            steady_cycle_count=None,
            boundary_shift=None,
            boundary_candidates_evaluated=None,
            composition_runtime_seconds=None,
            actions=(),
            cycle_actions=(),
            makespan=None,
            validation_ok=None,
            cycle=None,
            startup=None,
            closedown=None,
        )

    repeat_count = _repeat_count(instance, ratio)
    solved = _solve_cycle(
        instance,
        ratio,
        time_limit_seconds=time_limit_seconds,
        random_seed=random_seed,
        num_search_workers=num_search_workers,
    )
    composition_started = time.monotonic()
    candidates, boundary_candidates = _select_cycle_boundaries(
        instance,
        ratio,
        repeat_count,
        solved,
    )
    problem = to_cluster_problem(instance)
    transition_budget_default = min(float(time_limit_seconds), 5.0)
    startup_budget = (
        transition_budget_default
        if startup_time_limit_seconds is None
        else float(startup_time_limit_seconds)
    )
    closedown_budget = (
        transition_budget_default
        if closedown_time_limit_seconds is None
        else float(closedown_time_limit_seconds)
    )
    _require_positive_number("startup_time_limit_seconds", startup_budget)
    _require_positive_number("closedown_time_limit_seconds", closedown_budget)
    transition_jobs: dict[tuple[int, str], object] = {}
    transition_wall_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2 * len(candidates)) as executor:
        for candidate_index, candidate in enumerate(candidates):
            startup_boundary = (
                min(repeat_count, candidate.pipeline_depth)
                * candidate.solved.period
            )
            closedown_boundary = repeat_count * candidate.solved.period
            transition_jobs[candidate_index, "startup"] = executor.submit(
                solve_startup,
                problem,
                candidate.actions,
                boundary_time=startup_boundary,
                time_limit_seconds=startup_budget,
                random_seed=random_seed + 2 * candidate_index,
                num_search_workers=num_search_workers,
            )
            transition_jobs[candidate_index, "closedown"] = executor.submit(
                solve_closedown,
                problem,
                candidate.actions,
                boundary_time=closedown_boundary,
                time_limit_seconds=closedown_budget,
                random_seed=random_seed + 2 * candidate_index + 1,
                num_search_workers=num_search_workers,
            )
    transition_wall_runtime = time.monotonic() - transition_wall_started

    outcomes: list[
        tuple[
            tuple[int, int, int],
            _BoundaryCandidate,
            tuple[Mapping[str, object], ...],
            TransitionSolveResult,
            TransitionSolveResult,
            bool,
        ]
    ] = []
    for candidate_index, candidate in enumerate(candidates):
        try:
            startup_solve = transition_jobs[candidate_index, "startup"].result()
            closedown_solve = transition_jobs[candidate_index, "closedown"].result()
        except FeasibilityConsistencyError:
            raise
        except Exception:
            startup_solve = _failed_transition_result()
            closedown_solve = _failed_transition_result()
        startup_boundary = (
            min(repeat_count, candidate.pipeline_depth) * candidate.solved.period
        )
        actions = candidate.actions
        used_transition_cpsat = False
        if startup_solve.actions and closedown_solve.actions:
            merged = _merge_transition_solutions(
                candidate.actions,
                startup_solve.actions,
                closedown_solve.actions,
                startup_boundary=startup_boundary,
                startup_objective=startup_solve.objective,
            )
            transition_report = ValidatorSuite(problem).validate(
                merged,
                require_complete=True,
                exact_action_durations=True,
            )
            if transition_report.ok:
                actions = merged
                used_transition_cpsat = True
        makespan = max(int(action["end"]) for action in actions)
        outcomes.append(
            (
                (makespan, candidate.estimated_makespan, candidate.shift),
                candidate,
                actions,
                startup_solve,
                closedown_solve,
                used_transition_cpsat,
            )
        )

    (
        _,
        selected_candidate,
        actions,
        startup_solve,
        closedown_solve,
        used_transition_cpsat,
    ) = min(outcomes, key=lambda outcome: outcome[0])
    solved = selected_candidate.solved
    phase_counts = selected_candidate.phase_counts
    pipeline_depth = selected_candidate.pipeline_depth
    boundary_shift = selected_candidate.shift
    startup_boundary = min(repeat_count, pipeline_depth) * solved.period
    closedown_boundary = repeat_count * solved.period
    composition_runtime = max(
        0.0,
        time.monotonic() - composition_started - transition_wall_runtime,
    )
    report = ValidatorSuite(problem).validate(
        actions,
        require_complete=True,
        exact_action_durations=True,
    )
    expected_action_count = 2 * sum(
        item.wafer_count
        * (len(problem.routes[item.recipe_id].visits) + 1)
        for item in instance.workload
    )
    if len(actions) != expected_action_count:
        raise RuntimeError(
            "periodic composition is incomplete: "
            f"expected {expected_action_count} actions, got {len(actions)}"
        )
    if not report.ok:
        details = "; ".join(issue.message for issue in report.issues[:8])
        raise RuntimeError(f"periodic composition failed ValidatorSuite: {details}")

    makespan = max((int(action["end"]) for action in actions), default=0)
    startup_periods = min(repeat_count, pipeline_depth)
    startup_end = (
        startup_solve.objective
        if startup_solve.objective is not None
        else min(makespan, startup_periods * solved.period)
    )
    closedown_duration = (
        closedown_solve.objective
        if closedown_solve.objective is not None
        else max(0, makespan - closedown_boundary)
    )
    steady_cycle_count = max(0, repeat_count - pipeline_depth)

    return PeriodicResult(
        status="FEASIBLE",
        ratio=ratio,
        repeat_count=repeat_count,
        period=solved.period,
        pipeline_depth_periods=pipeline_depth,
        steady_cycle_count=steady_cycle_count,
        boundary_shift=boundary_shift,
        boundary_candidates_evaluated=boundary_candidates,
        composition_runtime_seconds=composition_runtime,
        actions=actions,
        cycle_actions=solved.actions,
        makespan=makespan,
        validation_ok=True,
        cycle=PeriodicComponentResult(
            status=solved.status,
            objective=solved.period,
            best_bound=solved.best_bound,
            runtime_seconds=solved.runtime_seconds,
            action_count=len(solved.actions),
            method="cpsat_periodic_circuit",
        ),
        startup=PeriodicComponentResult(
            status=startup_solve.status,
            objective=startup_end,
            best_bound=startup_solve.best_bound,
            runtime_seconds=startup_solve.runtime_seconds,
            action_count=phase_counts["startup"],
            method=(
                startup_solve.method
                if used_transition_cpsat
                else "cyclic_unroll_from_empty_fallback"
            ),
        ),
        closedown=PeriodicComponentResult(
            status=closedown_solve.status,
            objective=closedown_duration,
            best_bound=closedown_solve.best_bound,
            runtime_seconds=closedown_solve.runtime_seconds,
            action_count=phase_counts["closedown"],
            method=(
                closedown_solve.method
                if used_transition_cpsat
                else "cyclic_unroll_to_empty_fallback"
            ),
        ),
    )


def _failed_transition_result() -> TransitionSolveResult:
    return TransitionSolveResult(
        status="UNKNOWN",
        objective=None,
        best_bound=None,
        runtime_seconds=0.0,
        actions=(),
        method="cyclic_unroll_fallback",
    )


def _merge_transition_solutions(
    derived: tuple[Mapping[str, object], ...],
    startup: tuple[Mapping[str, object], ...],
    closedown: tuple[Mapping[str, object], ...],
    *,
    startup_boundary: int,
    startup_objective: int | None,
) -> tuple[Mapping[str, object], ...]:
    def key(action: Mapping[str, object]) -> tuple[str, int, str, int]:
        return (
            str(action["route_id"]),
            int(action["wafer_index"]),
            str(action["action_type"]),
            int(action["step_index"]),
        )

    startup_by_key = {key(action): action for action in startup}
    closedown_by_key = {key(action): action for action in closedown}
    shift = (
        startup_boundary if startup_objective is None else startup_objective
    ) - startup_boundary
    merged: list[Mapping[str, object]] = []
    for reference in derived:
        if str(reference.get("periodic_phase")) == "closedown":
            selected = closedown_by_key[key(reference)]
            merged.append(
                MappingProxyType(
                    {
                        **dict(selected),
                        "start": int(selected["start"]) + shift,
                        "end": int(selected["end"]) + shift,
                    }
                )
            )
        else:
            merged.append(startup_by_key[key(reference)])
    merged.sort(
        key=lambda action: (
            int(action["start"]),
            int(action["end"]),
            str(action["tm_id"]),
            str(action["route_id"]),
            int(action["wafer_index"]),
            str(action["action_type"]),
        )
    )
    return tuple(merged)


def _solve_cycle(
    instance: SchedulingInstance,
    ratio: tuple[int, ...],
    *,
    time_limit_seconds: float,
    random_seed: int,
    num_search_workers: int,
) -> _SolvedCycle:
    cycle_instance = instance.model_copy(
        update={
            "workload": tuple(
                WorkloadItem(recipe_id=recipe.recipe_id, wafer_count=count)
                for recipe, count in zip(instance.recipes, ratio, strict=True)
            )
        }
    )
    problem = to_cluster_problem(cycle_instance)
    robot_ids = _require_supported_problem(problem)
    reference = build_safe_reference_schedule(problem)
    upper_bound = _as_integer_time(reference.makespan, "cycle reference makespan")
    pick_durations = {
        robot_id: _as_integer_time(
            problem.ClusterTool[robot_id].pick_time,
            f"Robot {robot_id} pick_time",
        )
        for robot_id in robot_ids
    }
    place_durations = {
        robot_id: _as_integer_time(
            problem.ClusterTool[robot_id].place_time,
            f"Robot {robot_id} place_time",
        )
        for robot_id in robot_ids
    }
    lower_bound = 0

    model = cp_model.CpModel()
    period = model.new_int_var(lower_bound, upper_bound, "period")
    module_ids = tuple(sorted(problem.Modules))
    module_index = {module_id: index for index, module_id in enumerate(module_ids)}
    io_id = problem.io_module_ids[0]
    io_location = model.new_constant(module_index[io_id])
    actions: list[_CycleAction] = []
    transfers: list[_CycleTransfer] = []
    visits: list[_CycleVisit] = []

    def new_action(
        action_type: str,
        token_key: WaferKey,
        edge_index: int,
        step_index: int,
        duration_by_robot: Mapping[str, int],
        location: cp_model.IntVar,
        assigned: Mapping[str, cp_model.IntVar],
    ) -> _CycleAction:
        node_id = len(actions)
        start = model.new_int_var(0, upper_bound, f"cycle_a{node_id}_start")
        end = model.new_int_var(0, upper_bound, f"cycle_a{node_id}_end")
        duration = model.new_int_var(
            min(duration_by_robot.values()),
            max(duration_by_robot.values()),
            f"cycle_a{node_id}_duration",
        )
        model.add(
            duration
            == sum(
                duration_by_robot[robot_id] * assigned[robot_id]
                for robot_id in robot_ids
            )
        )
        model.add(end == start + duration)
        model.add(end <= period)
        action = _CycleAction(
            node_id=node_id,
            action_type=action_type,
            token_key=token_key,
            edge_index=edge_index,
            step_index=step_index,
            duration=duration,
            location=location,
            assigned=assigned,
            start=start,
            end=end,
        )
        actions.append(action)
        return action

    for recipe, token_count in zip(instance.recipes, ratio, strict=True):
        route = problem.routes[recipe.recipe_id]
        for token_index in range(token_count):
            token_key = (recipe.recipe_id, token_index)
            visit_locations: list[cp_model.IntVar] = []
            visit_selected: list[Mapping[str, cp_model.IntVar]] = []
            for step_index, route_visit in enumerate(route.visits, start=1):
                candidate_indices = [module_index[item] for item in route_visit.module_ids]
                location = model.new_int_var_from_domain(
                    cp_model.Domain.from_values(candidate_indices),
                    f"cycle_{recipe.recipe_id}_{token_index}_s{step_index}_location",
                )
                selected: dict[str, cp_model.IntVar] = {}
                for module_id in route_visit.module_ids:
                    choice = model.new_bool_var(
                        f"cycle_{recipe.recipe_id}_{token_index}_s{step_index}_{module_id}"
                    )
                    model.add(location == module_index[module_id]).only_enforce_if(choice)
                    selected[module_id] = choice
                model.add_exactly_one(selected.values())
                visit_locations.append(location)
                visit_selected.append(MappingProxyType(selected))

            locations = [io_location, *visit_locations, io_location]
            token_transfers: list[_CycleTransfer] = []
            for edge_index in range(len(route.visits) + 1):
                assigned = _new_transfer_assignment(
                    model,
                    problem,
                    locations[edge_index],
                    locations[edge_index + 1],
                    module_ids,
                    robot_ids,
                    prefix=(
                        f"cycle_{recipe.recipe_id}_{token_index}_e{edge_index}"
                    ),
                )
                pick = new_action(
                    "pick",
                    token_key,
                    edge_index,
                    edge_index,
                    pick_durations,
                    locations[edge_index],
                    assigned,
                )
                place = new_action(
                    "place",
                    token_key,
                    edge_index,
                    edge_index + 1,
                    place_durations,
                    locations[edge_index + 1],
                    assigned,
                )
                wrap = model.new_bool_var(
                    f"cycle_{recipe.recipe_id}_{token_index}_e{edge_index}_hold_wrap"
                )
                model.add(place.start >= pick.end).only_enforce_if(wrap.negated())
                model.add(place.start + period >= pick.end).only_enforce_if(wrap)
                model.add(place.start < pick.end).only_enforce_if(wrap)
                transfer = _CycleTransfer(
                    token_key=token_key,
                    edge_index=edge_index,
                    pick=pick,
                    place=place,
                    wrap=wrap,
                )
                transfers.append(transfer)
                token_transfers.append(transfer)

            for step_offset, route_visit in enumerate(route.visits):
                incoming = token_transfers[step_offset]
                outgoing = token_transfers[step_offset + 1]
                wrap = model.new_bool_var(
                    f"cycle_{recipe.recipe_id}_{token_index}_s{step_offset + 1}_process_wrap"
                )
                process_time = _as_integer_time(
                    route_visit.process_time or 0,
                    "Recipe process_time",
                )
                model.add(
                    outgoing.pick.start >= incoming.place.end + process_time
                ).only_enforce_if(wrap.negated())
                model.add(
                    outgoing.pick.start + period >= incoming.place.end + process_time
                ).only_enforce_if(wrap)
                model.add(
                    outgoing.pick.start < incoming.place.end + process_time
                ).only_enforce_if(wrap)
                visits.append(
                    _CycleVisit(
                        token_key=token_key,
                        step_index=step_offset + 1,
                        location=visit_locations[step_offset],
                        selected=visit_selected[step_offset],
                        incoming_place=incoming.place,
                        outgoing_pick=outgoing.pick,
                        wrap=wrap,
                    )
                )

    if not actions:  # pragma: no cover - SchedulingInstance requires Recipes/wafers
        raise ValueError("periodic cycle contains no actions")
    model.add(actions[0].start == 0)
    _add_cyclic_pm_capacity(model, problem, visits, period, upper_bound)
    _add_cyclic_robot_capacity(
        model,
        problem,
        transfers,
        period,
        upper_bound,
    )
    circuit_arcs, circuit_wraps = _add_cyclic_robot_circuits(
        model,
        actions,
        period,
        problem,
    )
    model.minimize(period)
    _add_cycle_reference_hints(
        model,
        reference.actions,
        actions,
        visits,
        transfers,
        circuit_arcs,
        circuit_wraps,
        period,
        upper_bound,
        module_index,
    )
    model_error = model.validate()
    if model_error:
        raise RuntimeError(f"invalid periodic CP-SAT model: {model_error}")

    started = time.monotonic()
    incumbent_solver = _new_solver(
        time_limit_seconds=min(float(time_limit_seconds), 5.0),
        random_seed=random_seed,
        num_search_workers=1,
        fix_hints=True,
    )
    incumbent_status = incumbent_solver.solve(model)
    if incumbent_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if incumbent_status in (cp_model.MODEL_INVALID, cp_model.INFEASIBLE):
            raise FeasibilityConsistencyError(
                "serial cycle hint is inconsistent with the periodic model: "
                f"{incumbent_solver.status_name(incumbent_status)}"
            )
        raise TimeoutError("periodic cycle did not load its feasible hint in time")

    incumbent_runtime = time.monotonic() - started
    solver = _new_solver(
        time_limit_seconds=max(0.001, float(time_limit_seconds) - incumbent_runtime),
        random_seed=random_seed,
        num_search_workers=num_search_workers,
        fix_hints=False,
    )
    status_code = solver.solve(model)
    runtime = time.monotonic() - started
    if status_code in (cp_model.MODEL_INVALID, cp_model.INFEASIBLE):
        raise FeasibilityConsistencyError(
            "periodic model rejected a known feasible cycle: "
            f"{solver.status_name(status_code)}"
        )
    has_full_solution = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    solution_solver = solver if has_full_solution else incumbent_solver
    status = solver.status_name(status_code) if has_full_solution else "FEASIBLE"
    solved_period = int(solution_solver.value(period))

    decoded_actions = tuple(
        MappingProxyType(
            {
                "node_id": action.node_id,
                "action_type": action.action_type,
                "route_id": action.token_key[0],
                "token_index": action.token_key[1],
                "edge_index": action.edge_index,
                "step_index": action.step_index,
                "module_id": module_ids[solution_solver.value(action.location)],
                "tm_id": next(
                    robot_id
                    for robot_id, assigned in action.assigned.items()
                    if solution_solver.value(assigned)
                ),
                "start": int(solution_solver.value(action.start)),
                "end": int(solution_solver.value(action.end)),
            }
        )
        for action in actions
    )
    transfer_wraps = MappingProxyType(
        {
            (transfer.token_key, transfer.edge_index): int(
                solution_solver.value(transfer.wrap)
            )
            for transfer in transfers
        }
    )
    process_wraps = MappingProxyType(
        {
            (visit.token_key, visit.step_index): int(solution_solver.value(visit.wrap))
            for visit in visits
        }
    )
    best_bound = (
        solved_period
        if status_code == cp_model.OPTIMAL
        else _best_integer_bound(solver.best_objective_bound)
    )
    return _SolvedCycle(
        status=status,
        period=solved_period,
        best_bound=best_bound,
        runtime_seconds=runtime,
        actions=decoded_actions,
        transfer_wraps=transfer_wraps,
        process_wraps=process_wraps,
    )


def _add_cyclic_pm_capacity(
    model: cp_model.CpModel,
    problem: ClusterProblem,
    visits: list[_CycleVisit],
    period: cp_model.IntVar,
    upper_bound: int,
) -> None:
    intervals_by_module: dict[str, list[cp_model.IntervalVar]] = {
        module_id: []
        for module_id, module in problem.Modules.items()
        if module.type not in {ModuleType.IO, ModuleType.LP}
    }
    for visit in visits:
        wrap_time = _boolean_times_period(
            model,
            visit.wrap,
            period,
            upper_bound,
            f"visit_{visit.token_key[0]}_{visit.token_key[1]}_{visit.step_index}",
        )
        absolute_end = model.new_int_var(
            0,
            2 * upper_bound,
            f"visit_{visit.token_key[0]}_{visit.token_key[1]}_{visit.step_index}_end",
        )
        size = model.new_int_var(
            0,
            upper_bound,
            f"visit_{visit.token_key[0]}_{visit.token_key[1]}_{visit.step_index}_size",
        )
        model.add(absolute_end == visit.outgoing_pick.end + wrap_time)
        model.add(size == absolute_end - visit.incoming_place.start)
        model.add(size <= period)
        shifted_start = model.new_int_var(
            0,
            2 * upper_bound,
            f"visit_{visit.token_key[0]}_{visit.token_key[1]}_{visit.step_index}_shift_start",
        )
        shifted_end = model.new_int_var(
            0,
            3 * upper_bound,
            f"visit_{visit.token_key[0]}_{visit.token_key[1]}_{visit.step_index}_shift_end",
        )
        model.add(shifted_start == visit.incoming_place.start + period)
        model.add(shifted_end == absolute_end + period)
        for module_id, selected in visit.selected.items():
            intervals_by_module[module_id].append(
                model.new_optional_interval_var(
                    visit.incoming_place.start,
                    size,
                    absolute_end,
                    selected,
                    f"{module_id}_visit_base_{visit.token_key}_{visit.step_index}",
                )
            )
            intervals_by_module[module_id].append(
                model.new_optional_interval_var(
                    shifted_start,
                    size,
                    shifted_end,
                    selected,
                    f"{module_id}_visit_shift_{visit.token_key}_{visit.step_index}",
                )
            )

    for module_id, intervals in intervals_by_module.items():
        capacity = problem.Modules[module_id].capacity
        if capacity == 1:
            model.add_no_overlap(intervals)
        else:
            model.add_cumulative(intervals, [1] * len(intervals), capacity)


def _add_cyclic_robot_capacity(
    model: cp_model.CpModel,
    problem: ClusterProblem,
    transfers: list[_CycleTransfer],
    period: cp_model.IntVar,
    upper_bound: int,
) -> None:
    for robot_id, robot in problem.ClusterTool.items():
        intervals: list[cp_model.IntervalVar] = []
        for transfer in transfers:
            prefix = (
                f"{robot_id}_transfer_{transfer.token_key[0]}_"
                f"{transfer.token_key[1]}_{transfer.edge_index}"
            )
            wrap_time = _boolean_times_period(
                model,
                transfer.wrap,
                period,
                upper_bound,
                prefix,
            )
            absolute_end = model.new_int_var(0, 2 * upper_bound, f"{prefix}_end")
            size = model.new_int_var(0, upper_bound, f"{prefix}_size")
            model.add(absolute_end == transfer.place.end + wrap_time)
            model.add(size == absolute_end - transfer.pick.start)
            model.add(size <= period)
            shifted_start = model.new_int_var(
                0, 2 * upper_bound, f"{prefix}_shift_start"
            )
            shifted_end = model.new_int_var(
                0, 3 * upper_bound, f"{prefix}_shift_end"
            )
            model.add(shifted_start == transfer.pick.start + period)
            model.add(shifted_end == absolute_end + period)
            presence = transfer.pick.assigned[robot_id]
            intervals.append(
                model.new_optional_interval_var(
                    transfer.pick.start,
                    size,
                    absolute_end,
                    presence,
                    f"{prefix}_base",
                )
            )
            intervals.append(
                model.new_optional_interval_var(
                    shifted_start,
                    size,
                    shifted_end,
                    presence,
                    f"{prefix}_shift",
                )
            )
        capacity = 1 if robot.arm_type is TMArmType.SINGLE_ARM else 2
        model.add_cumulative(intervals, [1] * len(intervals), capacity)


def _add_cyclic_robot_circuits(
    model: cp_model.CpModel,
    actions: list[_CycleAction],
    period: cp_model.IntVar,
    problem: ClusterProblem,
) -> tuple[
    dict[tuple[str, int, int], cp_model.IntVar],
    dict[tuple[str, int, int], cp_model.IntVar],
]:
    arc_vars: dict[tuple[str, int, int], cp_model.IntVar] = {}
    wrap_vars: dict[tuple[str, int, int], cp_model.IntVar] = {}
    for robot_id, robot in problem.ClusterTool.items():
        travel_time = _as_integer_time(
            robot.travel_times,
            f"Robot {robot_id} travel_time",
        )
        arcs: list[tuple[int, int, cp_model.IntVar]] = []
        used = model.new_bool_var(f"cycle_{robot_id}_used")
        model.add_max_equality(
            used,
            [action.assigned[robot_id] for action in actions],
        )
        robot_wraps: list[cp_model.IntVar] = []
        for action in actions:
            arcs.append(
                (
                    action.node_id,
                    action.node_id,
                    action.assigned[robot_id].negated(),
                )
            )
            for right in actions:
                if action.node_id == right.node_id:
                    continue
                key = (robot_id, action.node_id, right.node_id)
                arc = model.new_bool_var(
                    f"cycle_{robot_id}_arc_{action.node_id}_{right.node_id}"
                )
                wrap = model.new_bool_var(
                    f"cycle_{robot_id}_wrap_{action.node_id}_{right.node_id}"
                )
                model.add(arc <= action.assigned[robot_id])
                model.add(arc <= right.assigned[robot_id])
                model.add(wrap <= arc)
                arcs.append((action.node_id, right.node_id, arc))
                arc_vars[key] = arc
                wrap_vars[key] = wrap
                robot_wraps.append(wrap)

                left_domain = set(_domain_values(action.location))
                right_domain = set(_domain_values(right.location))
                if not left_domain & right_domain:
                    travel = travel_time
                elif len(left_domain) == len(right_domain) == 1:
                    travel = 0
                else:
                    same = model.new_bool_var(
                        f"cycle_{robot_id}_same_{action.node_id}_{right.node_id}"
                    )
                    model.add(action.location == right.location).only_enforce_if(same)
                    model.add(action.location != right.location).only_enforce_if(
                        same.negated()
                    )
                    travel = travel_time * (1 - same)

                model.add(right.start >= action.end + travel).only_enforce_if(
                    [arc, wrap.negated()]
                )
                model.add(
                    right.start + period >= action.end + travel
                ).only_enforce_if(wrap)

        model.add_circuit(arcs)
        model.add(sum(robot_wraps) == used)
    return arc_vars, wrap_vars


def _add_cycle_reference_hints(
    model: cp_model.CpModel,
    reference_actions: tuple[dict[str, object], ...],
    actions: list[_CycleAction],
    visits: list[_CycleVisit],
    transfers: list[_CycleTransfer],
    circuit_arcs: Mapping[tuple[str, int, int], cp_model.IntVar],
    circuit_wraps: Mapping[tuple[str, int, int], cp_model.IntVar],
    period: cp_model.IntVar,
    upper_bound: int,
    module_index: Mapping[str, int],
) -> None:
    by_key = {
        (
            str(action["route_id"]),
            int(action["wafer_index"]),
            str(action["action_type"]),
            int(action["step_index"]),
        ): action
        for action in reference_actions
    }
    order: list[tuple[int, int, int]] = []
    hinted_assignments: set[int] = set()
    for action in actions:
        reference = by_key[
            action.token_key[0],
            action.token_key[1],
            action.action_type,
            action.step_index,
        ]
        start = _as_integer_time(reference["start"], "cycle hint start")
        end = _as_integer_time(reference["end"], "cycle hint end")
        model.add_hint(action.start, start)
        model.add_hint(action.end, end)
        reference_robot = str(reference["tm_id"])
        for robot_id, assigned in action.assigned.items():
            if assigned.index not in hinted_assignments:
                model.add_hint(assigned, int(robot_id == reference_robot))
                hinted_assignments.add(assigned.index)
        order.append((start, end, action.node_id))
    model.add_hint(period, upper_bound)

    for visit in visits:
        reference = by_key[
            visit.token_key[0],
            visit.token_key[1],
            "place",
            visit.step_index,
        ]
        module_id = str(reference["module_id"])
        model.add_hint(visit.location, module_index[module_id])
        for candidate, selected in visit.selected.items():
            model.add_hint(selected, int(candidate == module_id))
        model.add_hint(visit.wrap, 0)
    for transfer in transfers:
        model.add_hint(transfer.wrap, 0)

    for robot_id in sorted({key[0] for key in circuit_arcs}):
        ordered_nodes = [
            node_id
            for _, _, node_id in sorted(order)
            if str(
                by_key[
                    actions[node_id].token_key[0],
                    actions[node_id].token_key[1],
                    actions[node_id].action_type,
                    actions[node_id].step_index,
                ]["tm_id"]
            )
            == robot_id
        ]
        selected_arcs = set(
            zip(
                ordered_nodes,
                [*ordered_nodes[1:], ordered_nodes[0]],
                strict=True,
            )
        ) if ordered_nodes else set()
        wrap_arc = (
            (ordered_nodes[-1], ordered_nodes[0]) if ordered_nodes else None
        )
        for key, variable in circuit_arcs.items():
            if key[0] != robot_id:
                continue
            edge = (key[1], key[2])
            model.add_hint(variable, int(edge in selected_arcs))
            model.add_hint(circuit_wraps[key], int(edge == wrap_arc))


def _select_cycle_boundaries(
    instance: SchedulingInstance,
    ratio: tuple[int, ...],
    repeat_count: int,
    solved: _SolvedCycle,
) -> tuple[tuple[_BoundaryCandidate, ...], int]:
    problem = to_cluster_problem(instance)
    candidate_shifts = sorted(
        {
            value
            for action in solved.actions
            for value in (int(action["start"]), int(action["end"]))
            if value < solved.period
        }
    )
    ranked: list[tuple[tuple[int, int, int], _BoundaryCandidate]] = []
    evaluated = 0
    for shift in candidate_shifts:
        rotated = _rotate_cycle_boundary(instance, ratio, solved, shift)
        actions, phase_counts, depth = _materialize_finite_schedule(
            instance,
            ratio,
            repeat_count,
            rotated,
        )
        # A feasible cyclic incumbent can place an entire ratio batch inside
        # one period.  That finite unroll is valid, but it has no explicit
        # boundary transition and therefore does not satisfy this solver's
        # explicit startup + closedown result contract.  A short finite batch
        # may legitimately have no steady actions, so only the two transitions
        # are required here.  CP-SAT may return a zero-depth incumbent on one
        # platform and an overlapping one on another, so enforce the contract
        # before ranking candidates by makespan.
        if depth <= 0 or any(
            phase_counts[phase] <= 0 for phase in ("startup", "closedown")
        ):
            continue
        report = ValidatorSuite(problem).validate(
            actions,
            require_complete=True,
            exact_action_durations=True,
        )
        if not report.ok:
            continue
        evaluated += 1
        makespan = max((int(action["end"]) for action in actions), default=0)
        # Makespan is primary; shallower fill and deterministic smaller shifts
        # break ties without changing the steady-state objective.
        key = (makespan, depth, shift)
        ranked.append(
            (
                key,
                _BoundaryCandidate(
                    solved=rotated,
                    actions=actions,
                    phase_counts=phase_counts,
                    pipeline_depth=depth,
                    shift=shift,
                    estimated_makespan=makespan,
                ),
            )
        )
    if not ranked:
        raise RuntimeError("no cycle boundary rotation produced a valid finite schedule")
    ranked.sort(key=lambda item: item[0])
    return tuple(item[1] for item in ranked[:2]), evaluated


def _rotate_cycle_boundary(
    instance: SchedulingInstance,
    ratio: tuple[int, ...],
    solved: _SolvedCycle,
    shift: int,
) -> _SolvedCycle:
    period = solved.period
    rotated_actions: list[Mapping[str, object]] = []
    for action in solved.actions:
        old_start = int(action["start"])
        duration = int(action["end"]) - old_start
        start = (old_start - shift) % period
        rotated_actions.append(
            MappingProxyType(
                {
                    **dict(action),
                    "start": start,
                    "end": start + duration,
                }
            )
        )
    by_key = {
        (
            str(action["route_id"]),
            int(action["token_index"]),
            str(action["action_type"]),
            int(action["edge_index"]),
        ): action
        for action in rotated_actions
    }
    transfer_wraps: dict[tuple[WaferKey, int], int] = {}
    process_wraps: dict[tuple[WaferKey, int], int] = {}
    for recipe, lane_count in zip(
        instance.recipes,
        ratio,
        strict=True,
    ):
        for lane in range(lane_count):
            token_key = (recipe.recipe_id, lane)
            for edge_index in range(len(recipe.steps) + 1):
                pick = by_key[recipe.recipe_id, lane, "pick", edge_index]
                place = by_key[recipe.recipe_id, lane, "place", edge_index]
                wrap = int(int(place["start"]) < int(pick["end"]))
                if int(place["start"]) + wrap * period < int(pick["end"]):
                    raise RuntimeError("rotated cycle breaks a transfer dependency")
                transfer_wraps[token_key, edge_index] = wrap
            for step_index, recipe_step in enumerate(recipe.steps, start=1):
                place = by_key[recipe.recipe_id, lane, "place", step_index - 1]
                pick = by_key[recipe.recipe_id, lane, "pick", step_index]
                required = int(place["end"]) + recipe_step.process_time
                wrap = int(int(pick["start"]) < required)
                if int(pick["start"]) + wrap * period < required:
                    raise RuntimeError("rotated cycle breaks a process dependency")
                process_wraps[token_key, step_index] = wrap
    return _SolvedCycle(
        status=solved.status,
        period=period,
        best_bound=solved.best_bound,
        runtime_seconds=solved.runtime_seconds,
        actions=tuple(rotated_actions),
        transfer_wraps=MappingProxyType(transfer_wraps),
        process_wraps=MappingProxyType(process_wraps),
    )


def _materialize_finite_schedule(
    instance: SchedulingInstance,
    ratio: tuple[int, ...],
    repeat_count: int,
    solved: _SolvedCycle,
) -> tuple[tuple[Mapping[str, object], ...], dict[str, int], int]:
    template_by_key = {
        (
            str(action["route_id"]),
            int(action["token_index"]),
            str(action["action_type"]),
            int(action["edge_index"]),
        ): action
        for action in solved.actions
    }
    raw_actions: list[dict[str, object]] = []
    maximum_period_offset = 0
    for recipe, lanes in zip(instance.recipes, ratio, strict=True):
        for introduction_period in range(repeat_count):
            for lane in range(lanes):
                wafer_key = (recipe.recipe_id, introduction_period * lanes + lane)
                edge_period = introduction_period
                for edge_index in range(len(recipe.steps) + 1):
                    pick = template_by_key[
                        recipe.recipe_id,
                        lane,
                        "pick",
                        edge_index,
                    ]
                    place = template_by_key[
                        recipe.recipe_id,
                        lane,
                        "place",
                        edge_index,
                    ]
                    pick_period = edge_period
                    place_period = edge_period + solved.transfer_wraps[
                        ((recipe.recipe_id, lane), edge_index)
                    ]
                    maximum_period_offset = max(
                        maximum_period_offset,
                        pick_period - introduction_period,
                        place_period - introduction_period,
                    )
                    raw_actions.append(
                        _finite_action(pick, wafer_key, pick_period, solved.period)
                    )
                    raw_actions.append(
                        _finite_action(place, wafer_key, place_period, solved.period)
                    )
                    if edge_index < len(recipe.steps):
                        edge_period = place_period + solved.process_wraps[
                            ((recipe.recipe_id, lane), edge_index + 1)
                        ]

    raw_actions.sort(
        key=lambda action: (
            int(action["start"]),
            int(action["end"]),
            int(action["template_node_id"]),
        )
    )
    free_arms = {
        robot_id: [
            f"arm{index}"
            for index in range(1 if robot.arm_kind.value == "single_arm" else 2)
        ]
        for robot_id, robot in instance.topology.robots.items()
    }
    wafer_arms: dict[tuple[str, WaferKey], str] = {}
    phase_counts = {"startup": 0, "steady": 0, "closedown": 0}
    decoded: list[Mapping[str, object]] = []
    for action in raw_actions:
        wafer_key = (str(action["route_id"]), int(action["wafer_index"]))
        robot_id = str(action["tm_id"])
        arm_key = (robot_id, wafer_key)
        if action["action_type"] == "pick":
            if not free_arms[robot_id]:
                raise RuntimeError("periodic composition exceeds Robot arm capacity")
            arm_id = free_arms[robot_id].pop(0)
            wafer_arms[arm_key] = arm_id
        else:
            try:
                arm_id = wafer_arms.pop(arm_key)
            except KeyError as exc:
                raise RuntimeError(
                    f"periodic Place occurs before Pick for wafer {wafer_key}"
                ) from exc
            free_arms[robot_id].append(arm_id)
            free_arms[robot_id].sort()

        occurrence_period = int(action["occurrence_period"])
        if occurrence_period < maximum_period_offset:
            phase = "startup"
        elif occurrence_period >= repeat_count:
            phase = "closedown"
        else:
            phase = "steady"
        phase_counts[phase] += 1
        decoded.append(
            MappingProxyType(
                {
                    "action_type": action["action_type"],
                    "tm_id": robot_id,
                    "module_id": action["module_id"],
                    "route_id": wafer_key[0],
                    "wafer_index": wafer_key[1],
                    "step_index": action["step_index"],
                    "arm_id": arm_id,
                    "start": action["start"],
                    "end": action["end"],
                    "periodic_phase": phase,
                }
            )
        )
    if wafer_arms:
        raise RuntimeError(f"periodic composition leaves wafers on Robot: {wafer_arms}")
    return tuple(decoded), phase_counts, maximum_period_offset


def _finite_action(
    template: Mapping[str, object],
    wafer_key: WaferKey,
    occurrence_period: int,
    period: int,
) -> dict[str, object]:
    return {
        "action_type": template["action_type"],
        "module_id": template["module_id"],
        "tm_id": template["tm_id"],
        "route_id": wafer_key[0],
        "wafer_index": wafer_key[1],
        "step_index": template["step_index"],
        "start": int(template["start"]) + occurrence_period * period,
        "end": int(template["end"]) + occurrence_period * period,
        "occurrence_period": occurrence_period,
        "template_node_id": template["node_id"],
    }


def _boolean_times_period(
    model: cp_model.CpModel,
    boolean: cp_model.IntVar,
    period: cp_model.IntVar,
    upper_bound: int,
    prefix: str,
) -> cp_model.IntVar:
    product = model.new_int_var(0, upper_bound, f"{prefix}_wrap_time")
    model.add_multiplication_equality(product, [boolean, period])
    return product


def _repeat_count(instance: SchedulingInstance, ratio: tuple[int, ...]) -> int:
    counts_by_recipe = {item.recipe_id: item.wafer_count for item in instance.workload}
    factors = {
        counts_by_recipe[recipe.recipe_id] // part
        for recipe, part in zip(instance.recipes, ratio, strict=True)
    }
    if len(factors) != 1:
        raise ValueError("workload is not an exact multiple of its periodic ratio")
    return next(iter(factors))
