"""Direct non-cyclic CP-SAT solver for atmospheric Cluster Tools.

The formulation borrows the circuit encoding used by the reference CP-SAT
projects, but keeps the constraints aligned with ``ValidatorSuite``:

* every transfer chooses one Robot that reaches both endpoint Modules;
* one optional-node circuit orders the actions assigned to each Robot;
* candidate PM selection is a decision variable shared by Place and Pick;
* a PM is occupied from Place.start through the matching Pick.end;
* a Robot holds a wafer from Pick.start through Place.end, with capacity one or
  two according to its arm type;
* sequence-dependent travel is charged only when consecutive Robot actions are
  at different Modules.

Cleaning, JIT/residency, Load Locks, and random processing times are outside
this solver's current contract.  Robot changes are possible only at explicit
BUFFER visits because Pick and Place of one transfer share the same Robot.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ortools.sat.python import cp_model

from cluster_toolkit.cluster_generator.heuristic import build_safe_reference_schedule
from cluster_toolkit.cluster_generator.pipeline_models import SchedulingInstance
from cluster_toolkit.cluster_generator.problem_adapter import to_cluster_problem
from cluster_toolkit.problem import (
    ClusterProblem,
    ModuleLocation,
    ModuleType,
    TMArmType,
    WaferKey,
)
from cluster_toolkit.validator import ValidatorSuite


SOLVER_NAME = "cpsat_direct"
SOLVER_VERSION = "0.2.0"


class FeasibilityConsistencyError(RuntimeError):
    """A solver model contradicted the generator's validated feasibility witness."""


@dataclass(frozen=True, slots=True)
class CpSatResult:
    """One direct CP-SAT attempt, including proof and validation information."""

    status: str
    actions: tuple[Mapping[str, object], ...]
    makespan: int | None
    best_bound: int | None
    runtime_seconds: float
    validation_ok: bool | None
    solver_name: str = SOLVER_NAME
    solver_version: str = SOLVER_VERSION


@dataclass(frozen=True, slots=True)
class _Action:
    node_id: int
    action_type: str
    wafer_key: WaferKey
    step_index: int
    duration: cp_model.IntVar
    location: cp_model.IntVar
    assigned: Mapping[str, cp_model.IntVar]
    start: cp_model.IntVar
    end: cp_model.IntVar


@dataclass(frozen=True, slots=True)
class _Visit:
    wafer_key: WaferKey
    step_index: int
    location: cp_model.IntVar
    selected: Mapping[str, cp_model.IntVar]
    place: _Action
    pick: _Action


def solve_instance(
    instance: SchedulingInstance,
    *,
    time_limit_seconds: float = 1800,
    random_seed: int = 0,
    num_search_workers: int = 1,
) -> CpSatResult:
    """Solve one canonical instance after the minimal execution-model adapter."""

    return solve(
        to_cluster_problem(instance),
        time_limit_seconds=time_limit_seconds,
        random_seed=random_seed,
        num_search_workers=num_search_workers,
    )


def solve(
    problem: ClusterProblem,
    *,
    time_limit_seconds: float = 1800,
    random_seed: int = 0,
    num_search_workers: int = 1,
) -> CpSatResult:
    """Minimize makespan for a direct atmospheric problem.

    ``OPTIMAL`` is a proof for the complete supported problem, not merely for a
    cycle component.  Every feasible solution is checked by ``ValidatorSuite``
    before it is returned.
    """

    _require_positive_number("time_limit_seconds", time_limit_seconds)
    _require_non_negative_integer("random_seed", random_seed)
    _require_positive_integer("num_search_workers", num_search_workers)
    robot_ids = _require_supported_problem(problem)

    reference = build_safe_reference_schedule(problem)
    horizon = _as_integer_time(reference.makespan, "reference makespan")
    model = cp_model.CpModel()
    module_ids = tuple(sorted(problem.Modules))
    module_index = {module_id: index for index, module_id in enumerate(module_ids)}

    actions: list[_Action] = []
    visits: list[_Visit] = []
    transfers: list[tuple[_Action, _Action]] = []
    final_places: list[_Action] = []

    def new_action(
        action_type: str,
        wafer_key: WaferKey,
        step_index: int,
        duration_by_robot: Mapping[str, int],
        location: cp_model.IntVar,
        assigned: Mapping[str, cp_model.IntVar],
    ) -> _Action:
        node_id = len(actions)
        start = model.new_int_var(0, horizon, f"a{node_id}_start")
        end = model.new_int_var(0, horizon, f"a{node_id}_end")
        minimum_duration = min(duration_by_robot.values())
        maximum_duration = max(duration_by_robot.values())
        duration = model.new_int_var(
            minimum_duration,
            maximum_duration,
            f"a{node_id}_duration",
        )
        model.add(
            duration
            == sum(
                duration_by_robot[robot_id] * assigned[robot_id]
                for robot_id in robot_ids
            )
        )
        model.add(end == start + duration)
        action = _Action(
            node_id=node_id,
            action_type=action_type,
            wafer_key=wafer_key,
            step_index=step_index,
            duration=duration,
            location=location,
            assigned=assigned,
            start=start,
            end=end,
        )
        actions.append(action)
        return action

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

    initial_wafers = problem.initial_state.to_snapshot().wafers_by_key
    for wafer_key in sorted(initial_wafers):
        initial_wafer = initial_wafers[wafer_key]
        route = problem.routes[initial_wafer.route_id]
        source_module_id = initial_wafer.location.module_id
        source_location = model.new_constant(module_index[source_module_id])
        previous_location = source_location
        previous_place: _Action | None = None
        wafer_picks: list[_Action] = []
        visit_drafts: list[
            tuple[
                int,
                cp_model.IntVar,
                Mapping[str, cp_model.IntVar],
                _Action,
            ]
        ] = []

        for route_offset, route_visit in enumerate(route.visits, start=1):
            candidate_indices = [module_index[item] for item in route_visit.module_ids]
            location = model.new_int_var_from_domain(
                cp_model.Domain.from_values(candidate_indices),
                f"w{wafer_key[0]}_{wafer_key[1]}_s{route_offset}_location",
            )
            selected: dict[str, cp_model.IntVar] = {}
            for module_id in route_visit.module_ids:
                choice = model.new_bool_var(
                    f"w{wafer_key[0]}_{wafer_key[1]}_s{route_offset}_{module_id}"
                )
                model.add(location == module_index[module_id]).only_enforce_if(choice)
                selected[module_id] = choice
            model.add_exactly_one(selected.values())

            assigned = _new_transfer_assignment(
                model,
                problem,
                previous_location,
                location,
                module_ids,
                robot_ids,
                prefix=(
                    f"w{wafer_key[0]}_{wafer_key[1]}_e{route_offset - 1}"
                ),
            )

            pick = new_action(
                "pick",
                wafer_key,
                route_offset - 1,
                pick_durations,
                previous_location,
                assigned,
            )
            wafer_picks.append(pick)
            place = new_action(
                "place",
                wafer_key,
                route_offset,
                place_durations,
                location,
                assigned,
            )
            model.add(place.start >= pick.end)
            transfers.append((pick, place))

            if previous_place is not None:
                previous_process_time = _as_integer_time(
                    route.visits[route_offset - 2].process_time or 0,
                    "Recipe process_time",
                )
                model.add(pick.start >= previous_place.end + previous_process_time)

            visit_drafts.append(
                (
                    route_offset,
                    location,
                    MappingProxyType(selected),
                    place,
                )
            )
            previous_location = location
            previous_place = place

        final_pick = new_action(
            "pick",
            wafer_key,
            len(route.visits),
            pick_durations,
            previous_location,
            _new_transfer_assignment(
                model,
                problem,
                previous_location,
                model.new_constant(module_index[problem.return_module_id(initial_wafer)]),
                module_ids,
                robot_ids,
                prefix=f"w{wafer_key[0]}_{wafer_key[1]}_e{len(route.visits)}",
            ),
        )
        wafer_picks.append(final_pick)
        assert previous_place is not None
        model.add(
            final_pick.start
            >= previous_place.end
            + _as_integer_time(route.visits[-1].process_time or 0, "Recipe process_time")
        )
        return_module_id = problem.return_module_id(initial_wafer)
        final_place = new_action(
            "place",
            wafer_key,
            len(route.visits) + 1,
            place_durations,
            model.new_constant(module_index[return_module_id]),
            final_pick.assigned,
        )
        model.add(final_place.start >= final_pick.end)
        transfers.append((final_pick, final_place))
        final_places.append(final_place)

        # A PM remains occupied from the Place into that visit until the next
        # Pick has completed.  The incoming Pick belongs to the previous
        # Module and therefore must not close this occupancy interval.
        for draft, outgoing_pick in zip(
            visit_drafts,
            wafer_picks[1:],
            strict=True,
        ):
            (
                route_offset,
                location,
                selected,
                place,
            ) = draft
            visits.append(
                _Visit(
                    wafer_key=wafer_key,
                    step_index=route_offset,
                    location=location,
                    selected=selected,
                    place=place,
                    pick=outgoing_pick,
                )
            )

    _add_module_capacity_constraints(model, problem, visits, horizon)
    _add_robot_holding_capacity(model, problem, transfers, horizon)
    circuit_arcs = _add_robot_circuits(
        model,
        actions,
        problem,
    )

    makespan = model.new_int_var(0, horizon, "makespan")
    for action in actions:
        model.add(action.end <= makespan)
    for action in final_places:
        model.add(makespan >= action.end)
    model.minimize(makespan)

    _add_reference_hints(
        model,
        reference.actions,
        actions,
        visits,
        circuit_arcs,
        makespan,
        horizon,
        module_index,
    )
    model_error = model.validate()
    if model_error:
        raise RuntimeError(f"invalid direct CP-SAT model: {model_error}")

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
                "fixed direct hint contradicted a known feasible serial witness: "
                f"{incumbent_solver.status_name(incumbent_status)}"
            )
        runtime_seconds = time.monotonic() - started
        return CpSatResult(
            status=incumbent_solver.status_name(incumbent_status),
            actions=(),
            makespan=None,
            best_bound=None,
            runtime_seconds=runtime_seconds,
            validation_ok=None,
        )

    incumbent_runtime = time.monotonic() - started
    remaining_seconds = max(0.001, float(time_limit_seconds) - incumbent_runtime)
    solver = _new_solver(
        time_limit_seconds=remaining_seconds,
        random_seed=random_seed,
        num_search_workers=num_search_workers,
        fix_hints=False,
    )
    status_code = solver.solve(model)
    runtime_seconds = time.monotonic() - started
    if status_code in (cp_model.MODEL_INVALID, cp_model.INFEASIBLE):
        raise FeasibilityConsistencyError(
            "full CP-SAT model rejected a known feasible incumbent: "
            f"{solver.status_name(status_code)}"
        )
    has_full_solution = status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    solution_solver = solver if has_full_solution else incumbent_solver
    status = solver.status_name(status_code) if has_full_solution else "FEASIBLE"

    result_actions = _decode_actions(
        solution_solver,
        actions,
        circuit_arcs,
        module_ids=module_ids,
        problem=problem,
    )
    report = ValidatorSuite(problem).validate(
        result_actions,
        require_complete=True,
        exact_action_durations=True,
    )
    if not report.ok:
        details = "; ".join(issue.message for issue in report.issues[:5])
        raise RuntimeError(f"CP-SAT produced an invalid schedule: {details}")
    expected_action_count = 2 * sum(
        len(problem.routes[wafer.route_id].visits) + 1
        for wafer in initial_wafers.values()
    )
    if len(result_actions) != expected_action_count:
        raise RuntimeError(
            "CP-SAT decoded an incomplete schedule: "
            f"expected {expected_action_count} actions, got {len(result_actions)}"
        )

    return CpSatResult(
        status=status,
        actions=result_actions,
        makespan=int(solution_solver.value(makespan)),
        best_bound=(
            int(solution_solver.value(makespan))
            if status_code == cp_model.OPTIMAL
            else _best_integer_bound(solver.best_objective_bound)
        ),
        runtime_seconds=runtime_seconds,
        validation_ok=True,
    )


def _new_solver(
    *,
    time_limit_seconds: float,
    random_seed: int,
    num_search_workers: int,
    fix_hints: bool,
) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.random_seed = random_seed
    solver.parameters.num_search_workers = num_search_workers
    solver.parameters.fix_variables_to_their_hinted_value = fix_hints
    return solver


def _add_module_capacity_constraints(
    model: cp_model.CpModel,
    problem: ClusterProblem,
    visits: list[_Visit],
    horizon: int,
) -> None:
    intervals_by_module: dict[str, list[cp_model.IntervalVar]] = {
        module_id: []
        for module_id, module in problem.Modules.items()
        if module.type not in {ModuleType.IO, ModuleType.LP}
    }
    for visit in visits:
        occupancy_size = model.new_int_var(
            0,
            horizon,
            f"w{visit.wafer_key[0]}_{visit.wafer_key[1]}_s{visit.step_index}_occupancy",
        )
        model.add(occupancy_size == visit.pick.end - visit.place.start)
        for module_id, selected in visit.selected.items():
            interval = model.new_optional_interval_var(
                visit.place.start,
                occupancy_size,
                visit.pick.end,
                selected,
                f"{module_id}_w{visit.wafer_key[0]}_{visit.wafer_key[1]}_s{visit.step_index}",
            )
            intervals_by_module[module_id].append(interval)

    for module_id, intervals in intervals_by_module.items():
        capacity = problem.Modules[module_id].capacity
        if capacity == 1:
            model.add_no_overlap(intervals)
        else:
            model.add_cumulative(intervals, [1] * len(intervals), capacity)


def _add_robot_holding_capacity(
    model: cp_model.CpModel,
    problem: ClusterProblem,
    transfers: list[tuple[_Action, _Action]],
    horizon: int,
) -> None:
    for robot_id, robot in problem.ClusterTool.items():
        holding_intervals: list[cp_model.IntervalVar] = []
        for pick, place in transfers:
            size = model.new_int_var(0, horizon, f"transfer_{pick.node_id}_holding")
            model.add(size == place.end - pick.start)
            holding_intervals.append(
                model.new_optional_interval_var(
                    pick.start,
                    size,
                    place.end,
                    pick.assigned[robot_id],
                    f"{robot_id}_transfer_{pick.node_id}",
                )
            )
        capacity = 1 if robot.arm_type is TMArmType.SINGLE_ARM else 2
        model.add_cumulative(
            holding_intervals,
            [1] * len(holding_intervals),
            capacity,
        )


def _add_robot_circuits(
    model: cp_model.CpModel,
    actions: list[_Action],
    problem: ClusterProblem,
) -> dict[tuple[str, int, int], cp_model.IntVar]:
    node_count = len(actions)
    depot = node_count
    arc_vars: dict[tuple[str, int, int], cp_model.IntVar] = {}

    for robot_id, robot in problem.ClusterTool.items():
        travel_time = _as_integer_time(
            robot.travel_times,
            f"Robot {robot_id} travel_time",
        )
        arcs: list[tuple[int, int, cp_model.IntVar]] = []
        used = model.new_bool_var(f"{robot_id}_used")
        model.add_max_equality(
            used,
            [action.assigned[robot_id] for action in actions],
        )
        arcs.append((depot, depot, used.negated()))
        for action in actions:
            arcs.append(
                (action.node_id, action.node_id, action.assigned[robot_id].negated())
            )

        for left in range(node_count + 1):
            for right in range(node_count + 1):
                if left == right:
                    continue
                arc = model.new_bool_var(f"{robot_id}_arc_{left}_{right}")
                arcs.append((left, right, arc))
                arc_vars[robot_id, left, right] = arc
                if left != depot:
                    model.add(arc <= actions[left].assigned[robot_id])
                if right != depot:
                    model.add(arc <= actions[right].assigned[robot_id])
                if left == depot or right == depot:
                    continue

                left_action = actions[left]
                right_action = actions[right]
                model.add(right_action.start >= left_action.end).only_enforce_if(arc)
                if travel_time <= 0:
                    continue

                left_domain = set(_domain_values(left_action.location))
                right_domain = set(_domain_values(right_action.location))
                common = left_domain & right_domain
                if not common:
                    model.add(
                        right_action.start >= left_action.end + travel_time
                    ).only_enforce_if(arc)
                elif len(left_domain) == len(right_domain) == 1:
                    continue
                else:
                    same_location = model.new_bool_var(
                        f"{robot_id}_same_location_{left}_{right}"
                    )
                    model.add(
                        left_action.location == right_action.location
                    ).only_enforce_if(same_location)
                    model.add(
                        left_action.location != right_action.location
                    ).only_enforce_if(same_location.negated())
                    model.add(
                        right_action.start
                        >= left_action.end + travel_time * (1 - same_location)
                    ).only_enforce_if(arc)

        model.add_circuit(arcs)
    return arc_vars


def _new_transfer_assignment(
    model: cp_model.CpModel,
    problem: ClusterProblem,
    source: cp_model.IntVar,
    target: cp_model.IntVar,
    module_ids: tuple[str, ...],
    robot_ids: tuple[str, ...],
    *,
    prefix: str,
) -> Mapping[str, cp_model.IntVar]:
    source_domain = set(_domain_values(source))
    target_domain = set(_domain_values(target))
    assigned: dict[str, cp_model.IntVar] = {}
    for robot_id in robot_ids:
        choice = model.new_bool_var(f"{prefix}_{robot_id}")
        assigned[robot_id] = choice
        reachable = set(problem.ClusterTool[robot_id].module_ids)
        allowed_pairs = [
            (source_index, target_index)
            for source_index in source_domain
            for target_index in target_domain
            if module_ids[source_index] in reachable
            and module_ids[target_index] in reachable
        ]
        if allowed_pairs:
            model.add_allowed_assignments(
                [source, target],
                allowed_pairs,
            ).only_enforce_if(choice)
        else:
            model.add(choice == 0)
    model.add_exactly_one(assigned.values())
    return MappingProxyType(assigned)


def _add_reference_hints(
    model: cp_model.CpModel,
    reference_actions: tuple[dict[str, object], ...],
    actions: list[_Action],
    visits: list[_Visit],
    circuit_arcs: Mapping[tuple[str, int, int], cp_model.IntVar],
    makespan: cp_model.IntVar,
    horizon: int,
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
    action_order: list[tuple[int, int, int]] = []
    hinted_assignments: set[int] = set()
    for action in actions:
        reference = by_key[
            action.wafer_key[0],
            action.wafer_key[1],
            action.action_type,
            action.step_index,
        ]
        start = _as_integer_time(reference["start"], "reference action start")
        end = _as_integer_time(reference["end"], "reference action end")
        model.add_hint(action.start, start)
        model.add_hint(action.end, end)
        reference_robot = str(reference["tm_id"])
        for robot_id, assigned in action.assigned.items():
            if assigned.index not in hinted_assignments:
                model.add_hint(assigned, int(robot_id == reference_robot))
                hinted_assignments.add(assigned.index)
        action_order.append((start, end, action.node_id))

    for visit in visits:
        reference = by_key[
            visit.wafer_key[0],
            visit.wafer_key[1],
            "place",
            visit.step_index,
        ]
        module_id = str(reference["module_id"])
        model.add_hint(visit.location, module_index[module_id])
        for candidate, selected in visit.selected.items():
            model.add_hint(selected, int(candidate == module_id))

    depot = len(actions)
    for robot_id in sorted({key[0] for key in circuit_arcs}):
        ordered_nodes = [
            node_id
            for _, _, node_id in sorted(action_order)
            if str(
                by_key[
                    actions[node_id].wafer_key[0],
                    actions[node_id].wafer_key[1],
                    actions[node_id].action_type,
                    actions[node_id].step_index,
                ]["tm_id"]
            )
            == robot_id
        ]
        selected_set = set(
            zip(
                [depot, *ordered_nodes],
                [*ordered_nodes, depot],
                strict=True,
            )
        ) if ordered_nodes else set()
        for (arc_robot, left, right), arc_var in circuit_arcs.items():
            if arc_robot == robot_id:
                model.add_hint(arc_var, int((left, right) in selected_set))
    model.add_hint(makespan, horizon)


def _decode_actions(
    solver: cp_model.CpSolver,
    actions: list[_Action],
    circuit_arcs: Mapping[tuple[str, int, int], cp_model.IntVar],
    *,
    module_ids: tuple[str, ...],
    problem: ClusterProblem,
) -> tuple[Mapping[str, object], ...]:
    depot = len(actions)
    decoded: list[Mapping[str, object]] = []
    decoded_nodes: set[int] = set()
    for robot_id, robot in sorted(problem.ClusterTool.items()):
        successor = {
            left: right
            for (arc_robot, left, right), variable in circuit_arcs.items()
            if arc_robot == robot_id and solver.value(variable)
        }
        ordered: list[_Action] = []
        current = successor.get(depot)
        while current is not None and current != depot and len(ordered) < len(actions):
            ordered.append(actions[current])
            decoded_nodes.add(current)
            current = successor.get(current)
        assigned_count = sum(
            solver.value(action.assigned[robot_id]) for action in actions
        )
        if len(ordered) != assigned_count or (ordered and current != depot):
            raise RuntimeError(
                f"decoded Robot {robot_id} circuit does not contain its assigned actions"
            )

        arm_capacity = 1 if robot.arm_type is TMArmType.SINGLE_ARM else 2
        available_arms = [f"arm{index}" for index in range(arm_capacity)]
        wafer_arms: dict[WaferKey, str] = {}
        for action in ordered:
            if action.action_type == "pick":
                if not available_arms:
                    raise RuntimeError(
                        "decoded Robot arm capacity is inconsistent with CP-SAT"
                    )
                arm_id = available_arms.pop(0)
                wafer_arms[action.wafer_key] = arm_id
            else:
                arm_id = wafer_arms.pop(action.wafer_key)
                available_arms.append(arm_id)
                available_arms.sort()
            decoded.append(
                MappingProxyType(
                    {
                        "action_type": action.action_type,
                        "tm_id": robot_id,
                        "module_id": module_ids[solver.value(action.location)],
                        "route_id": action.wafer_key[0],
                        "wafer_index": action.wafer_key[1],
                        "step_index": action.step_index,
                        "arm_id": arm_id,
                        "start": int(solver.value(action.start)),
                        "end": int(solver.value(action.end)),
                    }
                )
            )
        if wafer_arms:
            raise RuntimeError(
                f"decoded wafers remain on Robot {robot_id}: {sorted(wafer_arms)}"
            )
    if decoded_nodes != set(range(len(actions))):
        raise RuntimeError("decoded Robot circuits do not cover every action exactly once")
    decoded.sort(
        key=lambda action: (
            int(action["start"]),
            int(action["end"]),
            str(action["tm_id"]),
            str(action["route_id"]),
            int(action["wafer_index"]),
            str(action["action_type"]),
        )
    )
    return tuple(decoded)


def _require_supported_problem(problem: ClusterProblem) -> tuple[str, ...]:
    if problem.just_in_time is not None or problem.cleaning is not None:
        raise NotImplementedError("direct CP-SAT does not support JIT or cleaning constraints")
    unsupported_modules = sorted(
        module_id
        for module_id, module in problem.Modules.items()
        if module.type not in {
            ModuleType.IO,
            ModuleType.PM,
            ModuleType.AL,
            ModuleType.BUFFER,
        }
    )
    if unsupported_modules:
        raise NotImplementedError(
            "direct CP-SAT supports atmospheric Modules only; unsupported Modules: "
            f"{unsupported_modules}"
        )
    if len(problem.io_module_ids) != 1:
        raise NotImplementedError("direct CP-SAT requires exactly one virtual IO")
    robot_ids = tuple(sorted(problem.ClusterTool))

    snapshot = problem.initial_state.to_snapshot()
    if any(
        snapshot.tm_positions.get(robot_id) is not None
        or snapshot.tm_arms.get(robot_id)
        for robot_id in robot_ids
    ):
        raise NotImplementedError(
            "direct CP-SAT requires empty Robots with no initial positions"
        )
    source_id = problem.io_module_ids[0]
    for wafer in snapshot.wafers_by_key.values():
        if not isinstance(wafer.location, ModuleLocation):
            raise NotImplementedError("all wafers must initially be in the virtual IO")
        if wafer.location.module_id != source_id or wafer.step_index != 0:
            raise NotImplementedError("all wafers must start at step 0 in the virtual IO")

    for robot_id, robot in problem.ClusterTool.items():
        for label, value in (
            (f"Robot {robot_id} pick_time", robot.pick_time),
            (f"Robot {robot_id} place_time", robot.place_time),
            (f"Robot {robot_id} travel_time", robot.travel_times),
        ):
            _as_integer_time(value, label)
    for route in problem.routes.values():
        for visit in route.visits:
            _as_integer_time(visit.process_time or 0, "Recipe process_time")
        layers = [
            (source_id,),
            *(visit.module_ids for visit in route.visits),
            (source_id,),
        ]
        for sources, targets in zip(layers, layers[1:]):
            for source in sources:
                for target in targets:
                    if not any(
                        source in robot.module_ids and target in robot.module_ids
                        for robot in problem.ClusterTool.values()
                    ):
                        raise NotImplementedError(
                            f"no Robot can execute transfer {source} -> {target}"
                        )
    return robot_ids


def _domain_values(variable: cp_model.IntVar) -> tuple[int, ...]:
    domain = list(variable.proto.domain)
    values: list[int] = []
    for start, end in zip(domain[::2], domain[1::2], strict=True):
        values.extend(range(start, end + 1))
    return tuple(values)


def _best_integer_bound(value: float) -> int | None:
    if not math.isfinite(value):
        return None
    return max(0, int(math.ceil(value - 1e-9)))


def _as_integer_time(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{label} must be a non-negative integer")
    return int(numeric)


def _require_positive_number(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")


def _require_non_negative_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_positive_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
