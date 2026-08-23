from __future__ import annotations

from dataclasses import dataclass

from cluster_toolkit.cluster_engine import ADVANCE, ClusterEngine, PickAction, PlaceAction
from cluster_toolkit.problem import ClusterProblem, ModuleType, WaferKey
from cluster_toolkit.validator import ValidatorSuite


@dataclass(frozen=True, slots=True)
class HeuristicResult:
    actions: tuple[dict[str, object], ...]
    makespan: float
    lower_bound: float
    assignments: dict[tuple[WaferKey, int], str]
    pm_loads: dict[str, float]
    average_legal_actions: float
    multiple_choice_state_ratio: float


def build_safe_reference_schedule(problem: ClusterProblem) -> HeuristicResult:
    """Complete wafers serially through any connected multi-robot topology.

    This is deliberately a feasibility witness, not a competitive scheduling
    policy. Serial completion prevents resource cycles while still exercising
    Load Lock conversion, Buffer handoff, candidate PMs, and every Robot type.
    """

    wafers = sorted(
        problem.initial_state.wafers,
        key=lambda wafer: (wafer.priority, wafer.route_id, wafer.wafer_index),
    )
    assignments: dict[tuple[WaferKey, int], str] = {}
    pm_loads = {
        module_id: 0.0
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.PM
    }
    engine = ClusterEngine(problem)
    state = engine.reset()
    actions: list[dict[str, object]] = []
    legal_action_counts: list[int] = []

    for initial_wafer in wafers:
        key = initial_wafer.wafer_key
        route = problem.routes[initial_wafer.route_id]
        visits = list(enumerate(route.visits, start=1))
        visits.append((len(route.visits) + 1, None))

        for step_index, visit in visits:
            runtime_wafer = state.wafers[key]
            current_module = runtime_wafer.module_id
            if current_module is None:
                raise RuntimeError(f"wafer {key!r} is not in a Module before step {step_index}")

            if visit is None:
                candidates = (problem.return_module_id(initial_wafer),)
            else:
                candidates = visit.module_ids
            target_module, robot_id = _select_transfer(
                problem,
                current_module,
                candidates,
                pm_loads,
            )
            if visit is not None and problem.Modules[target_module].type is ModuleType.PM:
                assignments[(key, step_index)] = target_module
                pm_loads[target_module] += float(visit.process_time or 0.0)

            pick = PickAction(robot_id=robot_id, wafer_key=key)
            _advance_for_action(engine, pick)
            legal_action_counts.append(len(engine.available_actions()))
            pick_record = engine.step(pick)
            assert pick_record is not None
            actions.append(pick_record.to_dict())
            while state.wafers[key].robot_id is None:
                if ADVANCE not in engine.available_actions():
                    raise RuntimeError(f"reference scheduler cannot finish Pick for {key!r}")
                engine.step(ADVANCE)

            place = PlaceAction(wafer_key=key, target_module_id=target_module)
            _advance_for_action(engine, place)
            legal_action_counts.append(len(engine.available_actions()))
            place_record = engine.step(place)
            assert place_record is not None
            actions.append(place_record.to_dict())
            while state.wafers[key].module_id != target_module:
                if ADVANCE not in engine.available_actions():
                    raise RuntimeError(f"reference scheduler cannot finish Place for {key!r}")
                engine.step(ADVANCE)

    if not engine.is_complete():
        raise RuntimeError("safe reference scheduler did not complete the problem")
    validator_report = ValidatorSuite(problem).validate(
        actions,
        require_complete=True,
        exact_action_durations=True,
    )
    if not validator_report.ok:
        raise RuntimeError(
            f"ValidatorSuite rejected safe reference schedule: {validator_report.issues}"
        )

    robot_work: dict[str, float] = {robot_id: 0.0 for robot_id in problem.ClusterTool}
    for action in actions:
        robot_work[str(action["tm_id"])] += float(action["end"]) - float(action["start"])
    longest_process = max(
        (
            float(visit.process_time or 0.0)
            for route in problem.routes.values()
            for visit in route.visits
        ),
        default=0.0,
    )
    lower_bound = max(
        longest_process,
        max(pm_loads.values(), default=0.0),
        max(robot_work.values(), default=0.0),
    )
    multiple_choice_count = sum(count > 1 for count in legal_action_counts)
    return HeuristicResult(
        actions=tuple(actions),
        makespan=float(state.time),
        lower_bound=float(lower_bound),
        assignments=assignments,
        pm_loads=pm_loads,
        average_legal_actions=(
            sum(legal_action_counts) / len(legal_action_counts)
            if legal_action_counts
            else 0.0
        ),
        multiple_choice_state_ratio=(
            multiple_choice_count / len(legal_action_counts)
            if legal_action_counts
            else 0.0
        ),
    )


def _select_transfer(
    problem: ClusterProblem,
    current_module: str,
    candidates: tuple[str, ...],
    pm_loads: dict[str, float],
) -> tuple[str, str]:
    options: list[tuple[float, str, str]] = []
    for target_module in candidates:
        for robot_id, robot in problem.ClusterTool.items():
            if current_module not in robot.module_ids or target_module not in robot.module_ids:
                continue
            load = (
                pm_loads[target_module]
                if problem.Modules[target_module].type is ModuleType.PM
                else 0.0
            )
            options.append((load, target_module, robot_id))
    if not options:
        raise RuntimeError(
            f"no Robot can transfer from {current_module} to any of {list(candidates)}"
        )
    _, target_module, robot_id = min(options)
    return target_module, robot_id


def _advance_for_action(
    engine: ClusterEngine,
    action: PickAction | PlaceAction,
) -> None:
    for _ in range(100_000):
        if action in engine.available_actions():
            return
        if ADVANCE not in engine.available_actions():
            raise RuntimeError(f"safe heuristic cannot dispatch action: {action!r}")
        engine.step(ADVANCE)
    raise RuntimeError(f"safe heuristic timed out waiting for action: {action!r}")
