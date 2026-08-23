"""Greedy rolling-horizon branch search for cluster-tool scheduling.

This module ports the decision rule from the standalone
``branch-and-search`` project onto the repository's event engine.  At every
decision point it enumerates feasible action sequences up to a fixed horizon,
selects the leaf with the smallest elapsed time, commits only the first
action, and then replans.

The event engine remains the source of truth for Pick/Place timing and state
transitions.  The completed action sequence is independently checked with
``ValidatorSuite`` before it is returned.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from cluster_rl.action_mask import ActionSafetyFilter
from cluster_toolkit.cluster_engine import (
    ADVANCE,
    ClusterEngine,
    EngineAction,
    PickAction,
    PlaceAction,
)
from cluster_toolkit.cluster_generator.pipeline_models import SchedulingInstance
from cluster_toolkit.cluster_generator.problem_adapter import to_cluster_problem
from cluster_toolkit.problem import ClusterProblem
from cluster_toolkit.validator import ValidatorSuite


SOLVER_NAME = "branch_search"
SOLVER_VERSION = "0.1.1"


class BranchSearchExhaustedError(RuntimeError):
    """The bounded policy search ended normally without a safe incumbent."""


@dataclass(frozen=True, slots=True)
class BranchSearchResult:
    """One validated schedule produced by rolling-horizon branch search."""

    actions: tuple[Mapping[str, object], ...]
    makespan: float
    planning_horizon: int
    iterations: int
    nodes_expanded: int
    runtime_seconds: float
    validation_ok: bool = True
    solver_name: str = SOLVER_NAME
    solver_version: str = SOLVER_VERSION


@dataclass(frozen=True, slots=True)
class _SearchChoice:
    first_action: EngineAction
    nodes_expanded: int


def solve_instance(
    instance: SchedulingInstance,
    *,
    planning_horizon: int = 5,
    safety_lookahead_depth: int = 2,
    time_limit_seconds: float | None = None,
) -> BranchSearchResult:
    """Solve one canonical generated instance."""

    return solve(
        to_cluster_problem(instance),
        planning_horizon=planning_horizon,
        safety_lookahead_depth=safety_lookahead_depth,
        time_limit_seconds=time_limit_seconds,
    )


def solve(
    problem: ClusterProblem,
    *,
    planning_horizon: int = 5,
    safety_lookahead_depth: int = 2,
    time_limit_seconds: float | None = None,
) -> BranchSearchResult:
    """Build a valid schedule with greedy rolling-horizon branch search."""

    _require_integer("planning_horizon", planning_horizon, minimum=1)
    _require_integer(
        "safety_lookahead_depth",
        safety_lookahead_depth,
        minimum=0,
    )
    if time_limit_seconds is not None:
        if (
            isinstance(time_limit_seconds, bool)
            or not isinstance(time_limit_seconds, (int, float))
            or time_limit_seconds <= 0
            or not math.isfinite(float(time_limit_seconds))
        ):
            raise ValueError("time_limit_seconds must be a positive number")

    started = time.monotonic()
    deadline = (
        None
        if time_limit_seconds is None
        else started + float(time_limit_seconds)
    )
    engine = ClusterEngine(problem)
    engine.reset()
    safety_filter = ActionSafetyFilter(
        problem,
        lookahead_depth=safety_lookahead_depth,
    )
    actions: list[Mapping[str, object]] = []
    iterations = 0
    nodes_expanded = 0

    if not _advance_until_decision(engine, safety_filter):
        raise BranchSearchExhaustedError(
            "initial state has no safe feasible continuation"
        )

    while not engine.is_complete():
        _check_deadline(deadline)
        choice = _choose_first_action(
            engine,
            safety_filter,
            planning_horizon=planning_horizon,
            deadline=deadline,
        )
        nodes_expanded += choice.nodes_expanded
        iterations += 1

        record = engine.step(choice.first_action)
        if record is None:
            raise RuntimeError("branch search selected an internal Advance action")
        actions.append(MappingProxyType(record.to_dict()))
        if not _advance_until_decision(engine, safety_filter):
            raise BranchSearchExhaustedError(
                "branch search reached a deadlock before completing all wafers"
            )

    runtime_seconds = time.monotonic() - started
    result_actions = tuple(actions)
    report = ValidatorSuite(problem).validate(
        result_actions,
        require_complete=True,
        exact_action_durations=True,
    )
    if not report.ok:
        details = "; ".join(issue.message for issue in report.issues[:5])
        raise RuntimeError(f"branch search produced an invalid schedule: {details}")

    return BranchSearchResult(
        actions=result_actions,
        makespan=float(engine.state.time),
        planning_horizon=planning_horizon,
        iterations=iterations,
        nodes_expanded=nodes_expanded,
        runtime_seconds=runtime_seconds,
    )


def _choose_first_action(
    root: ClusterEngine,
    safety_filter: ActionSafetyFilter,
    *,
    planning_horizon: int,
    deadline: float | None,
) -> _SearchChoice:
    """Enumerate one horizon and return the first action of its best path."""

    # Each stack item is already advanced to either completion or the next
    # state with at least one non-Advance action.  Event advancement is free,
    # matching the standalone scheduler's event-driven joint-action step.
    stack: list[
        tuple[ClusterEngine, int, tuple[EngineAction, ...]]
    ] = [(ActionSafetyFilter.fork_engine(root), 0, ())]
    best_score: tuple[float, tuple[tuple[object, ...], ...]] | None = None
    best_first_action: EngineAction | None = None
    nodes_expanded = 0
    seen: dict[
        tuple[int, tuple[object, ...]],
        tuple[tuple[object, ...], ...],
    ] = {}

    while stack:
        _check_deadline(deadline)
        engine, depth, path = stack.pop()

        if depth == planning_horizon or engine.is_complete():
            path_key = tuple(_action_key(action) for action in path)
            score = (float(engine.state.time), path_key)
            if best_score is None or score < best_score:
                best_score = score
                best_first_action = path[0] if path else None
            continue

        if best_score is not None and engine.state.time > best_score[0]:
            continue

        state_key = (
            depth,
            ActionSafetyFilter.state_signature(engine),
        )
        path_key = tuple(_action_key(action) for action in path)
        previous_path = seen.get(state_key)
        if previous_path is not None and previous_path <= path_key:
            continue
        seen[state_key] = path_key

        candidates = _transfer_actions(engine, safety_filter)
        for action in reversed(candidates):
            next_engine = ActionSafetyFilter.fork_engine(engine)
            next_engine.step(action)
            nodes_expanded += 1
            if not _advance_until_decision(next_engine, safety_filter):
                continue
            stack.append((next_engine, depth + 1, (*path, action)))

    if best_first_action is None:
        raise BranchSearchExhaustedError(
            "no safe feasible branch found within the planning horizon"
        )
    return _SearchChoice(
        first_action=best_first_action,
        nodes_expanded=nodes_expanded,
    )


def _advance_until_decision(
    engine: ClusterEngine,
    safety_filter: ActionSafetyFilter,
) -> bool:
    """Advance event time while no safe Pick/Place decision is available."""

    while not engine.is_complete():
        safe_actions = safety_filter.safe_actions(engine)
        if any(action != ADVANCE for action in safe_actions):
            return True
        if ADVANCE not in safe_actions:
            return False
        engine.step(ADVANCE)
    return True


def _transfer_actions(
    engine: ClusterEngine,
    safety_filter: ActionSafetyFilter,
) -> tuple[EngineAction, ...]:
    return tuple(
        sorted(
            (
                action
                for action in safety_filter.safe_actions(engine)
                if action != ADVANCE
            ),
            key=_action_key,
        )
    )


def _action_key(action: EngineAction) -> tuple[object, ...]:
    """Give actions a stable order independent of object identity."""

    if isinstance(action, PlaceAction):
        return (0, action.wafer_key[0], action.wafer_key[1], action.target_module_id)
    if isinstance(action, PickAction):
        return (1, action.wafer_key[0], action.wafer_key[1], action.robot_id)
    return (2,)


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("branch search exceeded time_limit_seconds")


def _require_integer(name: str, value: int, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m baseline.branch_search",
        description="Solve a generated cluster-tool instance with branch search.",
    )
    parser.add_argument("problem", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--planning-horizon", type=int, default=5)
    parser.add_argument("--safety-lookahead-depth", type=int, default=2)
    parser.add_argument("--time-limit-seconds", type=float)
    args = parser.parse_args(argv)

    if args.problem.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the problem file")

    try:
        instance = SchedulingInstance.model_validate_json(
            args.problem.read_text(encoding="utf-8")
        )
        result = solve_instance(
            instance,
            planning_horizon=args.planning_horizon,
            safety_lookahead_depth=args.safety_lookahead_depth,
            time_limit_seconds=args.time_limit_seconds,
        )
        args.output.write_text(
            json.dumps(
                [dict(action) for action in result.actions],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "solver": result.solver_name,
                "makespan": result.makespan,
                "planning_horizon": result.planning_horizon,
                "iterations": result.iterations,
                "nodes_expanded": result.nodes_expanded,
                "runtime_seconds": result.runtime_seconds,
                "action_count": len(result.actions),
                "validation_ok": result.validation_ok,
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
