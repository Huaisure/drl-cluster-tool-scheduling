from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from baseline.branch_search import (
    SOLVER_VERSION as BRANCH_SEARCH_VERSION,
    BranchSearchExhaustedError,
    solve_instance as solve_branch_search,
)
from baseline.cpsat.direct import (
    SOLVER_VERSION as DIRECT_VERSION,
    FeasibilityConsistencyError,
)
from baseline.cpsat.direct import solve_instance as solve_direct
from baseline.cpsat.periodic import (
    SOLVER_VERSION as PERIODIC_VERSION,
    periodic_ratio,
    solve_periodic_instance,
)
from baseline.genetic import SOLVER_VERSION as GENETIC_VERSION
from baseline.genetic import solve_instance as solve_genetic

from cluster_toolkit.validator import ValidatorSuite

from .pipeline_models import SchedulingInstance
from .problem_adapter import to_cluster_problem
from .production import load_run
from .production_models import ProductionPlan, ProductionRunSpec
from .solutions import (
    ComponentResult,
    GlobalOptimalityStatus,
    InstanceSolutions,
    SolutionRecord,
    SolverStatus,
    TerminationReason,
    ValidationStatus,
    WorkflowStatus,
)


VALIDATOR_NAME = "validator_suite"
VALIDATOR_VERSION = "2.0.0"


@dataclass(frozen=True, slots=True)
class SolverTask:
    instance_id: str
    solver_name: Literal["cpsat_direct", "cpsat_periodic", "genetic", "branch_search"]
    attempt: Literal["short", "long"]
    seed: int
    time_limit_seconds: float
    num_search_workers: int = 1
    planning_horizon: int | None = None
    startup_time_limit_seconds: float | None = None
    closedown_time_limit_seconds: float | None = None

    @property
    def solution_id(self) -> str:
        suffix = f"seed-{self.seed}"
        if self.planning_horizon is not None:
            suffix = f"horizon-{self.planning_horizon}"
        return f"{self.attempt}-{suffix}"

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def soft_wall_limit(self) -> float:
        if self.solver_name == "cpsat_periodic":
            return self.time_limit_seconds + max(
                self.startup_time_limit_seconds or 0,
                self.closedown_time_limit_seconds or 0,
            )
        return self.time_limit_seconds


def run_labeling(run_root: str | Path) -> dict[str, int]:
    """Run every missing short attempt, eligible promotion, then reduce."""

    root = Path(run_root)
    spec, plan = load_run(root)
    short_tasks = _short_tasks(root, spec, plan)
    short_completed = _run_tasks(root, spec, short_tasks)
    long_tasks = _long_tasks(root, spec, plan)
    long_completed = _run_tasks(root, spec, long_tasks)
    reduced = reduce_run(root)
    return {
        "short_tasks_completed": short_completed,
        "long_tasks_completed": long_completed,
        "instances_reduced": reduced,
    }


def reduce_run(run_root: str | Path) -> int:
    root = Path(run_root)
    _, plan = load_run(root)
    for entry in plan.entries:
        InstanceSolutions(root / "instances" / entry.instance_id).reduce()
    return len(plan.entries)


def run_status(run_root: str | Path) -> dict[str, object]:
    root = Path(run_root)
    spec, plan = load_run(root)
    short = _short_tasks(root, spec, plan)
    long = _long_tasks(root, spec, plan)
    expected = [*short, *long]
    existing = sum(_record_path(root, task).is_file() for task in expected)
    usable = 0
    quarantined = 0
    for entry in plan.entries:
        index_path = root / "instances" / entry.instance_id / "solution_index.json"
        if not index_path.is_file():
            continue
        index = json.loads(index_path.read_text(encoding="utf-8"))
        usable += bool(index.get("usable"))
        quarantined += bool(index.get("quarantined"))
    return {
        "run_id": spec.run_id,
        "instance_count": len(plan.entries),
        "expected_attempt_count": len(expected),
        "terminal_attempt_count": existing,
        "pending_attempt_count": len(expected) - existing,
        "usable_instance_count": usable,
        "quarantined_instance_count": quarantined,
        "complete": existing == len(expected),
    }


def _short_tasks(
    root: Path,
    spec: ProductionRunSpec,
    plan: ProductionPlan,
) -> list[SolverTask]:
    tasks: list[SolverTask] = []
    for entry in plan.entries:
        tasks.append(
            SolverTask(
                instance_id=entry.instance_id,
                solver_name="cpsat_direct",
                attempt="short",
                seed=0,
                time_limit_seconds=spec.budgets.direct_short_seconds,
                num_search_workers=spec.cpsat_workers,
            )
        )
        instance = _load_instance(root, entry.instance_id)
        if periodic_ratio(instance) is not None:
            tasks.append(
                SolverTask(
                    instance_id=entry.instance_id,
                    solver_name="cpsat_periodic",
                    attempt="short",
                    seed=0,
                    time_limit_seconds=(
                        spec.budgets.periodic_cycle_short_seconds
                    ),
                    startup_time_limit_seconds=(
                        spec.budgets.periodic_transition_short_seconds
                    ),
                    closedown_time_limit_seconds=(
                        spec.budgets.periodic_transition_short_seconds
                    ),
                    num_search_workers=spec.cpsat_workers,
                )
            )
        for seed in spec.genetic_seeds:
            tasks.append(
                SolverTask(
                    instance_id=entry.instance_id,
                    solver_name="genetic",
                    attempt="short",
                    seed=seed,
                    time_limit_seconds=spec.budgets.genetic_seconds,
                )
            )
        for horizon in spec.branch_search_horizons:
            tasks.append(
                SolverTask(
                    instance_id=entry.instance_id,
                    solver_name="branch_search",
                    attempt="short",
                    seed=0,
                    planning_horizon=horizon,
                    time_limit_seconds=spec.budgets.branch_search_seconds,
                )
            )
    return tasks


def _long_tasks(
    root: Path,
    spec: ProductionRunSpec,
    plan: ProductionPlan,
) -> list[SolverTask]:
    tasks: list[SolverTask] = []
    for entry in plan.entries:
        instance = _load_instance(root, entry.instance_id)
        eligible = periodic_ratio(instance) is not None
        direct_short = _load_record(
            root,
            SolverTask(
                instance_id=entry.instance_id,
                solver_name="cpsat_direct",
                attempt="short",
                seed=0,
                time_limit_seconds=spec.budgets.direct_short_seconds,
                num_search_workers=spec.cpsat_workers,
            ),
        )
        if (
            not eligible
            and direct_short is not None
            and (
                direct_short.status is not SolverStatus.OPTIMAL
                or direct_short.validation_status is not ValidationStatus.VALID
            )
        ):
            tasks.append(
                SolverTask(
                    instance_id=entry.instance_id,
                    solver_name="cpsat_direct",
                    attempt="long",
                    seed=0,
                    time_limit_seconds=spec.budgets.direct_long_seconds,
                    num_search_workers=spec.cpsat_workers,
                )
            )
        if eligible:
            periodic_short_task = SolverTask(
                instance_id=entry.instance_id,
                solver_name="cpsat_periodic",
                attempt="short",
                seed=0,
                time_limit_seconds=spec.budgets.periodic_cycle_short_seconds,
                startup_time_limit_seconds=(
                    spec.budgets.periodic_transition_short_seconds
                ),
                closedown_time_limit_seconds=(
                    spec.budgets.periodic_transition_short_seconds
                ),
                num_search_workers=spec.cpsat_workers,
            )
            periodic_short = _load_record(root, periodic_short_task)
            if periodic_short is not None and (
                periodic_short.status is not SolverStatus.FEASIBLE
                or periodic_short.validation_status is not ValidationStatus.VALID
                or set(periodic_short.components) != {"cycle", "startup", "closedown"}
                or any(
                    component.status is not SolverStatus.OPTIMAL
                    for component in periodic_short.components.values()
                )
            ):
                tasks.append(
                    SolverTask(
                        instance_id=entry.instance_id,
                        solver_name="cpsat_periodic",
                        attempt="long",
                        seed=0,
                        time_limit_seconds=(
                            spec.budgets.periodic_cycle_long_seconds
                        ),
                        startup_time_limit_seconds=(
                            spec.budgets.periodic_transition_long_seconds
                        ),
                        closedown_time_limit_seconds=(
                            spec.budgets.periodic_transition_long_seconds
                        ),
                        num_search_workers=spec.cpsat_workers,
                    )
                )
    return tasks


def _run_tasks(
    root: Path,
    spec: ProductionRunSpec,
    tasks: list[SolverTask],
) -> int:
    pending = [task for task in tasks if not _record_path(root, task).is_file()]
    if not pending:
        return 0
    completed = 0
    with ThreadPoolExecutor(max_workers=spec.max_parallel_tasks) as executor:
        futures = {
            executor.submit(
                _supervise_task,
                root,
                task,
                spec.budgets.hard_kill_grace_seconds,
            ): task
            for task in pending
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
    return completed


def _supervise_task(root: Path, task: SolverTask, grace_seconds: float) -> None:
    if _record_path(root, task).is_file():
        return
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_solver_worker,
        args=(str(root), asdict(task)),
        name=f"{task.instance_id}-{task.solver_name}-{task.solution_id}",
    )
    process.start()
    process.join(task.soft_wall_limit + grace_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        if not _record_path(root, task).is_file():
            _write_terminal_failure(
                root,
                task,
                termination_reason=TerminationReason.INTERRUPTED,
                runtime_seconds=task.soft_wall_limit + grace_seconds,
                error="solver subprocess exceeded hard deadline",
            )
    elif process.exitcode != 0 and not _record_path(root, task).is_file():
        _write_terminal_failure(
            root,
            task,
            termination_reason=TerminationReason.ERROR,
            runtime_seconds=0,
            error=f"solver subprocess exited with code {process.exitcode}",
        )


def _solver_worker(root_value: str, task_values: dict[str, object]) -> None:
    root = Path(root_value)
    task = SolverTask(**task_values)
    started = time.monotonic()
    try:
        instance = _load_instance(root, task.instance_id)
        if task.solver_name == "cpsat_direct":
            result = solve_direct(
                instance,
                time_limit_seconds=task.time_limit_seconds,
                random_seed=task.seed,
                num_search_workers=task.num_search_workers,
            )
            record, actions = _direct_record(task, result)
        elif task.solver_name == "cpsat_periodic":
            result = solve_periodic_instance(
                instance,
                time_limit_seconds=task.time_limit_seconds,
                startup_time_limit_seconds=task.startup_time_limit_seconds,
                closedown_time_limit_seconds=task.closedown_time_limit_seconds,
                random_seed=task.seed,
                num_search_workers=task.num_search_workers,
            )
            record, actions = _periodic_record(task, result)
        elif task.solver_name == "genetic":
            result = solve_genetic(
                instance,
                seed=task.seed,
                time_limit_seconds=task.time_limit_seconds,
            )
            record, actions = _heuristic_record(
                task,
                actions=result.actions,
                makespan=result.makespan,
                runtime_seconds=result.runtime_seconds,
                termination_reason=TerminationReason(result.termination_reason),
                solver_version=GENETIC_VERSION,
            )
        else:
            assert task.planning_horizon is not None
            result = solve_branch_search(
                instance,
                planning_horizon=task.planning_horizon,
                time_limit_seconds=task.time_limit_seconds,
            )
            record, actions = _heuristic_record(
                task,
                actions=result.actions,
                makespan=result.makespan,
                runtime_seconds=result.runtime_seconds,
                termination_reason=TerminationReason.NORMAL,
                solver_version=result.solver_version,
            )
        if actions:
            report = ValidatorSuite(to_cluster_problem(instance)).validate(
                actions,
                require_complete=True,
                exact_action_durations=True,
            )
            if not report.ok:
                details = "; ".join(issue.message for issue in report.issues[:8])
                record = SolutionRecord.model_validate(
                    record.model_copy(
                        update={
                            "validation_status": ValidationStatus.INVALID,
                            "global_optimality_status": (
                                GlobalOptimalityStatus.UNPROVEN
                            ),
                            "strong_sample_signals": {
                                **record.strong_sample_signals,
                                "validation_error": details,
                            },
                        }
                    ).model_dump()
                )
        InstanceSolutions(root / "instances" / task.instance_id).write(
            record,
            list(actions) if actions else None,
        )
    except FeasibilityConsistencyError as exc:
        _write_infeasible_consistency_failure(
            root,
            task,
            runtime_seconds=time.monotonic() - started,
            error=str(exc),
        )
    except BranchSearchExhaustedError as exc:
        _write_terminal_failure(
            root,
            task,
            termination_reason=TerminationReason.NORMAL,
            runtime_seconds=time.monotonic() - started,
            error=str(exc),
            signal_name="search_exhausted",
            solver_version=BRANCH_SEARCH_VERSION,
        )
    except TimeoutError as exc:
        _write_terminal_failure(
            root,
            task,
            termination_reason=TerminationReason.TIME_LIMIT,
            runtime_seconds=time.monotonic() - started,
            error=str(exc),
        )
    except Exception as exc:
        _write_terminal_failure(
            root,
            task,
            termination_reason=TerminationReason.ERROR,
            runtime_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}",
        )


def _direct_record(task: SolverTask, result) -> tuple[SolutionRecord, tuple]:
    try:
        status = SolverStatus(result.status)
    except ValueError:
        status = SolverStatus.UNKNOWN
    validation = (
        ValidationStatus.VALID
        if result.validation_ok is True
        else ValidationStatus.NOT_RUN
    )
    termination = (
        TerminationReason.NORMAL
        if status is SolverStatus.OPTIMAL
        else TerminationReason.TIME_LIMIT
        if status in {SolverStatus.FEASIBLE, SolverStatus.UNKNOWN}
        else TerminationReason.NORMAL
    )
    optimal = status is SolverStatus.OPTIMAL and validation is ValidationStatus.VALID
    return (
        SolutionRecord(
            instance_id=task.instance_id,
            solution_id=task.solution_id,
            solver_name=task.solver_name,
            solver_version=result.solver_version,
            solver_config_hash=task.config_hash,
            seed=task.seed,
            status=status,
            termination_reason=termination,
            validation_status=validation,
            validator_name=VALIDATOR_NAME if validation is not ValidationStatus.NOT_RUN else None,
            validator_version=(
                VALIDATOR_VERSION if validation is not ValidationStatus.NOT_RUN else None
            ),
            workflow_status=WorkflowStatus.TERMINAL,
            global_optimality_status=(
                GlobalOptimalityStatus.PROVEN_OPTIMAL
                if optimal
                else GlobalOptimalityStatus.UNPROVEN
            ),
            makespan=result.makespan,
            best_bound=result.best_bound,
            best_bound_scope=(
                "full_problem" if result.best_bound is not None else None
            ),
            runtime_seconds=result.runtime_seconds,
            time_limit_seconds=task.time_limit_seconds,
            strong_sample_signals={
                "no_incumbent": not bool(result.actions),
                "unproven": status is not SolverStatus.OPTIMAL,
            },
        ),
        result.actions,
    )


def _periodic_record(task: SolverTask, result) -> tuple[SolutionRecord, tuple]:
    if result.status == "NOT_ELIGIBLE":
        return (
            SolutionRecord(
                instance_id=task.instance_id,
                solution_id=task.solution_id,
                solver_name=task.solver_name,
                solver_version=result.solver_version,
                solver_config_hash=task.config_hash,
                seed=task.seed,
                status=SolverStatus.UNKNOWN,
                termination_reason=TerminationReason.NOT_ELIGIBLE,
                validation_status=ValidationStatus.NOT_RUN,
                runtime_seconds=0,
                time_limit_seconds=task.time_limit_seconds,
            ),
            (),
        )
    components = {}
    for name in ("cycle", "startup", "closedown"):
        component = getattr(result, name)
        if component is None:
            continue
        components[name] = ComponentResult(
            status=SolverStatus(component.status),
            objective=component.objective,
            best_bound=component.best_bound,
            runtime_seconds=component.runtime_seconds,
        )
    all_optimal = bool(components) and all(
        component.status is SolverStatus.OPTIMAL
        for component in components.values()
    )
    runtime = sum(component.runtime_seconds for component in components.values())
    runtime += result.composition_runtime_seconds or 0
    return (
        SolutionRecord(
            instance_id=task.instance_id,
            solution_id=task.solution_id,
            solver_name=task.solver_name,
            solver_version=result.solver_version,
            solver_config_hash=task.config_hash,
            seed=task.seed,
            status=SolverStatus.FEASIBLE,
            termination_reason=(
                TerminationReason.NORMAL
                if all_optimal
                else TerminationReason.TIME_LIMIT
            ),
            validation_status=ValidationStatus.VALID,
            validator_name=VALIDATOR_NAME,
            validator_version=VALIDATOR_VERSION,
            workflow_status=WorkflowStatus.TERMINAL,
            makespan=result.makespan,
            runtime_seconds=runtime,
            time_limit_seconds=task.soft_wall_limit,
            components=components,
            strong_sample_signals={
                "unproven": True,
                "component_unproven": not all_optimal,
            },
        ),
        result.actions,
    )


def _heuristic_record(
    task: SolverTask,
    *,
    actions,
    makespan: float,
    runtime_seconds: float,
    termination_reason: TerminationReason,
    solver_version: str,
) -> tuple[SolutionRecord, tuple]:
    integer_makespan = int(round(float(makespan)))
    return (
        SolutionRecord(
            instance_id=task.instance_id,
            solution_id=task.solution_id,
            solver_name=task.solver_name,
            solver_version=solver_version,
            solver_config_hash=task.config_hash,
            seed=task.seed,
            status=SolverStatus.FEASIBLE,
            termination_reason=termination_reason,
            validation_status=ValidationStatus.VALID,
            validator_name=VALIDATOR_NAME,
            validator_version=VALIDATOR_VERSION,
            workflow_status=WorkflowStatus.TERMINAL,
            makespan=integer_makespan,
            runtime_seconds=runtime_seconds,
            time_limit_seconds=task.time_limit_seconds,
            strong_sample_signals={"unproven": True},
        ),
        actions,
    )


def _write_terminal_failure(
    root: Path,
    task: SolverTask,
    *,
    termination_reason: TerminationReason,
    runtime_seconds: float,
    error: str,
    signal_name: str = "error",
    solver_version: str | None = None,
) -> None:
    if _record_path(root, task).is_file():
        return
    _orphan_action_path(root, task).unlink(missing_ok=True)
    record = SolutionRecord(
        instance_id=task.instance_id,
        solution_id=task.solution_id,
        solver_name=task.solver_name,
        solver_version=solver_version or _solver_version(task),
        solver_config_hash=task.config_hash,
        seed=task.seed,
        status=SolverStatus.UNKNOWN,
        termination_reason=termination_reason,
        validation_status=ValidationStatus.NOT_RUN,
        workflow_status=WorkflowStatus.TERMINAL,
        runtime_seconds=max(0.0, runtime_seconds),
        time_limit_seconds=task.time_limit_seconds,
        strong_sample_signals={
            "no_incumbent": True,
            signal_name: error[:4000],
        },
    )
    InstanceSolutions(root / "instances" / task.instance_id).write(record)


def _write_infeasible_consistency_failure(
    root: Path,
    task: SolverTask,
    *,
    runtime_seconds: float,
    error: str,
) -> None:
    if _record_path(root, task).is_file():
        return
    _orphan_action_path(root, task).unlink(missing_ok=True)
    record = SolutionRecord(
        instance_id=task.instance_id,
        solution_id=task.solution_id,
        solver_name=task.solver_name,
        solver_version=_solver_version(task),
        solver_config_hash=task.config_hash,
        seed=task.seed,
        status=SolverStatus.INFEASIBLE,
        termination_reason=TerminationReason.ERROR,
        validation_status=ValidationStatus.NOT_RUN,
        workflow_status=WorkflowStatus.TERMINAL,
        runtime_seconds=max(0.0, runtime_seconds),
        time_limit_seconds=task.time_limit_seconds,
        strong_sample_signals={
            "feasibility_consistency_failure": True,
            "error": error[:4000],
        },
    )
    InstanceSolutions(root / "instances" / task.instance_id).write(record)


def _solver_version(task: SolverTask) -> str:
    return {
        "cpsat_direct": DIRECT_VERSION,
        "cpsat_periodic": PERIODIC_VERSION,
        "genetic": GENETIC_VERSION,
        "branch_search": BRANCH_SEARCH_VERSION,
    }[task.solver_name]


def _load_instance(root: Path, instance_id: str) -> SchedulingInstance:
    return SchedulingInstance.model_validate_json(
        (root / "instances" / instance_id / "problem.json").read_text(
            encoding="utf-8"
        )
    )


def _record_path(root: Path, task: SolverTask) -> Path:
    return (
        root
        / "instances"
        / task.instance_id
        / "solutions"
        / task.solver_name
        / f"{task.solution_id}.solution.json"
    )


def _orphan_action_path(root: Path, task: SolverTask) -> Path:
    return (
        root
        / "instances"
        / task.instance_id
        / "solutions"
        / task.solver_name
        / f"{task.solution_id}.actions.json.gz"
    )


def _load_record(root: Path, task: SolverTask) -> SolutionRecord | None:
    path = _record_path(root, task)
    if not path.is_file():
        return None
    return SolutionRecord.model_validate_json(path.read_text(encoding="utf-8"))
