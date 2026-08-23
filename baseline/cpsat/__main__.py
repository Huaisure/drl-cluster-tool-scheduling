from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from cluster_toolkit.cluster_generator.pipeline_models import SchedulingInstance

from .direct import solve_instance
from .periodic import PeriodicResult, solve_periodic_instance
from .portfolio import solve_cpsat_instance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m baseline.cpsat",
        description="Solve one canonical SchedulingInstance with CP-SAT.",
    )
    parser.add_argument("problem", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("auto", "periodic", "direct"),
        default="auto",
    )
    parser.add_argument("--time-limit", type=float, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    if args.problem.resolve() == args.output.resolve():
        parser.error("--output must not overwrite the problem file")
    try:
        instance = SchedulingInstance.model_validate_json(
            args.problem.read_text(encoding="utf-8")
        )
        if args.mode == "auto":
            routed = solve_cpsat_instance(
                instance,
                time_limit_seconds=args.time_limit,
                random_seed=args.seed,
                num_search_workers=args.workers,
            )
            method = routed.method
            result = routed.result
        elif args.mode == "periodic":
            method = "periodic"
            result = solve_periodic_instance(
                instance,
                time_limit_seconds=args.time_limit,
                random_seed=args.seed,
                num_search_workers=args.workers,
            )
            if result.status == "NOT_ELIGIBLE":
                raise ValueError("problem workload ratio is not periodic-eligible")
        else:
            method = "direct"
            result = solve_instance(
                instance,
                time_limit_seconds=args.time_limit,
                random_seed=args.seed,
                num_search_workers=args.workers,
            )
        if result.validation_ok is not True:
            raise RuntimeError(f"solver did not return a validated solution: {result.status}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps([dict(action) for action in result.actions], indent=2),
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    summary: dict[str, object] = {
        "instance_id": instance.instance_id,
        "method": method,
        "status": result.status,
        "makespan": result.makespan,
        "action_count": len(result.actions),
        "validation_ok": result.validation_ok,
        "output": str(args.output),
    }
    if isinstance(result, PeriodicResult):
        summary.update(
            {
                "ratio": result.ratio,
                "repeat_count": result.repeat_count,
                "period": result.period,
                "pipeline_depth_periods": result.pipeline_depth_periods,
                "steady_cycle_count": result.steady_cycle_count,
                "boundary_shift": result.boundary_shift,
                "boundary_candidates_evaluated": result.boundary_candidates_evaluated,
                "cycle_status": result.cycle.status if result.cycle else None,
                "cycle_best_bound": result.cycle.best_bound if result.cycle else None,
            }
        )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
