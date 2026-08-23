from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cluster_toolkit.cluster_generator.labeling import (
    reduce_run,
    run_labeling,
    run_status,
)
from cluster_toolkit.cluster_generator.production import (
    default_run_id,
    materialize_plan,
)
from cluster_toolkit.cluster_generator.production_models import (
    ProductionRunSpec,
    SolverBudgets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, label, resume, reduce, and inspect a production corpus"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="materialize an immutable run and problems")
    plan.add_argument("output", type=Path)
    plan.add_argument("--spec", type=Path)
    plan.add_argument("--run-id")
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--count", type=int, default=100)
    plan.add_argument(
        "--topology-count",
        type=int,
        default=32,
        help="number of fixed archetype × arm variants (maximum 32 by default)",
    )
    plan.add_argument("--max-parallel-tasks", type=int, default=4)
    plan.add_argument("--cpsat-workers", type=int, default=1)
    plan.add_argument(
        "--quick",
        action="store_true",
        help="use two-second solver budgets for an end-to-end smoke run",
    )

    for name in ("run", "resume", "reduce", "status"):
        command = commands.add_parser(name)
        command.add_argument("run_root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        if args.spec is not None:
            spec = ProductionRunSpec.model_validate_json(
                args.spec.read_text(encoding="utf-8")
            )
        else:
            budgets = (
                SolverBudgets(
                    direct_short_seconds=2,
                    direct_long_seconds=2,
                    periodic_cycle_short_seconds=2,
                    periodic_cycle_long_seconds=2,
                    periodic_transition_short_seconds=2,
                    periodic_transition_long_seconds=2,
                    genetic_seconds=2,
                    branch_search_seconds=2,
                    hard_kill_grace_seconds=2,
                )
                if args.quick
                else SolverBudgets()
            )
            spec = ProductionRunSpec(
                run_id=args.run_id or default_run_id(args.seed, args.count),
                master_seed=args.seed,
                instance_count=args.count,
                topology_count=args.topology_count,
                max_parallel_tasks=args.max_parallel_tasks,
                cpsat_workers=args.cpsat_workers,
                budgets=budgets,
            )
        plan_result = materialize_plan(args.output, spec)
        result: object = {
            "run_id": spec.run_id,
            "instance_count": len(plan_result.entries),
            "run_root": str(args.output),
        }
    elif args.command in {"run", "resume"}:
        result = run_labeling(args.run_root)
    elif args.command == "reduce":
        result = {"instances_reduced": reduce_run(args.run_root)}
    else:
        result = run_status(args.run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
