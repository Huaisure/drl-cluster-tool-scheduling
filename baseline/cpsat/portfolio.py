"""Routing between periodic and direct CP-SAT for canonical instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from cluster_toolkit.cluster_generator.pipeline_models import SchedulingInstance

from .direct import CpSatResult, solve_instance
from .periodic import PeriodicResult, periodic_ratio, solve_periodic_instance


@dataclass(frozen=True, slots=True)
class RoutedCpSatResult:
    method: Literal["periodic", "direct"]
    result: PeriodicResult | CpSatResult


def solve_cpsat_instance(
    instance: SchedulingInstance,
    *,
    time_limit_seconds: float = 1800,
    random_seed: int = 0,
    num_search_workers: int = 1,
) -> RoutedCpSatResult:
    """Use periodic CP-SAT exactly when the workload ratio is supported."""

    if periodic_ratio(instance) is not None:
        return RoutedCpSatResult(
            method="periodic",
            result=solve_periodic_instance(
                instance,
                time_limit_seconds=time_limit_seconds,
                random_seed=random_seed,
                num_search_workers=num_search_workers,
            ),
        )
    return RoutedCpSatResult(
        method="direct",
        result=solve_instance(
            instance,
            time_limit_seconds=time_limit_seconds,
            random_seed=random_seed,
            num_search_workers=num_search_workers,
        ),
    )
