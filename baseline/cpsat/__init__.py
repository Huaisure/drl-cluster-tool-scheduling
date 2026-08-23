"""CP-SAT baselines for canonical scheduling instances.

The public entry point intentionally does not import the historical scenario
scripts under :mod:`baseline.cpsat.solver`.  Those scripts use their own JSON
schema and optional cleaning/JIT constraints, while ``solve_instance`` targets
the canonical data-pipeline schema through the execution-model adapter.
"""

from .direct import CpSatResult, solve, solve_instance
from .periodic import (
    PeriodicComponentResult,
    PeriodicResult,
    periodic_ratio,
    solve_periodic_instance,
)
from .portfolio import RoutedCpSatResult, solve_cpsat_instance
from .transitions import TransitionSolveResult, solve_closedown, solve_startup

__all__ = [
    "CpSatResult",
    "PeriodicComponentResult",
    "PeriodicResult",
    "RoutedCpSatResult",
    "TransitionSolveResult",
    "periodic_ratio",
    "solve",
    "solve_cpsat_instance",
    "solve_closedown",
    "solve_instance",
    "solve_periodic_instance",
    "solve_startup",
]
