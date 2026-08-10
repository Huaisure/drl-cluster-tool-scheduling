"""Reproducible structurally-feasible Cluster Tool benchmark generation."""

from .generator import DatasetGenerator
from .heuristic import HeuristicResult, build_safe_reference_schedule
from .models import (
    DatasetManifest,
    GenerationAudit,
    GenerationConfig,
    IntRange,
    ManifestEntry,
    ProblemGenerationConfig,
    RLGenerationConfig,
    RouteAudit,
    RouteWitness,
)
from .validation import validate_generated_instance
from .problem_generator import GeneratedBenchmark, ProblemGenerator
from .rl_dataset import RLDatasetGenerator

__all__ = [
    "DatasetGenerator",
    "DatasetManifest",
    "GenerationAudit",
    "GenerationConfig",
    "GeneratedBenchmark",
    "HeuristicResult",
    "IntRange",
    "ManifestEntry",
    "ProblemGenerationConfig",
    "RLGenerationConfig",
    "RLDatasetGenerator",
    "RouteAudit",
    "RouteWitness",
    "ProblemGenerator",
    "build_safe_reference_schedule",
    "validate_generated_instance",
]
