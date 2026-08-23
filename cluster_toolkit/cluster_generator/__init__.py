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
from .corpus import InstanceCorpus
from .pipeline import GeneratedInstance, InstanceGenerator, PERIODIC_RATIOS, WAFER_RANGES
from .pipeline_catalog import PipelineCatalog
from .pipeline_models import (
    InstanceGenerationRequest,
    RecipeGenerationProfile,
    SchedulingInstance,
    TopologyTemplate,
)
from .problem_adapter import to_cluster_problem
from .topology_family import (
    ATMOSPHERIC_FAMILY_ID,
    AtmosphericTopologyRequest,
    generate_atmospheric_topology,
)
from .production_models import ProductionPlan, ProductionRunSpec, SolverBudgets
from .production import default_run_id, load_run, materialize_plan
from .solutions import (
    GlobalOptimalityStatus,
    InstanceSolutions,
    LabelingStatus,
    SolutionIndex,
    SolutionRecord,
    SolverStatus,
    TerminationReason,
    ValidationStatus,
    WorkflowStatus,
)

__all__ = [
    "DatasetGenerator",
    "DatasetManifest",
    "GenerationAudit",
    "GenerationConfig",
    "GeneratedBenchmark",
    "GeneratedInstance",
    "HeuristicResult",
    "IntRange",
    "InstanceCorpus",
    "InstanceGenerationRequest",
    "InstanceGenerator",
    "InstanceSolutions",
    "LabelingStatus",
    "ManifestEntry",
    "ProblemGenerationConfig",
    "PipelineCatalog",
    "PERIODIC_RATIOS",
    "RLGenerationConfig",
    "RLDatasetGenerator",
    "RecipeGenerationProfile",
    "RouteAudit",
    "RouteWitness",
    "SchedulingInstance",
    "SolutionIndex",
    "SolutionRecord",
    "SolverStatus",
    "TerminationReason",
    "GlobalOptimalityStatus",
    "ValidationStatus",
    "WorkflowStatus",
    "TopologyTemplate",
    "to_cluster_problem",
    "ATMOSPHERIC_FAMILY_ID",
    "AtmosphericTopologyRequest",
    "generate_atmospheric_topology",
    "WAFER_RANGES",
    "ProblemGenerator",
    "ProductionPlan",
    "ProductionRunSpec",
    "SolverBudgets",
    "build_safe_reference_schedule",
    "validate_generated_instance",
    "default_run_id",
    "load_run",
    "materialize_plan",
]
