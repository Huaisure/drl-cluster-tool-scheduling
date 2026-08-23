from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .pipeline_models import IntInterval, InstanceGenerationRequest


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SolverBudgets(_StrictModel):
    direct_short_seconds: float = 600
    direct_long_seconds: float = 1800
    periodic_cycle_short_seconds: float = 300
    periodic_cycle_long_seconds: float = 600
    periodic_transition_short_seconds: float = 300
    periodic_transition_long_seconds: float = 1200
    genetic_seconds: float = 600
    branch_search_seconds: float = 600
    hard_kill_grace_seconds: float = 60

    @field_validator("*", mode="before")
    @classmethod
    def _validate_budget(cls, value: object) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise ValueError("solver budgets must be positive finite numbers")
        return float(value)


class ProductionRunSpec(_StrictModel):
    schema_version: Literal[1, 2] = 2
    run_id: str
    master_seed: int
    instance_count: int = 100
    topology_count: int = 32
    profile_id: str = "atmospheric_linear_default"
    # Empty means all archetypes installed in the topology JSON catalog.
    topology_archetypes: tuple[str, ...] = ()
    cell_count_weights: dict[int, float] = Field(
        default_factory=lambda: {1: 0.50, 2: 0.40, 3: 0.10}
    )
    recipe_counts: tuple[int, ...] = (1, 2, 3)
    wafer_scales: tuple[str, ...] = ("small", "medium", "large", "xlarge")
    wafer_ranges: dict[str, IntInterval] = Field(
        default_factory=lambda: {
            "small": IntInterval(minimum=10, maximum=25),
            "medium": IntInterval(minimum=26, maximum=50),
            "large": IntInterval(minimum=51, maximum=100),
            "xlarge": IntInterval(minimum=101, maximum=200),
        }
    )
    periodic_fraction: float = 0.50
    route_pattern_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "local": 0.30,
            "single_transition": 0.40,
            "multi_transition": 0.30,
        }
    )
    genetic_seeds: tuple[int, ...] = (0, 1, 2, 3)
    branch_search_horizons: tuple[int, ...] = (1, 3, 5)
    max_parallel_tasks: int = 4
    cpsat_workers: int = 1
    budgets: SolverBudgets = Field(default_factory=SolverBudgets)

    @field_validator("run_id", "profile_id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("run_id and profile_id must be non-empty strings")
        return value

    @field_validator("topology_archetypes")
    @classmethod
    def _validate_topology_archetypes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("topology_archetypes must not contain duplicates")
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError("topology_archetypes must contain non-empty strings")
        return value

    @field_validator(
        "master_seed",
        "instance_count",
        "topology_count",
        "max_parallel_tasks",
        "cpsat_workers",
        mode="before",
    )
    @classmethod
    def _validate_integer(cls, value: object, info) -> int:
        minimum = 0 if info.field_name == "master_seed" else 1
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
        ):
            raise ValueError(f"{info.field_name} must be an integer >= {minimum}")
        return value

    @field_validator("periodic_fraction", mode="before")
    @classmethod
    def _validate_fraction(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("periodic_fraction must be between 0 and 1")
        result = float(value)
        if not 0 <= result <= 1:
            raise ValueError("periodic_fraction must be between 0 and 1")
        return result

    @model_validator(mode="after")
    def _validate_distributions(self) -> "ProductionRunSpec":
        if set(self.cell_count_weights) != {1, 2, 3}:
            raise ValueError("cell_count_weights must define Cells 1, 2, and 3")
        if any(weight < 0 for weight in self.cell_count_weights.values()) or sum(
            self.cell_count_weights.values()
        ) <= 0:
            raise ValueError("cell_count_weights must contain positive total weight")
        if not self.recipe_counts or any(
            recipe_count not in {1, 2, 3} for recipe_count in self.recipe_counts
        ):
            raise ValueError("recipe_counts supports only 1, 2, and 3")
        if set(self.wafer_scales) - set(self.wafer_ranges):
            raise ValueError("every wafer scale needs a configured range")
        if set(self.route_pattern_weights) - {
            "local",
            "single_transition",
            "multi_transition",
        }:
            raise ValueError("unsupported route pattern")
        if sum(self.route_pattern_weights.values()) <= 0:
            raise ValueError("route_pattern_weights needs positive total weight")
        if self.schema_version == 2:
            positive_cell_counts = sum(
                weight > 0 for weight in self.cell_count_weights.values()
            )
            if self.topology_count < positive_cell_counts:
                raise ValueError(
                    "topology_count must cover every Cell count with positive weight"
                )
        if len(set(self.genetic_seeds)) != len(self.genetic_seeds):
            raise ValueError("genetic_seeds must be unique")
        if self.branch_search_horizons != (1, 3, 5):
            raise ValueError("branch_search_horizons must be exactly (1, 3, 5)")
        return self


class ProductionPlanEntry(_StrictModel):
    ordinal: int
    request: InstanceGenerationRequest
    instance_id: str
    topology_cell_count: int
    # Added after the initial schema-v1 pilot. Keep it optional so status and
    # resume can still read immutable plans created before this provenance
    # field existed; newly materialized plans always populate it.
    topology_seed: int | None = None
    topology_archetype_id: str | None = None
    robot_arm_profile_id: str | None = None
    periodic_requested: bool
    route_pattern: Literal["local", "single_transition", "multi_transition"]


class ProductionPlan(_StrictModel):
    schema_version: Literal[1] = 1
    run_id: str
    entries: tuple[ProductionPlanEntry, ...]
