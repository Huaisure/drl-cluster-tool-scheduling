from __future__ import annotations

import math
import random
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


GENERATOR_NAME = "cluster_generator"
GENERATOR_VERSION = "0.5.0"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IntRange(_StrictModel):
    minimum: int
    maximum: int

    @field_validator("minimum", "maximum", mode="before")
    @classmethod
    def _validate_bound(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("range bounds must be non-negative integers")
        return value

    @model_validator(mode="after")
    def _validate_order(self) -> "IntRange":
        if self.maximum < self.minimum:
            raise ValueError("range maximum must be greater than or equal to minimum")
        return self

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.minimum, self.maximum)


_PROFILE_DEFAULTS: dict[str, dict[str, IntRange]] = {
    "small": {
        "route_count": IntRange(minimum=1, maximum=2),
        "total_wafers": IntRange(minimum=10, maximum=20),
        "route_steps": IntRange(minimum=2, maximum=5),
        "process_time": IntRange(minimum=50, maximum=100),
    },
    "medium": {
        "route_count": IntRange(minimum=2, maximum=4),
        "total_wafers": IntRange(minimum=21, maximum=50),
        "route_steps": IntRange(minimum=4, maximum=10),
        "process_time": IntRange(minimum=50, maximum=300),
    },
    "large": {
        "route_count": IntRange(minimum=4, maximum=6),
        "total_wafers": IntRange(minimum=51, maximum=75),
        "route_steps": IntRange(minimum=8, maximum=20),
        "process_time": IntRange(minimum=100, maximum=600),
    },
}


class GenerationConfig(_StrictModel):
    profile: Literal["small", "medium", "large"] = "small"
    instance_count: int = 10
    seed: int = 0

    route_count: IntRange | None = None
    total_wafers: IntRange | None = None
    route_steps: IntRange | None = None
    process_time: IntRange | None = None

    candidate_probability: float = 0.25
    max_candidates: int = 3

    pick_time: IntRange = Field(default_factory=lambda: IntRange(minimum=1, maximum=10))
    place_time: IntRange = Field(default_factory=lambda: IntRange(minimum=1, maximum=10))
    travel_time: IntRange = Field(default_factory=lambda: IntRange(minimum=1, maximum=10))
    pump_time: IntRange = Field(default_factory=lambda: IntRange(minimum=5, maximum=30))
    vent_time: IntRange = Field(default_factory=lambda: IntRange(minimum=5, maximum=30))
    pm_capacity: IntRange = Field(default_factory=lambda: IntRange(minimum=1, maximum=2))
    ll_capacity: IntRange = Field(default_factory=lambda: IntRange(minimum=1, maximum=2))
    lp_spare_percent: IntRange = Field(default_factory=lambda: IntRange(minimum=0, maximum=25))
    max_lp_capacity: int | None = None

    @field_validator("instance_count", "max_candidates", mode="before")
    @classmethod
    def _validate_positive_int(cls, value: object, info) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("seed must be an integer")
        return value

    @field_validator("candidate_probability", mode="before")
    @classmethod
    def _validate_probability(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("candidate_probability must be between 0 and 1")
        probability = float(value)
        if not 0 <= probability <= 1:
            raise ValueError("candidate_probability must be between 0 and 1")
        return probability

    @field_validator("max_lp_capacity", mode="before")
    @classmethod
    def _validate_optional_capacity(cls, value: object) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("max_lp_capacity must be a positive integer or null")
        return value

    @model_validator(mode="after")
    def _apply_profile_defaults(self) -> "GenerationConfig":
        defaults = _PROFILE_DEFAULTS[self.profile]
        for field_name in ("route_count", "total_wafers", "route_steps", "process_time"):
            if getattr(self, field_name) is None:
                setattr(self, field_name, defaults[field_name].model_copy(deep=True))

        assert self.route_count is not None
        assert self.total_wafers is not None
        if self.total_wafers.minimum < self.route_count.maximum:
            raise ValueError(
                "total_wafers.minimum must be at least route_count.maximum so every Route has a wafer"
            )
        return self


class ProblemGenerationConfig(_StrictModel):
    """Serializable domain distributions used by :class:`ProblemGenerator`."""

    max_attempts: int = 512
    topology_weights_by_split: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "train": {"simple": 0.10, "single_vacuum": 0.60, "dual_vacuum": 0.30},
            "validation": {"simple": 0.05, "single_vacuum": 0.45, "dual_vacuum": 0.50},
            "test": {"simple": 0.00, "single_vacuum": 0.30, "dual_vacuum": 0.70},
        }
    )
    ll_count_weights: dict[int, float] = Field(
        default_factory=lambda: {1: 0.30, 2: 0.70}
    )
    atm_arm_weights: dict[str, float] = Field(
        default_factory=lambda: {"single_arm": 0.70, "dual_arm": 0.30}
    )
    vtm_arm_weights: dict[str, float] = Field(
        default_factory=lambda: {"single_arm": 0.30, "dual_arm": 0.70}
    )
    pm_count_weights: dict[int, float] = Field(
        default_factory=lambda: {3: 0.15, 4: 0.35, 5: 0.35, 6: 0.15}
    )
    buffer_count_weights: dict[int, float] = Field(
        default_factory=lambda: {1: 0.30, 2: 0.70}
    )
    pm_capability_count_weights: dict[int, float] = Field(
        default_factory=lambda: {2: 0.70, 3: 0.30}
    )
    process_support_count_weights: dict[int, float] = Field(
        default_factory=lambda: {1: 0.50, 2: 0.45, 3: 0.035, 4: 0.015}
    )
    recipe_count_weights: dict[int, float] = Field(
        default_factory=lambda: {
            1: 0.20,
            2: 0.25,
            3: 0.25,
            4: 0.15,
            5: 0.10,
            6: 0.05,
        }
    )
    recipe_length_weights: dict[int, float] = Field(
        default_factory=lambda: {
            1: 0.10,
            2: 0.233333,
            3: 0.233334,
            4: 0.233333,
            5: 0.075,
            6: 0.075,
            7: 0.025,
            8: 0.025,
        }
    )
    reentry_weights: dict[str, float] = Field(
        default_factory=lambda: {"none": 0.65, "twice": 0.30, "deep": 0.05}
    )
    deep_reentry_count_weights: dict[int, float] = Field(
        default_factory=lambda: {3: 0.80, 4: 0.20}
    )
    cooling_count_weights: dict[int, float] = Field(
        default_factory=lambda: {1: 0.80, 2: 0.18, 3: 0.02}
    )
    priority_mode_weights: dict[str, float] = Field(
        default_factory=lambda: {"none": 0.50, "recipe": 0.30, "wave": 0.20}
    )
    mixed_unit_layout_weights: dict[str, float] = Field(
        default_factory=lambda: {"pre": 0.30, "post": 0.30, "both": 0.40}
    )
    pm_time_anchor_weights: dict[int, float] = Field(
        default_factory=lambda: {30: 0.20, 50: 0.30, 300: 0.30, 600: 0.20}
    )

    cross_unit_probability: float = 0.60
    vtm2_only_probability_given_cross: float = 0.30
    cooling_probability: float = 0.25
    process_time_jitter: float = 0.10
    recipe_count_perturbation: float = 0.30
    recipe_count_max_ratio: float = 2.0

    easy_wafer_range: tuple[int, int] = (10, 25)
    medium_wafer_range: tuple[int, int] = (26, 50)
    hard_wafer_range: tuple[int, int] = (51, 75)
    al_time_range: tuple[int, int] = (10, 20)
    transfer_time_range: tuple[int, int] = (8, 15)
    ll_common_time_range: tuple[int, int] = (10, 20)
    ll_tail_time_range: tuple[int, int] = (21, 30)
    cooling_time_range: tuple[int, int] = (30, 60)
    ll_tail_probability: float = 0.10

    @field_validator("max_attempts", mode="before")
    @classmethod
    def _validate_max_attempts(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("max_attempts must be a positive integer")
        return value

    @field_validator(
        "cross_unit_probability",
        "vtm2_only_probability_given_cross",
        "cooling_probability",
        "process_time_jitter",
        "recipe_count_perturbation",
        "ll_tail_probability",
        mode="before",
    )
    @classmethod
    def _validate_fraction(cls, value: object, info) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        fraction = float(value)
        if not 0 <= fraction <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return fraction

    @field_validator("recipe_count_max_ratio", mode="before")
    @classmethod
    def _validate_ratio(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("recipe_count_max_ratio must be a number")
        ratio = float(value)
        if ratio < 1:
            raise ValueError("recipe_count_max_ratio must be at least 1")
        return ratio

    @field_validator(
        "ll_count_weights",
        "atm_arm_weights",
        "vtm_arm_weights",
        "pm_count_weights",
        "buffer_count_weights",
        "pm_capability_count_weights",
        "process_support_count_weights",
        "recipe_count_weights",
        "recipe_length_weights",
        "reentry_weights",
        "deep_reentry_count_weights",
        "cooling_count_weights",
        "priority_mode_weights",
        "mixed_unit_layout_weights",
        "pm_time_anchor_weights",
    )
    @classmethod
    def _validate_weights(cls, value: dict[object, float], info) -> dict[object, float]:
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
            for weight in value.values()
        ):
            raise ValueError(f"{info.field_name} weights must be finite and non-negative")
        if sum(float(weight) for weight in value.values()) <= 0:
            raise ValueError(f"{info.field_name} must contain a positive weight")
        return value

    @field_validator("topology_weights_by_split")
    @classmethod
    def _validate_topology_weights(
        cls,
        value: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        expected_splits = {"train", "validation", "test"}
        expected_families = {"simple", "single_vacuum", "dual_vacuum"}
        if set(value) != expected_splits:
            raise ValueError("topology_weights_by_split must define train, validation, and test")
        for split, weights in value.items():
            if set(weights) != expected_families:
                raise ValueError(
                    f"topology_weights_by_split.{split} must define every topology family"
                )
            if any(weight < 0 or not math.isfinite(weight) for weight in weights.values()):
                raise ValueError("topology weights must be finite and non-negative")
            if sum(weights.values()) <= 0:
                raise ValueError(f"topology_weights_by_split.{split} must contain a positive weight")
        return value

    @field_validator(
        "easy_wafer_range",
        "medium_wafer_range",
        "hard_wafer_range",
        "al_time_range",
        "transfer_time_range",
        "ll_common_time_range",
        "ll_tail_time_range",
        "cooling_time_range",
    )
    @classmethod
    def _validate_int_range(cls, value: tuple[int, int], info) -> tuple[int, int]:
        if len(value) != 2 or value[0] < 0 or value[1] < value[0]:
            raise ValueError(f"{info.field_name} must be an ascending non-negative pair")
        return value

    @model_validator(mode="after")
    def _validate_domain_keys(self) -> "ProblemGenerationConfig":
        expected_keys = {
            "ll_count_weights": {1, 2},
            "atm_arm_weights": {"single_arm", "dual_arm"},
            "vtm_arm_weights": {"single_arm", "dual_arm"},
            "pm_count_weights": {3, 4, 5, 6},
            "buffer_count_weights": {1, 2},
            "pm_capability_count_weights": {2, 3},
            "process_support_count_weights": {1, 2, 3, 4},
            "recipe_count_weights": {1, 2, 3, 4, 5, 6},
            "recipe_length_weights": {1, 2, 3, 4, 5, 6, 7, 8},
            "reentry_weights": {"none", "twice", "deep"},
            "deep_reentry_count_weights": {3, 4},
            "cooling_count_weights": {1, 2, 3},
            "priority_mode_weights": {"none", "recipe", "wave"},
            "mixed_unit_layout_weights": {"pre", "post", "both"},
            "pm_time_anchor_weights": {30, 50, 300, 600},
        }
        for field_name, keys in expected_keys.items():
            if set(getattr(self, field_name)) != keys:
                raise ValueError(f"{field_name} must define exactly {sorted(keys, key=str)}")
        return self


class RouteWitness(_StrictModel):
    start_lp: str
    end_lp: str
    witness_path: tuple[str, ...]


class RouteAudit(_StrictModel):
    witnesses: tuple[RouteWitness, ...]


class GenerationAudit(_StrictModel):
    module_counts: dict[str, int]
    robot_count: int
    route_count: int
    wafer_count: int
    min_route_steps: int
    max_route_steps: int
    candidate_step_count: int
    routes: dict[str, RouteAudit]


class ManifestEntry(_StrictModel):
    file: str
    instance_seed: int
    audit: GenerationAudit


class DatasetManifest(_StrictModel):
    schema_version: Literal[1] = 1
    generator: dict[str, str]
    template: dict[str, str]
    config: dict[str, Any]
    instances: tuple[ManifestEntry, ...]


class RLGenerationConfig(_StrictModel):
    """Configuration for a procedural PPO curriculum dataset."""

    instance_count: int = 100
    seed: int = 0
    split: Literal["train", "validation", "test"] = "train"
    difficulty: Literal["easy", "medium", "hard", "edge"] | None = None
    difficulty_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "easy": 0.30,
            "medium": 0.40,
            "hard": 0.20,
            "edge": 0.10,
        }
    )
    materialize_problems: bool = True
    include_reference_actions: bool = True

    @field_validator("instance_count", mode="before")
    @classmethod
    def _validate_instance_count(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("instance_count must be a positive integer")
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_rl_seed(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("seed must be an integer")
        return value

    @field_validator("difficulty_weights")
    @classmethod
    def _validate_difficulty_weights(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        expected = {"easy", "medium", "hard", "edge"}
        if set(value) != expected:
            raise ValueError(
                "difficulty_weights must contain easy, medium, hard, and edge exactly"
            )
        normalized: dict[str, float] = {}
        for name, raw_weight in value.items():
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise ValueError(f"difficulty_weights.{name} must be a number")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"difficulty_weights.{name} must be finite and non-negative")
            normalized[name] = weight
        if sum(normalized.values()) <= 0:
            raise ValueError("difficulty_weights must contain a positive weight")
        return normalized

    @model_validator(mode="after")
    def _validate_materialization(self) -> "RLGenerationConfig":
        if self.split != "train" and not self.materialize_problems:
            raise ValueError("validation and test datasets must materialize problem JSON files")
        if not self.materialize_problems and self.include_reference_actions:
            raise ValueError("reference actions require materialized problem JSON files")
        return self
