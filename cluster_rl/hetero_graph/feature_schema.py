"""Node feature definitions used by the cluster-tool graph builder.

Feature order is part of the model input contract. Add, remove, or reorder a
feature here first, then update the corresponding row construction in
``ClusterHeteroGraphBuilder._build_nodes``.
"""

from __future__ import annotations

from dataclasses import dataclass


# All graph time features use this shared physical unit. Environment state and
# event calculations remain in seconds.
TIME_SCALE_SECONDS = 100.0


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Name and meaning of one scalar node feature."""

    name: str
    description: str


GLOBAL_FEATURES = (
    FeatureSpec(
        "completed_wafer_ratio",
        "Completed wafer count divided by total wafer count.",
    ),
    FeatureSpec(
        "completed_step_ratio",
        "Sum of completed wafer steps divided by all required steps.",
    ),
    FeatureSpec(
        "remaining_process_ratio",
        "Remaining process work divided by initial total process work.",
    ),
)

WAFER_FEATURES = (
    FeatureSpec(
        "route_progress",
        "Completed route steps divided by route visits plus the final LP return.",
    ),
    FeatureSpec(
        "process_remaining",
        "Processing time remaining divided by TIME_SCALE_SECONDS.",
    ),
    FeatureSpec(
        "is_ready",
        "1 when processing is complete and the wafer is the FIFO head at its location.",
    ),
    FeatureSpec(
        "is_complete",
        "1 after the wafer has completed its final return to LP, otherwise 0.",
    ),
    FeatureSpec(
        "remaining_step_ratio",
        "Remaining route steps divided by route visits plus the final LP return.",
    ),
    FeatureSpec(
        "remaining_process_time",
        "Future process time divided by TIME_SCALE_SECONDS, excluding the current visit.",
    ),
    FeatureSpec(
        "holding_rank",
        "One-based order in the robot holding list; 0 when not held.",
    ),
)

ROUTE_STEP_FEATURES = (
    FeatureSpec(
        "process_time",
        "Required processing time divided by TIME_SCALE_SECONDS.",
    ),
    FeatureSpec(
        "residency_time",
        "Maximum residency time divided by TIME_SCALE_SECONDS; 0 when unspecified.",
    ),
    FeatureSpec(
        "has_residency_limit",
        "1 when residency_time is explicitly configured, otherwise 0.",
    ),
    FeatureSpec(
        "step_progress",
        "One-based step index divided by route visits plus the final LP return.",
    ),
    FeatureSpec(
        "is_return_to_lp",
        "1 for the synthetic final return-to-LP step, otherwise 0.",
    ),
)

MODULE_FEATURES = (
    FeatureSpec("is_lp", "1 for an LP module, otherwise 0."),
    FeatureSpec("is_pm", "1 for a PM module, otherwise 0."),
    FeatureSpec("is_ll", "1 for an LL module, otherwise 0."),
    FeatureSpec("capacity", "Maximum number of wafers in the module."),
    FeatureSpec(
        "occupancy_ratio",
        "Current wafer count divided by module capacity.",
    ),
    FeatureSpec(
        "available_ratio",
        "Slots available after pending Place reservations, divided by capacity.",
    ),
    FeatureSpec(
        "is_full",
        "1 when physical occupancy plus Place reservations reaches capacity.",
    ),
)

ROBOT_FEATURES = (
    FeatureSpec("pick_time", "Pick duration divided by TIME_SCALE_SECONDS."),
    FeatureSpec("place_time", "Place duration divided by TIME_SCALE_SECONDS."),
    FeatureSpec("travel_time", "Travel time divided by TIME_SCALE_SECONDS."),
    FeatureSpec("arm_capacity", "Number of wafers the robot can hold."),
    FeatureSpec(
        "held_ratio",
        "Number of held wafers divided by arm capacity.",
    ),
    FeatureSpec(
        "available_ratio",
        "Available robot hands divided by arm capacity.",
    ),
    FeatureSpec("is_full", "1 when the robot has no available hand."),
    FeatureSpec("is_idle", "1 when the robot has no pending operation."),
    FeatureSpec(
        "is_travel_to_pick",
        "1 while the robot travels to a pending Pick source.",
    ),
    FeatureSpec("is_picking", "1 from Pick.start until Pick.end."),
    FeatureSpec(
        "is_travel_to_place",
        "1 while the robot travels to a pending Place target.",
    ),
    FeatureSpec("is_placing", "1 from Place.start until Place.end."),
    FeatureSpec(
        "time_to_operation_start",
        "Time until operation start divided by TIME_SCALE_SECONDS; 0 once started.",
    ),
    FeatureSpec(
        "time_to_operation_end",
        "Time until operation end divided by TIME_SCALE_SECONDS; 0 when idle.",
    ),
)

GLOBAL_FEATURE_NAMES = tuple(feature.name for feature in GLOBAL_FEATURES)
WAFER_FEATURE_NAMES = tuple(feature.name for feature in WAFER_FEATURES)
ROUTE_STEP_FEATURE_NAMES = tuple(
    feature.name for feature in ROUTE_STEP_FEATURES
)
MODULE_FEATURE_NAMES = tuple(feature.name for feature in MODULE_FEATURES)
ROBOT_FEATURE_NAMES = tuple(feature.name for feature in ROBOT_FEATURES)
