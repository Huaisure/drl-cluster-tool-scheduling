from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

EdgeType: TypeAlias = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class NodeStore:
    """Nodes of one type.

    ``ids[i]`` and ``features[i]`` must always describe the same entity.
    Stable entity ordering is important because policy logits reuse these
    indexes as environment actions.
    """

    ids: tuple[object, ...]
    features: NDArray[np.float32]
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError("node features must have shape [node, feature]")
        if features.shape != (len(self.ids), len(self.feature_names)):
            raise ValueError("node feature shape must match ids and feature_names")
        object.__setattr__(self, "features", features)


@dataclass(frozen=True, slots=True)
class EdgeStore:
    """Directed edges of one relation type."""

    edge_index: NDArray[np.int64]

    def __post_init__(self) -> None:
        edge_index = np.asarray(self.edge_index, dtype=np.int64)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge]")
        object.__setattr__(self, "edge_index", edge_index)


@dataclass(frozen=True, slots=True)
class HeteroGraph:
    """Library-independent heterogeneous graph at one decision point."""

    nodes: dict[str, NodeStore]
    edges: dict[EdgeType, EdgeStore]
    pick_action_mask: NDArray[np.bool_]
    place_action_mask: NDArray[np.bool_]
    can_advance: bool

    def __post_init__(self) -> None:
        pick_action_mask = np.asarray(self.pick_action_mask, dtype=np.bool_)
        place_action_mask = np.asarray(self.place_action_mask, dtype=np.bool_)
        wafer_nodes = self.nodes.get("wafer")
        module_nodes = self.nodes.get("module")
        robot_nodes = self.nodes.get("robot")
        wafer_count = len(wafer_nodes.ids) if wafer_nodes is not None else 0
        module_count = len(module_nodes.ids) if module_nodes is not None else 0
        robot_count = len(robot_nodes.ids) if robot_nodes is not None else 0
        if pick_action_mask.shape != (wafer_count, robot_count):
            raise ValueError("pick_action_mask must have shape [wafer, robot]")
        if place_action_mask.shape != (wafer_count, module_count):
            raise ValueError("place_action_mask must have shape [wafer, module]")

        for (source_type, _, target_type), edge_store in self.edges.items():
            if source_type not in self.nodes or target_type not in self.nodes:
                raise ValueError("edge type references an unknown node type")
            if edge_store.edge_index.size == 0:
                continue
            source_count = len(self.nodes[source_type].ids)
            target_count = len(self.nodes[target_type].ids)
            invalid_source = (
                edge_store.edge_index[0].min() < 0
                or edge_store.edge_index[0].max() >= source_count
            )
            invalid_target = (
                edge_store.edge_index[1].min() < 0
                or edge_store.edge_index[1].max() >= target_count
            )
            if invalid_source or invalid_target:
                raise ValueError("edge_index contains an invalid node index")

        object.__setattr__(self, "pick_action_mask", pick_action_mask)
        object.__setattr__(self, "place_action_mask", place_action_mask)
        object.__setattr__(self, "can_advance", bool(self.can_advance))
