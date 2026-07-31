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
    features: NDArray[np.float32]
    feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        edge_index = np.asarray(self.edge_index, dtype=np.int64)
        features = np.asarray(self.features, dtype=np.float32)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge]")
        if features.shape != (
            edge_index.shape[1],
            len(self.feature_names),
        ):
            raise ValueError("edge feature shape must match edges and feature_names")
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "features", features)


@dataclass(frozen=True, slots=True)
class HeteroGraph:
    """Library-independent heterogeneous graph at one decision point."""

    nodes: dict[str, NodeStore]
    edges: dict[EdgeType, EdgeStore]
    graph_features: NDArray[np.float32]
    graph_feature_names: tuple[str, ...]
    action_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        graph_features = np.asarray(self.graph_features, dtype=np.float32)
        action_mask = np.asarray(self.action_mask, dtype=np.bool_)
        if graph_features.shape != (len(self.graph_feature_names),):
            raise ValueError("graph feature shape must match graph_feature_names")
        if action_mask.ndim != 1:
            raise ValueError("action_mask must be one-dimensional")

        for (source_type, _, target_type), edge_store in self.edges.items():
            if source_type not in self.nodes or target_type not in self.nodes:
                raise ValueError("edge type references an unknown node type")
            if edge_store.edge_index.size == 0:
                continue
            source_count = len(self.nodes[source_type].ids)
            target_count = len(self.nodes[target_type].ids)
            if edge_store.edge_index[0].min() < 0 or edge_store.edge_index[0].max() >= source_count or edge_store.edge_index[1].min() < 0 or edge_store.edge_index[1].max() >= target_count:
                raise ValueError("edge_index contains an invalid node index")

        object.__setattr__(self, "graph_features", graph_features)
        object.__setattr__(self, "action_mask", action_mask)
