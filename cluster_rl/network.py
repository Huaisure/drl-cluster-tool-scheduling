from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import HGTConv
from torch_geometric.utils import to_dense_batch

from cluster_rl.hetero_graph.builder import (
    EDGE_TYPES,
    GLOBAL,
    MODULE,
    NODE_TYPES,
    ROBOT,
    ROUTE_STEP,
    WAFER,
    ClusterHeteroGraphBuilder,
)
from cluster_rl.hetero_graph.feature_schema import (
    GLOBAL_FEATURE_NAMES,
    MODULE_FEATURE_NAMES,
    ROBOT_FEATURE_NAMES,
    ROUTE_STEP_FEATURE_NAMES,
    WAFER_FEATURE_NAMES,
)
from cluster_rl.hetero_graph.schema import HeteroGraph
from problem import ClusterProblem, WaferKey

PICK_ACTION = 0
PLACE_ACTION = 1
ADVANCE_ACTION = 2
ACTION_TYPE_COUNT = 3
PAD_ACTION = 3

NODE_FEATURE_DIMS = {
    GLOBAL: len(GLOBAL_FEATURE_NAMES),
    WAFER: len(WAFER_FEATURE_NAMES),
    ROUTE_STEP: len(ROUTE_STEP_FEATURE_NAMES),
    MODULE: len(MODULE_FEATURE_NAMES),
    ROBOT: len(ROBOT_FEATURE_NAMES),
}


@dataclass(frozen=True)
class EncodedObservation:
    """One graph state and its compact legal-action description."""

    graph: HeteroData
    env_action_indices: np.ndarray
    action_kind: np.ndarray
    action_entity: np.ndarray
    action_target: np.ndarray

    @property
    def action_count(self) -> int:
        return int(self.env_action_indices.shape[0])


@dataclass(frozen=True)
class EntityBatch:
    """Batched heterogeneous graphs and padded action queries."""

    graph: HeteroData
    action_valid: Tensor
    env_action_indices: Tensor
    action_kind: Tensor
    action_entity: Tensor
    action_target: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.action_valid.shape[0])

    @property
    def action_mask(self) -> Tensor:
        """Mask real legal actions against batch padding."""

        return self.action_valid

    def to(self, device: torch.device | str) -> EntityBatch:
        return EntityBatch(
            graph=self.graph.to(device),
            action_valid=self.action_valid.to(device),
            env_action_indices=self.env_action_indices.to(device),
            action_kind=self.action_kind.to(device),
            action_entity=self.action_entity.to(device),
            action_target=self.action_target.to(device),
        )

    def to_model_actions(self, env_actions: Tensor) -> Tensor:
        """Map flat environment actions to compact legal-action indexes."""

        if env_actions.shape != (self.batch_size,):
            raise ValueError("env_actions must contain one action per batch item")
        env_actions = env_actions.to(
            self.action_valid.device,
            dtype=torch.long,
        )
        matches = (
            self.env_action_indices == env_actions[:, None]
        ) & self.action_valid
        if not torch.all(matches.sum(dim=1) == 1):
            raise ValueError("env_actions contains an action that is not legal")
        return matches.to(dtype=torch.long).argmax(dim=1)

    def to_env_actions(self, model_actions: Tensor) -> Tensor:
        """Map compact legal-action indexes back to flat Env actions."""

        model_actions = self._validated_model_actions(model_actions)
        return self.env_action_indices.gather(
            1,
            model_actions[:, None],
        ).squeeze(1)

    def _validated_model_actions(self, actions: Tensor) -> Tensor:
        if actions.shape != (self.batch_size,):
            raise ValueError("model_actions must contain one action per batch item")
        actions = actions.to(self.action_valid.device, dtype=torch.long)
        in_range = (actions >= 0) & (actions < self.action_valid.shape[1])
        safe_actions = actions.clamp(0, self.action_valid.shape[1] - 1)
        valid = self.action_valid.gather(1, safe_actions[:, None]).squeeze(1)
        if not torch.all(in_range & valid):
            raise ValueError("model_actions contains a padded or out-of-range action")
        return actions


@dataclass(frozen=True)
class PolicyValueOutput:
    logits: Tensor
    value: Tensor


@dataclass(frozen=True)
class TransformerConfig:
    model_dim: int = 128
    num_heads: int = 8
    hgt_layers: int = 2
    num_layers: int = 2
    feedforward_dim: int = 512
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if self.num_heads <= 0 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.hgt_layers <= 0 or self.num_layers <= 0:
            raise ValueError("hgt_layers and num_layers must be positive")
        if self.feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class ClusterObservationEncoder:
    """Build the canonical heterogeneous graph used by the policy."""

    def __init__(
        self,
        problem: ClusterProblem,
        wafer_keys: Sequence[WaferKey],
        module_ids: Sequence[str],
        time_scale: float | None = None,
    ) -> None:
        del time_scale
        self.builder = ClusterHeteroGraphBuilder(
            problem,
            wafer_keys,
            module_ids,
        )

    @property
    def problem(self) -> ClusterProblem:
        return self.builder.problem

    @property
    def wafer_keys(self) -> tuple[WaferKey, ...]:
        return self.builder.wafer_keys

    @property
    def module_ids(self) -> tuple[str, ...]:
        return self.builder.module_ids

    @classmethod
    def from_env(
        cls,
        env: Any,
        time_scale: float | None = None,
    ) -> ClusterObservationEncoder:
        return cls(
            env.problem,
            env.wafer_keys,
            env.module_ids,
            time_scale,
        )

    def encode(
        self,
        observation: Mapping[str, Any],
    ) -> EncodedObservation:
        graph = self.builder.build(observation)
        return _encode_graph(graph)


def _encode_graph(graph: HeteroGraph) -> EncodedObservation:
    data = HeteroData()
    for node_type in NODE_TYPES:
        store = graph.nodes[node_type]
        data[node_type].x = torch.from_numpy(store.features.copy())
        data[node_type].num_nodes = len(store.ids)
    for edge_type in EDGE_TYPES:
        data[edge_type].edge_index = torch.from_numpy(
            graph.edges[edge_type].edge_index.copy()
        )

    wafer_count = len(graph.nodes[WAFER].ids)
    robot_count = len(graph.nodes[ROBOT].ids)
    module_count = len(graph.nodes[MODULE].ids)
    pick_count = wafer_count * robot_count
    place_count = wafer_count * module_count
    flat_action_mask = np.concatenate(
        (
            graph.pick_action_mask.reshape(-1),
            graph.place_action_mask.reshape(-1),
            np.asarray([graph.can_advance]),
        )
    )
    env_action_indices = np.flatnonzero(flat_action_mask).astype(
        np.int64,
        copy=False,
    )
    if not env_action_indices.size:
        raise ValueError("cannot encode a state without legal actions")

    action_kind = np.full(
        env_action_indices.shape,
        ADVANCE_ACTION,
        dtype=np.int64,
    )
    action_entity = np.zeros_like(env_action_indices)
    action_target = np.zeros_like(env_action_indices)
    pick = env_action_indices < pick_count
    place = (env_action_indices >= pick_count) & (
        env_action_indices < pick_count + place_count
    )
    action_kind[pick] = PICK_ACTION
    action_entity[pick], action_target[pick] = np.divmod(
        env_action_indices[pick],
        robot_count,
    )
    action_kind[place] = PLACE_ACTION
    action_entity[place], action_target[place] = np.divmod(
        env_action_indices[place] - pick_count,
        module_count,
    )
    return EncodedObservation(
        graph=data,
        env_action_indices=env_action_indices,
        action_kind=action_kind,
        action_entity=action_entity,
        action_target=action_target,
    )


def collate_observations(
    encoders: Sequence[ClusterObservationEncoder],
    observations: Sequence[Mapping[str, Any]],
    *,
    device: torch.device | str | None = None,
) -> EntityBatch:
    """Build and batch graphs from potentially different problem instances."""

    if not encoders or len(encoders) != len(observations):
        raise ValueError(
            "encoders and observations must have the same non-zero length"
        )
    encoded = [
        encoder.encode(observation)
        for encoder, observation in zip(encoders, observations)
    ]
    return collate_encoded_observations(encoded, device=device)


def collate_encoded_observations(
    encoded: Sequence[EncodedObservation],
    *,
    device: torch.device | str | None = None,
) -> EntityBatch:
    """Batch observations already encoded by CPU environment workers."""

    if not encoded:
        raise ValueError("encoded must not be empty")

    graph = Batch.from_data_list([item.graph for item in encoded])
    return _collate_encoded_batch(encoded, graph, device)


def collate_encoded_observations_fast(
    encoded: Sequence[EncodedObservation],
    *,
    device: torch.device | str | None = None,
) -> EntityBatch:
    """Batch the fixed training graph schema without PyG slice metadata."""

    if not encoded:
        raise ValueError("encoded must not be empty")

    batch_size = len(encoded)
    graph = HeteroData()
    node_ptrs: dict[str, Tensor] = {}

    for node_type in NODE_TYPES:
        features = [item.graph[node_type].x for item in encoded]
        counts = torch.tensor(
            [feature.shape[0] for feature in features],
            dtype=torch.long,
        )
        ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        graph[node_type].x = torch.cat(features, dim=0)
        graph[node_type].batch = torch.repeat_interleave(
            torch.arange(batch_size), counts
        )
        graph[node_type].ptr = ptr
        graph[node_type].num_nodes = ptr[-1].item()
        node_ptrs[node_type] = ptr

    for edge_type in EDGE_TYPES:
        source_type, _, target_type = edge_type
        edges = [item.graph[edge_type].edge_index for item in encoded]
        counts = torch.tensor(
            [edge.shape[1] for edge in edges],
            dtype=torch.long,
        )
        edge_index = torch.cat(edges, dim=1)
        if edge_index.numel():
            edge_index = edge_index + torch.stack(
                (
                    torch.repeat_interleave(
                        node_ptrs[source_type][:-1], counts
                    ),
                    torch.repeat_interleave(
                        node_ptrs[target_type][:-1], counts
                    ),
                )
            )
        graph[edge_type].edge_index = edge_index

    return _collate_encoded_batch(encoded, graph, device)


def _collate_encoded_batch(
    encoded: Sequence[EncodedObservation],
    graph: HeteroData,
    device: torch.device | str | None,
) -> EntityBatch:
    """Pad action descriptions and attach them to an already batched graph."""

    max_actions = max(item.action_count for item in encoded)
    batch_size = len(encoded)
    action_valid = torch.zeros(batch_size, max_actions, dtype=torch.bool)
    env_action_indices = torch.full(
        (batch_size, max_actions),
        -1,
        dtype=torch.long,
    )
    action_kind = torch.full(
        (batch_size, max_actions), PAD_ACTION, dtype=torch.long
    )
    action_entity = torch.zeros(batch_size, max_actions, dtype=torch.long)
    action_target = torch.zeros_like(action_entity)

    for index, item in enumerate(encoded):
        action_count = item.action_count
        action_valid[index, :action_count] = True
        env_action_indices[index, :action_count] = torch.from_numpy(
            item.env_action_indices
        )
        action_kind[index, :action_count] = torch.from_numpy(item.action_kind)
        action_entity[index, :action_count] = torch.from_numpy(
            item.action_entity
        )
        action_target[index, :action_count] = torch.from_numpy(
            item.action_target
        )

    return EntityBatch(
        graph=graph,
        action_valid=action_valid,
        env_action_indices=env_action_indices,
        action_kind=action_kind,
        action_entity=action_entity,
        action_target=action_target,
    ).to(device or "cpu")


class ClusterActorCritic(nn.Module):
    """HGT graph encoder with a Transformer action-query decoder."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        model_dim = self.config.model_dim
        metadata = (list(NODE_TYPES), list(EDGE_TYPES))

        self.node_encoders = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(feature_dim, model_dim),
                    nn.GELU(),
                    nn.LayerNorm(model_dim),
                )
                for node_type, feature_dim in NODE_FEATURE_DIMS.items()
            }
        )
        self.hgt_layers = nn.ModuleList(
            HGTConv(
                model_dim,
                model_dim,
                metadata,
                heads=self.config.num_heads,
            )
            for _ in range(self.config.hgt_layers)
        )
        self.hgt_norms = nn.ModuleList(
            nn.ModuleDict(
                {node_type: nn.LayerNorm(model_dim) for node_type in NODE_TYPES}
            )
            for _ in range(self.config.hgt_layers)
        )
        self.hgt_dropout = nn.Dropout(self.config.dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.feedforward_dim,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(model_dim),
        )
        self.action_type_embedding = nn.Embedding(
            ACTION_TYPE_COUNT,
            model_dim,
        )
        self.advance_query = nn.Parameter(torch.empty(model_dim))
        self.actor_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        nn.init.normal_(self.advance_query, std=0.02)

    def forward(self, batch: EntityBatch) -> PolicyValueOutput:
        self._validate_batch(batch)
        x_dict = {
            node_type: self.node_encoders[node_type](
                batch.graph[node_type].x
            )
            for node_type in NODE_TYPES
        }
        for layer, norms in zip(self.hgt_layers, self.hgt_norms):
            updated = layer(x_dict, batch.graph.edge_index_dict)
            x_dict = {
                node_type: norms[node_type](
                    x_dict[node_type]
                    + self.hgt_dropout(updated[node_type])
                )
                for node_type in NODE_TYPES
            }

        dense_nodes = {
            node_type: to_dense_batch(
                x_dict[node_type],
                batch.graph[node_type].batch,
                batch_size=batch.batch_size,
            )
            for node_type in NODE_TYPES
        }
        memory = torch.cat(
            [dense_nodes[node_type][0] for node_type in NODE_TYPES],
            dim=1,
        )
        memory_valid = torch.cat(
            [dense_nodes[node_type][1] for node_type in NODE_TYPES],
            dim=1,
        )
        queries = self._action_queries(batch, dense_nodes)
        decoded = self.decoder(
            queries,
            memory,
            tgt_key_padding_mask=~batch.action_valid,
            memory_key_padding_mask=~memory_valid,
        )
        logits = self.actor_head(decoded).squeeze(-1)
        logits = logits.masked_fill(~batch.action_valid, -torch.inf)
        global_state = dense_nodes[GLOBAL][0][:, 0]
        value = self.value_head(global_state).squeeze(-1)
        return PolicyValueOutput(logits=logits, value=value)

    def _action_queries(
        self,
        batch: EntityBatch,
        dense_nodes: dict[str, tuple[Tensor, Tensor]],
    ) -> Tensor:
        batch_index = torch.arange(
            batch.batch_size,
            device=batch.action_valid.device,
        )[:, None]
        wafer = dense_nodes[WAFER][0][
            batch_index,
            batch.action_entity.clamp_max(dense_nodes[WAFER][0].shape[1] - 1),
        ]
        module = dense_nodes[MODULE][0][
            batch_index,
            batch.action_target.clamp_max(dense_nodes[MODULE][0].shape[1] - 1),
        ]
        robot = dense_nodes[ROBOT][0][
            batch_index,
            batch.action_target.clamp_max(dense_nodes[ROBOT][0].shape[1] - 1),
        ]
        global_state = dense_nodes[GLOBAL][0][:, :1].expand(
            -1,
            batch.action_valid.shape[1],
            -1,
        )
        pick = batch.action_kind == PICK_ACTION
        place = batch.action_kind == PLACE_ACTION
        advance = batch.action_kind == ADVANCE_ACTION
        queries = torch.zeros_like(wafer)
        queries = torch.where(pick[..., None], wafer + robot, queries)
        queries = torch.where(place[..., None], wafer + module, queries)
        queries = torch.where(
            advance[..., None],
            global_state + self.advance_query,
            queries,
        )
        safe_kind = batch.action_kind.clamp_max(ADVANCE_ACTION)
        queries = queries + self.action_type_embedding(safe_kind)
        return queries.masked_fill(~batch.action_valid[..., None], 0.0)

    @staticmethod
    def _validate_batch(batch: EntityBatch) -> None:
        expected_shape = batch.action_valid.shape
        if batch.action_valid.ndim != 2:
            raise ValueError("action_valid must have shape [batch, action]")
        for name in (
            "env_action_indices",
            "action_kind",
            "action_entity",
            "action_target",
        ):
            if getattr(batch, name).shape != expected_shape:
                raise ValueError(f"{name} has an invalid shape")
        if torch.any(batch.env_action_indices[batch.action_valid] < 0):
            raise ValueError("legal actions must identify an Env action")
        if set(batch.graph.node_types) != set(NODE_TYPES):
            raise ValueError("graph has unexpected node types")
        if set(batch.graph.edge_types) != set(EDGE_TYPES):
            raise ValueError("graph has unexpected edge types")
