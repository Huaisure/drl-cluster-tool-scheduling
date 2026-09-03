"""Shared relational message passing and one score per anonymous candidate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .graph import EDGE_TYPES, NODE_TYPES, NUMERIC_WIDTH, IRGraph


@dataclass(frozen=True)
class IRBatch:
    node_types: Tensor
    node_features: Tensor
    edge_index: Tensor
    edge_types: Tensor
    graph_index: Tensor
    action_nodes: Tensor
    action_valid: Tensor


def collate_graphs(graphs: list[IRGraph], device: torch.device | str = "cpu") -> IRBatch:
    if not graphs:
        raise ValueError("cannot batch zero graphs")
    counts = [len(g.node_types) for g in graphs]
    offsets = np.cumsum([0, *counts[:-1]])
    actions = np.zeros((len(graphs), max(1, max(g.action_count for g in graphs))), dtype=np.int64)
    valid = np.zeros(actions.shape, dtype=np.bool_)
    for i, (graph, offset) in enumerate(zip(graphs, offsets)):
        actions[i, :graph.action_count] = graph.action_nodes + offset
        valid[i, :graph.action_count] = True
    arrays = (
        np.concatenate([g.node_types for g in graphs]),
        np.concatenate([g.node_features for g in graphs]),
        np.concatenate([g.edge_index + offset for g, offset in zip(graphs, offsets)], axis=1),
        np.concatenate([g.edge_types for g in graphs]),
        np.repeat(np.arange(len(graphs), dtype=np.int64), counts), actions, valid,
    )
    return IRBatch(*(torch.as_tensor(array, device=device) for array in arrays))


class _MessageLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.relations = nn.Embedding(2 * len(EDGE_TYPES), width)
        self.message = nn.Linear(width, width)
        self.update = nn.Sequential(nn.Linear(2 * width + 1, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, h: Tensor, edges: Tensor, types: Tensor) -> Tensor:
        source, target = edges
        relation = self.relations(types)
        messages = torch.nn.functional.silu(self.message(h[source]) + relation)
        totals = torch.zeros_like(h).index_add_(0, target, messages)
        degree = h.new_zeros((len(h), 1)).index_add_(0, target, h.new_ones((len(target), 1)))
        # Mean plus degree retains multiplicity (capacity/ownership counts).
        update = self.update(torch.cat((h, totals / degree.clamp_min(1), degree.log1p()), dim=-1))
        return self.norm(h + update)


@dataclass(frozen=True)
class PolicyOutput:
    logits: Tensor
    value: Tensor


class IRActorCritic(nn.Module):
    def __init__(self, width: int = 64, layers: int = 4) -> None:
        super().__init__()
        if width < 8 or layers < 1:
            raise ValueError("width must be >= 8 and layers must be >= 1")
        self.width, self.depth = width, layers
        self.node_kind = nn.Embedding(len(NODE_TYPES), width)
        self.numeric = nn.Linear(NUMERIC_WIDTH, width)
        self.layers = nn.ModuleList(_MessageLayer(width) for _ in range(layers))
        self.actor = nn.Sequential(nn.Linear(2 * width + 1, width), nn.SiLU(), nn.Linear(width, 1))
        self.critic = nn.Sequential(nn.Linear(width + 1, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, batch: IRBatch) -> PolicyOutput:
        h = self.node_kind(batch.node_types) + self.numeric(batch.node_features)
        for layer in self.layers:
            h = layer(h, batch.edge_index, batch.edge_types)
        size = batch.action_nodes.shape[0]
        pooled = h.new_zeros((size, self.width)).index_add_(0, batch.graph_index, h)
        count = h.new_zeros((size, 1)).index_add_(0, batch.graph_index, h.new_ones((len(h), 1)))
        context = torch.cat((pooled / count.clamp_min(1), count.log1p()), dim=-1)
        candidates = h[batch.action_nodes]
        expanded = context[:, None, :].expand(-1, candidates.shape[1], -1)
        logits = self.actor(torch.cat((candidates, expanded), dim=-1)).squeeze(-1)
        return PolicyOutput(logits.masked_fill(~batch.action_valid, -torch.inf), self.critic(context).squeeze(-1))
