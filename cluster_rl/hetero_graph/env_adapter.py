from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from cluster_rl.cluster_env import ClusterEnv

from .builder import ClusterHeteroGraphBuilder, MODULE, WAFER
from .schema import HeteroGraph


@dataclass(frozen=True, slots=True)
class ActionRef:
    """Semantic view of an integer environment action."""

    kind: Literal["pick", "place"]
    entity_type: str
    entity_index: int
    entity_id: object


class GraphEnvAdapter:
    """Expose graph observations without moving transition logic out of Env.

    ``ClusterEnv`` remains responsible for legal actions, time advancement,
    rewards and terminal states. This adapter only translates each resulting
    decision snapshot into a heterogeneous graph.
    """

    def __init__(
        self,
        env: ClusterEnv,
        builder: ClusterHeteroGraphBuilder | None = None,
    ) -> None:
        self.env = env
        self.builder = builder or ClusterHeteroGraphBuilder.from_env(env)
        self.raw_observation: dict[str, Any] | None = None

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, **kwargs: Any) -> tuple[HeteroGraph, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.raw_observation = observation
        return self.builder.build(observation, info), info

    def step(
        self,
        action: int,
    ) -> tuple[HeteroGraph, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.raw_observation = observation
        graph = self.builder.build(observation, info)
        return graph, reward, terminated, truncated, info

    def decode_action(self, action: int) -> ActionRef:
        """Map Env's integer action back to the graph entity it selects."""

        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside the action space")
        wafer_count = len(self.env.wafer_keys)
        if action < wafer_count:
            return ActionRef(
                kind="pick",
                entity_type=WAFER,
                entity_index=action,
                entity_id=self.env.wafer_keys[action],
            )

        module_index = action - wafer_count
        return ActionRef(
            kind="place",
            entity_type=MODULE,
            entity_index=module_index,
            entity_id=self.env.module_ids[module_index],
        )

    def encode_action(
        self,
        kind: Literal["pick", "place"],
        entity_index: int,
    ) -> int:
        """Map a wafer/module node index to Env's integer action."""

        count = len(self.env.wafer_keys) if kind == "pick" else len(self.env.module_ids)
        if not 0 <= entity_index < count:
            raise ValueError(f"{kind} entity_index is outside the graph")
        if kind == "pick":
            return entity_index
        return len(self.env.wafer_keys) + entity_index
