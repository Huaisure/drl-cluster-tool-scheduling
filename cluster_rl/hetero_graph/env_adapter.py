from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from cluster_rl.cluster_env import ADVANCE, PICK, PLACE, ClusterEnv

from .builder import ClusterHeteroGraphBuilder, MODULE, WAFER
from .schema import HeteroGraph


@dataclass(frozen=True, slots=True)
class ActionRef:
    """Semantic view of a flat environment action."""

    kind: Literal["pick", "place", "advance"]
    entity_type: str | None
    entity_index: int | None
    entity_id: object | None
    target_type: str | None
    target_index: int | None
    target_id: object | None


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
        return self.builder.build(observation), info

    def step(
        self,
        action: int,
    ) -> tuple[HeteroGraph, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.raw_observation = observation
        graph = self.builder.build(observation)
        return graph, reward, terminated, truncated, info

    def decode_action(self, action: int) -> ActionRef:
        """Map an Env action back to the selected entity and robot."""

        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside the action space")
        kind, entity_index, robot_index = self.env._decode_action(int(action))
        if kind == ADVANCE:
            return ActionRef(ADVANCE, None, None, None, None, None, None)

        assert entity_index is not None and robot_index is not None
        if kind == PICK:
            return ActionRef(
                kind=PICK,
                entity_type=WAFER,
                entity_index=entity_index,
                entity_id=self.env.wafer_keys[entity_index],
                target_type="robot",
                target_index=robot_index,
                target_id=self.builder.robot_ids[robot_index],
            )

        return ActionRef(
            kind=PLACE,
            entity_type=WAFER,
            entity_index=entity_index,
            entity_id=self.env.wafer_keys[entity_index],
            target_type=MODULE,
            target_index=robot_index,
            target_id=self.env.module_ids[robot_index],
        )

    def encode_action(
        self,
        kind: Literal["pick", "place", "advance"],
        entity_index: int | None = None,
        target_index: int | None = None,
    ) -> int:
        """Map graph entity and robot indexes to an Env action."""

        if kind == ADVANCE:
            if entity_index is not None or target_index is not None:
                raise ValueError("advance does not select an entity or robot")
            return int(self.action_space.n) - 1
        if kind not in {PICK, PLACE}:
            raise ValueError(f"unknown action kind: {kind!r}")
        if entity_index is None or target_index is None:
            raise ValueError(f"{kind} requires entity_index and target_index")

        count = (
            len(self.env.wafer_keys)
        )
        if not 0 <= entity_index < count:
            raise ValueError(f"{kind} entity_index is outside the graph")
        target_count = (
            len(self.builder.robot_ids)
            if kind == PICK
            else len(self.env.module_ids)
        )
        if not 0 <= target_index < target_count:
            raise ValueError("target_index is outside the graph")
        if kind == PICK:
            return entity_index * len(self.builder.robot_ids) + target_index
        return (
            len(self.env.wafer_keys) * len(self.builder.robot_ids)
            + entity_index * len(self.env.module_ids)
            + target_index
        )
