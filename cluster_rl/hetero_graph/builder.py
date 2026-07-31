from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from problem import ClusterProblem, ModuleType, TMArmType, WaferKey

from .schema import EdgeStore, EdgeType, HeteroGraph, NodeStore

WAFER = "wafer"
MODULE = "module"
ROBOT = "robot"

LOCATED_IN: EdgeType = (WAFER, "located_in", MODULE)
HELD_BY: EdgeType = (WAFER, "held_by", ROBOT)
CAN_MOVE_TO: EdgeType = (WAFER, "can_move_to", MODULE)
CAN_ACCESS: EdgeType = (ROBOT, "can_access", MODULE)


def _edge_index(edges: list[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


class ClusterHeteroGraphBuilder:
    """Build a graph snapshot from one ``ClusterEnv`` observation.

    This class is the main extension point. The default features and relations
    are intentionally small examples; replace the ``_build_*`` methods as the
    graph design evolves.
    """

    def __init__(
        self,
        problem: ClusterProblem,
        wafer_keys: Sequence[WaferKey],
        module_ids: Sequence[str],
    ) -> None:
        self.problem = problem
        self.wafer_keys = tuple(wafer_keys)
        self.module_ids = tuple(module_ids)
        self.robot_ids = tuple(problem.ClusterTool)
        self._module_index = {module_id: index for index, module_id in enumerate(self.module_ids)}
        self._robot_index = {robot_id: index for index, robot_id in enumerate(self.robot_ids)}
        self._lp_id = next(module_id for module_id in self.module_ids if problem.Modules[module_id].type is ModuleType.LP)

        if set(self.module_ids) != set(problem.Modules):
            raise ValueError("module_ids must match problem.Modules")
        if set(self.wafer_keys) != set(problem.initial_state.to_snapshot().wafers_by_key):
            raise ValueError("wafer_keys must match the initial wafers")

    @classmethod
    def from_env(cls, env: Any) -> ClusterHeteroGraphBuilder:
        return cls(env.problem, env.wafer_keys, env.module_ids)

    def build(
        self,
        observation: Mapping[str, Any],
        info: Mapping[str, Any] | None = None,
    ) -> HeteroGraph:
        """Create one graph for the current decision state."""

        wafer_module = np.asarray(observation["wafer_module"], dtype=np.int64)
        wafer_step = np.asarray(observation["wafer_step"], dtype=np.int64)
        process_remaining = np.asarray(observation["process_remaining"], dtype=np.float32)
        action_mask = np.asarray(observation["action_mask"], dtype=np.bool_)
        robot_module = int(observation["robot_module"])
        self._validate_observation(
            wafer_module,
            wafer_step,
            process_remaining,
            action_mask,
            robot_module,
        )

        nodes = self._build_nodes(
            wafer_module,
            wafer_step,
            process_remaining,
        )
        edges = self._build_edges(
            wafer_module,
            wafer_step,
            robot_module,
        )
        return HeteroGraph(
            nodes=nodes,
            edges=edges,
            graph_features=np.asarray(
                [float((info or {}).get("time", 0.0))],
                dtype=np.float32,
            ),
            graph_feature_names=("time",),
            action_mask=action_mask,
        )

    def _build_nodes(
        self,
        wafer_module: np.ndarray,
        wafer_step: np.ndarray,
        process_remaining: np.ndarray,
    ) -> dict[str, NodeStore]:
        """Define node features here.

        TODO: add the scheduling state needed by your model, such as remaining
        route work, module cleaning state, load-lock state, or robot-arm state.
        """

        wafer_features = np.empty((len(self.wafer_keys), 3), dtype=np.float32)
        for index, (route_id, _) in enumerate(self.wafer_keys):
            completed_step = len(self.problem.routes[route_id].visits) + 1
            wafer_features[index] = (
                wafer_step[index] / completed_step,
                process_remaining[index],
                float(wafer_module[index] == len(self.module_ids)),
            )

        module_features = np.empty((len(self.module_ids), 4), dtype=np.float32)
        for index, module_id in enumerate(self.module_ids):
            module = self.problem.Modules[module_id]
            module_features[index] = (
                module.capacity,
                float(module.type is ModuleType.LP),
                float(module.type is ModuleType.PM),
                float(module.type is ModuleType.LL),
            )

        robot_features = np.empty((len(self.robot_ids), 3), dtype=np.float32)
        for index, robot_id in enumerate(self.robot_ids):
            robot = self.problem.ClusterTool[robot_id]
            robot_features[index] = (
                robot.pick_time,
                robot.place_time,
                float(robot.arm_type is TMArmType.SINGLE_ARM),
            )

        return {
            WAFER: NodeStore(
                ids=self.wafer_keys,
                features=wafer_features,
                feature_names=(
                    "route_progress",
                    "process_remaining",
                    "on_robot",
                ),
            ),
            MODULE: NodeStore(
                ids=self.module_ids,
                features=module_features,
                feature_names=("capacity", "is_lp", "is_pm", "is_ll"),
            ),
            ROBOT: NodeStore(
                ids=self.robot_ids,
                features=robot_features,
                feature_names=("pick_time", "place_time", "is_single_arm"),
            ),
        }

    def _build_edges(
        self,
        wafer_module: np.ndarray,
        wafer_step: np.ndarray,
        robot_module: int,
    ) -> dict[EdgeType, EdgeStore]:
        """Define graph relations here.

        Static relations come from ``problem``; dynamic relations come from
        the current observation. Add reverse relations explicitly if the
        selected GNN library does not add them for you.
        """

        located_in: list[tuple[int, int]] = []
        held_by: list[tuple[int, int]] = []
        can_move_to: list[tuple[int, int]] = []
        candidate_times: list[tuple[float]] = []

        for wafer_index, (route_id, _) in enumerate(self.wafer_keys):
            if wafer_module[wafer_index] < len(self.module_ids):
                located_in.append((wafer_index, int(wafer_module[wafer_index])))
            else:
                held_by.append((wafer_index, 0))

            route = self.problem.routes[route_id]
            next_step = int(wafer_step[wafer_index]) + 1
            if next_step <= len(route.visits):
                visit = route.visits[next_step - 1]
                targets = visit.module_ids
                process_time = visit.process_time or 0.0
            elif next_step == len(route.visits) + 1:
                targets = (self._lp_id,)
                process_time = 0.0
            else:
                targets = ()
                process_time = 0.0

            for module_id in targets:
                can_move_to.append((wafer_index, self._module_index[module_id]))
                candidate_times.append((process_time,))

        can_access = [(self._robot_index[robot_id], self._module_index[module_id]) for robot_id, robot in self.problem.ClusterTool.items() for module_id in robot.module_ids]

        edges = {
            LOCATED_IN: EdgeStore(
                edge_index=_edge_index(located_in),
                features=np.empty((len(located_in), 0), dtype=np.float32),
            ),
            HELD_BY: EdgeStore(
                edge_index=_edge_index(held_by),
                features=np.empty((len(held_by), 0), dtype=np.float32),
            ),
            CAN_MOVE_TO: EdgeStore(
                edge_index=_edge_index(can_move_to),
                features=np.asarray(candidate_times, dtype=np.float32).reshape(-1, 1),
                feature_names=("process_time",),
            ),
            CAN_ACCESS: EdgeStore(
                edge_index=_edge_index(can_access),
                features=np.empty((len(can_access), 0), dtype=np.float32),
            ),
        }

        if robot_module < len(self.module_ids):
            edges[(ROBOT, "located_at", MODULE)] = EdgeStore(
                edge_index=np.asarray([[0], [robot_module]], dtype=np.int64),
                features=np.empty((1, 0), dtype=np.float32),
            )
        return edges

    def _validate_observation(
        self,
        wafer_module: np.ndarray,
        wafer_step: np.ndarray,
        process_remaining: np.ndarray,
        action_mask: np.ndarray,
        robot_module: int,
    ) -> None:
        wafer_count = len(self.wafer_keys)
        module_count = len(self.module_ids)
        if wafer_module.shape != (wafer_count,) or wafer_step.shape != (wafer_count,) or process_remaining.shape != (wafer_count,):
            raise ValueError("observation has invalid wafer array shapes")
        if action_mask.shape != (wafer_count + module_count,):
            raise ValueError("action_mask must follow [wafer, module] order")
        if np.any((wafer_module < 0) | (wafer_module > module_count)):
            raise ValueError("wafer_module contains an invalid index")
        if not 0 <= robot_module <= module_count:
            raise ValueError("robot_module contains an invalid index")
