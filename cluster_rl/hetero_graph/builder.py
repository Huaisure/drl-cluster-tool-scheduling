from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np

from cluster_rl.cluster_env import LoadLockSide, RobotPhase
from problem import (
    ClusterProblem,
    LoadLockState,
    ModuleType,
    RouteVisit,
    TMArmType,
    WaferKey,
)

from .feature_schema import (
    GLOBAL_FEATURE_NAMES,
    MODULE_FEATURE_NAMES,
    ROBOT_FEATURE_NAMES,
    ROUTE_STEP_FEATURE_NAMES,
    TIME_SCALE_SECONDS,
    WAFER_FEATURE_NAMES,
)
from .schema import EdgeStore, EdgeType, HeteroGraph, NodeStore

RouteStepKey: TypeAlias = tuple[str, int]

GLOBAL = "global"
WAFER = "wafer"
ROUTE_STEP = "route_step"
MODULE = "module"
ROBOT = "robot"

LOCATED_IN: EdgeType = (WAFER, "located_in", MODULE)
CONTAINS: EdgeType = (MODULE, "contains", WAFER)
HELD_BY: EdgeType = (WAFER, "held_by", ROBOT)
HOLDS: EdgeType = (ROBOT, "holds", WAFER)
AT_STEP: EdgeType = (WAFER, "at_step", ROUTE_STEP)
CURRENT_FOR: EdgeType = (ROUTE_STEP, "current_for", WAFER)
NEXT_STEP: EdgeType = (WAFER, "next_step", ROUTE_STEP)
NEXT_FOR: EdgeType = (ROUTE_STEP, "next_for", WAFER)
CAN_RUN_ON: EdgeType = (ROUTE_STEP, "can_run_on", MODULE)
SUPPORTS_STEP: EdgeType = (MODULE, "supports_step", ROUTE_STEP)
PRECEDES: EdgeType = (ROUTE_STEP, "precedes", ROUTE_STEP)
FOLLOWS: EdgeType = (ROUTE_STEP, "follows", ROUTE_STEP)
CAN_ACCESS: EdgeType = (ROBOT, "can_access", MODULE)
ACCESSIBLE_BY: EdgeType = (MODULE, "accessible_by", ROBOT)
ACCESSES_ATMOSPHERE: EdgeType = (ROBOT, "accesses_atmosphere", MODULE)
ATMOSPHERE_ACCESSIBLE_BY: EdgeType = (
    MODULE,
    "atmosphere_accessible_by",
    ROBOT,
)
ACCESSES_VACUUM: EdgeType = (ROBOT, "accesses_vacuum", MODULE)
VACUUM_ACCESSIBLE_BY: EdgeType = (MODULE, "vacuum_accessible_by", ROBOT)
LOCATED_AT: EdgeType = (ROBOT, "located_at", MODULE)
HAS_ROBOT: EdgeType = (MODULE, "has_robot", ROBOT)
OPERATES_ON: EdgeType = (ROBOT, "operates_on", WAFER)
OPERATION_OF: EdgeType = (WAFER, "operation_of", ROBOT)
OPERATION_AT: EdgeType = (ROBOT, "operation_at", MODULE)
HAS_OPERATION: EdgeType = (MODULE, "has_operation", ROBOT)
RETURNS_TO: EdgeType = (WAFER, "returns_to", MODULE)
RETURN_DESTINATION_OF: EdgeType = (
    MODULE,
    "return_destination_of",
    WAFER,
)

NODE_TYPES = (GLOBAL, WAFER, ROUTE_STEP, MODULE, ROBOT)
EDGE_TYPES = (
    LOCATED_IN,
    CONTAINS,
    HELD_BY,
    HOLDS,
    AT_STEP,
    CURRENT_FOR,
    NEXT_STEP,
    NEXT_FOR,
    CAN_RUN_ON,
    SUPPORTS_STEP,
    PRECEDES,
    FOLLOWS,
    CAN_ACCESS,
    ACCESSIBLE_BY,
    ACCESSES_ATMOSPHERE,
    ATMOSPHERE_ACCESSIBLE_BY,
    ACCESSES_VACUUM,
    VACUUM_ACCESSIBLE_BY,
    LOCATED_AT,
    HAS_ROBOT,
    OPERATES_ON,
    OPERATION_OF,
    OPERATION_AT,
    HAS_OPERATION,
    RETURNS_TO,
    RETURN_DESTINATION_OF,
    *((GLOBAL, "contextualizes", node_type) for node_type in NODE_TYPES[1:]),
    *((node_type, "summarizes_into", GLOBAL) for node_type in NODE_TYPES[1:]),
)


def _edge_store(edges: list[tuple[int, int]]) -> EdgeStore:
    edge_index = (
        np.asarray(edges, dtype=np.int64).T
        if edges
        else np.empty((2, 0), dtype=np.int64)
    )
    return EdgeStore(edge_index=edge_index)


def _reverse(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(target, source) for source, target in edges]


class ClusterHeteroGraphBuilder:
    """Build one heterogeneous graph from a ``ClusterEnv`` observation.

    Location indexes follow the environment contract:

    - ``wafer_loc`` uses modules first, followed by robots;
    - ``robot_loc`` uses module indexes and ``module_count`` for unknown;
    - ``action_mask`` is flat in entity-major order and ends with ADVANCE;
    - graph Pick actions use shape ``[wafer_count, robot_count]``;
    - graph Place actions use shape ``[wafer_count, module_count]``.

    A route with ``N`` visits owns route-step nodes ``1..N+1``. The last node
    is a synthetic zero-time step for the wafer's final return to LP.
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
        self.robot_ids = tuple(sorted(problem.ClusterTool))
        self._max_arm_capacity = max(
            1 if robot.arm_type is TMArmType.SINGLE_ARM else 2
            for robot in problem.ClusterTool.values()
        )

        snapshot = problem.initial_state.to_snapshot()
        if set(self.module_ids) != set(problem.Modules):
            raise ValueError("module_ids must match problem.Modules")
        if set(self.wafer_keys) != set(
            snapshot.wafers_by_key
        ):
            raise ValueError("wafer_keys must match the initial wafers")
        self.return_module_ids = tuple(
            problem.return_module_id(snapshot.wafers_by_key[key])
            for key in self.wafer_keys
        )
        self._return_modules_by_route = {
            route_id: tuple(
                sorted(
                    {
                        return_module_id
                        for key, return_module_id in zip(
                            self.wafer_keys,
                            self.return_module_ids,
                        )
                        if key[0] == route_id
                    }
                )
            )
            for route_id in problem.routes
        }

        self.route_step_ids: tuple[RouteStepKey, ...] = tuple(
            (route_id, step)
            for route_id in sorted(problem.routes)
            for step in range(1, len(problem.routes[route_id].visits) + 2)
        )
        self._module_index = {
            module_id: index
            for index, module_id in enumerate(self.module_ids)
        }
        self._robot_index = {
            robot_id: index
            for index, robot_id in enumerate(self.robot_ids)
        }
        self._route_step_index = {
            route_step: index
            for index, route_step in enumerate(self.route_step_ids)
        }
        self._total_process_time = sum(
            visit.process_time or 0.0
            for route_id, _ in self.wafer_keys
            for visit in problem.routes[route_id].visits
        )

    @classmethod
    def from_env(cls, env: Any) -> ClusterHeteroGraphBuilder:
        return cls(env.problem, env.wafer_keys, env.module_ids)

    def build(
        self,
        observation: Mapping[str, Any],
    ) -> HeteroGraph:
        wafer_loc = np.asarray(observation["wafer_loc"], dtype=np.int64)
        wafer_step = np.asarray(observation["wafer_step"], dtype=np.int64)
        process_remaining = np.asarray(
            observation["process_remaining"], dtype=np.float32
        )
        wafer_priority = np.asarray(
            observation["wafer_priority"], dtype=np.float32
        )
        wafer_index_feature = np.asarray(
            observation["wafer_index"], dtype=np.float32
        )
        ll_pump_time = np.asarray(
            observation["ll_pump_time"], dtype=np.float32
        )
        ll_vent_time = np.asarray(
            observation["ll_vent_time"], dtype=np.float32
        )
        ll_last_pick_side = np.asarray(
            observation["ll_last_pick_side"], dtype=np.int64
        )
        ll_empty_transition_progress = np.asarray(
            observation["ll_empty_transition_progress"], dtype=np.float32
        )
        ll_occupied_exit_side = np.asarray(
            observation["ll_occupied_exit_side"], dtype=np.int64
        )
        ll_occupied_transition_progress = np.asarray(
            observation["ll_occupied_transition_progress"], dtype=np.float32
        )
        robot_loc = np.asarray(observation["robot_loc"], dtype=np.int64)
        robot_holding = np.asarray(
            observation["robot_holding"], dtype=np.int64
        )
        robot_phase = np.asarray(
            observation["robot_phase"], dtype=np.int64
        )
        operation_wafer = np.asarray(
            observation["robot_operation_wafer"], dtype=np.int64
        )
        operation_module = np.asarray(
            observation["robot_operation_module"], dtype=np.int64
        )
        time_to_operation_start = np.asarray(
            observation["time_to_operation_start"], dtype=np.float32
        )
        time_to_operation_end = np.asarray(
            observation["time_to_operation_end"], dtype=np.float32
        )
        flat_action_mask = np.asarray(
            observation["action_mask"], dtype=np.bool_
        )

        self._validate_observation(
            wafer_loc,
            wafer_step,
            process_remaining,
            wafer_priority,
            wafer_index_feature,
            ll_pump_time,
            ll_vent_time,
            ll_last_pick_side,
            ll_empty_transition_progress,
            ll_occupied_exit_side,
            ll_occupied_transition_progress,
            robot_loc,
            robot_holding,
            robot_phase,
            operation_wafer,
            operation_module,
            time_to_operation_start,
            time_to_operation_end,
            flat_action_mask,
        )

        pick_count = len(self.wafer_keys) * len(self.robot_ids)
        place_count = len(self.wafer_keys) * len(self.module_ids)
        pick_action_mask = flat_action_mask[:pick_count].reshape(
            len(self.wafer_keys), len(self.robot_ids)
        )
        place_action_mask = flat_action_mask[
            pick_count : pick_count + place_count
        ].reshape(len(self.wafer_keys), len(self.module_ids))

        return HeteroGraph(
            nodes=self._build_nodes(
                wafer_loc,
                wafer_step,
                process_remaining,
                wafer_priority,
                wafer_index_feature,
                ll_pump_time,
                ll_vent_time,
                ll_last_pick_side,
                ll_empty_transition_progress,
                ll_occupied_exit_side,
                ll_occupied_transition_progress,
                robot_holding,
                robot_phase,
                operation_module,
                time_to_operation_start,
                time_to_operation_end,
            ),
            edges=self._build_edges(
                wafer_loc,
                wafer_step,
                robot_loc,
                robot_holding,
                operation_wafer,
                operation_module,
            ),
            pick_action_mask=pick_action_mask,
            place_action_mask=place_action_mask,
            can_advance=bool(flat_action_mask[-1]),
        )

    def _build_nodes(
        self,
        wafer_loc: np.ndarray,
        wafer_step: np.ndarray,
        process_remaining: np.ndarray,
        wafer_priority: np.ndarray,
        wafer_index_feature: np.ndarray,
        ll_pump_time: np.ndarray,
        ll_vent_time: np.ndarray,
        ll_last_pick_side: np.ndarray,
        ll_empty_transition_progress: np.ndarray,
        ll_occupied_exit_side: np.ndarray,
        ll_occupied_transition_progress: np.ndarray,
        robot_holding: np.ndarray,
        robot_phase: np.ndarray,
        operation_module: np.ndarray,
        time_to_operation_start: np.ndarray,
        time_to_operation_end: np.ndarray,
    ) -> dict[str, NodeStore]:
        module_count = len(self.module_ids)
        wafer_count = len(self.wafer_keys)
        holding_rank = np.zeros(wafer_count, dtype=np.float32)
        for row in robot_holding:
            for rank, wafer_index in enumerate(row, start=1):
                if wafer_index < wafer_count:
                    holding_rank[wafer_index] = rank

        wafer_features = np.empty(
            (len(self.wafer_keys), len(WAFER_FEATURE_NAMES)),
            dtype=np.float32,
        )
        for index, (route_id, _) in enumerate(self.wafer_keys):
            route = self.problem.routes[route_id]
            process_times = tuple(
                visit.process_time or 0.0 for visit in route.visits
            )
            completed_step = len(route.visits) + 1
            step = int(wafer_step[index])
            wafer_features[index] = (
                step / completed_step,
                process_remaining[index] / TIME_SCALE_SECONDS,
                float(process_remaining[index] == 0.0),
                float(step == completed_step),
                (completed_step - step) / completed_step,
                sum(process_times[step:]) / TIME_SCALE_SECONDS,
                holding_rank[index],
                wafer_priority[index],
                wafer_index_feature[index],
            )

        physical_occupancy = np.bincount(
            wafer_loc[wafer_loc < module_count],
            minlength=module_count,
        )
        place_reservations = np.zeros(module_count, dtype=np.int64)
        for phase, module_index in zip(robot_phase, operation_module):
            if module_index >= module_count:
                continue
            if phase == RobotPhase.PLACING:
                physical_occupancy[module_index] += 1
            elif phase == RobotPhase.TRAVEL_TO_PLACE:
                place_reservations[module_index] += 1
        committed_occupancy = physical_occupancy + place_reservations
        module_features = np.empty(
            (module_count, len(MODULE_FEATURE_NAMES)),
            dtype=np.float32,
        )
        for index, module_id in enumerate(self.module_ids):
            module = self.problem.Modules[module_id]
            capacity = module.capacity
            available = capacity - committed_occupancy[index]
            module_features[index] = (
                float(module.type is ModuleType.IO),
                float(
                    module.type
                    in (ModuleType.PM, ModuleType.AL, ModuleType.BUFFER)
                ),
                float(module.type is ModuleType.LL),
                capacity,
                physical_occupancy[index] / capacity,
                available / capacity,
                float(available <= 0),
                ll_pump_time[index] / TIME_SCALE_SECONDS,
                ll_vent_time[index] / TIME_SCALE_SECONDS,
                float(ll_last_pick_side[index] == LoadLockSide.ATMOSPHERE),
                float(ll_last_pick_side[index] == LoadLockSide.VACUUM),
                ll_empty_transition_progress[index],
                float(
                    ll_occupied_exit_side[index] == LoadLockSide.ATMOSPHERE
                ),
                float(ll_occupied_exit_side[index] == LoadLockSide.VACUUM),
                ll_occupied_transition_progress[index],
            )

        held_count = np.sum(robot_holding < wafer_count, axis=1)
        robot_features = np.empty(
            (len(self.robot_ids), len(ROBOT_FEATURE_NAMES)),
            dtype=np.float32,
        )
        for index, robot_id in enumerate(self.robot_ids):
            robot = self.problem.ClusterTool[robot_id]
            arm_capacity = (
                1 if robot.arm_type is TMArmType.SINGLE_ARM else 2
            )
            available = arm_capacity - held_count[index]
            robot_features[index] = (
                robot.pick_time / TIME_SCALE_SECONDS,
                robot.place_time / TIME_SCALE_SECONDS,
                robot.travel_times / TIME_SCALE_SECONDS,
                arm_capacity,
                held_count[index] / arm_capacity,
                available / arm_capacity,
                float(available == 0),
                float(robot_phase[index] == RobotPhase.IDLE),
                float(robot_phase[index] == RobotPhase.TRAVEL_TO_PICK),
                float(robot_phase[index] == RobotPhase.PICKING),
                float(robot_phase[index] == RobotPhase.TRAVEL_TO_PLACE),
                float(robot_phase[index] == RobotPhase.PLACING),
                time_to_operation_start[index] / TIME_SCALE_SECONDS,
                time_to_operation_end[index] / TIME_SCALE_SECONDS,
            )

        route_step_features = np.empty(
            (len(self.route_step_ids), len(ROUTE_STEP_FEATURE_NAMES)),
            dtype=np.float32,
        )
        for index, (route_id, step) in enumerate(self.route_step_ids):
            route = self.problem.routes[route_id]
            is_return_to_source = step == len(route.visits) + 1
            visit = None if is_return_to_source else route.visits[step - 1]
            residency_time = None if visit is None else self._residency_time(visit)
            route_step_features[index] = (
                (
                    0.0
                    if visit is None
                    else (visit.process_time or 0.0) / TIME_SCALE_SECONDS
                ),
                (residency_time or 0.0) / TIME_SCALE_SECONDS,
                float(residency_time is not None),
                step / (len(route.visits) + 1),
                float(is_return_to_source),
            )

        completed_steps = 0
        total_steps = 0
        completed_wafers = 0
        remaining_process_time = 0.0
        for index, (route_id, _) in enumerate(self.wafer_keys):
            route = self.problem.routes[route_id]
            completed_step = len(route.visits) + 1
            step = int(wafer_step[index])
            completed_steps += step
            total_steps += completed_step
            completed_wafers += step == completed_step
            remaining_process_time += process_remaining[index] + sum(
                visit.process_time or 0.0 for visit in route.visits[step:]
            )
        global_features = np.asarray(
            [
                completed_wafers / len(self.wafer_keys),
                completed_steps / total_steps,
                (
                    remaining_process_time / self._total_process_time
                    if self._total_process_time > 0
                    else 0.0
                ),
            ],
            dtype=np.float32,
        ).reshape(1, -1)

        return {
            GLOBAL: NodeStore(
                ids=("system",),
                features=global_features,
                feature_names=GLOBAL_FEATURE_NAMES,
            ),
            WAFER: NodeStore(
                ids=self.wafer_keys,
                features=wafer_features,
                feature_names=WAFER_FEATURE_NAMES,
            ),
            ROUTE_STEP: NodeStore(
                ids=self.route_step_ids,
                features=route_step_features,
                feature_names=ROUTE_STEP_FEATURE_NAMES,
            ),
            MODULE: NodeStore(
                ids=self.module_ids,
                features=module_features,
                feature_names=MODULE_FEATURE_NAMES,
            ),
            ROBOT: NodeStore(
                ids=self.robot_ids,
                features=robot_features,
                feature_names=ROBOT_FEATURE_NAMES,
            ),
        }

    def _residency_time(self, visit: RouteVisit) -> float | None:
        """Resolve a route-step residency limit from local and global rules."""

        if visit.residency_time is not None:
            return visit.residency_time
        if self.problem.just_in_time is None:
            return None

        module_types = {
            self.problem.Modules[module_id].type
            for module_id in visit.module_ids
        }
        if module_types == {ModuleType.PM}:
            value = self.problem.just_in_time.pm_residency_time
            if value is not None:
                return value
        if module_types == {ModuleType.LL}:
            value = self.problem.just_in_time.ll_residency_time
            if value is not None:
                return value
        return self.problem.just_in_time.residency_time

    def _build_edges(
        self,
        wafer_loc: np.ndarray,
        wafer_step: np.ndarray,
        robot_loc: np.ndarray,
        robot_holding: np.ndarray,
        operation_wafer: np.ndarray,
        operation_module: np.ndarray,
    ) -> dict[EdgeType, EdgeStore]:
        module_count = len(self.module_ids)
        wafer_count = len(self.wafer_keys)
        located_in: list[tuple[int, int]] = []
        at_step: list[tuple[int, int]] = []
        next_step: list[tuple[int, int]] = []

        for wafer_index, (route_id, _) in enumerate(self.wafer_keys):
            location = int(wafer_loc[wafer_index])
            if location < module_count:
                located_in.append((wafer_index, location))

            step = int(wafer_step[wafer_index])
            completed_step = len(self.problem.routes[route_id].visits) + 1
            if step > 0:
                at_step.append(
                    (wafer_index, self._route_step_index[(route_id, step)])
                )
            if step < completed_step:
                next_step.append(
                    (
                        wafer_index,
                        self._route_step_index[(route_id, step + 1)],
                    )
                )

        held_by = [
            (int(wafer_index), robot_index)
            for robot_index, row in enumerate(robot_holding)
            for wafer_index in row
            if wafer_index < wafer_count
        ]
        operates_on = [
            (robot_index, int(wafer_index))
            for robot_index, wafer_index in enumerate(operation_wafer)
            if wafer_index < wafer_count
        ]
        operation_at = [
            (robot_index, int(module_index))
            for robot_index, module_index in enumerate(operation_module)
            if module_index < module_count
        ]
        returns_to = [
            (wafer_index, self._module_index[return_module_id])
            for wafer_index, return_module_id in enumerate(
                self.return_module_ids
            )
        ]

        can_run_on: list[tuple[int, int]] = []
        precedes: list[tuple[int, int]] = []
        for route_step_index, (route_id, step) in enumerate(
            self.route_step_ids
        ):
            route = self.problem.routes[route_id]
            targets = (
                self._return_modules_by_route[route_id]
                if step == len(route.visits) + 1
                else route.visits[step - 1].module_ids
            )
            can_run_on.extend(
                (route_step_index, self._module_index[module_id])
                for module_id in targets
            )
            next_key = (route_id, step + 1)
            if next_key in self._route_step_index:
                precedes.append(
                    (route_step_index, self._route_step_index[next_key])
                )

        can_access: list[tuple[int, int]] = []
        accesses_atmosphere: list[tuple[int, int]] = []
        accesses_vacuum: list[tuple[int, int]] = []
        for robot_id, robot in self.problem.ClusterTool.items():
            robot_index = self._robot_index[robot_id]
            for module_id in robot.module_ids:
                module_index = self._module_index[module_id]
                load_lock = self.problem.Modules[module_id].load_lock
                if load_lock is None:
                    can_access.append((robot_index, module_index))
                    continue
                required_state = load_lock.tm_required_states.get(robot_id)
                if required_state is LoadLockState.ATMOSPHERE:
                    accesses_atmosphere.append((robot_index, module_index))
                elif required_state is LoadLockState.VACUUM:
                    accesses_vacuum.append((robot_index, module_index))
        located_at = [
            (robot_index, int(module_index))
            for robot_index, module_index in enumerate(robot_loc)
            if module_index < module_count
        ]

        edges = {
            LOCATED_IN: _edge_store(located_in),
            CONTAINS: _edge_store(_reverse(located_in)),
            HELD_BY: _edge_store(held_by),
            HOLDS: _edge_store(_reverse(held_by)),
            AT_STEP: _edge_store(at_step),
            CURRENT_FOR: _edge_store(_reverse(at_step)),
            NEXT_STEP: _edge_store(next_step),
            NEXT_FOR: _edge_store(_reverse(next_step)),
            CAN_RUN_ON: _edge_store(can_run_on),
            SUPPORTS_STEP: _edge_store(_reverse(can_run_on)),
            PRECEDES: _edge_store(precedes),
            FOLLOWS: _edge_store(_reverse(precedes)),
            CAN_ACCESS: _edge_store(can_access),
            ACCESSIBLE_BY: _edge_store(_reverse(can_access)),
            ACCESSES_ATMOSPHERE: _edge_store(accesses_atmosphere),
            ATMOSPHERE_ACCESSIBLE_BY: _edge_store(
                _reverse(accesses_atmosphere)
            ),
            ACCESSES_VACUUM: _edge_store(accesses_vacuum),
            VACUUM_ACCESSIBLE_BY: _edge_store(_reverse(accesses_vacuum)),
            LOCATED_AT: _edge_store(located_at),
            HAS_ROBOT: _edge_store(_reverse(located_at)),
            OPERATES_ON: _edge_store(operates_on),
            OPERATION_OF: _edge_store(_reverse(operates_on)),
            OPERATION_AT: _edge_store(operation_at),
            HAS_OPERATION: _edge_store(_reverse(operation_at)),
            RETURNS_TO: _edge_store(returns_to),
            RETURN_DESTINATION_OF: _edge_store(_reverse(returns_to)),
        }
        for node_type, count in (
            (WAFER, len(self.wafer_keys)),
            (ROUTE_STEP, len(self.route_step_ids)),
            (MODULE, len(self.module_ids)),
            (ROBOT, len(self.robot_ids)),
        ):
            from_global = [(0, index) for index in range(count)]
            edges[(GLOBAL, "contextualizes", node_type)] = _edge_store(
                from_global
            )
            edges[(node_type, "summarizes_into", GLOBAL)] = _edge_store(
                _reverse(from_global)
            )
        return edges

    def _validate_observation(
        self,
        wafer_loc: np.ndarray,
        wafer_step: np.ndarray,
        process_remaining: np.ndarray,
        wafer_priority: np.ndarray,
        wafer_index_feature: np.ndarray,
        ll_pump_time: np.ndarray,
        ll_vent_time: np.ndarray,
        ll_last_pick_side: np.ndarray,
        ll_empty_transition_progress: np.ndarray,
        ll_occupied_exit_side: np.ndarray,
        ll_occupied_transition_progress: np.ndarray,
        robot_loc: np.ndarray,
        robot_holding: np.ndarray,
        robot_phase: np.ndarray,
        operation_wafer: np.ndarray,
        operation_module: np.ndarray,
        time_to_operation_start: np.ndarray,
        time_to_operation_end: np.ndarray,
        action_mask: np.ndarray,
    ) -> None:
        wafer_count = len(self.wafer_keys)
        module_count = len(self.module_ids)
        robot_count = len(self.robot_ids)

        if (
            wafer_loc.shape != (wafer_count,)
            or wafer_step.shape != (wafer_count,)
            or process_remaining.shape != (wafer_count,)
            or wafer_priority.shape != (wafer_count,)
            or wafer_index_feature.shape != (wafer_count,)
        ):
            raise ValueError("observation has invalid wafer array shapes")
        if robot_loc.shape != (robot_count,):
            raise ValueError("robot_loc must contain one index per robot")
        load_lock_arrays = (
            ll_pump_time,
            ll_vent_time,
            ll_last_pick_side,
            ll_empty_transition_progress,
            ll_occupied_exit_side,
            ll_occupied_transition_progress,
        )
        if any(array.shape != (module_count,) for array in load_lock_arrays):
            raise ValueError("LL observation arrays must contain one value per module")
        if robot_holding.shape != (robot_count, self._max_arm_capacity):
            raise ValueError("robot_holding has an invalid shape")
        robot_arrays = (
            robot_phase,
            operation_wafer,
            operation_module,
            time_to_operation_start,
            time_to_operation_end,
        )
        if any(array.shape != (robot_count,) for array in robot_arrays):
            raise ValueError("robot operation arrays must contain one value per robot")
        expected_action_count = (
            wafer_count * robot_count + wafer_count * module_count + 1
        )
        if action_mask.shape != (expected_action_count,):
            raise ValueError(
                "action_mask must be flat transport actions followed by ADVANCE"
            )
        if np.any((wafer_loc < 0) | (wafer_loc >= module_count + robot_count)):
            raise ValueError("wafer_loc contains an invalid location index")
        if np.any((robot_loc < 0) | (robot_loc > module_count)):
            raise ValueError("robot_loc contains an invalid module index")
        if np.any((robot_holding < 0) | (robot_holding > wafer_count)):
            raise ValueError("robot_holding contains an invalid wafer index")
        held_wafers = robot_holding[robot_holding < wafer_count]
        if len(set(map(int, held_wafers))) != len(held_wafers):
            raise ValueError("a wafer cannot be held more than once")
        for robot_index, row in enumerate(robot_holding):
            capacity = (
                1
                if self.problem.ClusterTool[
                    self.robot_ids[robot_index]
                ].arm_type is TMArmType.SINGLE_ARM
                else 2
            )
            if np.count_nonzero(row < wafer_count) > capacity:
                raise ValueError("robot_holding exceeds robot arm capacity")
            if any(
                row[index] == wafer_count and row[index + 1] < wafer_count
                for index in range(len(row) - 1)
            ):
                raise ValueError("robot_holding must use a contiguous prefix")
        if np.any((robot_phase < 0) | (robot_phase >= len(RobotPhase))):
            raise ValueError("robot_phase contains an invalid phase")
        if np.any((operation_wafer < 0) | (operation_wafer > wafer_count)):
            raise ValueError("robot_operation_wafer contains an invalid index")
        if np.any((operation_module < 0) | (operation_module > module_count)):
            raise ValueError("robot_operation_module contains an invalid index")
        operation_times = np.concatenate(
            (time_to_operation_start, time_to_operation_end)
        )
        if np.any(~np.isfinite(operation_times)) or np.any(operation_times < 0):
            raise ValueError("robot operation times must be finite and non-negative")
        idle = robot_phase == RobotPhase.IDLE
        if np.any(operation_wafer[idle] != wafer_count) or np.any(
            operation_module[idle] != module_count
        ):
            raise ValueError("idle robots must use operation sentinels")
        busy = ~idle
        if np.any(operation_wafer[busy] >= wafer_count) or np.any(
            operation_module[busy] >= module_count
        ):
            raise ValueError("busy robots must identify their operation entities")
        if np.any(~np.isfinite(process_remaining)) or np.any(
            process_remaining < 0
        ):
            raise ValueError(
                "process_remaining must contain finite non-negative values"
            )
        for name, feature in (
            ("wafer_priority", wafer_priority),
            ("wafer_index", wafer_index_feature),
        ):
            if np.any(~np.isfinite(feature)) or np.any(
                (feature < 0) | (feature > 1)
            ):
                raise ValueError(f"{name} must contain values in [0, 1]")
        ll_times = np.concatenate((ll_pump_time, ll_vent_time))
        if np.any(~np.isfinite(ll_times)) or np.any(ll_times < 0):
            raise ValueError("LL transition times must be finite and non-negative")
        for name, progress in (
            ("ll_empty_transition_progress", ll_empty_transition_progress),
            (
                "ll_occupied_transition_progress",
                ll_occupied_transition_progress,
            ),
        ):
            if np.any(~np.isfinite(progress)) or np.any(
                (progress < 0) | (progress > 1)
            ):
                raise ValueError(f"{name} must contain values in [0, 1]")
        for name, side in (
            ("ll_last_pick_side", ll_last_pick_side),
            ("ll_occupied_exit_side", ll_occupied_exit_side),
        ):
            if np.any((side < 0) | (side >= len(LoadLockSide))):
                raise ValueError(f"{name} contains an invalid side")
        for index, (route_id, _) in enumerate(self.wafer_keys):
            completed_step = len(self.problem.routes[route_id].visits) + 1
            if not 0 <= wafer_step[index] <= completed_step:
                raise ValueError(f"wafer_step[{index}] is outside its route")
