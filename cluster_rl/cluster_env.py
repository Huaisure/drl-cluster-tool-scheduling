from __future__ import annotations

from copy import copy, deepcopy
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Literal, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cluster_toolkit.cluster_engine import (
    ADVANCE as ENGINE_ADVANCE,
    ClusterEngine,
    EngineAction,
    PickAction,
    PlaceAction,
)
from cluster_toolkit.problem import ClusterProblem, ModuleType, TMArmType, WaferKey


PICK = "pick"
PLACE = "place"
ADVANCE = "advance"


class RobotPhase(IntEnum):
    """Observable phase of one Robot's current operation."""

    IDLE = 0
    TRAVEL_TO_PICK = 1
    PICKING = 2
    TRAVEL_TO_PLACE = 3
    PLACING = 4


class LoadLockSide(IntEnum):
    """Categorical side used by module-aligned LL observation arrays."""

    NONE = 0
    ATMOSPHERE = 1
    VACUUM = 2


class ClusterEnv(gym.Env[dict[str, Any], int]):
    """Thin Gym adapter around :class:`cluster_engine.ClusterEngine`.

    The flat action layout is::

        Pick(wafer, robot) | Place(wafer, module) | Advance

    A Place identifies its wafer explicitly.  The Robot is inferred from the
    Engine state because the selected wafer is already held by exactly one
    Robot.
    """

    def __init__(
        self,
        problem: ClusterProblem,
        *,
        safety_lookahead_depth: int = 2,
    ) -> None:
        if safety_lookahead_depth < 0:
            raise ValueError("safety_lookahead_depth must be non-negative")
        self.problem = problem
        self.engine = ClusterEngine(problem)
        self.safety_lookahead_depth = safety_lookahead_depth

        snapshot = problem.initial_state.to_snapshot()
        self._robot_ids = tuple(sorted(problem.ClusterTool))
        self._wafer_keys = tuple(sorted(snapshot.wafers_by_key))
        self._module_ids = tuple(sorted(problem.Modules))
        self._wafer_index = {
            wafer_key: index for index, wafer_key in enumerate(self._wafer_keys)
        }
        self._robot_index = {
            robot_id: index for index, robot_id in enumerate(self._robot_ids)
        }
        self._module_index = {
            module_id: index for index, module_id in enumerate(self._module_ids)
        }
        self._robot_modules = {
            robot_id: frozenset(problem.ClusterTool[robot_id].module_ids)
            for robot_id in self._robot_ids
        }
        self._arm_capacities = {
            robot_id: (
                1
                if problem.ClusterTool[robot_id].arm_type is TMArmType.SINGLE_ARM
                else 2
            )
            for robot_id in self._robot_ids
        }
        self._transfer_robots_cache: dict[
            tuple[str, int, str, str],
            frozenset[str],
        ] = {}
        self._initial_wafers = snapshot.wafers_by_key
        self._return_module_ids = tuple(
            problem.return_module_id(snapshot.wafers_by_key[key])
            for key in self._wafer_keys
        )

        wafer_count = len(self._wafer_keys)
        robot_count = len(self._robot_ids)
        module_count = len(self._module_ids)
        self._pick_action_count = wafer_count * robot_count
        self._place_action_count = wafer_count * module_count
        self._max_arm_capacity = max(
            self._arm_capacities.values(),
            default=1,
        )

        self.action_space = spaces.Discrete(
            self._pick_action_count + self._place_action_count + 1
        )
        self.observation_space = spaces.Dict(
            {
                "wafer_loc": spaces.MultiDiscrete(
                    np.full(wafer_count, module_count + robot_count)
                ),
                "wafer_step": spaces.MultiDiscrete(
                    np.asarray(
                        [
                            len(problem.routes[route_id].visits) + 2
                            for route_id, _ in self._wafer_keys
                        ]
                    )
                ),
                "process_remaining": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(wafer_count,),
                    dtype=np.float32,
                ),
                "wafer_priority": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(wafer_count,),
                    dtype=np.float32,
                ),
                "wafer_index": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(wafer_count,),
                    dtype=np.float32,
                ),
                "ll_pump_time": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(module_count,),
                    dtype=np.float32,
                ),
                "ll_vent_time": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(module_count,),
                    dtype=np.float32,
                ),
                "ll_last_pick_side": spaces.MultiDiscrete(
                    np.full(module_count, len(LoadLockSide))
                ),
                "ll_empty_transition_progress": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(module_count,),
                    dtype=np.float32,
                ),
                "ll_occupied_exit_side": spaces.MultiDiscrete(
                    np.full(module_count, len(LoadLockSide))
                ),
                "ll_occupied_transition_progress": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(module_count,),
                    dtype=np.float32,
                ),
                "robot_loc": spaces.MultiDiscrete(
                    np.full(robot_count, module_count + 1)
                ),
                "robot_holding": spaces.MultiDiscrete(
                    np.full(
                        (robot_count, self._max_arm_capacity),
                        wafer_count + 1,
                    )
                ),
                "robot_phase": spaces.MultiDiscrete(
                    np.full(robot_count, len(RobotPhase))
                ),
                "robot_operation_wafer": spaces.MultiDiscrete(
                    np.full(robot_count, wafer_count + 1)
                ),
                "robot_operation_module": spaces.MultiDiscrete(
                    np.full(robot_count, module_count + 1)
                ),
                "time_to_operation_start": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(robot_count,),
                    dtype=np.float32,
                ),
                "time_to_operation_end": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(robot_count,),
                    dtype=np.float32,
                ),
                "legal_action_mask": spaces.MultiBinary(int(self.action_space.n)),
                "action_mask": spaces.MultiBinary(int(self.action_space.n)),
            }
        )
        self._actions: list[dict[str, object]] = []

    @property
    def wafer_keys(self) -> tuple[WaferKey, ...]:
        return self._wafer_keys

    @property
    def module_ids(self) -> tuple[str, ...]:
        return self._module_ids

    @property
    def return_module_ids(self) -> tuple[str, ...]:
        return self._return_module_ids

    @property
    def actions(self) -> tuple[Mapping[str, object], ...]:
        return tuple(MappingProxyType(action) for action in self._actions)

    @property
    def time(self) -> float:
        return self.engine.state.time

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        super().reset(seed=seed)
        self.engine.reset()
        self._actions = []
        return self._observation(), {"time": self.time}

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside the action space")

        action_index = int(action)
        if not self._action_mask()[action_index]:
            raise ValueError(f"action {action_index} is not allowed in the current state")

        previous_time = self.time
        engine_action = self._engine_action(action_index)
        record = self.engine.step(engine_action)
        if record is not None:
            self._actions.append(record.to_dict())

        terminated = self.engine.is_complete()
        observation = self._observation()
        truncated = not terminated and not observation["action_mask"].any()
        info: dict[str, Any] = {
            "time": self.time,
            "action_mask": observation["action_mask"],
            "legal_action_mask": observation["legal_action_mask"],
        }
        if terminated:
            info.update(is_success=True, reason="completed")
        elif truncated:
            reason = (
                "deadlock"
                if not observation["legal_action_mask"].any()
                else "safety_deadlock"
            )
            info.update(is_success=False, reason=reason)
        return (
            observation,
            previous_time - self.time,
            terminated,
            truncated,
            info,
        )

    def _observation(self) -> dict[str, Any]:
        state = self.engine.state
        legal_action_mask = self._legal_action_mask()
        action_mask = self._action_mask()
        return {
            "wafer_loc": np.asarray(
                [
                    self._module_index[wafer.module_id]
                    if wafer.module_id is not None
                    else len(self._module_ids) + self._robot_index[wafer.robot_id]
                    for wafer in (state.wafers[key] for key in self._wafer_keys)
                ],
                dtype=np.int64,
            ),
            "wafer_step": np.asarray(
                [state.wafers[key].step_index for key in self._wafer_keys],
                dtype=np.int64,
            ),
            "process_remaining": np.asarray(
                [
                    max(0.0, state.wafers[key].ready_at - state.time)
                    for key in self._wafer_keys
                ],
                dtype=np.float32,
            ),
            "wafer_priority": self._normalized_priorities(),
            "wafer_index": self._normalized_wafer_indexes(),
            **self._load_lock_observation(),
            "robot_loc": np.asarray(
                [
                    self._module_index[state.robots[robot_id].module_id]
                    if state.robots[robot_id].module_id is not None
                    else len(self._module_ids)
                    for robot_id in self._robot_ids
                ],
                dtype=np.int64,
            ),
            **self._robot_observation(),
            "legal_action_mask": legal_action_mask,
            "action_mask": action_mask,
        }

    def _load_lock_observation(self) -> dict[str, np.ndarray]:
        module_count = len(self._module_ids)
        pump_time = np.zeros(module_count, dtype=np.float32)
        vent_time = np.zeros(module_count, dtype=np.float32)
        last_pick_side = np.zeros(module_count, dtype=np.int64)
        empty_progress = np.zeros(module_count, dtype=np.float32)
        occupied_exit_side = np.zeros(module_count, dtype=np.int64)
        occupied_progress = np.zeros(module_count, dtype=np.float32)
        side_index = {
            "atmosphere": LoadLockSide.ATMOSPHERE,
            "vacuum": LoadLockSide.VACUUM,
        }

        for index, module_id in enumerate(self._module_ids):
            if self.problem.Modules[module_id].load_lock is None:
                continue
            load_lock = self.engine.load_lock_observation(module_id)
            pump_time[index] = load_lock.pump_time
            vent_time[index] = load_lock.vent_time
            last_pick_side[index] = side_index[load_lock.last_pick_side]
            empty_progress[index] = load_lock.empty_transition_progress
            if load_lock.occupied_exit_side is not None:
                occupied_exit_side[index] = side_index[
                    load_lock.occupied_exit_side
                ]
            occupied_progress[index] = load_lock.occupied_transition_progress

        return {
            "ll_pump_time": pump_time,
            "ll_vent_time": vent_time,
            "ll_last_pick_side": last_pick_side,
            "ll_empty_transition_progress": empty_progress,
            "ll_occupied_exit_side": occupied_exit_side,
            "ll_occupied_transition_progress": occupied_progress,
        }

    def _robot_observation(self) -> dict[str, np.ndarray]:
        state = self.engine.state
        robot_count = len(self._robot_ids)
        wafer_sentinel = len(self._wafer_keys)
        module_sentinel = len(self._module_ids)
        holding = np.full(
            (robot_count, self._max_arm_capacity),
            wafer_sentinel,
            dtype=np.int64,
        )
        phase = np.full(robot_count, RobotPhase.IDLE, dtype=np.int64)
        operation_wafer = np.full(robot_count, wafer_sentinel, dtype=np.int64)
        operation_module = np.full(robot_count, module_sentinel, dtype=np.int64)
        time_to_start = np.zeros(robot_count, dtype=np.float32)
        time_to_end = np.zeros(robot_count, dtype=np.float32)

        for robot_index, robot_id in enumerate(self._robot_ids):
            held = [self._wafer_index[key] for key in state.robots[robot_id].holding]
            holding[robot_index, : len(held)] = held

        for operation in state.pending_operations:
            robot_index = self._robot_index[operation.robot_id]
            operation_wafer[robot_index] = self._wafer_index[operation.wafer_key]
            operation_module[robot_index] = self._module_index[operation.module_id]
            time_to_start[robot_index] = max(0.0, operation.start - state.time)
            time_to_end[robot_index] = max(0.0, operation.end - state.time)
            if operation.action_type == PICK:
                phase[robot_index] = (
                    RobotPhase.PICKING
                    if operation.started
                    else RobotPhase.TRAVEL_TO_PICK
                )
            else:
                phase[robot_index] = (
                    RobotPhase.PLACING
                    if operation.started
                    else RobotPhase.TRAVEL_TO_PLACE
                )

        return {
            "robot_holding": holding,
            "robot_phase": phase,
            "robot_operation_wafer": operation_wafer,
            "robot_operation_module": operation_module,
            "time_to_operation_start": time_to_start,
            "time_to_operation_end": time_to_end,
        }

    def _normalized_priorities(self) -> np.ndarray:
        priorities = np.asarray(
            [self._initial_wafers[key].priority for key in self._wafer_keys],
            dtype=np.float32,
        )
        maximum = float(priorities.max()) if priorities.size else 0.0
        return priorities / max(1.0, maximum)

    def _normalized_wafer_indexes(self) -> np.ndarray:
        recipe_maximums: dict[str, int] = {}
        for route_id, wafer_index in self._wafer_keys:
            recipe_maximums[route_id] = max(
                recipe_maximums.get(route_id, 0),
                wafer_index,
            )
        return np.asarray(
            [
                wafer_index / max(1, recipe_maximums[route_id])
                for route_id, wafer_index in self._wafer_keys
            ],
            dtype=np.float32,
        )

    def _action_mask(self) -> np.ndarray:
        """Return actions that pass local rules and bounded lookahead."""

        actions = self._safe_available_actions(
            self.engine,
            self.safety_lookahead_depth,
        )
        return self._encode_action_mask(actions)

    def _legal_action_mask(self) -> np.ndarray:
        """Return current Engine actions after Env admission ordering."""

        return self._encode_action_mask(self._env_available_actions(self.engine))

    def _encode_action_mask(
        self,
        actions: tuple[EngineAction, ...],
    ) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for action in actions:
            mask[self._encode_engine_action(action)] = 1
        return mask

    def _env_available_actions(
        self,
        engine: ClusterEngine | None = None,
    ) -> tuple[EngineAction, ...]:
        """Apply the Env-only same-Recipe index tie-break to Engine actions."""

        engine = self.engine if engine is None else engine
        actions = engine.available_actions()
        source_picks = [
            action
            for action in actions
            if isinstance(action, PickAction)
            and self._is_source_pick(action, engine)
        ]
        minimum_indexes: dict[tuple[int, str], int] = {}
        for action in source_picks:
            initial = self._initial_wafers[action.wafer_key]
            key = (initial.priority, initial.route_id)
            minimum_indexes[key] = min(
                minimum_indexes.get(key, initial.wafer_index),
                initial.wafer_index,
            )
        return tuple(
            action
            for action in actions
            if not isinstance(action, PickAction)
            or not self._is_source_pick(action, engine)
            or self._initial_wafers[action.wafer_key].wafer_index
            == minimum_indexes[
                (
                    self._initial_wafers[action.wafer_key].priority,
                    self._initial_wafers[action.wafer_key].route_id,
                )
            ]
        )

    def _is_source_pick(
        self,
        action: PickAction,
        engine: ClusterEngine | None = None,
    ) -> bool:
        engine = self.engine if engine is None else engine
        wafer = engine.state.wafers[action.wafer_key]
        return (
            wafer.step_index == 0
            and wafer.module_id is not None
            and self.problem.Modules[wafer.module_id].type
            in {ModuleType.IO, ModuleType.LP}
        )

    def _safe_available_actions(
        self,
        engine: ClusterEngine,
        lookahead_depth: int,
    ) -> tuple[EngineAction, ...]:
        actions = self._statically_safe_actions(engine)
        if lookahead_depth == 0:
            return actions
        return tuple(
            action
            for action in actions
            if self._action_has_safe_continuation(
                engine,
                action,
                lookahead_depth,
            )
        )

    def _statically_safe_actions(
        self,
        engine: ClusterEngine,
    ) -> tuple[EngineAction, ...]:
        return tuple(
            action
            for action in self._env_available_actions(engine)
            if not isinstance(action, PickAction)
            or (
                self._robot_can_reach_next_target(engine, action)
                and not self._pick_fills_robot_with_blocked_wafers(engine, action)
            )
        )

    def _pick_fills_robot_with_blocked_wafers(
        self,
        engine: ClusterEngine,
        action: PickAction,
    ) -> bool:
        """Detect a guaranteed full-arm wait cycle before simulating it.

        Once this Pick completes, a full Robot cannot unload any held wafer if
        all of their destinations are full of wafers that only this same Robot
        can transfer onward.  Other Robots are treated optimistically: one
        topologically valid transfer is enough to leave the Pick available.
        """

        robot = engine.state.robots[action.robot_id]
        held_after_pick = (*robot.holding, action.wafer_key)
        if len(held_after_pick) < self._arm_capacities[action.robot_id]:
            return False
        return all(
            self._wafer_destinations_require_same_robot_release(
                engine,
                wafer_key,
                action.robot_id,
                action.wafer_key,
            )
            for wafer_key in held_after_pick
        )

    def _wafer_destinations_require_same_robot_release(
        self,
        engine: ClusterEngine,
        wafer_key: WaferKey,
        robot_id: str,
        released_wafer_key: WaferKey,
    ) -> bool:
        wafer = engine.state.wafers[wafer_key]
        targets = self._next_targets(
            wafer.route_id,
            wafer.step_index,
            wafer.return_module_id,
        )
        return bool(targets) and all(
            self._full_target_requires_same_robot_release(
                engine,
                target_id,
                robot_id,
                released_wafer_key,
            )
            for target_id in targets
        )

    def _full_target_requires_same_robot_release(
        self,
        engine: ClusterEngine,
        module_id: str,
        robot_id: str,
        released_wafer_key: WaferKey,
    ) -> bool:
        occupants = engine.state.module_occupants[module_id]
        released_count = int(released_wafer_key in occupants)
        if (
            len(occupants) - released_count
            < self.problem.Modules[module_id].capacity
        ):
            return False

        for wafer_key in occupants:
            if wafer_key == released_wafer_key:
                continue
            if any(
                operation.action_type == PICK
                and operation.wafer_key == wafer_key
                and operation.robot_id != robot_id
                for operation in engine.state.pending_operations
            ):
                return False
            if any(
                candidate != robot_id
                for candidate in self._transfer_robot_ids(
                    engine,
                    wafer_key,
                    module_id,
                )
            ):
                return False
        return True

    def _transfer_robot_ids(
        self,
        engine: ClusterEngine,
        wafer_key: WaferKey,
        source_module_id: str,
    ) -> frozenset[str]:
        wafer = engine.state.wafers[wafer_key]
        key = (
            wafer.route_id,
            wafer.step_index,
            wafer.return_module_id,
            source_module_id,
        )
        if key not in self._transfer_robots_cache:
            targets = self._next_targets(
                wafer.route_id,
                wafer.step_index,
                wafer.return_module_id,
            )
            self._transfer_robots_cache[key] = frozenset(
                robot_id
                for robot_id, robot_modules in self._robot_modules.items()
                if source_module_id in robot_modules
                and any(target_id in robot_modules for target_id in targets)
            )
        return self._transfer_robots_cache[key]

    def _robot_can_reach_next_target(
        self,
        engine: ClusterEngine,
        action: PickAction,
    ) -> bool:
        wafer = engine.state.wafers[action.wafer_key]
        robot_modules = self._robot_modules[action.robot_id]
        return any(
            module_id in robot_modules
            for module_id in self._next_targets(
                wafer.route_id,
                wafer.step_index,
                wafer.return_module_id,
            )
        )

    def _next_targets(
        self,
        route_id: str,
        step_index: int,
        return_module_id: str,
    ) -> tuple[str, ...]:
        route = self.problem.routes[route_id]
        next_step = step_index + 1
        if 1 <= next_step <= len(route.visits):
            return route.visits[next_step - 1].module_ids
        if next_step == len(route.visits) + 1:
            return (return_module_id,)
        return ()

    def _action_has_safe_continuation(
        self,
        engine: ClusterEngine,
        action: EngineAction,
        remaining_depth: int,
    ) -> bool:
        next_engine = self._fork_engine(engine)
        next_engine.step(action)
        if action != ENGINE_ADVANCE:
            remaining_depth -= 1
        return self._state_has_safe_continuation(
            next_engine,
            remaining_depth,
        )

    def _state_has_safe_continuation(
        self,
        engine: ClusterEngine,
        remaining_depth: int,
    ) -> bool:
        if engine.is_complete():
            return True

        actions = self._statically_safe_actions(engine)
        while actions == (ENGINE_ADVANCE,):
            engine.step(ENGINE_ADVANCE)
            if engine.is_complete():
                return True
            actions = self._statically_safe_actions(engine)

        if not actions:
            return False
        if remaining_depth <= 0:
            return True
        return any(
            self._action_has_safe_continuation(
                engine,
                action,
                remaining_depth,
            )
            for action in sorted(
                actions,
                key=lambda candidate: self._search_priority(engine, candidate),
            )
        )

    def _search_priority(
        self,
        engine: ClusterEngine,
        action: EngineAction,
    ) -> int:
        if isinstance(action, PlaceAction):
            return 0
        if isinstance(action, PickAction):
            return 2 if self._is_source_pick(action, engine) else 1
        return 3

    @staticmethod
    def _fork_engine(engine: ClusterEngine) -> ClusterEngine:
        fork = copy(engine)
        fork._state = deepcopy(engine.state)
        return fork

    def _engine_action(self, action: int) -> EngineAction:
        kind, wafer_index, target_index = self._decode_action(action)
        if kind == PICK:
            assert wafer_index is not None and target_index is not None
            return PickAction(
                robot_id=self._robot_ids[target_index],
                wafer_key=self._wafer_keys[wafer_index],
            )
        if kind == PLACE:
            assert wafer_index is not None and target_index is not None
            return PlaceAction(
                wafer_key=self._wafer_keys[wafer_index],
                target_module_id=self._module_ids[target_index],
            )
        return ENGINE_ADVANCE

    def _encode_engine_action(self, action: EngineAction) -> int:
        if isinstance(action, PickAction):
            return (
                self._wafer_index[action.wafer_key] * len(self._robot_ids)
                + self._robot_index[action.robot_id]
            )
        if isinstance(action, PlaceAction):
            return (
                self._pick_action_count
                + self._wafer_index[action.wafer_key] * len(self._module_ids)
                + self._module_index[action.target_module_id]
            )
        return int(self.action_space.n) - 1

    def _decode_action(
        self,
        action: int,
    ) -> tuple[Literal["pick", "place", "advance"], int | None, int | None]:
        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError(f"action {action!r} is outside the action space")
        if action_index == self.action_space.n - 1:
            return ADVANCE, None, None
        if action_index < self._pick_action_count:
            wafer_index, robot_index = divmod(
                action_index,
                len(self._robot_ids),
            )
            return PICK, wafer_index, robot_index
        wafer_index, module_index = divmod(
            action_index - self._pick_action_count,
            len(self._module_ids),
        )
        return PLACE, wafer_index, module_index
