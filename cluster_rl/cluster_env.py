from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Iterable, Literal, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from problem import (
    ClusterProblem,
    TMArmType,
    WaferKey,
)

PICK = "pick"
PLACE = "place"
ADVANCE = "advance"
EPS = 1e-6


class RobotPhase(IntEnum):
    """Observable phase of one robot's pending operation."""

    IDLE = 0
    TRAVEL_TO_PICK = 1
    PICKING = 2
    TRAVEL_TO_PLACE = 3
    PLACING = 4


@dataclass(slots=True)
class _WaferState:
    step_index: int
    module_id: str | None
    robot_id: str | None
    ready_at: float


@dataclass(slots=True)
class _RobotState:
    module_id: str | None
    ready_at: float
    holding: list[int]


@dataclass(slots=True)
class _PendingEvent:
    action_type: Literal["pick", "place"]
    robot_index: int
    wafer_index: int
    module_index: int
    start: float
    end: float
    started: bool = False


class ClusterEnv(gym.Env[dict[str, Any], int]):
    def __init__(self, problem: ClusterProblem) -> None:
        self.problem = problem

        self._robot_ids = tuple(sorted(problem.ClusterTool))
        snapshot = problem.initial_state.to_snapshot()
        self._wafer_keys = tuple(sorted(snapshot.wafers_by_key))
        self._return_module_ids = tuple(
            problem.return_module_id(snapshot.wafers_by_key[key])
            for key in self._wafer_keys
        )
        self._module_ids = tuple(sorted(problem.Modules))
        self._module_index = {
            module_id: index
            for index, module_id in enumerate(self._module_ids)
        }
        self._robot_index = {
            robot_id: index
            for index, robot_id in enumerate(self._robot_ids)
        }

        wafer_count = len(self._wafer_keys)
        module_count = len(self._module_ids)
        robot_count = len(self._robot_ids)
        self._max_arm_capacity = max(
            1 if robot.arm_type is TMArmType.SINGLE_ARM else 2
            for robot in problem.ClusterTool.values()
        )

        self.action_space = spaces.Discrete(
            (wafer_count + module_count) * robot_count + 1
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
                        ],
                    )
                ),
                "process_remaining": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(wafer_count,),
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
                "deadlock_safe_mask": spaces.MultiBinary(int(self.action_space.n)),
                "action_mask": spaces.MultiBinary(int(self.action_space.n)),
            }
        )

        self._time = 0.0
        self._robots: list[_RobotState] = []
        self._wafers: list[_WaferState] = []
        self._actions: list[dict[str, object]] = []
        self._pending_queue: list[_PendingEvent] = []

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

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, float]]:
        super().reset(seed=seed)
        snapshot = self.problem.initial_state.to_snapshot()
        self._time = 0.0
        self._robots = []
        self._wafers = []
        self._pending_queue = []

        for robot_id in self._robot_ids:
            self._robots.append(
                _RobotState(
                    module_id=snapshot.tm_positions.get(robot_id),
                    ready_at=0.0,
                    holding=[],
                )
            )

        for key in self._wafer_keys:
            snapshot_wafer = snapshot.wafers_by_key[key]
            in_module = snapshot_wafer.location.kind == "module"
            self._wafers.append(
                _WaferState(
                    step_index=snapshot_wafer.step_index,
                    module_id=snapshot_wafer.location.module_id if in_module else None,
                    robot_id=snapshot_wafer.location.robot_id if not in_module else None,
                    ready_at=snapshot_wafer.process_end_time or 0.0,
                )
            )
            if not in_module:
                robot_index = self._robot_ids.index(snapshot_wafer.location.robot_id)
                self._robots[robot_index].holding.append(len(self._wafers) - 1)

        self._actions = []
        return self._observation(), {"time": self._time}

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self._wafers:
            raise RuntimeError("reset() must be called before step()")

        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside the action space")

        action_index = int(action)
        if not self._action_mask()[action_index]:
            raise ValueError(f"action {action_index} is not allowed in the current state")

        previous_time = self._time
        action_type, entity_index, robot_index = self._decode_action(action_index)
        if action_type == PICK:
            assert entity_index is not None and robot_index is not None
            self._pick(entity_index, robot_index)
        elif action_type == PLACE:
            assert entity_index is not None and robot_index is not None
            self._place(entity_index, robot_index)
        else:
            self._advance()

        completed = self._complete()
        reward = self._reward(previous_time)
        info: dict[str, Any] = {"time": self._time}
        if completed:
            info.update(
                is_success=True,
                reason="completed",
            )

        observation = self._observation()
        deadlocked = not completed and not observation["legal_action_mask"].any()
        if deadlocked:
            info.update(
                is_success=False,
                reason="deadlock",
            )
        info["action_mask"] = observation["action_mask"]
        info["legal_action_mask"] = observation["legal_action_mask"]
        info["deadlock_safe_mask"] = observation["deadlock_safe_mask"]
        return observation, reward, completed, deadlocked, info

    def _reward(self, previous_time: float) -> float:
        """Charge elapsed physical time so an episode return is ``-makespan``."""

        return previous_time - self._time

    def _observation(self) -> dict[str, Any]:
        robot_module = [robot.module_id for robot in self._robots]
        legal_action_mask, deadlock_safe_mask, action_mask = self._action_masks()
        return {
            "wafer_loc": np.asarray(
                [
                    self._robot_index[wafer.robot_id] + len(self._module_ids)
                    if wafer.module_id is None
                    else self._module_index[wafer.module_id]
                    for wafer in self._wafers
                ],
                dtype=np.int64,
            ),
            "wafer_step": np.asarray(
                [wafer.step_index for wafer in self._wafers], dtype=np.int64
            ),
            "process_remaining": np.asarray(
                [
                    max(0.0, wafer.ready_at - self._time)
                    for wafer in self._wafers
                ],
                dtype=np.float32,
            ),
            "robot_loc": np.asarray(
                [
                    self._module_index[module_id]
                    if module_id is not None
                    else len(self._module_ids)
                    for module_id in robot_module
                ],
                dtype=np.int64,
            ),
            **self._robot_operation_observation(),
            "legal_action_mask": legal_action_mask,
            "deadlock_safe_mask": deadlock_safe_mask,
            "action_mask": action_mask,
        }

    def _robot_operation_observation(self) -> dict[str, np.ndarray]:
        """Project ordered holdings and pending events into robot arrays."""

        robot_count = len(self._robot_ids)
        wafer_sentinel = len(self._wafer_keys)
        module_sentinel = len(self._module_ids)
        robot_holding = np.full(
            (robot_count, self._max_arm_capacity),
            wafer_sentinel,
            dtype=np.int64,
        )
        for robot_index, robot in enumerate(self._robots):
            robot_holding[robot_index, : len(robot.holding)] = robot.holding

        robot_phase = np.full(
            robot_count,
            RobotPhase.IDLE,
            dtype=np.int64,
        )
        operation_wafer = np.full(
            robot_count,
            wafer_sentinel,
            dtype=np.int64,
        )
        operation_module = np.full(
            robot_count,
            module_sentinel,
            dtype=np.int64,
        )
        time_to_start = np.zeros(robot_count, dtype=np.float32)
        time_to_end = np.zeros(robot_count, dtype=np.float32)

        for event in self._pending_queue:
            robot_index = event.robot_index
            if operation_wafer[robot_index] != wafer_sentinel:
                raise RuntimeError("a robot cannot have multiple pending operations")

            if event.action_type == PICK:
                phase = (
                    RobotPhase.PICKING
                    if event.started
                    else RobotPhase.TRAVEL_TO_PICK
                )
            else:
                phase = (
                    RobotPhase.PLACING
                    if event.started
                    else RobotPhase.TRAVEL_TO_PLACE
                )

            robot_phase[robot_index] = phase
            operation_wafer[robot_index] = event.wafer_index
            operation_module[robot_index] = event.module_index
            time_to_start[robot_index] = max(0.0, event.start - self._time)
            time_to_end[robot_index] = max(0.0, event.end - self._time)

        return {
            "robot_holding": robot_holding,
            "robot_phase": robot_phase,
            "robot_operation_wafer": operation_wafer,
            "robot_operation_module": operation_module,
            "time_to_operation_start": time_to_start,
            "time_to_operation_end": time_to_end,
        }

    def _action_mask(self) -> np.ndarray:
        """Return physical legality after applying the cheap deadlock shield."""

        return self._action_masks()[2]

    def _action_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return physical, deadlock-safety, and combined action masks."""

        legal_mask = self._legal_action_mask()
        safe_mask = np.ones(self.action_space.n, dtype=np.int8)
        if not self._pending_queue:
            wafer_count = len(self._wafer_keys)
            robot_count = len(self._robot_ids)
            for wafer_index in range(wafer_count):
                for robot_index in range(robot_count):
                    action = wafer_index * robot_count + robot_index
                    if legal_mask[action] and self._pick_creates_closed_wait_cycle(
                        wafer_index,
                        robot_index,
                    ):
                        safe_mask[action] = 0

            for module_index in range(len(self._module_ids)):
                entity_index = wafer_count + module_index
                for robot_index in range(robot_count):
                    action = entity_index * robot_count + robot_index
                    if legal_mask[action] and self._place_creates_closed_wait_cycle(
                        module_index,
                        robot_index,
                    ):
                        safe_mask[action] = 0

        action_mask = legal_mask & safe_mask
        if legal_mask.any() and not action_mask.any():
            safe_mask[legal_mask.astype(bool)] = 1
            action_mask = legal_mask.copy()
        return legal_mask, safe_mask, action_mask

    def _legal_action_mask(self) -> np.ndarray:
        """Return the flat physical-legality mask before safety filtering."""

        mask = np.zeros(self.action_space.n, dtype=np.int8)
        if self._complete():
            return mask

        robot_count = len(self._robot_ids)
        wafer_count = len(self._wafer_keys)
        for wafer_index in range(wafer_count):
            for robot_index in range(robot_count):
                if self._can_pick(wafer_index, robot_index):
                    mask[wafer_index * robot_count + robot_index] = 1

        for module_index in range(len(self._module_ids)):
            entity_index = wafer_count + module_index
            for robot_index in range(robot_count):
                if self._place_candidate(module_index, robot_index) is not None:
                    mask[entity_index * robot_count + robot_index] = 1

        mask[-1] = self._next_event_time() is not None
        return mask

    def _place_creates_closed_wait_cycle(
        self,
        module_index: int,
        robot_index: int,
    ) -> bool:
        """Check whether one Place closes a FIFO resource-wait cycle.

        This shield intentionally models only module occupancy and FIFO heads. It
        is cheap enough for rollout collection and catches the common ring
        deadlock without pretending to solve the full residual scheduling
        problem.
        """

        wafer_index = self._place_candidate(module_index, robot_index)
        if wafer_index is None:
            return False

        return self._move_creates_closed_wait_cycle(wafer_index, module_index)

    def _pick_creates_closed_wait_cycle(
        self,
        wafer_index: int,
        robot_index: int,
    ) -> bool:
        """Check whether every destination after one Pick closes a wait cycle."""

        destinations = [
            self._module_index[target_module_id]
            for target_module_id in self._targets(wafer_index)
            if self._can_achieve(robot_index, target_module_id)
            and self._has_capacity_after_pick(
                self._module_index[target_module_id],
                wafer_index,
            )
        ]
        return bool(destinations) and all(
            self._move_creates_closed_wait_cycle(wafer_index, module_index)
            for module_index in destinations
        )

    def _move_creates_closed_wait_cycle(
        self,
        wafer_index: int,
        module_index: int,
    ) -> bool:
        """Check the module wait graph after moving a wafer to one target."""

        module_locations = [wafer.module_id for wafer in self._wafers]
        step_indexes = [wafer.step_index for wafer in self._wafers]
        placed_module_id = self._module_ids[module_index]
        module_locations[wafer_index] = placed_module_id
        step_indexes[wafer_index] += 1

        occupancy = {
            module_id: sum(location == module_id for location in module_locations)
            for module_id in self._module_ids
        }
        wait_graph = {module_id: set() for module_id in self._module_ids}

        for source_module_id in self._module_ids:
            candidates = [
                index
                for index, location in enumerate(module_locations)
                if location == source_module_id
                and step_indexes[index]
                < len(self.problem.routes[self._wafer_keys[index][0]].visits) + 1
            ]
            if not candidates:
                continue

            head = min(
                candidates,
                key=lambda index: (
                    self._wafer_keys[index][1],
                    self._wafer_keys[index][0],
                ),
            )
            targets = self._targets_for_step(head, step_indexes[head])
            full_targets = {
                target
                for target in targets
                if occupancy[target] - int(target == source_module_id)
                >= self.problem.Modules[target].capacity
            }
            if targets and len(full_targets) == len(targets):
                wait_graph[source_module_id].update(full_targets)

        reachable = self._reachable_modules(wait_graph, placed_module_id)
        reverse_graph = {module_id: set() for module_id in self._module_ids}
        for source, targets in wait_graph.items():
            for target in targets:
                reverse_graph[target].add(source)
        component = reachable & self._reachable_modules(
            reverse_graph,
            placed_module_id,
        )
        has_cycle = len(component) > 1 or placed_module_id in wait_graph[
            placed_module_id
        ]
        return has_cycle and all(
            wait_graph[module_id] <= component for module_id in component
        )

    @staticmethod
    def _reachable_modules(
        graph: Mapping[str, set[str]],
        start: str,
    ) -> set[str]:
        reachable: set[str] = set()
        pending = [start]
        while pending:
            module_id = pending.pop()
            if module_id in reachable:
                continue
            reachable.add(module_id)
            pending.extend(graph[module_id] - reachable)
        return reachable

    def _decode_action(
        self,
        action: int,
    ) -> tuple[
        Literal["pick", "place", "advance"],
        int | None,
        int | None,
    ]:
        """Decode a flat action into its operation, entity, and robot indexes."""

        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError(f"action {action!r} is outside the action space")
        if action_index == self.action_space.n - 1:
            return ADVANCE, None, None

        entity_index, robot_index = divmod(action_index, len(self._robot_ids))
        if entity_index < len(self._wafer_keys):
            return PICK, entity_index, robot_index
        return PLACE, entity_index - len(self._wafer_keys), robot_index

    def _targets(self, wafer_index: int) -> tuple[str, ...]:
        """返回wafer index对应的下一个步骤的module ids，可能有多个module，返回对应的元组"""
        return self._targets_for_step(
            wafer_index,
            self._wafers[wafer_index].step_index,
        )

    def _targets_for_step(
        self,
        wafer_index: int,
        step_index: int,
    ) -> tuple[str, ...]:
        """Return targets following an explicit current step index."""

        route = self.problem.routes[self._wafer_keys[wafer_index][0]]
        next_step = step_index + 1
        if next_step <= len(route.visits):
            return route.visits[next_step - 1].module_ids
        if next_step == len(route.visits) + 1:
            return (self._return_module_ids[wafer_index],)
        return ()

    def _can_achieve(self, robot_index: int, module_id: str | Iterable[str]) -> bool:
        ct = self.problem.ClusterTool[self._robot_ids[robot_index]]
        if isinstance(module_id, str):
            return module_id in ct.module_ids
        return all(module in ct.module_ids for module in module_id)

    def _arm_capacity(self, robot_index: int) -> int:
        robot = self.problem.ClusterTool[self._robot_ids[robot_index]]
        return 1 if robot.arm_type is TMArmType.SINGLE_ARM else 2

    def _is_fifo_head(self, wafer_index: int) -> bool:
        wafer = self._wafers[wafer_index]
        if wafer.module_id is None:
            return False

        candidates = [
            index
            for index, other in enumerate(self._wafers)
            if other.module_id == wafer.module_id
            and other.step_index
            < len(self.problem.routes[self._wafer_keys[index][0]].visits) + 1
        ]
        if not candidates:
            return False
        return wafer_index == min(
            candidates,
            key=lambda index: (
                self._wafer_keys[index][1],
                self._wafer_keys[index][0],
            ),
        )

    def _is_pick_reserved(self, wafer_index: int) -> bool:
        return any(
            event.action_type == PICK and event.wafer_index == wafer_index
            for event in self._pending_queue
        )

    def _has_capacity_after_pick(
        self,
        module_index: int,
        wafer_index: int,
    ) -> bool:
        """Check target capacity after the selected wafer leaves its source."""

        module_id = self._module_ids[module_index]
        occupancy = self._module_occupancy(
            module_index,
            include_reservations=True,
        )
        if self._wafers[wafer_index].module_id == module_id:
            occupancy -= 1
        return occupancy < self.problem.Modules[module_id].capacity

    def _can_pick(self, wafer_index: int, robot_index: int) -> bool:
        wafer = self._wafers[wafer_index]
        robot = self._robots[robot_index]
        route = self.problem.routes[self._wafer_keys[wafer_index][0]]
        if (
            wafer.module_id is None
            or wafer.step_index >= len(route.visits) + 1
            or wafer.ready_at > self._time + EPS
            or not self._is_fifo_head(wafer_index)
            or self._is_pick_reserved(wafer_index)
            or robot.ready_at > self._time + EPS
            or len(robot.holding) >= self._arm_capacity(robot_index)
            or not self._can_achieve(robot_index, wafer.module_id)
        ):
            return False

        return any(
            self._can_achieve(robot_index, target_module_id)
            and self._has_capacity_after_pick(
                self._module_index[target_module_id],
                wafer_index,
            )
            for target_module_id in self._targets(wafer_index)
        )

    def _place_candidate(
        self,
        module_index: int,
        robot_index: int,
    ) -> int | None:
        robot = self._robots[robot_index]
        module_id = self._module_ids[module_index]
        if (
            robot.ready_at > self._time + EPS
            or not self._can_achieve(robot_index, module_id)
            or not self._has_capacity(module_index)
        ):
            return None

        return next(
            (
                wafer_index
                for wafer_index in robot.holding
                if module_id in self._targets(wafer_index)
            ),
            None,
        )

    def _module_occupancy(
        self,
        module_index: int,
        *,
        include_reservations: bool = False,
    ) -> int:
        """Return physical occupancy, optionally including pending places."""

        module_id = self._module_ids[module_index]
        occupancy = sum(
            wafer.module_id == module_id for wafer in self._wafers
        )
        occupancy += sum(
            event.action_type == PLACE
            and event.module_index == module_index
            and (event.started or include_reservations)
            for event in self._pending_queue
        )
        return occupancy

    def _has_capacity(self, module_index: int) -> bool:
        """Check capacity after accounting for dispatched Place reservations."""

        module_id = self._module_ids[module_index]
        return self._module_occupancy(
            module_index,
            include_reservations=True,
        ) < self.problem.Modules[module_id].capacity

    def _pick(self, wafer_index: int, robot_index: int) -> None:
        if not self._can_pick(wafer_index, robot_index):
            raise ValueError(
                f"Pick({wafer_index}, {robot_index}) is not allowed in the current state"
            )

        wafer = self._wafers[wafer_index]
        robot = self._robots[robot_index]
        module_id = wafer.module_id
        robot_id = self._robot_ids[robot_index]
        assert module_id is not None

        start = self._time + self._travel_time(robot_index, module_id)
        end = start + self._pick_time(robot_index)
        self._record(
            PICK,
            wafer_index,
            robot_id,
            module_id,
            wafer.step_index,
            start,
            end,
        )
        robot.ready_at = end
        self._pending_queue.append(
            _PendingEvent(
                PICK,
                robot_index,
                wafer_index,
                self._module_index[module_id],
                start,
                end,
            )
        )
        self._apply_events_at(self._time)

    def _place(self, module_index: int, robot_index: int) -> None:
        wafer_index = self._place_candidate(module_index, robot_index)
        if wafer_index is None:
            raise ValueError(
                f"Place({module_index}, {robot_index}) is not allowed in the current state"
            )
        wafer = self._wafers[wafer_index]
        robot = self._robots[robot_index]
        module_id = self._module_ids[module_index]
        next_step = wafer.step_index + 1
        robot_id = self._robot_ids[robot_index]

        start = self._time + self._travel_time(robot_index, module_id)
        end = start + self._place_time(robot_index)
        self._record(
            PLACE,
            wafer_index,
            robot_id,
            module_id,
            next_step,
            start,
            end,
        )

        robot.ready_at = end
        self._pending_queue.append(
            _PendingEvent(
                PLACE,
                robot_index,
                wafer_index,
                module_index,
                start,
                end,
            )
        )
        self._apply_events_at(self._time)

    def _advance(self) -> bool:
        """Advance to and apply the next physical event boundary.

        Event effects follow ``[start, end)`` semantics.  End effects for
        operations that were already active are applied before start effects
        at the same timestamp, so a resource released at ``t`` can be occupied
        by another operation starting at ``t``.

        Returns ``False`` when no future event can change the state.
        """

        self._apply_events_at(self._time)
        next_event_time = self._next_event_time()
        if next_event_time is None:
            return False

        self._time = next_event_time
        self._apply_events_at(self._time)
        return True

    def _next_event_time(self) -> float | None:
        """Return the next strictly future boundary without changing state."""

        future_times: list[float] = []
        for event in self._pending_queue:
            if not event.started and event.start > self._time + EPS:
                future_times.append(event.start)
            if event.end > self._time + EPS:
                future_times.append(event.end)

        for wafer in self._wafers:
            if wafer.module_id is not None and wafer.ready_at > self._time + EPS:
                future_times.append(wafer.ready_at)

        return min(future_times) if future_times else None

    def _apply_events_at(self, event_time: float) -> None:
        """Apply every pending boundary at ``event_time`` atomically."""

        ending = [
            event
            for event in self._pending_queue
            if event.started and event.end <= event_time + EPS
        ]
        for event in ending:
            self._finish_event(event)
            self._pending_queue.remove(event)

        starting = [
            event
            for event in self._pending_queue
            if not event.started and event.start <= event_time + EPS
        ]
        for event in starting:
            self._start_event(event)

        zero_duration = [
            event
            for event in self._pending_queue
            if event.started and event.end <= event_time + EPS
        ]
        for event in zero_duration:
            self._finish_event(event)
            self._pending_queue.remove(event)

    def _start_event(self, event: _PendingEvent) -> None:
        robot = self._robots[event.robot_index]
        robot.module_id = self._module_ids[event.module_index]

        if event.action_type == PICK:
            if event.wafer_index not in robot.holding:
                robot.holding.append(event.wafer_index)
        elif event.action_type != PLACE:
            raise ValueError(f"unknown pending action type: {event.action_type!r}")

        event.started = True

    def _finish_event(self, event: _PendingEvent) -> None:
        wafer = self._wafers[event.wafer_index]
        robot = self._robots[event.robot_index]
        module_id = self._module_ids[event.module_index]

        if event.action_type == PICK:
            wafer.module_id = None
            wafer.robot_id = self._robot_ids[event.robot_index]
            return

        if event.action_type != PLACE:
            raise ValueError(f"unknown pending action type: {event.action_type!r}")

        wafer.step_index += 1
        wafer.module_id = module_id
        wafer.robot_id = None
        robot.holding.remove(event.wafer_index)

        route = self.problem.routes[self._wafer_keys[event.wafer_index][0]]
        process_time = (
            route.visits[wafer.step_index - 1].process_time or 0.0
            if wafer.step_index <= len(route.visits)
            else 0.0
        )
        wafer.ready_at = event.end + process_time

    def _pick_time(self, robot_index: int):
        return self.problem.ClusterTool[self._robot_ids[robot_index]].pick_time

    def _place_time(self, robot_index: int):
        return self.problem.ClusterTool[self._robot_ids[robot_index]].place_time

    def _travel_time(self, robot_index: int, dst_module_id: str) -> float:
        src_module_id = self._robots[robot_index].module_id
        if src_module_id is None or src_module_id == dst_module_id:
            return 0.0
        return self.problem.ClusterTool[self._robot_ids[robot_index]].travel_time(
            src_module_id,
            dst_module_id,
        )

    def _record(
        self,
        action_type: str,
        wafer_index: int,
        robot_id: str,
        module_id: str,
        step_index: int,
        start: float,
        end: float,
    ) -> None:
        route_id, wafer_number = self._wafer_keys[wafer_index]
        self._actions.append(
            {
                "action_type": action_type,
                "tm_id": robot_id,
                "module_id": module_id,
                "route_id": route_id,
                "wafer_index": wafer_number,
                "step_index": step_index,
                "start": start,
                "end": end,
            }
        )

    def _complete(self) -> bool:
        return all(
            wafer.module_id == return_module_id
            and wafer.step_index
            == len(self.problem.routes[key[0]].visits) + 1
            for key, wafer, return_module_id in zip(
                self._wafer_keys,
                self._wafers,
                self._return_module_ids,
            )
        )
