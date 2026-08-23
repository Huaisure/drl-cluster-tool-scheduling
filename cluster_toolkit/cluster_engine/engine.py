from __future__ import annotations

from cluster_toolkit.problem import (
    ClusterProblem,
    LoadLockState,
    ModuleLocation,
    ModuleType,
    RobotLocation,
    TMArmType,
    WaferInitialState,
    WaferKey,
)

from .models import (
    ADVANCE,
    AdvanceAction,
    DispatchRecord,
    EngineAction,
    IllegalActionError,
    PickAction,
    PlaceAction,
)
from .state import (
    ClusterState,
    LoadLockObservation,
    LoadLockRuntimeState,
    PendingOperation,
    RobotState,
    Side,
    WaferState,
)


EPS = 1e-9


class ClusterEngine:
    """Minimal stateful discrete-event core for RL environments."""

    def __init__(self, problem: ClusterProblem) -> None:
        self.problem = problem
        self._initial_wafers = problem.initial_state.to_snapshot().wafers_by_key
        self._state: ClusterState | None = None

    @property
    def state(self) -> ClusterState:
        if self._state is None:
            raise RuntimeError("reset() must be called before accessing Engine state")
        return self._state

    def reset(self) -> ClusterState:
        """Reset this Engine to the Problem's initial snapshot."""

        snapshot = self.problem.initial_state.to_snapshot()
        robots = {
            robot_id: RobotState(
                module_id=snapshot.tm_positions.get(robot_id),
            )
            for robot_id in self.problem.ClusterTool
        }
        for robot_id, arms in snapshot.tm_arms.items():
            robots[robot_id].holding.extend(
                wafer_key
                for _, wafer_key in sorted(arms.items())
            )

        wafers: dict[WaferKey, WaferState] = {}
        for wafer_key, initial in snapshot.wafers_by_key.items():
            module_id: str | None = None
            robot_id: str | None = None
            if isinstance(initial.location, ModuleLocation):
                module_id = initial.location.module_id
            elif isinstance(initial.location, RobotLocation):
                robot_id = initial.location.robot_id

            wafers[wafer_key] = WaferState(
                route_id=initial.route_id,
                wafer_index=initial.wafer_index,
                step_index=initial.step_index,
                module_id=module_id,
                robot_id=robot_id,
                ready_at=float(initial.process_end_time or 0.0),
                return_module_id=self.problem.return_module_id(initial),
            )

        load_locks = {
            module_id: LoadLockRuntimeState(
                last_pick_side=self._side(module.load_lock.initial_state),
                last_pick_end=0.0,
            )
            for module_id, module in self.problem.Modules.items()
            if module.load_lock is not None
        }
        for module_id, occupants in snapshot.module_occupants.items():
            if module_id not in load_locks or not occupants:
                continue
            load_lock = self.problem.Modules[module_id].load_lock
            assert load_lock is not None
            load_locks[module_id].occupied_exit_side = self._side(
                load_lock.initial_state
            )
            load_locks[module_id].occupied_ready_at = max(
                wafers[wafer_key].ready_at
                for wafer_key in occupants
            )
            load_locks[module_id].occupied_transition_duration = load_locks[
                module_id
            ].occupied_ready_at

        self._state = ClusterState(
            time=0.0,
            wafers=wafers,
            robots=robots,
            module_occupants={
                module_id: set(snapshot.module_occupants.get(module_id, ()))
                for module_id in self.problem.Modules
            },
            load_locks=load_locks,
        )
        return self._state

    def load_lock_observation(self, module_id: str) -> LoadLockObservation:
        """Return a model-facing snapshot for one conversion Load Lock.

        Empty-LL progress starts at the last ``Pick.end`` and measures when
        the opposite side becomes placeable. Occupied-LL progress starts at
        the entering ``Place.end`` and measures when the configured exit side
        becomes pickable. Inactive progress is zero.
        """

        try:
            runtime = self.state.load_locks[module_id]
        except KeyError as exc:
            raise ValueError(
                f"Module {module_id!r} is not a conversion Load Lock"
            ) from exc

        load_lock = self.problem.Modules[module_id].load_lock
        assert load_lock is not None
        pump_time = float(load_lock.atmosphere_to_vacuum_time)
        vent_time = float(load_lock.vacuum_to_atmosphere_time)
        if runtime.occupied_exit_side is None:
            opposite = self._opposite_side(runtime.last_pick_side)
            duration = self._ll_transition_duration(
                module_id,
                runtime.last_pick_side,
                opposite,
            )
            empty_progress = self._transition_progress(
                self.state.time - runtime.last_pick_end,
                duration,
            )
            occupied_progress = 0.0
        else:
            empty_progress = 0.0
            occupied_progress = self._transition_progress(
                self.state.time - runtime.occupied_transition_start,
                runtime.occupied_transition_duration,
            )

        return LoadLockObservation(
            pump_time=pump_time,
            vent_time=vent_time,
            last_pick_side=runtime.last_pick_side,
            empty_transition_progress=empty_progress,
            occupied_exit_side=runtime.occupied_exit_side,
            occupied_transition_progress=occupied_progress,
        )

    def available_actions(self) -> tuple[EngineAction, ...]:
        """Return semantic actions allowed by physics and admission priority."""

        state = self.state
        if self.is_complete():
            return ()

        pick_actions: list[PickAction] = []

        for wafer_key, wafer in sorted(state.wafers.items()):
            for robot_id in sorted(state.robots):
                if not self._can_pick(wafer_key, robot_id):
                    continue
                pick_actions.append(
                    PickAction(robot_id=robot_id, wafer_key=wafer_key)
                )

        source_picks = [
            action
            for action in pick_actions
            if self._is_initial_source_wafer(state.wafers[action.wafer_key])
        ]
        if source_picks:
            minimum_priority = min(
                self._initial_wafer(action.wafer_key).priority
                for action in source_picks
            )
            pick_actions = [
                action
                for action in pick_actions
                if (
                    action not in source_picks
                    or self._initial_wafer(action.wafer_key).priority
                    == minimum_priority
                )
            ]

        actions: list[EngineAction] = list(pick_actions)

        for robot_id, robot in sorted(state.robots.items()):
            for wafer_key in tuple(robot.holding):
                wafer = state.wafers[wafer_key]
                for module_id in self._next_targets(wafer):
                    if self._can_place(wafer_key, module_id):
                        actions.append(
                            PlaceAction(
                                wafer_key=wafer_key,
                                target_module_id=module_id,
                            )
                        )

        if self.next_event_time() is not None:
            actions.append(ADVANCE)
        return tuple(actions)

    def step(self, action: EngineAction) -> DispatchRecord | None:
        """Apply one available action to this Engine in place."""

        if action not in self.available_actions():
            raise IllegalActionError(f"action is not available: {action!r}")
        if isinstance(action, PickAction):
            return self._dispatch_pick(action)
        if isinstance(action, PlaceAction):
            return self._dispatch_place(action)
        if isinstance(action, AdvanceAction):
            self._advance()
            return None
        raise TypeError(f"unsupported action: {action!r}")

    def next_event_time(self) -> float | None:
        """Return the next future boundary that can change action availability."""

        state = self.state
        times: list[float] = []
        for operation in state.pending_operations:
            if not operation.started and operation.start > state.time + EPS:
                times.append(operation.start)
            if operation.end > state.time + EPS:
                times.append(operation.end)

        times.extend(
            wafer.ready_at
            for wafer in state.wafers.values()
            if wafer.module_id is not None and wafer.ready_at > state.time + EPS
        )

        for module_id, runtime in state.load_locks.items():
            if self._reserved_occupancy(module_id):
                continue
            opposite = self._opposite_side(runtime.last_pick_side)
            ready_at = runtime.last_pick_end + self._ll_transition_duration(
                module_id,
                runtime.last_pick_side,
                opposite,
            )
            if ready_at > state.time + EPS:
                times.append(ready_at)

        return min(times) if times else None

    def is_complete(self) -> bool:
        state = self.state
        if state.pending_operations:
            return False
        for wafer in state.wafers.values():
            route = self.problem.routes[wafer.route_id]
            if wafer.step_index != len(route.visits) + 1:
                return False
            if wafer.module_id != wafer.return_module_id:
                return False
        return True

    def is_deadlocked(self) -> bool:
        return not self.is_complete() and not self.available_actions()

    def _is_initial_source_wafer(self, wafer: WaferState) -> bool:
        return (
            wafer.step_index == 0
            and wafer.module_id is not None
            and self.problem.Modules[wafer.module_id].type
            in {ModuleType.IO, ModuleType.LP}
        )

    def _initial_wafer(self, wafer_key: WaferKey) -> WaferInitialState:
        return self._initial_wafers[wafer_key]

    def _can_pick(self, wafer_key: WaferKey, robot_id: str) -> bool:
        state = self.state
        wafer = state.wafers[wafer_key]
        robot = state.robots[robot_id]
        if wafer.module_id is None:
            return False
        if wafer.step_index >= len(self.problem.routes[wafer.route_id].visits) + 1:
            return False
        if wafer.ready_at > state.time + EPS:
            return False
        if self._wafer_reserved_for_pick(wafer_key):
            return False
        if not self._robot_idle(robot_id):
            return False
        if len(robot.holding) >= self._arm_capacity(robot_id):
            return False
        if wafer.module_id not in self.problem.ClusterTool[robot_id].module_ids:
            return False
        if (
            wafer.last_place_robot_id is not None
            and wafer.last_place_robot_id != robot_id
            and self.problem.Modules[wafer.module_id].type
            not in {ModuleType.BUFFER, ModuleType.LL}
        ):
            return False
        if wafer_key not in state.module_occupants[wafer.module_id]:
            return False
        return self._ll_pick_allowed(wafer.module_id, robot_id)

    def _can_place(self, wafer_key: WaferKey, module_id: str) -> bool:
        state = self.state
        wafer = state.wafers[wafer_key]
        robot_id = wafer.robot_id
        if robot_id is None or wafer_key not in state.robots[robot_id].holding:
            return False
        if not self._robot_idle(robot_id):
            return False
        if module_id not in self._next_targets(wafer):
            return False
        if module_id not in self.problem.ClusterTool[robot_id].module_ids:
            return False
        if self._reserved_occupancy(module_id) >= self.problem.Modules[module_id].capacity:
            return False
        return self._ll_place_allowed(module_id, robot_id)

    def _dispatch_pick(self, action: PickAction) -> DispatchRecord:
        state = self.state
        wafer = state.wafers[action.wafer_key]
        assert wafer.module_id is not None
        module_id = wafer.module_id
        robot = state.robots[action.robot_id]
        start = state.time + self._travel_time(action.robot_id, module_id)
        end = start + float(self.problem.ClusterTool[action.robot_id].pick_time)
        operation = PendingOperation(
            action_type="pick",
            robot_id=action.robot_id,
            wafer_key=action.wafer_key,
            module_id=module_id,
            start=start,
            end=end,
        )
        robot.ready_at = end
        state.pending_operations.append(operation)
        self._apply_events_at(state.time)
        return DispatchRecord(
            action_type="pick",
            robot_id=action.robot_id,
            module_id=module_id,
            wafer_key=action.wafer_key,
            step_index=wafer.step_index,
            start=start,
            end=end,
        )

    def _dispatch_place(self, action: PlaceAction) -> DispatchRecord:
        state = self.state
        wafer = state.wafers[action.wafer_key]
        assert wafer.robot_id is not None
        robot_id = wafer.robot_id
        robot = state.robots[robot_id]
        start = state.time + self._travel_time(robot_id, action.target_module_id)
        end = start + float(self.problem.ClusterTool[robot_id].place_time)
        operation = PendingOperation(
            action_type="place",
            robot_id=robot_id,
            wafer_key=action.wafer_key,
            module_id=action.target_module_id,
            start=start,
            end=end,
        )
        robot.ready_at = end
        state.pending_operations.append(operation)
        self._apply_events_at(state.time)
        return DispatchRecord(
            action_type="place",
            robot_id=robot_id,
            module_id=action.target_module_id,
            wafer_key=action.wafer_key,
            step_index=wafer.step_index + 1,
            start=start,
            end=end,
        )

    def _advance(self) -> None:
        state = self.state
        self._apply_events_at(state.time)
        next_time = self.next_event_time()
        if next_time is None:
            raise IllegalActionError("Advance has no future event")
        state.time = next_time
        self._apply_events_at(next_time)

    def _apply_events_at(self, event_time: float) -> None:
        state = self.state
        ending = [
            operation
            for operation in state.pending_operations
            if operation.started and operation.end <= event_time + EPS
        ]
        for operation in ending:
            self._finish_operation(operation)
            state.pending_operations.remove(operation)

        starting = [
            operation
            for operation in state.pending_operations
            if not operation.started and operation.start <= event_time + EPS
        ]
        for operation in starting:
            self._start_operation(operation)

        zero_duration = [
            operation
            for operation in state.pending_operations
            if operation.started and operation.end <= event_time + EPS
        ]
        for operation in zero_duration:
            self._finish_operation(operation)
            state.pending_operations.remove(operation)

    def _start_operation(self, operation: PendingOperation) -> None:
        state = self.state
        robot = state.robots[operation.robot_id]
        robot.module_id = operation.module_id
        if operation.action_type == "pick":
            robot.holding.append(operation.wafer_key)
        else:
            state.module_occupants[operation.module_id].add(operation.wafer_key)
        operation.started = True

    def _finish_operation(self, operation: PendingOperation) -> None:
        state = self.state
        wafer = state.wafers[operation.wafer_key]
        robot = state.robots[operation.robot_id]
        if operation.action_type == "pick":
            state.module_occupants[operation.module_id].remove(operation.wafer_key)
            wafer.module_id = None
            wafer.robot_id = operation.robot_id
            if operation.module_id in state.load_locks:
                runtime = state.load_locks[operation.module_id]
                runtime.last_pick_side = self._ll_robot_side(
                    operation.module_id,
                    operation.robot_id,
                )
                runtime.last_pick_end = operation.end
                runtime.occupied_exit_side = None
                runtime.occupied_ready_at = 0.0
                runtime.occupied_transition_start = 0.0
                runtime.occupied_transition_duration = 0.0
            return

        wafer.step_index += 1
        wafer.module_id = operation.module_id
        wafer.robot_id = None
        wafer.last_place_robot_id = operation.robot_id
        robot.holding.remove(operation.wafer_key)
        wafer.ready_at = operation.end + self._process_time(wafer)
        if operation.module_id in state.load_locks:
            self._configure_occupied_load_lock(operation, wafer)

    def _configure_occupied_load_lock(
        self,
        operation: PendingOperation,
        wafer: WaferState,
    ) -> None:
        module_id = operation.module_id
        runtime = self.state.load_locks[module_id]
        entry_side = self._ll_robot_side(module_id, operation.robot_id)
        exit_side = self._next_destination_side(module_id, wafer)
        transition_duration = self._ll_transition_duration(
            module_id,
            entry_side,
            exit_side,
        )
        runtime.occupied_exit_side = exit_side
        runtime.occupied_transition_start = operation.end
        runtime.occupied_transition_duration = transition_duration
        runtime.occupied_ready_at = operation.end + transition_duration
        wafer.ready_at = max(wafer.ready_at, runtime.occupied_ready_at)

    def _next_destination_side(self, load_lock_id: str, wafer: WaferState) -> Side:
        module = self.problem.Modules[load_lock_id]
        assert module.load_lock is not None
        sides: set[Side] = set()
        targets = self._next_targets(wafer)
        for target_module_id in targets:
            target_sides = {
                self._side(required_state)
                for robot_id, required_state in module.load_lock.tm_required_states.items()
                if target_module_id in self.problem.ClusterTool[robot_id].module_ids
            }
            if len(target_sides) != 1:
                raise ValueError(
                    f"Load Lock {load_lock_id} cannot infer one side for target "
                    f"{target_module_id}: {sorted(target_sides)}"
                )
            sides.update(target_sides)
        if len(sides) != 1:
            raise ValueError(
                f"Load Lock {load_lock_id} next Route candidates must share one side"
            )
        return next(iter(sides))

    def _ll_pick_allowed(self, module_id: str, robot_id: str) -> bool:
        if module_id not in self.state.load_locks:
            return True
        runtime = self.state.load_locks[module_id]
        side = self._ll_robot_side(module_id, robot_id)
        return (
            runtime.occupied_exit_side == side
            and self.state.time + EPS >= runtime.occupied_ready_at
        )

    def _ll_place_allowed(self, module_id: str, robot_id: str) -> bool:
        if module_id not in self.state.load_locks:
            return True
        runtime = self.state.load_locks[module_id]
        side = self._ll_robot_side(module_id, robot_id)
        if side == runtime.last_pick_side:
            return True
        ready_at = runtime.last_pick_end + self._ll_transition_duration(
            module_id,
            runtime.last_pick_side,
            side,
        )
        return self.state.time + EPS >= ready_at

    def _ll_robot_side(self, module_id: str, robot_id: str) -> Side:
        load_lock = self.problem.Modules[module_id].load_lock
        assert load_lock is not None
        try:
            return self._side(load_lock.tm_required_states[robot_id])
        except KeyError as exc:
            raise ValueError(
                f"Load Lock {module_id} has no side configured for Robot {robot_id}"
            ) from exc

    def _ll_transition_duration(
        self,
        module_id: str,
        source: Side,
        target: Side,
    ) -> float:
        if source == target:
            return 0.0
        load_lock = self.problem.Modules[module_id].load_lock
        assert load_lock is not None
        if source == "atmosphere":
            return float(load_lock.atmosphere_to_vacuum_time)
        return float(load_lock.vacuum_to_atmosphere_time)

    @staticmethod
    def _transition_progress(elapsed: float, duration: float) -> float:
        if duration <= EPS:
            return 1.0
        return min(1.0, max(0.0, elapsed / duration))

    def _next_targets(self, wafer: WaferState) -> tuple[str, ...]:
        route = self.problem.routes[wafer.route_id]
        next_step = wafer.step_index + 1
        if 1 <= next_step <= len(route.visits):
            return route.visits[next_step - 1].module_ids
        if next_step == len(route.visits) + 1:
            return (wafer.return_module_id,)
        return ()

    def _process_time(self, wafer: WaferState) -> float:
        route = self.problem.routes[wafer.route_id]
        if not 1 <= wafer.step_index <= len(route.visits):
            return 0.0
        return float(route.visits[wafer.step_index - 1].process_time or 0.0)

    def _travel_time(self, robot_id: str, target_module_id: str) -> float:
        source_module_id = self.state.robots[robot_id].module_id
        if source_module_id is None or source_module_id == target_module_id:
            return 0.0
        return float(
            self.problem.ClusterTool[robot_id].travel_time(
                source_module_id,
                target_module_id,
            )
        )

    def _robot_idle(self, robot_id: str) -> bool:
        return (
            self.state.robots[robot_id].ready_at <= self.state.time + EPS
            and not any(
                operation.robot_id == robot_id
                for operation in self.state.pending_operations
            )
        )

    def _arm_capacity(self, robot_id: str) -> int:
        arm_type = self.problem.ClusterTool[robot_id].arm_type
        return 1 if arm_type is TMArmType.SINGLE_ARM else 2

    def _wafer_reserved_for_pick(self, wafer_key: WaferKey) -> bool:
        return any(
            operation.action_type == "pick" and operation.wafer_key == wafer_key
            for operation in self.state.pending_operations
        )

    def _reserved_occupancy(self, module_id: str) -> int:
        return len(self.state.module_occupants[module_id]) + sum(
            operation.action_type == "place"
            and operation.module_id == module_id
            and not operation.started
            for operation in self.state.pending_operations
        )

    @staticmethod
    def _side(state: LoadLockState) -> Side:
        if state is LoadLockState.ATMOSPHERE:
            return "atmosphere"
        return "vacuum"

    @staticmethod
    def _opposite_side(side: Side) -> Side:
        return "vacuum" if side == "atmosphere" else "atmosphere"
