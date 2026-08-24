from __future__ import annotations

from itertools import count

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
_STATE_REVISION_COUNTER = count()


class ClusterEngine:
    """Minimal stateful discrete-event core for RL environments."""

    def __init__(self, problem: ClusterProblem) -> None:
        self.problem = problem
        self._initial_wafers = problem.initial_state.to_snapshot().wafers_by_key
        self._wafer_keys = tuple(sorted(self._initial_wafers))
        self._robot_ids = tuple(sorted(problem.ClusterTool))
        self._route_visit_counts = {
            route_id: len(route.visits)
            for route_id, route in problem.routes.items()
        }
        self._arm_capacities = {
            robot_id: (
                1
                if problem.ClusterTool[robot_id].arm_type is TMArmType.SINGLE_ARM
                else 2
            )
            for robot_id in self._robot_ids
        }
        self._robot_ids_by_module = {
            module_id: tuple(
                robot_id
                for robot_id in self._robot_ids
                if module_id in problem.ClusterTool[robot_id].module_ids
            )
            for module_id in problem.Modules
        }
        source_priorities: dict[int, list[WaferKey]] = {}
        for wafer_key, initial in self._initial_wafers.items():
            if (
                initial.step_index == 0
                and isinstance(initial.location, ModuleLocation)
                and problem.Modules[initial.location.module_id].type
                in {ModuleType.IO, ModuleType.LP}
            ):
                source_priorities.setdefault(initial.priority, []).append(wafer_key)
        self._source_wafer_keys_by_priority = tuple(
            (priority, tuple(sorted(wafer_keys)))
            for priority, wafer_keys in sorted(source_priorities.items())
        )
        self._state: ClusterState | None = None
        self._state_revision = next(_STATE_REVISION_COUNTER)
        self._shared_wafer_keys: set[WaferKey] = set()
        self._shared_module_occupant_ids: set[str] = set()

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
        self._shared_wafer_keys.clear()
        self._shared_module_occupant_ids.clear()
        self._touch_state()
        return self._state

    def _touch_state(self) -> None:
        self._state_revision = next(_STATE_REVISION_COUNTER)

    def _state_cache_key(self) -> tuple[object, ...]:
        """Return a compact key for Engine-managed state plus common mutations.

        Pending operations, wafers, occupants, and Load Locks are covered by
        ``_state_revision``.  Time and Robot fields stay in the key because
        callers historically adjust those public runtime fields directly in
        diagnostics and tests.
        """

        state = self.state
        return (
            self._state_revision,
            state.time,
            tuple(
                (
                    robot_id,
                    robot.module_id,
                    robot.ready_at,
                    tuple(robot.holding),
                )
                for robot_id in self._robot_ids
                for robot in (state.robots[robot_id],)
            ),
        )

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

        return self._available_actions(recipe_fifo=False)

    def _available_actions(
        self,
        *,
        recipe_fifo: bool,
    ) -> tuple[EngineAction, ...]:
        """Build actions, optionally applying Env same-Recipe admission early."""

        state = self.state
        if self.is_complete():
            return ()

        pending_robot_ids = {
            operation.robot_id for operation in state.pending_operations
        }
        reserved_pick_wafer_keys = {
            operation.wafer_key
            for operation in state.pending_operations
            if operation.action_type == "pick"
        }
        reserved_place_counts: dict[str, int] = {}
        for operation in state.pending_operations:
            if operation.action_type == "place" and not operation.started:
                reserved_place_counts[operation.module_id] = (
                    reserved_place_counts.get(operation.module_id, 0) + 1
                )
        idle_robot_ids = {
            robot_id
            for robot_id, robot in state.robots.items()
            if robot_id not in pending_robot_ids
            and robot.ready_at <= state.time + EPS
        }
        pick_capable_robot_ids = {
            robot_id
            for robot_id in idle_robot_ids
            if len(state.robots[robot_id].holding) < self._arm_capacities[robot_id]
        }

        pick_entries: list[tuple[PickAction, bool, int]] = []
        minimum_source_priority: int | None = None
        admitted_source_groups: set[tuple[int, str]] = set()
        candidate_wafer_keys = sorted(
            wafer_key
            for occupants in state.module_occupants.values()
            for wafer_key in occupants
            if state.wafers[wafer_key].module_id is not None
        )
        for wafer_key in candidate_wafer_keys:
            wafer = state.wafers[wafer_key]
            module_id = wafer.module_id
            assert module_id is not None
            if (
                wafer.step_index >= self._route_visit_counts[wafer.route_id] + 1
                or wafer.ready_at > state.time + EPS
                or wafer_key in reserved_pick_wafer_keys
            ):
                continue
            is_source = self._is_initial_source_wafer(wafer)
            priority = self._initial_wafer(wafer_key).priority
            source_group = (priority, wafer.route_id)
            if recipe_fifo and is_source and source_group in admitted_source_groups:
                continue
            source_action_found = False
            for robot_id in self._robot_ids_by_module[module_id]:
                if robot_id not in pick_capable_robot_ids:
                    continue
                if (
                    wafer.last_place_robot_id is not None
                    and wafer.last_place_robot_id != robot_id
                    and self.problem.Modules[module_id].type
                    not in {ModuleType.BUFFER, ModuleType.LL}
                ):
                    continue
                if not self._ll_pick_allowed(module_id, robot_id):
                    continue
                action = PickAction(robot_id=robot_id, wafer_key=wafer_key)
                pick_entries.append((action, is_source, priority))
                if is_source:
                    source_action_found = True
                    minimum_source_priority = (
                        priority
                        if minimum_source_priority is None
                        else min(minimum_source_priority, priority)
                    )
            if recipe_fifo and source_action_found:
                admitted_source_groups.add(source_group)

        pick_actions = [
            action
            for action, is_source, priority in pick_entries
            if not is_source or priority == minimum_source_priority
        ]

        actions: list[EngineAction] = list(pick_actions)

        for robot_id in self._robot_ids:
            robot = state.robots[robot_id]
            if robot_id not in idle_robot_ids:
                continue
            for wafer_key in tuple(robot.holding):
                wafer = state.wafers[wafer_key]
                for module_id in self._next_targets(wafer):
                    if module_id not in self.problem.ClusterTool[robot_id].module_ids:
                        continue
                    occupancy = len(state.module_occupants[module_id])
                    occupancy += reserved_place_counts.get(module_id, 0)
                    if occupancy >= self.problem.Modules[module_id].capacity:
                        continue
                    if not self._ll_place_allowed(module_id, robot_id):
                        continue
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

        if isinstance(action, PickAction):
            if not self._can_pick(action.wafer_key, action.robot_id):
                raise IllegalActionError(f"action is not available: {action!r}")
            if not self._source_priority_allows(action):
                raise IllegalActionError(f"action is not available: {action!r}")
            return self._dispatch_pick(action)
        if isinstance(action, PlaceAction):
            if not self._can_place(action.wafer_key, action.target_module_id):
                raise IllegalActionError(f"action is not available: {action!r}")
            return self._dispatch_place(action)
        if isinstance(action, AdvanceAction):
            self._advance()
            return None
        raise TypeError(f"unsupported action: {action!r}")

    def _source_priority_allows(self, action: PickAction) -> bool:
        wafer = self.state.wafers[action.wafer_key]
        if not self._is_initial_source_wafer(wafer):
            return True
        priority = self._initial_wafer(action.wafer_key).priority
        for lower_priority, wafer_keys in self._source_wafer_keys_by_priority:
            if lower_priority >= priority:
                break
            for wafer_key in wafer_keys:
                source = self.state.wafers[wafer_key]
                if not self._is_initial_source_wafer(source):
                    continue
                for robot_id in self._robot_ids_by_module[source.module_id]:
                    if self._can_pick(wafer_key, robot_id):
                        return False
        return True

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
        self._touch_state()
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
        self._touch_state()
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
        self._apply_events_at(self.state.time)
        next_time = self.next_event_time()
        if next_time is None:
            raise IllegalActionError("Advance has no future event")
        self._advance_to(next_time)

    def _advance_to(self, next_time: float) -> None:
        state = self.state
        state.time = next_time
        self._apply_events_at(next_time)
        self._touch_state()

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
            self._mutable_module_occupants(operation.module_id).add(
                operation.wafer_key
            )
        operation.started = True

    def _finish_operation(self, operation: PendingOperation) -> None:
        state = self.state
        wafer = self._mutable_wafer(operation.wafer_key)
        robot = state.robots[operation.robot_id]
        if operation.action_type == "pick":
            self._mutable_module_occupants(operation.module_id).remove(
                operation.wafer_key
            )
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

    def _mutable_wafer(self, wafer_key: WaferKey) -> WaferState:
        wafer = self.state.wafers[wafer_key]
        if wafer_key not in self._shared_wafer_keys:
            return wafer
        wafer = WaferState(
            route_id=wafer.route_id,
            wafer_index=wafer.wafer_index,
            step_index=wafer.step_index,
            module_id=wafer.module_id,
            robot_id=wafer.robot_id,
            ready_at=wafer.ready_at,
            return_module_id=wafer.return_module_id,
            last_place_robot_id=wafer.last_place_robot_id,
        )
        self.state.wafers[wafer_key] = wafer
        self._shared_wafer_keys.remove(wafer_key)
        return wafer

    def _mutable_module_occupants(self, module_id: str) -> set[WaferKey]:
        occupants = self.state.module_occupants[module_id]
        if module_id not in self._shared_module_occupant_ids:
            return occupants
        occupants = set(occupants)
        self.state.module_occupants[module_id] = occupants
        self._shared_module_occupant_ids.remove(module_id)
        return occupants

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
