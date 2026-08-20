from __future__ import annotations

from copy import copy

from cluster_toolkit.cluster_engine import (
    ADVANCE,
    ClusterEngine,
    ClusterState,
    EngineAction,
    LoadLockRuntimeState,
    PendingOperation,
    PickAction,
    PlaceAction,
    RobotState,
    WaferState,
)
from cluster_toolkit.problem import ClusterProblem, ModuleType, TMArmType, WaferKey


class ActionSafetyFilter:
    """Filter Engine actions using sound reachability and wait-cycle proofs.

    Each Robot has an independent search budget. Event advancement is free,
    and an action consumes only the budget of the Robot that owns it.
    """

    def __init__(
        self,
        problem: ClusterProblem,
        *,
        lookahead_depth: int = 2,
    ) -> None:
        if lookahead_depth < 0:
            raise ValueError("lookahead_depth must be non-negative")
        snapshot = problem.initial_state.to_snapshot()
        self.problem = problem
        self.lookahead_depth = lookahead_depth
        self._initial_wafers = snapshot.wafers_by_key
        self._robot_ids = tuple(sorted(problem.ClusterTool))
        self._robot_index = {
            robot_id: index for index, robot_id in enumerate(self._robot_ids)
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
            tuple[str, int, str, str, str | None],
            frozenset[str],
        ] = {}
        self._cached_state_signature: tuple[object, ...] | None = None
        self._cached_safe_actions: tuple[EngineAction, ...] = ()

    def available_actions(
        self,
        engine: ClusterEngine,
    ) -> tuple[EngineAction, ...]:
        """Apply the Env same-Recipe source-wafer admission order."""

        actions = engine.available_actions()
        source_picks = [
            action
            for action in actions
            if isinstance(action, PickAction)
            and self.is_source_pick(engine, action)
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
            or not self.is_source_pick(engine, action)
            or self._initial_wafers[action.wafer_key].wafer_index
            == minimum_indexes[
                (
                    self._initial_wafers[action.wafer_key].priority,
                    self._initial_wafers[action.wafer_key].route_id,
                )
            ]
        )

    def safe_actions(
        self,
        engine: ClusterEngine,
    ) -> tuple[EngineAction, ...]:
        state_signature = self.state_signature(engine)
        if state_signature == self._cached_state_signature:
            return self._cached_safe_actions

        actions = self.static_safe_actions(engine)
        if self.lookahead_depth == 0:
            safe_actions = actions
        else:
            memo: dict[tuple[object, ...], bool] = {}
            safe_actions = tuple(
                action
                for action in actions
                if self.action_has_safe_continuation(
                    engine,
                    action,
                    self.lookahead_depth,
                    memo,
                )
            )
        self._cached_state_signature = state_signature
        self._cached_safe_actions = safe_actions
        return safe_actions

    def static_safe_actions(
        self,
        engine: ClusterEngine,
    ) -> tuple[EngineAction, ...]:
        return tuple(
            action
            for action in self.available_actions(engine)
            if not isinstance(action, PickAction)
            or (
                self.robot_can_reach_next_target(engine, action)
                and not self.pick_closes_wait_cycle(engine, action)
            )
        )

    def action_has_safe_continuation(
        self,
        engine: ClusterEngine,
        action: EngineAction,
        remaining_depth: int,
        memo: dict[tuple[object, ...], bool] | None = None,
    ) -> bool:
        """Prove that one bounded continuation avoids every known deadlock."""

        focus_robot_id = self.action_robot_id(engine, action)
        return self._state_has_safe_continuation(
            self.engine_after_action(engine, action),
            (remaining_depth,) * len(self._robot_ids),
            focus_robot_id,
            self.watched_modules_for_action(engine, action),
            {} if memo is None else memo,
        )

    def _state_has_safe_continuation(
        self,
        engine: ClusterEngine,
        remaining_by_robot: tuple[int, ...],
        focus_robot_id: str | None,
        watched_module_ids: frozenset[str],
        memo: dict[tuple[object, ...], bool],
    ) -> bool:
        key = (
            focus_robot_id,
            watched_module_ids,
            remaining_by_robot,
            self.state_signature(engine),
        )
        if key in memo:
            return memo[key]

        result = self._uncached_state_has_safe_continuation(
            engine,
            remaining_by_robot,
            focus_robot_id,
            watched_module_ids,
            memo,
        )
        memo[key] = result
        return result

    def _uncached_state_has_safe_continuation(
        self,
        engine: ClusterEngine,
        remaining_by_robot: tuple[int, ...],
        focus_robot_id: str | None,
        watched_module_ids: frozenset[str],
        memo: dict[tuple[object, ...], bool],
    ) -> bool:
        if engine.is_complete():
            return True
        if self.has_deadlock_proof(engine):
            return False

        static_actions = self.static_safe_actions(engine)
        actions = tuple(
            action
            for action in static_actions
            if focus_robot_id is None
            or self.action_robot_id(engine, action) == focus_robot_id
            or self.external_action_can_release_watched_resource(
                engine,
                action,
                watched_module_ids,
            )
        )
        if not actions:
            if (
                static_actions
                and focus_robot_id is not None
                and not engine.state.robots[focus_robot_id].holding
                and not any(
                    operation.robot_id == focus_robot_id
                    for operation in engine.state.pending_operations
                )
            ):
                # Upstream work may later reach an idle focus Robot. Treat the
                # unexpanded continuation as unknown rather than deadlocked.
                return True
            return False

        # Reaching any Robot's horizon without a deadlock proof is optimistic:
        # bounded search may reject only branches that are already proven bad.
        for action in actions:
            robot_id = self.action_robot_id(engine, action)
            if (
                robot_id is not None
                and remaining_by_robot[self._robot_index[robot_id]] <= 0
            ):
                return True

        for action in sorted(
            actions,
            key=lambda candidate: self.search_priority(engine, candidate),
        ):
            if action == ADVANCE:
                next_remaining = remaining_by_robot
            else:
                robot_id = self.action_robot_id(engine, action)
                assert robot_id is not None
                robot_index = self._robot_index[robot_id]
                mutable_remaining = list(remaining_by_robot)
                mutable_remaining[robot_index] -= 1
                next_remaining = tuple(mutable_remaining)
            if self._state_has_safe_continuation(
                self.engine_after_action(engine, action),
                next_remaining,
                focus_robot_id,
                watched_module_ids
                | self.watched_modules_for_action(engine, action),
                memo,
            ):
                return True
        return False

    def watched_modules_for_action(
        self,
        engine: ClusterEngine,
        action: EngineAction,
    ) -> frozenset[str]:
        if isinstance(action, PickAction):
            return frozenset(
                self.next_targets_for_wafer(engine, action.wafer_key)
            )
        if isinstance(action, PlaceAction):
            return frozenset({action.target_module_id})
        return frozenset()

    def external_action_can_release_watched_resource(
        self,
        engine: ClusterEngine,
        action: EngineAction,
        watched_module_ids: frozenset[str],
    ) -> bool:
        """Return whether a non-focus action can create watched capacity."""

        if action == ADVANCE:
            return True
        if isinstance(action, PickAction):
            return (
                engine.state.wafers[action.wafer_key].module_id
                in watched_module_ids
            )
        if action.target_module_id in watched_module_ids:
            return True

        robot_id = self.action_robot_id(engine, action)
        assert robot_id is not None
        robot = engine.state.robots[robot_id]
        if len(robot.holding) < self._arm_capacities[robot_id]:
            return False
        return any(
            robot_id
            in self._transfer_robot_ids(engine, wafer_key, module_id)
            for module_id in watched_module_ids
            for wafer_key in engine.state.module_occupants[module_id]
        )

    @staticmethod
    def state_signature(engine: ClusterEngine) -> tuple[object, ...]:
        """Return an exact immutable key for one Engine runtime state."""

        state = engine.state
        return (
            state.time,
            tuple(
                (
                    wafer_key,
                    wafer.step_index,
                    wafer.module_id,
                    wafer.robot_id,
                    wafer.ready_at,
                )
                for wafer_key, wafer in sorted(state.wafers.items())
            ),
            tuple(
                (
                    robot_id,
                    robot.module_id,
                    robot.ready_at,
                    tuple(robot.holding),
                )
                for robot_id, robot in sorted(state.robots.items())
            ),
            tuple(
                (module_id, tuple(sorted(occupants)))
                for module_id, occupants in sorted(
                    state.module_occupants.items()
                )
            ),
            tuple(
                (
                    module_id,
                    runtime.last_pick_side,
                    runtime.last_pick_end,
                    runtime.occupied_exit_side,
                    runtime.occupied_ready_at,
                    runtime.occupied_transition_start,
                    runtime.occupied_transition_duration,
                )
                for module_id, runtime in sorted(state.load_locks.items())
            ),
            tuple(
                (
                    operation.action_type,
                    operation.robot_id,
                    operation.wafer_key,
                    operation.module_id,
                    operation.start,
                    operation.end,
                    operation.started,
                )
                for operation in state.pending_operations
            ),
        )

    def has_deadlock_proof(self, engine: ClusterEngine) -> bool:
        """Return whether a closed Robot or single-arm Module wait set exists."""

        return bool(
            self.closed_wait_robot_ids(engine)
            or self.closed_single_arm_module_ids(engine)
        )

    def closed_single_arm_module_ids(
        self,
        engine: ClusterEngine,
    ) -> frozenset[str]:
        """Find full Module sets that one single-arm Robot cannot drain."""

        pending_picks = {
            operation.wafer_key
            for operation in engine.state.pending_operations
            if operation.action_type == "pick"
        }
        pending_places = {
            operation.wafer_key: operation.module_id
            for operation in engine.state.pending_operations
            if operation.action_type == "place"
        }
        occupants = {
            module_id: {
                wafer_key: engine.state.wafers[wafer_key].step_index
                for wafer_key in wafer_keys
                if wafer_key not in pending_picks
            }
            for module_id, wafer_keys in engine.state.module_occupants.items()
        }
        for wafer_key, module_id in pending_places.items():
            occupants[module_id][wafer_key] = (
                engine.state.wafers[wafer_key].step_index + 1
            )

        full_modules = {
            module_id
            for module_id, wafer_steps in occupants.items()
            if len(wafer_steps) >= self.problem.Modules[module_id].capacity
        }
        closed_modules: set[str] = set()
        for robot_id in self._robot_ids:
            if self._arm_capacities[robot_id] != 1:
                continue
            candidates = {
                module_id
                for module_id in full_modules
                if occupants[module_id]
                and all(
                    self._transfer_robot_ids(
                        engine,
                        wafer_key,
                        module_id,
                        step_index=step_index,
                    )
                    == frozenset({robot_id})
                    for wafer_key, step_index in occupants[module_id].items()
                )
            }
            while True:
                escaping_modules = {
                    module_id
                    for module_id in candidates
                    if any(
                        any(
                            target_id == module_id
                            or target_id not in candidates
                            for target_id in self.next_targets(
                                engine.state.wafers[wafer_key].route_id,
                                step_index,
                                engine.state.wafers[wafer_key].return_module_id,
                            )
                        )
                        for wafer_key, step_index in occupants[module_id].items()
                    )
                }
                if not escaping_modules:
                    break
                candidates.difference_update(escaping_modules)
            closed_modules.update(candidates)
        return frozenset(closed_modules)

    def pick_closes_wait_cycle(
        self,
        engine: ClusterEngine,
        action: PickAction,
    ) -> bool:
        robot = engine.state.robots[action.robot_id]
        held_after_pick = (*robot.holding, action.wafer_key)
        if len(held_after_pick) < self._arm_capacities[action.robot_id]:
            return False
        return action.robot_id in self.closed_wait_robot_ids(
            engine,
            holding_overrides={action.robot_id: held_after_pick},
            released_wafer_key=action.wafer_key,
        )

    def closed_wait_robot_ids(
        self,
        engine: ClusterEngine,
        *,
        holding_overrides: dict[str, tuple[WaferKey, ...]] | None = None,
        released_wafer_key: WaferKey | None = None,
    ) -> frozenset[str]:
        """Return full Robots that remain in a closed resource wait graph."""

        holding_overrides = holding_overrides or {}
        pending_robot_ids = {
            operation.robot_id
            for operation in engine.state.pending_operations
        }
        holding_by_robot: dict[str, tuple[WaferKey, ...]] = {}
        for robot_id, robot in engine.state.robots.items():
            if robot_id in pending_robot_ids:
                continue
            holding = holding_overrides.get(robot_id, tuple(robot.holding))
            if len(holding) >= self._arm_capacities[robot_id]:
                holding_by_robot[robot_id] = holding

        blocked_robot_ids = set(holding_by_robot)
        while True:
            escaped_robot_ids = {
                robot_id
                for robot_id in blocked_robot_ids
                if any(
                    self._wafer_has_external_destination(
                        engine,
                        wafer_key,
                        blocked_robot_ids,
                        released_wafer_key,
                    )
                    for wafer_key in holding_by_robot[robot_id]
                )
            }
            if not escaped_robot_ids:
                return frozenset(blocked_robot_ids)
            blocked_robot_ids.difference_update(escaped_robot_ids)

    def _wafer_has_external_destination(
        self,
        engine: ClusterEngine,
        wafer_key: WaferKey,
        blocked_robot_ids: set[str],
        released_wafer_key: WaferKey | None,
    ) -> bool:
        return any(
            self._target_has_external_capacity(
                engine,
                target_id,
                blocked_robot_ids,
                released_wafer_key,
            )
            for target_id in self.next_targets_for_wafer(engine, wafer_key)
        )

    def _target_has_external_capacity(
        self,
        engine: ClusterEngine,
        module_id: str,
        blocked_robot_ids: set[str],
        released_wafer_key: WaferKey | None,
    ) -> bool:
        pending_places = {
            operation.wafer_key: operation
            for operation in engine.state.pending_operations
            if operation.action_type == "place" and operation.module_id == module_id
        }
        if any(
            operation.action_type == "pick" and operation.module_id == module_id
            for operation in engine.state.pending_operations
        ):
            return True

        blockers = set(engine.state.module_occupants[module_id])
        blockers.update(pending_places)
        if released_wafer_key is not None:
            blockers.discard(released_wafer_key)
        if len(blockers) < self.problem.Modules[module_id].capacity:
            return True

        for wafer_key in blockers:
            step_index = engine.state.wafers[wafer_key].step_index + int(
                wafer_key in pending_places
            )
            if any(
                candidate not in blocked_robot_ids
                for candidate in self._transfer_robot_ids(
                    engine,
                    wafer_key,
                    module_id,
                    step_index=step_index,
                )
            ):
                return True
        return False

    def _transfer_robot_ids(
        self,
        engine: ClusterEngine,
        wafer_key: WaferKey,
        source_module_id: str,
        *,
        step_index: int | None = None,
    ) -> frozenset[str]:
        wafer = engine.state.wafers[wafer_key]
        module = self.problem.Modules[source_module_id]
        selected_step_index = wafer.step_index if step_index is None else step_index
        has_pending_place = any(
            operation.action_type == "place"
            and operation.module_id == source_module_id
            and operation.wafer_key == wafer_key
            for operation in engine.state.pending_operations
        )
        exit_side = (
            engine.state.load_locks[source_module_id].occupied_exit_side
            if module.load_lock is not None
            and wafer_key in engine.state.module_occupants[source_module_id]
            and not has_pending_place
            else None
        )
        key = (
            wafer.route_id,
            selected_step_index,
            wafer.return_module_id,
            source_module_id,
            exit_side,
        )
        if key not in self._transfer_robots_cache:
            targets = self.next_targets(
                wafer.route_id,
                selected_step_index,
                wafer.return_module_id,
            )
            self._transfer_robots_cache[key] = frozenset(
                robot_id
                for robot_id, robot_modules in self._robot_modules.items()
                if source_module_id in robot_modules
                and any(target_id in robot_modules for target_id in targets)
                and (
                    exit_side is None
                    or (
                        module.load_lock is not None
                        and module.load_lock.tm_required_states.get(robot_id)
                        is not None
                        and module.load_lock.tm_required_states[robot_id].value
                        == exit_side
                    )
                )
            )
        return self._transfer_robots_cache[key]

    def robot_can_reach_next_target(
        self,
        engine: ClusterEngine,
        action: PickAction,
    ) -> bool:
        return any(
            module_id in self._robot_modules[action.robot_id]
            for module_id in self.next_targets_for_wafer(
                engine,
                action.wafer_key,
            )
        )

    def next_targets_for_wafer(
        self,
        engine: ClusterEngine,
        wafer_key: WaferKey,
    ) -> tuple[str, ...]:
        wafer = engine.state.wafers[wafer_key]
        return self.next_targets(
            wafer.route_id,
            wafer.step_index,
            wafer.return_module_id,
        )

    def next_targets(
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

    def is_source_pick(
        self,
        engine: ClusterEngine,
        action: PickAction,
    ) -> bool:
        wafer = engine.state.wafers[action.wafer_key]
        return (
            wafer.step_index == 0
            and wafer.module_id is not None
            and self.problem.Modules[wafer.module_id].type
            in {ModuleType.IO, ModuleType.LP}
        )

    def action_robot_id(
        self,
        engine: ClusterEngine,
        action: EngineAction,
    ) -> str | None:
        if isinstance(action, PickAction):
            return action.robot_id
        if isinstance(action, PlaceAction):
            return engine.state.wafers[action.wafer_key].robot_id
        return None

    def search_priority(
        self,
        engine: ClusterEngine,
        action: EngineAction,
    ) -> int:
        if isinstance(action, PlaceAction):
            return 0
        if isinstance(action, PickAction):
            return 2 if self.is_source_pick(engine, action) else 1
        return 3

    def engine_after_action(
        self,
        engine: ClusterEngine,
        action: EngineAction,
    ) -> ClusterEngine:
        next_engine = self.fork_engine(engine)
        if isinstance(action, PickAction):
            next_engine._dispatch_pick(action)
        elif isinstance(action, PlaceAction):
            next_engine._dispatch_place(action)
        else:
            next_engine._advance()
        return next_engine

    @staticmethod
    def fork_engine(engine: ClusterEngine) -> ClusterEngine:
        fork = copy(engine)
        state = engine.state
        fork._state = ClusterState(
            time=state.time,
            wafers={
                wafer_key: WaferState(
                    route_id=wafer.route_id,
                    wafer_index=wafer.wafer_index,
                    step_index=wafer.step_index,
                    module_id=wafer.module_id,
                    robot_id=wafer.robot_id,
                    ready_at=wafer.ready_at,
                    return_module_id=wafer.return_module_id,
                )
                for wafer_key, wafer in state.wafers.items()
            },
            robots={
                robot_id: RobotState(
                    module_id=robot.module_id,
                    ready_at=robot.ready_at,
                    holding=list(robot.holding),
                )
                for robot_id, robot in state.robots.items()
            },
            module_occupants={
                module_id: set(occupants)
                for module_id, occupants in state.module_occupants.items()
            },
            load_locks={
                module_id: LoadLockRuntimeState(
                    last_pick_side=runtime.last_pick_side,
                    last_pick_end=runtime.last_pick_end,
                    occupied_exit_side=runtime.occupied_exit_side,
                    occupied_ready_at=runtime.occupied_ready_at,
                    occupied_transition_start=runtime.occupied_transition_start,
                    occupied_transition_duration=runtime.occupied_transition_duration,
                )
                for module_id, runtime in state.load_locks.items()
            },
            pending_operations=[
                PendingOperation(
                    action_type=operation.action_type,
                    robot_id=operation.robot_id,
                    wafer_key=operation.wafer_key,
                    module_id=operation.module_id,
                    start=operation.start,
                    end=operation.end,
                    started=operation.started,
                )
                for operation in state.pending_operations
            ],
        )
        return fork
