from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from problem import (
    ClusterProblem,
    ModuleLocation,
    ModuleType,
    TMArmType,
    WaferKey,
)
from validator import PICK, PLACE


@dataclass(slots=True)
class _WaferState:
    step_index: int
    module_id: str | None
    robot_id: str | None
    ready_at: float


class ClusterEnv(gym.Env[dict[str, Any], int]):
    """Minimal single-arm cluster-tool scheduling environment.

    Actions ``[0, N)`` pick a wafer. Actions ``[N, N + M)`` place the held
    wafer in a module. The extra module index in observations means either
    "wafer on robot" or "robot initially anywhere", depending on the field.
    """

    metadata = {"render_modes": []}

    def __init__(self, problem: ClusterProblem) -> None:
        self.problem = problem
        self._validate_problem()

        self._robot_ids = tuple(sorted(problem.ClusterTool))
        self._lp_id = next(module_id for module_id, module in problem.Modules.items() if module.type is ModuleType.LP)
        snapshot = problem.initial_state.to_snapshot()
        self._wafer_keys = tuple(sorted(snapshot.wafers_by_key))
        self._module_ids = tuple(sorted(problem.Modules))
        self._module_index = {module_id: index for index, module_id in enumerate(self._module_ids)}
        self._robot_index = {robot_id: index for index, robot_id in enumerate(self._robot_ids)}

        wafer_count = len(self._wafer_keys)
        module_count = len(self._module_ids)
        robot_count = len(self._robot_ids)

        self.action_space = spaces.Discrete([wafer_count + module_count])
        self.observation_space = spaces.Dict(
            {
                "wafer_loc": spaces.MultiDiscrete(np.full(wafer_count, module_count + robot_count)),
                "wafer_step": spaces.MultiDiscrete(
                    np.asarray(
                        [len(problem.routes[route_id].visits) for route_id, _ in self._wafer_keys],
                    )
                ),
                "process_remaining": spaces.Box(
                    low=0.0,
                    high=float("+inf"),
                    shape=(wafer_count,),
                    dtype=np.float32,
                ),
                "robot_loc": spaces.Discrete(module_count + 1),
            }
        )

        self._time = 0.0
        self._robot_loc: list[str] = []
        self._wafers: list[_WaferState] = []
        self._actions: list[dict[str, object]] = []

    @property
    def wafer_keys(self) -> tuple[WaferKey, ...]:
        return self._wafer_keys

    @property
    def module_ids(self) -> tuple[str, ...]:
        return self._module_ids

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
        self._robot_module = [snapshot.tm_positions.get(robot_id) for robot_id in self._robot_ids]
        self._wafers = [
            _WaferState(
                step_index=snapshot.wafers_by_key[key].step_index,
                module_id=snapshot.wafers_by_key[key].location.module_id,
                robot_id=snapshot.wafers_by_key[key].location.robot_id,
                ready_at=0.0,
            )
            for key in self._wafer_keys
        ]

        self._actions = []
        return self._observation(), {"time": self._time}

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if not self._wafers:
            raise RuntimeError("reset() must be called before step()")

        mask = self._action_mask()
        if self._complete() or not mask.any():
            raise RuntimeError("step() called after the episode ended")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside the action space")

        action = int(action)
        if not mask[action]:
            raise ValueError(f"action {action} is not allowed by the action mask")

        previous_time = self._time
        if action < len(self._wafer_keys):
            self._pick(action)
        else:
            self._place(action - len(self._wafer_keys))

        deadlocked = self._advance_to_decision()
        completed = self._complete()
        reward = previous_time - (self._failure_cost if deadlocked else self._time)
        info: dict[str, Any] = {"time": self._time}
        if completed or deadlocked:
            info.update(
                is_success=completed,
                reason="completed" if completed else "deadlock",
            )
        if deadlocked:
            info["failure_cost"] = self._failure_cost
        return self._observation(), reward, completed or deadlocked, False, info

    def _validate_problem(self) -> None:
        if len(self.problem.ClusterTool) != 1:
            raise ValueError("ClusterEnv requires exactly one TM")

        robot_id, robot = next(iter(self.problem.ClusterTool.items()))
        if robot.arm_type is not TMArmType.SINGLE_ARM:
            raise ValueError("ClusterEnv requires a single-arm TM")

        lp_ids = [module_id for module_id, module in self.problem.Modules.items() if module.type is ModuleType.LP]
        if len(lp_ids) != 1:
            raise ValueError("ClusterEnv requires exactly one LP")
        if any(module.type not in {ModuleType.LP, ModuleType.PM} for module in self.problem.Modules.values()):
            raise ValueError("ClusterEnv supports only LP and PM modules")

        unreachable = set(self.problem.Modules) - set(robot.module_ids)
        if unreachable:
            raise ValueError(f"TM {robot_id} cannot reach modules: {sorted(unreachable)}")
        for route_id, route in self.problem.routes.items():
            for visit in route.visits:
                if any(self.problem.Modules[module_id].type is not ModuleType.PM for module_id in visit.module_ids):
                    raise ValueError(f"Route {route_id} must contain only PM visits")

        snapshot = self.problem.initial_state.to_snapshot()
        if not snapshot.wafers_by_key:
            raise ValueError("ClusterEnv requires at least one wafer")
        lp_id = lp_ids[0]
        for wafer in snapshot.wafers_by_key.values():
            if not isinstance(wafer.location, ModuleLocation) or wafer.location.module_id != lp_id or wafer.step_index != 0 or (wafer.process_end_time or 0) > 0:
                raise ValueError("Every wafer must start ready in the LP at step_index 0")

    def _observation(self) -> dict[str, Any]:
        return {
            "wafer_loc": np.asarray(
                [self._robot_index[wafer.robot_id] if wafer.module_id is None else self._module_index[wafer.module_id] for wafer in self._wafers],
                dtype=np.int64,
            ),
            "wafer_step": np.asarray([wafer.step_index for wafer in self._wafers], dtype=np.int64),
            "process_remaining": np.asarray([max(0.0, wafer.ready_at - self._time) for wafer in self._wafers], dtype=np.float32),
            "robot_loc": np.asarray([self._module_index[_m] if _m is not None else len(self._module_ids) for _m in self._robot_module], dtype=np.int64),
            "action_mask": self._action_mask(),
        }

    def _action_mask(self) -> np.ndarray:
        """Build locally legal Pick or Place actions from the current state."""

        mask = np.zeros(self.action_space.n, dtype=np.int8)
        held_index = self._held_index()

        if held_index is None:
            for index, wafer in enumerate(self._wafers):
                if wafer.module_id is not None and wafer.ready_at <= self._time and any(self._has_capacity(module_id, excluding=index) for module_id in self._targets(index)):
                    mask[index] = 1
            return mask

        offset = len(self._wafer_keys)
        for module_id in self._targets(held_index):
            if self._has_capacity(module_id):
                mask[offset + self._module_index[module_id]] = 1
        return mask

    def _targets(self, wafer_index: int) -> tuple[str, ...]:
        wafer = self._wafers[wafer_index]
        route = self.problem.routes[self._wafer_keys[wafer_index][0]]
        next_step = wafer.step_index + 1
        if next_step <= len(route.visits):
            return route.visits[next_step - 1].module_ids
        if next_step == len(route.visits) + 1:
            return (self._lp_id,)
        return ()

    def _has_capacity(
        self,
        module_id: str,
        excluding: int | None = None,
    ) -> bool:
        occupants = sum(index != excluding and wafer.module_id == module_id for index, wafer in enumerate(self._wafers))
        return occupants < self.problem.Modules[module_id].capacity

    def _held_index(self) -> int | None:
        return next(
            (index for index, wafer in enumerate(self._wafers) if wafer.module_id is None),
            None,
        )

    def _pick(self, wafer_index: int) -> None:
        wafer = self._wafers[wafer_index]
        module_id = wafer.module_id
        assert module_id is not None

        start = self._time + self._travel_time(module_id)
        end = start + self.problem.ClusterTool[self._robot_id].pick_time
        self._record(PICK, wafer_index, module_id, wafer.step_index, start, end)
        wafer.module_id = None
        self._robot_module = module_id
        self._time = end

    def _place(self, module_index: int) -> None:
        wafer_index = self._held_index()
        assert wafer_index is not None
        wafer = self._wafers[wafer_index]
        module_id = self._module_ids[module_index]
        next_step = wafer.step_index + 1

        start = self._time + self._travel_time(module_id)
        end = start + self.problem.ClusterTool[self._robot_id].place_time
        self._record(PLACE, wafer_index, module_id, next_step, start, end)

        route = self.problem.routes[self._wafer_keys[wafer_index][0]]
        process_time = route.visits[next_step - 1].process_time or 0.0 if next_step <= len(route.visits) else 0.0
        wafer.step_index = next_step
        wafer.module_id = module_id
        wafer.ready_at = end + process_time
        self._robot_module = module_id
        self._time = end

    def _travel_time(self, module_id: str) -> float:
        if self._robot_module is None or self._robot_module == module_id:
            return 0.0
        return self.problem.ClusterTool[self._robot_id].travel_time(
            self._robot_module,
            module_id,
        )

    def _record(
        self,
        action_type: str,
        wafer_index: int,
        module_id: str,
        step_index: int,
        start: float,
        end: float,
    ) -> None:
        route_id, wafer_number = self._wafer_keys[wafer_index]
        self._actions.append(
            {
                "action_type": action_type,
                "tm_id": self._robot_id,
                "module_id": module_id,
                "route_id": route_id,
                "wafer_index": wafer_number,
                "step_index": step_index,
                "arm_id": "arm0",
                "start": start,
                "end": end,
            }
        )

    def _complete(self) -> bool:
        return all(wafer.module_id == self._lp_id and wafer.step_index == len(self.problem.routes[key[0]].visits) + 1 for key, wafer in zip(self._wafer_keys, self._wafers))
