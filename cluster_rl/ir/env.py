"""A Gymnasium Adapter over IR, with no cluster-tool business predicates."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cluster_toolkit.constraint_ir import (
    ConstraintIRV1, DecisionFrame, DiagnosticCode, ReferenceKernel, ReferenceValidator,
    SemanticError, SessionSnapshot,
)

from .graph import IRGraph, IRGraphEncoder, IRGraphSpace


def terminal_matches(problem: ConstraintIRV1, snapshot: SessionSnapshot) -> bool:
    """Fast rollout termination check; independent auditing stays in evaluation."""
    goal, state = problem.terminal_state, snapshot.kernel_snapshot
    if goal is None:
        raise ValueError("IR training requires an explicit terminal_state")
    if state.active_interval_ids or state.active_obligations:
        return False
    if any(event.tick > snapshot.tick for event in snapshot.schedule.events):
        return False
    if any(interval.end_tick > snapshot.tick for interval in snapshot.schedule.intervals):
        return False
    values = {item.cell_id: item.value for item in state.state_values}
    return (
        all(type(values[item.cell_id]) is type(item.value) and values[item.cell_id] == item.value
            for item in goal.state_values)
        and {(item.resource_id, item.owner_id, item.amount) for item in state.active_leases}
        == {(item.resource_id, item.owner_id, item.amount) for item in goal.leases}
    )


class IRSchedulingEnv(gym.Env[IRGraph, int]):
    """Select one compact Intent index, or the last index for an available Wait.

    No-choice states advance automatically to explicit events/deadlines or a
    declared elapsed-time guard threshold. Wait is ALSO available when other
    candidates exist, so prefetching is not forced. Limits are capped-episode
    failures (truncated=True, -1 bonus); PPO treats them as absorbing, not as
    arbitrary sampling cuts. Rollout-length cuts, in contrast, bootstrap.
    """

    metadata = {"render_modes": []}

    def __init__(self, problem: ConstraintIRV1, *, max_decisions: int = 1000,
                 max_time_seconds: float | None = None, reward_scale_seconds: float = 100.0) -> None:
        if max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        if not math.isfinite(reward_scale_seconds) or reward_scale_seconds <= 0:
            raise ValueError("reward_scale_seconds must be finite and positive")
        if max_time_seconds is not None and (not math.isfinite(max_time_seconds) or max_time_seconds <= 0):
            raise ValueError("max_time_seconds must be finite and positive")
        self.problem = problem
        self.encoder = IRGraphEncoder(problem)
        self.max_decisions = max_decisions
        self.max_tick = (None if max_time_seconds is None else
                         math.floor(max_time_seconds * problem.time_domain.ticks_per_unit))
        if self.max_tick == 0:
            raise ValueError("max_time_seconds must span at least one tick")
        self.reward_scale = reward_scale_seconds * problem.time_domain.ticks_per_unit
        domain_sizes = {item.id: len(item.rows) for item in problem.binding_domains}
        maximum = 1 + len(problem.intent_seeds) + sum(domain_sizes[r.binding_domain_id]
                                                     for r in problem.dynamic_intents)
        self.action_space = spaces.Discrete(maximum)
        self.observation_space = IRGraphSpace(maximum)
        self.session = None
        self.reason: str | None = None
        self.decisions = 0
        self.wait_tick: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.action_space.seed(seed)
        self.session = ReferenceKernel.start(self.problem)
        self.reason, self.decisions, self.wait_tick = None, 0, None
        self.reward_tick = 0
        self._settle()
        return self.observation, self._info()

    def step(self, action: int):
        if self.session is None or self.reason is not None:
            raise RuntimeError("reset an unfinished episode before calling step")
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise ValueError("action must be a compact integer index")
        if not 0 <= action < self.observation.action_count:
            raise ValueError("action is not available in this DecisionFrame")
        before = self.reward_tick
        self.decisions += 1
        if action == len(self.frame.intents):
            assert self.wait_tick is not None
            self._advance(self.wait_tick)
        else:
            candidate = self.frame.intents[action]
            self.session.commit(self.frame.frame_token, (candidate.candidate_key,))
        self._settle()
        after = self.snapshot.tick
        self.reward_tick = after
        reward = -0.5 * (after / (self.reward_scale + after) - before / (self.reward_scale + before))
        if self.reason == "success":
            reward += 1.0
        elif self.reason is not None:
            reward -= 1.0
        truncated = self.reason in {"decision_limit", "time_limit"}
        terminated = self.reason is not None and not truncated
        return self.observation, reward, terminated, truncated, self._info()

    def _advance(self, tick: int) -> None:
        if self.max_tick is not None and tick > self.max_tick:
            self.session.advance_to(self.max_tick)
            self.reason = "time_limit"
            return
        try:
            self.session.advance_to(tick)
        except SemanticError as error:
            if error.code is not DiagnosticCode.DEADLINE_MISSED:
                raise  # Do not turn infrastructure/semantic bugs into training examples.
            self.reason = "deadline_missed"

    def _next_tick(self, snapshot: SessionSnapshot) -> int | None:
        now = snapshot.tick
        ticks = {event.tick for event in snapshot.schedule.events}
        ticks.update(t for interval in snapshot.schedule.intervals for t in (interval.start_tick, interval.end_tick))
        ticks.update(item.deadline_tick for item in snapshot.kernel_snapshot.active_obligations
                     if item.deadline_tick is not None)
        values = {item.cell_id: item.value for item in snapshot.kernel_snapshot.state_values}
        # The reference Session only knows explicit boundaries. Resolve the one
        # time-dependent guard primitive here, without parsing business IDs.
        for seed in self.problem.intent_seeds:
            if seed.id not in snapshot.committed_intent_ids:
                for guard in seed.guards:
                    if getattr(guard, "operator", None) == "elapsed_at_least":
                        ticks.add(values[guard.cell_id] + guard.value)
        domains = {item.id: item for item in self.problem.binding_domains}
        for rule in self.problem.dynamic_intents:
            domain = domains[rule.binding_domain_id]
            for guard in rule.guards:
                if getattr(guard, "operator", None) != "elapsed_at_least":
                    continue
                for row in domain.rows:
                    bindings = dict(zip((p.name for p in domain.parameters), row.values))
                    cell_id = (guard.cell.value if guard.cell.kind == "literal" else
                               bindings[guard.cell.parameter])
                    ticks.add(values[cell_id] + guard.value)
        return min((tick for tick in ticks if tick > now), default=None)

    def _settle(self) -> None:
        while True:
            self.snapshot = self.session.snapshot()
            self.wait_tick = None
            if self.reason is None and terminal_matches(self.problem, self.snapshot):
                self.reason = "success"
            if self.reason is not None:
                self.frame = DecisionFrame(frame_token="", tick=self.snapshot.tick, intents=())
                break
            self.frame = self.session.frame()
            next_tick = self._next_tick(self.snapshot)
            if self.frame.intents:
                if self.decisions >= self.max_decisions:
                    self.reason = "decision_limit"
                    continue
                self.wait_tick = next_tick
                break
            if next_tick is None:
                self.reason = "decision_limit" if self.decisions >= self.max_decisions else "deadlock"
            else:
                self._advance(next_tick)
        self.observation = self.encoder.encode(
            self.snapshot, self.frame, self.wait_tick,
            decisions_remaining=self.max_decisions - self.decisions,
            time_remaining=None if self.max_tick is None else self.max_tick - self.snapshot.tick,
        )

    def _info(self) -> dict:
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        mask[:self.observation.action_count] = True
        return {
            "success": self.reason == "success", "termination_reason": self.reason,
            "tick": self.snapshot.tick, "elapsed_seconds": self.snapshot.tick / self.problem.time_domain.ticks_per_unit,
            "decisions": self.decisions, "action_mask": mask,
        }

    def audit(self):
        if self.session is None:
            raise RuntimeError("reset before auditing")
        return ReferenceValidator.validate_session(
            self.problem, self.snapshot, require_terminal=self.reason == "success",
        )
