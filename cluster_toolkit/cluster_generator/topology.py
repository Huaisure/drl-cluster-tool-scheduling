from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

from cluster_toolkit.problem import ClusterProblem, ModuleType


@dataclass(frozen=True, slots=True)
class ModuleGraph:
    problem: ClusterProblem
    adjacency: dict[str, frozenset[str]]
    lp_ids: tuple[str, ...]
    non_lp_ids: tuple[str, ...]

    @classmethod
    def from_problem(cls, problem: ClusterProblem) -> "ModuleGraph":
        adjacency: dict[str, set[str]] = {
            module_id: set()
            for module_id in problem.Modules
        }
        for robot in problem.ClusterTool.values():
            module_ids = tuple(robot.module_ids)
            for left_index, left in enumerate(module_ids):
                for right in module_ids[left_index + 1 :]:
                    if left == right:
                        continue
                    adjacency[left].add(right)
                    adjacency[right].add(left)

        lp_ids = tuple(
            module_id
            for module_id, module in sorted(problem.Modules.items())
            if module.type in {ModuleType.IO, ModuleType.LP}
        )
        non_lp_ids = tuple(
            module_id
            for module_id, module in sorted(problem.Modules.items())
            if module.type not in {ModuleType.IO, ModuleType.LP}
        )
        return cls(
            problem=problem,
            adjacency={
                module_id: frozenset(neighbors)
                for module_id, neighbors in adjacency.items()
            },
            lp_ids=lp_ids,
            non_lp_ids=non_lp_ids,
        )

    def connected(self, left: str, right: str) -> bool:
        return right in self.adjacency.get(left, ())

    def feasible_lengths(
        self,
        start_lp: str,
        minimum: int,
        maximum: int,
        *,
        end_lp: str | None = None,
    ) -> tuple[int, ...]:
        return tuple(
            length
            for length in range(minimum, maximum + 1)
            if self._has_closed_walk(start_lp, length, end_lp=end_lp)
        )

    def construct_closed_walk(
        self,
        start_lp: str,
        length: int,
        rng: random.Random,
        *,
        end_lp: str | None = None,
    ) -> tuple[tuple[str, ...], str]:
        if not self._has_closed_walk(start_lp, length, end_lp=end_lp):
            raise ValueError(
                f"no LP-to-LP Route with {length} internal steps starts at {start_lp} and visits a PM"
            )

        @lru_cache(maxsize=None)
        def can_finish(node: str, remaining_edges: int, seen_pm: bool) -> bool:
            if remaining_edges == 1:
                return seen_pm and any(
                    neighbor in self.lp_ids and (end_lp is None or neighbor == end_lp)
                    for neighbor in self.adjacency[node]
                )
            return any(
                can_finish(
                    neighbor,
                    remaining_edges - 1,
                    seen_pm or self.problem.Modules[neighbor].type is ModuleType.PM,
                )
                for neighbor in self.adjacency[node]
                if neighbor in self.non_lp_ids
            )

        path: list[str] = []
        current = start_lp
        seen_pm = False
        for step_index in range(length):
            remaining_edges = length - step_index
            options = [
                neighbor
                for neighbor in sorted(self.adjacency[current])
                if neighbor in self.non_lp_ids
                and can_finish(
                    neighbor,
                    remaining_edges,
                    seen_pm or self.problem.Modules[neighbor].type is ModuleType.PM,
                )
            ]
            if not options:  # pragma: no cover - protected by the initial DP check
                raise RuntimeError("closed-walk construction lost its feasibility witness")
            current = rng.choice(options)
            path.append(current)
            seen_pm = seen_pm or self.problem.Modules[current].type is ModuleType.PM

        final_lps = sorted(
            lp_id
            for lp_id in set(self.adjacency[current]).intersection(self.lp_ids)
            if end_lp is None or lp_id == end_lp
        )
        if not final_lps:  # pragma: no cover - protected by DP
            raise RuntimeError("closed-walk construction has no final LP")
        return tuple(path), rng.choice(final_lps)

    def expand_candidates(
        self,
        path: tuple[str, ...],
        start_lp: str,
        end_lp: str,
        *,
        probability: float,
        max_candidates: int,
        rng: random.Random,
    ) -> tuple[tuple[str, ...], ...]:
        visits: list[tuple[str, ...]] = []
        for index, selected in enumerate(path):
            previous = start_lp if index == 0 else path[index - 1]
            following = end_lp if index == len(path) - 1 else path[index + 1]
            module_type = self.problem.Modules[selected].type
            alternatives = [
                module_id
                for module_id in self.non_lp_ids
                if module_id != selected
                and self.problem.Modules[module_id].type is module_type
                and self.connected(previous, module_id)
                and self.connected(module_id, following)
            ]
            rng.shuffle(alternatives)
            chosen = [selected]
            for alternative in alternatives:
                if len(chosen) >= max_candidates:
                    break
                if rng.random() < probability:
                    chosen.append(alternative)
            visits.append(tuple(chosen))
        return tuple(visits)

    def candidate_witness(
        self,
        start_lp: str,
        visits: tuple[tuple[str, ...], ...],
        *,
        end_lp: str | None = None,
    ) -> tuple[tuple[str, ...], str] | None:
        frontier: dict[str, tuple[str, ...]] = {start_lp: ()}
        for candidates in visits:
            next_frontier: dict[str, tuple[str, ...]] = {}
            for candidate in sorted(candidates):
                if candidate in self.lp_ids:
                    continue
                for previous, path in sorted(frontier.items()):
                    if self.connected(previous, candidate):
                        next_frontier.setdefault(candidate, path + (candidate,))
                        break
            frontier = next_frontier
            if not frontier:
                return None

        for previous, path in sorted(frontier.items()):
            final_lps = sorted(
                lp_id
                for lp_id in set(self.adjacency[previous]).intersection(self.lp_ids)
                if end_lp is None or lp_id == end_lp
            )
            if final_lps:
                return path, final_lps[0]
        return None

    def _has_closed_walk(
        self,
        start_lp: str,
        length: int,
        *,
        end_lp: str | None = None,
    ) -> bool:
        if start_lp not in self.lp_ids or length <= 0:
            return False

        @lru_cache(maxsize=None)
        def can_finish(node: str, remaining_edges: int, seen_pm: bool) -> bool:
            if remaining_edges == 1:
                return seen_pm and any(
                    neighbor in self.lp_ids and (end_lp is None or neighbor == end_lp)
                    for neighbor in self.adjacency[node]
                )
            return any(
                can_finish(
                    neighbor,
                    remaining_edges - 1,
                    seen_pm or self.problem.Modules[neighbor].type is ModuleType.PM,
                )
                for neighbor in self.adjacency[node]
                if neighbor in self.non_lp_ids
            )

        return any(
            can_finish(
                neighbor,
                length,
                self.problem.Modules[neighbor].type is ModuleType.PM,
            )
            for neighbor in self.adjacency[start_lp]
            if neighbor in self.non_lp_ids
        )
