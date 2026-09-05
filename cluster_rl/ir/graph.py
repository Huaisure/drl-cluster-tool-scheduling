"""An anonymous relational graph of the declared program and its current state.

IDs are lookup keys ONLY: no strings, hashes, lexical tokens or ID embeddings
enter the network. Grounding finite binding rows preserves their correlations.
All declared plans remain visible, including plans not currently enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from gymnasium import spaces

from cluster_toolkit.constraint_ir import ConstraintIRV1, DecisionFrame, SessionSnapshot
from cluster_toolkit.constraint_ir.schema import canonical_digest


FEATURE_VERSION = "ir-graph-3"
NODE_TYPES = (
    "program", "current", "initial", "goal", "resource", "cell_bool", "cell_int",
    "cell_enum", "owner", "symbol", "obligation", "scope", "group", "coalesce",
    "plan", "selectable", "automatic", "activity", "boundary", "number", "bool",
    "time", "none", "lease", "state_fact", "candidate", "wait", "reservation",
    "delta", "scheduled_interval", "scheduled_event", "active", "consumed",
    "dependency", "scope_claim", "before", "after", "empty_work",
    "equal", "not_equal", "greater_equal", "elapsed_at_least", "present", "absent",
    "set_state", "increment_state", "set_current_tick", "acquire_lease",
    "release_lease", "create_obligation", "satisfy_obligation",
)
EDGE_TYPES = (
    "contains", "cell", "owner", "resource", "value", "minimum", "maximum",
    "enum_member", "start", "end", "time", "duration", "effect", "condition",
    "requires", "scope", "release", "source", "destination", "before", "after",
    "plan", "earliest", "latest", "priority", "coalesce", "amount", "trigger",
    "lag", "initial", "current", "goal", "remaining", "status", "view",
)
NODE_INDEX = {name: i for i, name in enumerate(NODE_TYPES)}
EDGE_INDEX = {name: i for i, name in enumerate(EDGE_TYPES)}
NUMERIC_FEATURES = (
    "scalar_value",
    "action_earliest_seconds",
    "action_latest_slack_seconds",
    "action_duration_seconds",
    "action_resource_count",
    "action_resource_amount",
    "action_state_change_count",
    "action_successor_count",
)
NUMERIC_INDEX = {name: index for index, name in enumerate(NUMERIC_FEATURES)}
NUMERIC_WIDTH = len(NUMERIC_FEATURES)


@dataclass(frozen=True)
class IRGraph:
    node_types: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_types: np.ndarray
    action_nodes: np.ndarray

    @property
    def action_count(self) -> int:
        return len(self.action_nodes)


class IRGraphSpace(spaces.Space[IRGraph]):
    """Variable-size graph space; valid actions are its compact action_nodes."""

    def __init__(self, max_actions: int) -> None:
        super().__init__()
        self.max_actions = max_actions

    def sample(self, mask=None) -> IRGraph:
        # A structurally valid sample, not a sampled scheduling problem.
        return IRGraph(np.array([NODE_INDEX["candidate"]], dtype=np.int64),
                       np.zeros((1, NUMERIC_WIDTH), dtype=np.float32),
                       np.empty((2, 0), dtype=np.int64), np.empty(0, dtype=np.int64),
                       np.array([0], dtype=np.int64))

    def contains(self, graph: object) -> bool:
        if not isinstance(graph, IRGraph):
            return False
        n, e = len(graph.node_types), len(graph.edge_types)
        return bool(
            n > 0 and graph.node_types.shape == (n,)
            and graph.node_features.shape == (n, NUMERIC_WIDTH)
            and graph.edge_index.shape == (2, e) and graph.action_nodes.ndim == 1
            and graph.action_count <= self.max_actions
            and all(a.dtype == np.int64 for a in (graph.node_types, graph.edge_index,
                                                   graph.edge_types, graph.action_nodes))
            and graph.node_features.dtype == np.float32
            and np.isfinite(graph.node_features).all()
            and ((graph.node_types >= 0) & (graph.node_types < len(NODE_TYPES))).all()
            and ((graph.edge_index >= 0) & (graph.edge_index < n)).all()
            and ((graph.edge_types >= 0) & (graph.edge_types < 2 * len(EDGE_TYPES))).all()
            and ((graph.action_nodes >= 0) & (graph.action_nodes < n)).all()
        )


class _Builder:
    def __init__(self, ticks_per_second: int) -> None:
        self.scale = ticks_per_second
        self.types: list[int] = []
        self.features: list[list[float]] = []
        self.edges: list[tuple[int, int]] = []
        self.relations: list[int] = []
        self.identities: dict[tuple[str, str], int] = {}

    def copy(self) -> _Builder:
        other = _Builder(self.scale)
        other.types, other.features = self.types.copy(), self.features.copy()
        other.edges, other.relations = self.edges.copy(), self.relations.copy()
        other.identities = self.identities.copy()
        return other

    def node(self, kind: str, value: float = 0.0) -> int:
        index = len(self.types)
        self.types.append(NODE_INDEX[kind])
        features = [0.0] * NUMERIC_WIDTH
        features[NUMERIC_INDEX["scalar_value"]] = math.asinh(value)
        self.features.append(features)
        return index

    def feature(self, node: int, name: str, value: float, *, time: bool = False) -> None:
        scaled = value / self.scale if time else value
        self.features[node][NUMERIC_INDEX[name]] = math.asinh(scaled)

    def entity(self, kind: str, key: str) -> int:
        identity = kind, key
        if identity not in self.identities:
            self.identities[identity] = self.node(kind)
        return self.identities[identity]

    def edge(self, source: int, target: int, kind: str) -> None:
        relation = EDGE_INDEX[kind]
        self.edges.extend(((source, target), (target, source)))
        self.relations.extend((relation, relation + len(EDGE_TYPES)))

    def child(self, parent: int, kind: str, relation: str = "contains", value: float = 0.0) -> int:
        child = self.node(kind, value)
        self.edge(parent, child, relation)
        return child

    def scalar(self, parent: int, value, relation: str = "value", *, time: bool = False) -> None:
        if value is None:
            self.child(parent, "none", relation)
        elif isinstance(value, str):
            self.edge(parent, self.entity("symbol", value), relation)
        else:
            kind = "time" if time else "bool" if isinstance(value, bool) else "number"
            self.child(parent, kind, relation, value / self.scale if time else value)

    def ref(self, parent: int, kind: str, key: str, relation: str) -> None:
        if kind == "cell":
            # Cell types were declared before any program/initial-state references.
            node = next(self.identities[(t, key)] for t in ("cell_bool", "cell_int", "cell_enum")
                        if (t, key) in self.identities)
        else:
            node = self.entity(kind, key)
        self.edge(parent, node, relation)

    @staticmethod
    def resolve(reference: dict, bindings: dict[str, str]) -> str:
        return reference["value"] if reference["kind"] == "literal" else bindings[reference["parameter"]]

    def target(self, node: int, data: dict, field: str, bindings: dict[str, str]) -> None:
        key = data.get(f"{field}_id")
        if key is None:
            key = self.resolve(data[field], bindings)
        self.ref(node, field, key, field if field in EDGE_INDEX else "value")

    def condition(self, parent: int, condition: dict, bindings: dict[str, str]) -> int:
        node = self.child(parent, condition["operator"], "condition")
        if condition["operator"] in {"present", "absent"}:
            self.target(node, condition, "resource", bindings)
            self.target(node, condition, "owner", bindings)
        else:
            self.target(node, condition, "cell", bindings)
            self.scalar(node, condition["value"], time=condition["operator"] == "elapsed_at_least")
            self.child(node, condition["view"], "view")
        return node

    def effect(self, parent: int, effect: dict, bindings: dict[str, str], *, now: int = 0) -> int:
        kind = effect["kind"]
        node = self.child(parent, kind, "effect")
        if kind in {"set_state", "increment_state", "set_current_tick"}:
            self.target(node, effect, "cell", bindings)
            if kind != "set_current_tick":
                self.scalar(node, effect["value"] if kind == "set_state" else effect["delta"])
        elif kind in {"acquire_lease", "release_lease"}:
            self.target(node, effect, "resource", bindings)
            self.target(node, effect, "owner", bindings)
            if kind == "acquire_lease":
                self.scalar(node, effect["amount"], "amount")
        elif kind in {"create_obligation", "satisfy_obligation"}:
            self.ref(node, "obligation", effect["obligation_id"], "value")
            if kind == "create_obligation":
                deadline = effect.get("deadline_offset", effect.get("deadline_tick"))
                if deadline is not None and "deadline_tick" in effect:
                    deadline -= now
                self.scalar(node, deadline, "latest", time=True)
                self.scalar(node, effect["priority"], "priority")
                if effect["coalesce_key"] is not None:
                    self.ref(node, "coalesce", effect["coalesce_key"], "coalesce")
                if effect["condition"] is not None:
                    self.condition(node, effect["condition"], bindings)
        else:
            raise ValueError(f"unencoded IR effect: {kind}")
        return node

    def facts(self, parent: int, states, leases, obligations=(), *, now: int = 0) -> None:
        for state in states:
            fact = self.child(parent, "state_fact")
            self.ref(fact, "cell", state.cell_id, "cell")
            self.scalar(fact, state.value)
        for lease in leases:
            fact = self.child(parent, "lease")
            self.ref(fact, "resource", lease.resource_id, "resource")
            self.ref(fact, "owner", lease.owner_id, "owner")
            self.scalar(fact, lease.amount, "amount")
        for obligation in obligations:
            fact = self.child(parent, "active")
            self.ref(fact, "obligation", obligation.id, "value")
            remaining = None if obligation.deadline_tick is None else obligation.deadline_tick - now
            self.scalar(fact, remaining, "latest", time=True)

    def finish(self, actions: list[int]) -> IRGraph:
        arrays = (
            np.array(self.types, dtype=np.int64), np.array(self.features, dtype=np.float32),
            np.array(self.edges, dtype=np.int64).reshape(-1, 2).T,
            np.array(self.relations, dtype=np.int64), np.array(actions, dtype=np.int64),
        )
        for array in arrays:
            array.flags.writeable = False
        return IRGraph(*arrays)


class IRGraphEncoder:
    """Cache the anonymous static program; append a fresh observation per decision."""

    def __init__(self, problem: ConstraintIRV1) -> None:
        if problem.time_domain.unit != "second":
            raise ValueError("IR training currently requires time_domain.unit='second'")
        if problem.terminal_state is None:
            raise ValueError("IR training requires an explicit terminal_state")
        self.problem = problem
        self.base = _Builder(problem.time_domain.ticks_per_unit)
        b = self.base
        self.root = b.node("program")
        for resource in problem.resources:
            node = b.entity("resource", resource.id)
            b.edge(self.root, node, "contains")
            b.scalar(node, resource.capacity, "maximum")
        for cell in problem.state_cells:
            node = b.entity(f"cell_{cell.value_type}", cell.id)
            b.edge(self.root, node, "contains")
            b.scalar(node, cell.minimum, "minimum")
            b.scalar(node, cell.maximum, "maximum")
            for value in cell.enum_values:
                b.scalar(node, value, "enum_member")
        initial = b.child(self.root, "initial", "initial")
        b.facts(initial, problem.initial_state.state_values, problem.initial_state.leases,
                problem.initial_state.obligations)
        goal = b.child(self.root, "goal", "goal")
        b.facts(goal, problem.terminal_state.state_values, problem.terminal_state.leases)
        b.child(goal, "empty_work")
        self.templates = {item.id: item for item in problem.operator_templates}
        self.auto: dict[str, list] = {}
        for rule in problem.automatic_rules:
            self.auto.setdefault(rule.trigger_operator_template_id, []).append(rule)
        self.plans: dict[str, int] = {}
        self.seed_plans: dict[str, int] = {}
        self.guarded_plans: dict[tuple[str, object], list[int]] = {}
        domains = {domain.id: domain for domain in problem.binding_domains}
        for rule in problem.dynamic_intents:
            domain = domains[rule.binding_domain_id]
            for row in domain.rows:
                values = dict(zip((item.name for item in domain.parameters), row.values))
                encoded = [{"parameter": key, "value": value} for key, value in sorted(values.items())]
                prefix = f"dynamic/{rule.id}/{canonical_digest(encoded)}/"
                plan = self._plan(rule, values)
                self.plans[prefix] = plan
                for guard in rule.guards:
                    data = guard.model_dump(mode="python")
                    if data.get("operator") != "equal" or "cell" not in data:
                        continue
                    cell_id = _Builder.resolve(data["cell"], values)
                    self.guarded_plans.setdefault((cell_id, data["value"]), []).append(plan)
        for seed in problem.intent_seeds:
            self.seed_plans[seed.id] = self._plan(seed, {item.parameter: item.value for item in seed.bindings})

    def _plan(self, source, bindings: dict[str, str]) -> int:
        b = self.base
        plan = b.child(self.root, "plan")
        b.scalar(plan, source.earliest_start_offset, "earliest", time=True)
        b.scalar(plan, source.latest_start_offset, "latest", time=True)
        for condition in source.guards:
            b.condition(plan, condition.model_dump(), bindings)
        for obligation in source.required_obligation_ids:
            b.ref(plan, "obligation", obligation, "requires")
        boundaries = self._bundle(plan, plan, source.operator_template_id, bindings, 0, ())
        if hasattr(source, "choice_scope_templates"):
            claims = [(f"{scope.scope_prefix}/{canonical_digest([bindings[p] for p in scope.identity_parameters])}",
                       scope.release_boundary_id) for scope in source.choice_scope_templates]
        else:
            claims = [(scope.scope_key, scope.release_boundary_id) for scope in source.choice_scope_claims]
            if source.alternative_group_id is not None:
                b.ref(plan, "group", source.alternative_group_id, "scope")
        for key, release in claims:
            claim = b.child(plan, "scope_claim", "scope")
            b.ref(claim, "scope", key, "value")
            if release is None:
                b.scalar(claim, None, "release")
            elif release in boundaries:
                b.edge(claim, boundaries[release][0], "release")
            else:
                raise ValueError(f"unencoded scope release boundary: {release}")
        return plan

    def _bundle(self, parent: int, plan: int, template_id: str, bindings: dict[str, str],
                offset: int, stack: tuple[str, ...]) -> dict[str, tuple[int, int]]:
        if template_id in stack:
            raise ValueError("IR observation does not support cyclic automatic emission graphs")
        b, template = self.base, self.templates[template_id]
        bundle = b.child(parent, template.origin)
        boundaries = {}
        for interval in template.intervals:
            activity = b.child(bundle, "activity")
            b.edge(plan, activity, "contains")
            b.scalar(activity, interval.duration, "duration", time=True)
            for use in interval.resource_uses:
                use_node = b.child(activity, "reservation", "resource")
                resource = b.resolve(use.resource.model_dump(), bindings)
                b.ref(use_node, "resource", resource, "resource")
                b.scalar(use_node, use.amount, "amount")
                b.edge(plan, use_node, "resource")
            for phase, effects, tick in (
                ("start", interval.start_effects, offset + interval.start_offset),
                ("end", interval.end_effects, offset + interval.start_offset + interval.duration),
            ):
                boundary = b.child(activity, "boundary", phase)
                b.scalar(boundary, tick, "time", time=True)
                boundaries[f"{interval.id}.{phase}"] = boundary, tick
                for effect in effects:
                    effect_node = b.effect(boundary, effect.model_dump(), bindings)
                    b.edge(plan, effect_node, "effect")
        for dependency in template.step_dependencies:
            node = b.child(bundle, "dependency")
            start = boundaries[f"{dependency.predecessor_step_id}.{dependency.predecessor_boundary}"][0]
            end = boundaries[f"{dependency.successor_step_id}.{dependency.successor_boundary}"][0]
            b.edge(node, start, "source")
            b.edge(node, end, "destination")
            b.scalar(node, dependency.minimum_lag, "lag", time=True)
        for rule in self.auto.get(template_id, ()):
            trigger, tick = boundaries[rule.trigger_boundary_id]
            values = {item.target_parameter: bindings[item.source_parameter] for item in rule.binding_forwards}
            self._bundle(trigger, plan, rule.emit_operator_template_id, values, tick, (*stack, template_id))
        return boundaries

    def encode(self, snapshot: SessionSnapshot, frame: DecisionFrame, wait_tick: int | None,
               *, decisions_remaining: int | None = None, time_remaining: int | None = None) -> IRGraph:
        b, now = self.base.copy(), snapshot.tick
        current = b.child(self.root, "current", "current")
        b.scalar(current, now, "time", time=True)
        b.scalar(current, decisions_remaining, "remaining")
        b.scalar(current, time_remaining, "latest", time=True)
        state = snapshot.kernel_snapshot
        b.facts(current, state.state_values, state.active_leases, state.active_obligations, now=now)
        for kind, keys in (("scope", snapshot.active_choice_scope_keys),
                           ("group", snapshot.committed_alternative_group_ids)):
            for key in keys:
                status = b.child(current, "active", "status")
                b.ref(status, kind, key, "value")
        for seed in snapshot.committed_intent_ids:
            if seed in self.seed_plans:
                status = b.child(current, "consumed", "status")
                b.edge(status, self.seed_plans[seed], "plan")
        for record in snapshot.commit_log:
            for selection in record.selections:
                for claim in selection.choice_scope_claims:
                    if claim.scope_key not in snapshot.active_choice_scope_keys:
                        continue
                    scope = b.entity("scope", claim.scope_key)
                    if claim.release_boundary_id is None:
                        b.scalar(scope, None, "release")
                    else:
                        release = next(event.tick for event in snapshot.schedule.events
                                       if event.origin_intent_id == selection.source_intent_id
                                       and event.boundary_id == claim.release_boundary_id)
                        if release > now:
                            b.scalar(scope, release - now, "release", time=True)
        for interval in snapshot.schedule.intervals:
            if interval.end_tick <= now:
                continue
            node = b.child(current, "scheduled_interval", "remaining")
            b.scalar(node, interval.start_tick - now, "start", time=True)
            b.scalar(node, interval.end_tick - now, "end", time=True)
            for use in interval.resource_uses:
                claim = b.child(node, "reservation", "resource")
                b.ref(claim, "resource", use.resource_id, "resource")
                b.scalar(claim, use.amount, "amount")
        for event in snapshot.schedule.events:
            if event.tick <= now:
                continue
            node = b.child(current, "scheduled_event", "remaining")
            b.scalar(node, event.tick - now, "time", time=True)
            for effect in event.effects:
                b.effect(node, effect.model_dump(), {}, now=now)
        actions = []
        for candidate in frame.intents:
            node = b.child(current, "candidate")
            actions.append(node)
            prefix = candidate.id.rsplit("/", 1)[0] + "/"
            plan = self.seed_plans.get(candidate.id, self.plans.get(prefix))
            if plan is None:
                raise ValueError("candidate has no declared observation plan")
            b.edge(node, plan, "plan")
            earliest = candidate.earliest_start_tick - now
            b.scalar(node, earliest, "earliest", time=True)
            latest = None if candidate.latest_start_tick is None else candidate.latest_start_tick - now
            b.scalar(node, latest, "latest", time=True)
            b.scalar(node, candidate.duration_ticks, "duration", time=True)
            b.feature(node, "action_earliest_seconds", earliest, time=True)
            if latest is not None:
                b.feature(
                    node, "action_latest_slack_seconds", latest - earliest,
                    time=True,
                )
            b.feature(
                node, "action_duration_seconds", candidate.duration_ticks,
                time=True,
            )
            b.feature(
                node, "action_resource_count", len(candidate.resource_footprint),
            )
            b.feature(
                node, "action_resource_amount",
                sum(use.amount for use in candidate.resource_footprint),
            )
            b.feature(
                node, "action_state_change_count",
                len(candidate.state_delta.state_values) + len(candidate.state_delta.leases),
            )
            successors = set()
            for change in candidate.state_delta.state_values:
                successors.update(
                    self.guarded_plans.get((change.cell_id, change.after), ())
                )
            b.feature(node, "action_successor_count", len(successors))
            for use in candidate.resource_footprint:
                claim = b.child(node, "reservation", "resource")
                b.ref(claim, "resource", use.resource_id, "resource")
                b.scalar(claim, use.amount, "amount")
                b.scalar(claim, use.start_tick - now, "start", time=True)
                b.scalar(claim, None if use.end_tick is None else use.end_tick - now, "end", time=True)
            for change in candidate.state_delta.state_values:
                delta = b.child(node, "delta", "effect")
                b.ref(delta, "cell", change.cell_id, "cell")
                b.scalar(delta, change.before, "before")
                b.scalar(delta, change.after, "after")
                # A candidate must be able to reason about the operation it
                # enables next. Linking by an explicit state transition and an
                # equal guard is generic IR dataflow, not an identity heuristic.
                for successor in self.guarded_plans.get((change.cell_id, change.after), ()):
                    b.edge(node, successor, "remaining")
            for change in candidate.state_delta.leases:
                delta = b.child(node, "delta", "effect")
                b.ref(delta, "resource", change.resource_id, "resource")
                b.ref(delta, "owner", change.owner_id, "owner")
                b.scalar(delta, change.before_amount, "before")
                b.scalar(delta, change.after_amount, "after")
        if wait_tick is not None:
            node = b.child(current, "wait")
            b.scalar(node, wait_tick - now, "duration", time=True)
            b.feature(
                node, "action_duration_seconds", wait_tick - now, time=True,
            )
            actions.append(node)
        return b.finish(actions)
