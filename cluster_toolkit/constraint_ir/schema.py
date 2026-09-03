from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)


Tick = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Scalar: TypeAlias = StrictBool | StrictInt | StrictStr


def _canonicalize(value: object) -> object:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {
            key: _canonicalize(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple, set)):
        items = [_canonicalize(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
        )
    return value


def canonical_json(value: object) -> str:
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _non_empty(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _unique_ids(items: tuple[object, ...], section: str) -> None:
    ids = [getattr(item, "id") for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{section} must not contain duplicate ids")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TimeDomain(_StrictModel):
    unit: str
    ticks_per_unit: PositiveInt

    @field_validator("unit")
    @classmethod
    def _validate_unit(cls, value: str) -> str:
        return _non_empty(value, "TimeDomain.unit")


class ResourceSpec(_StrictModel):
    id: str
    capacity: PositiveInt

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _non_empty(value, "ResourceSpec.id")


class StateCellSpec(_StrictModel):
    id: str
    value_type: Literal["bool", "int", "enum"]
    enum_values: tuple[str, ...] = ()
    minimum: StrictInt | None = None
    maximum: StrictInt | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _non_empty(value, "StateCellSpec.id")

    @model_validator(mode="after")
    def _validate_domain(self) -> "StateCellSpec":
        if self.value_type == "enum":
            if not self.enum_values:
                raise ValueError("enum StateCellSpec must define enum_values")
            if any(not value for value in self.enum_values):
                raise ValueError("StateCellSpec.enum_values must be non-empty strings")
            if len(set(self.enum_values)) != len(self.enum_values):
                raise ValueError("StateCellSpec.enum_values must not contain duplicates")
        elif self.enum_values:
            raise ValueError("only enum StateCellSpec may define enum_values")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("StateCellSpec.minimum must not exceed maximum")
        if self.value_type != "int" and (
            self.minimum is not None or self.maximum is not None
        ):
            raise ValueError("only int StateCellSpec may define minimum/maximum")
        return self

    def accepts(self, value: object) -> bool:
        if self.value_type == "bool":
            return isinstance(value, bool)
        if self.value_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            if self.minimum is not None and value < self.minimum:
                return False
            return self.maximum is None or value <= self.maximum
        return isinstance(value, str) and value in self.enum_values


class StateAssignment(_StrictModel):
    cell_id: str
    value: Scalar

    @field_validator("cell_id")
    @classmethod
    def _validate_cell_id(cls, value: str) -> str:
        return _non_empty(value, "StateAssignment.cell_id")


class StateCondition(_StrictModel):
    cell_id: str
    operator: Literal["equal", "not_equal", "greater_equal", "elapsed_at_least"]
    value: Scalar
    view: Literal["before", "after"] = "after"

    @field_validator("cell_id")
    @classmethod
    def _validate_cell_id(cls, value: str) -> str:
        return _non_empty(value, "StateCondition.cell_id")

    @model_validator(mode="after")
    def _validate_value(self) -> "StateCondition":
        if self.operator in {"greater_equal", "elapsed_at_least"}:
            if not isinstance(self.value, int) or isinstance(self.value, bool):
                raise ValueError(f"{self.operator} requires an integer value")
        return self


class LeaseCondition(_StrictModel):
    """Admission predicate on an existing resource-owner pair, not capacity."""

    resource_id: str
    owner_id: str
    operator: Literal["present", "absent"] = "present"

    @field_validator("resource_id", "owner_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"LeaseCondition.{info.field_name}")


class LeaseSpec(_StrictModel):
    resource_id: str
    owner_id: str
    amount: PositiveInt = 1

    @field_validator("resource_id", "owner_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"LeaseSpec.{info.field_name}")


class ObligationInstanceSpec(_StrictModel):
    id: str
    deadline_tick: Tick | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _non_empty(value, "ObligationInstanceSpec.id")


class InitialStateSpec(_StrictModel):
    state_values: tuple[StateAssignment, ...] = ()
    leases: tuple[LeaseSpec, ...] = ()
    obligations: tuple[ObligationInstanceSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_keys(self) -> "InitialStateSpec":
        cell_ids = [assignment.cell_id for assignment in self.state_values]
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("InitialStateSpec.state_values contains duplicate cells")
        lease_keys = [
            (lease.resource_id, lease.owner_id) for lease in self.leases
        ]
        if len(set(lease_keys)) != len(lease_keys):
            raise ValueError("InitialStateSpec.leases contains duplicate owners")
        _unique_ids(self.obligations, "InitialStateSpec.obligations")
        return self


class SetStateEffect(_StrictModel):
    kind: Literal["set_state"] = "set_state"
    cell_id: str
    value: Scalar

    @field_validator("cell_id")
    @classmethod
    def _validate_cell_id(cls, value: str) -> str:
        return _non_empty(value, "SetStateEffect.cell_id")


class IncrementStateEffect(_StrictModel):
    kind: Literal["increment_state"] = "increment_state"
    cell_id: str
    delta: StrictInt

    @field_validator("cell_id")
    @classmethod
    def _validate_cell_id(cls, value: str) -> str:
        return _non_empty(value, "IncrementStateEffect.cell_id")


class AcquireLeaseEffect(_StrictModel):
    kind: Literal["acquire_lease"] = "acquire_lease"
    resource_id: str
    owner_id: str
    amount: PositiveInt = 1

    @field_validator("resource_id", "owner_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"AcquireLeaseEffect.{info.field_name}")


class ReleaseLeaseEffect(_StrictModel):
    kind: Literal["release_lease"] = "release_lease"
    resource_id: str
    owner_id: str

    @field_validator("resource_id", "owner_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"ReleaseLeaseEffect.{info.field_name}")


class CreateObligationEffect(_StrictModel):
    kind: Literal["create_obligation"] = "create_obligation"
    obligation_id: str
    deadline_tick: Tick | None = None
    condition: StateCondition | None = None
    coalesce_key: str | None = None
    priority: StrictInt = 0

    @field_validator("obligation_id")
    @classmethod
    def _validate_obligation_id(cls, value: str) -> str:
        return _non_empty(value, "CreateObligationEffect.obligation_id")

    @field_validator("coalesce_key")
    @classmethod
    def _validate_coalesce_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, "CreateObligationEffect.coalesce_key")


class SatisfyObligationEffect(_StrictModel):
    kind: Literal["satisfy_obligation"] = "satisfy_obligation"
    obligation_id: str

    @field_validator("obligation_id")
    @classmethod
    def _validate_obligation_id(cls, value: str) -> str:
        return _non_empty(value, "SatisfyObligationEffect.obligation_id")


Effect: TypeAlias = Annotated[
    SetStateEffect
    | IncrementStateEffect
    | AcquireLeaseEffect
    | ReleaseLeaseEffect
    | CreateObligationEffect
    | SatisfyObligationEffect,
    Field(discriminator="kind"),
]


def canonical_effect_digest(effects: tuple[Effect, ...]) -> str:
    return canonical_digest(
        [effect.model_dump(mode="json") for effect in effects]
    )


class BindingAssignment(_StrictModel):
    parameter: str
    value: str

    @field_validator("parameter", "value")
    @classmethod
    def _validate_values(cls, value: str, info) -> str:
        return _non_empty(value, f"BindingAssignment.{info.field_name}")


class ChoiceScopeClaimSpec(_StrictModel):
    scope_key: str
    release_boundary_id: str | None = None

    @field_validator("scope_key")
    @classmethod
    def _validate_scope_key(cls, value: str) -> str:
        return _non_empty(value, "ChoiceScopeClaimSpec.scope_key")

    @field_validator("release_boundary_id")
    @classmethod
    def _validate_release_boundary_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(
            value,
            "ChoiceScopeClaimSpec.release_boundary_id",
        )


class ChoiceScopeTemplateSpec(_StrictModel):
    scope_prefix: str
    identity_parameters: tuple[str, ...]
    release_boundary_id: str | None = None

    @field_validator("scope_prefix")
    @classmethod
    def _validate_scope_prefix(cls, value: str) -> str:
        return _non_empty(value, "ChoiceScopeTemplateSpec.scope_prefix")

    @model_validator(mode="after")
    def _validate_identity(self) -> "ChoiceScopeTemplateSpec":
        if not self.identity_parameters:
            raise ValueError(
                "ChoiceScopeTemplateSpec.identity_parameters must not be empty"
            )
        if len(set(self.identity_parameters)) != len(self.identity_parameters):
            raise ValueError(
                "ChoiceScopeTemplateSpec.identity_parameters contains duplicates"
            )
        if self.release_boundary_id is not None:
            _non_empty(
                self.release_boundary_id,
                "ChoiceScopeTemplateSpec.release_boundary_id",
            )
        return self


class EventSpec(_StrictModel):
    id: str
    tick: Tick
    decision_round: Tick = 0
    effects: tuple[Effect, ...] = ()
    effect_digest: str | None = None
    audit_kind: str | None = None
    operator_instance_id: str | None = None
    operator_template_id: str | None = None
    boundary_id: str | None = None
    origin_intent_id: str | None = None
    origin_rule_id: str | None = None
    trigger_event_id: str | None = None
    bindings: tuple[BindingAssignment, ...] = ()

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _non_empty(value, "EventSpec.id")

    @field_validator("effect_digest")
    @classmethod
    def _validate_effect_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("EventSpec.effect_digest must be a SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def _validate_bindings(self) -> "EventSpec":
        names = [binding.parameter for binding in self.bindings]
        if len(set(names)) != len(names):
            raise ValueError("EventSpec.bindings contains duplicate parameters")
        return self


class ResourceUseSpec(_StrictModel):
    resource_id: str
    amount: PositiveInt = 1

    @field_validator("resource_id")
    @classmethod
    def _validate_resource_id(cls, value: str) -> str:
        return _non_empty(value, "ResourceUseSpec.resource_id")


class IntervalSpec(_StrictModel):
    id: str
    start_tick: Tick
    end_tick: Tick
    resource_uses: tuple[ResourceUseSpec, ...]
    audit_kind: str | None = None
    operator_instance_id: str | None = None
    operator_template_id: str | None = None
    template_interval_id: str | None = None
    origin_intent_id: str | None = None
    origin_rule_id: str | None = None
    trigger_event_id: str | None = None
    bindings: tuple[BindingAssignment, ...] = ()

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _non_empty(value, "IntervalSpec.id")

    @model_validator(mode="after")
    def _validate_interval(self) -> "IntervalSpec":
        if self.end_tick <= self.start_tick:
            raise ValueError("IntervalSpec must satisfy start_tick < end_tick")
        if not self.resource_uses:
            raise ValueError("IntervalSpec.resource_uses must not be empty")
        resource_ids = [item.resource_id for item in self.resource_uses]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("IntervalSpec.resource_uses contains duplicate resources")
        binding_names = [binding.parameter for binding in self.bindings]
        if len(set(binding_names)) != len(binding_names):
            raise ValueError("IntervalSpec.bindings contains duplicate parameters")
        return self


class ScheduleV1(_StrictModel):
    events: tuple[EventSpec, ...] = ()
    intervals: tuple[IntervalSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "ScheduleV1":
        _unique_ids(self.events, "ScheduleV1.events")
        _unique_ids(self.intervals, "ScheduleV1.intervals")
        return self

    def canonical_dict(self) -> dict[str, object]:
        value = _canonicalize(self.model_dump(mode="json"))
        assert isinstance(value, dict)
        return value

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    @property
    def schedule_hash(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))


class ParameterSpec(_StrictModel):
    name: str
    kind: Literal["resource", "state_cell", "owner", "id"]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _non_empty(value, "ParameterSpec.name")


class BindingRowSpec(_StrictModel):
    values: tuple[str, ...]

    @field_validator("values")
    @classmethod
    def _validate_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value for value in values):
            raise ValueError("BindingRowSpec.values must be non-empty strings")
        return values


class BindingDomainSpec(_StrictModel):
    id: str
    parameters: tuple[ParameterSpec, ...]
    rows: tuple[BindingRowSpec, ...]

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _non_empty(value, "BindingDomainSpec.id")

    @model_validator(mode="after")
    def _validate_table(self) -> "BindingDomainSpec":
        if not self.parameters:
            raise ValueError("BindingDomainSpec.parameters must not be empty")
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("BindingDomainSpec.parameters contains duplicates")
        for row in self.rows:
            if len(row.values) != len(self.parameters):
                raise ValueError("BindingDomainSpec row width must match parameters")
        row_values = [row.values for row in self.rows]
        if len(set(row_values)) != len(row_values):
            raise ValueError("BindingDomainSpec.rows contains duplicates")
        return self


class LiteralIdRef(_StrictModel):
    kind: Literal["literal"] = "literal"
    value: str

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: str) -> str:
        return _non_empty(value, "LiteralIdRef.value")


class ParameterIdRef(_StrictModel):
    kind: Literal["parameter"] = "parameter"
    parameter: str

    @field_validator("parameter")
    @classmethod
    def _validate_parameter(cls, value: str) -> str:
        return _non_empty(value, "ParameterIdRef.parameter")


IdRef: TypeAlias = Annotated[
    LiteralIdRef | ParameterIdRef,
    Field(discriminator="kind"),
]


class ResourceUseTemplate(_StrictModel):
    resource: IdRef
    amount: PositiveInt = 1


class AcquireLeaseTemplateEffect(_StrictModel):
    kind: Literal["acquire_lease"] = "acquire_lease"
    resource: IdRef
    owner: IdRef
    amount: PositiveInt = 1


class ReleaseLeaseTemplateEffect(_StrictModel):
    kind: Literal["release_lease"] = "release_lease"
    resource: IdRef
    owner: IdRef


class SetStateTemplateEffect(_StrictModel):
    kind: Literal["set_state"] = "set_state"
    cell: IdRef
    value: Scalar


class IncrementStateTemplateEffect(_StrictModel):
    kind: Literal["increment_state"] = "increment_state"
    cell: IdRef
    delta: StrictInt


class SetCurrentTickTemplateEffect(_StrictModel):
    kind: Literal["set_current_tick"] = "set_current_tick"
    cell: IdRef


class StateConditionTemplate(_StrictModel):
    cell: IdRef
    operator: Literal["equal", "not_equal", "greater_equal", "elapsed_at_least"]
    value: Scalar
    view: Literal["before", "after"] = "after"

    @model_validator(mode="after")
    def _validate_value(self) -> "StateConditionTemplate":
        if self.operator in {"greater_equal", "elapsed_at_least"}:
            if not isinstance(self.value, int) or isinstance(self.value, bool):
                raise ValueError(f"{self.operator} requires an integer value")
        return self


class LeaseConditionTemplate(_StrictModel):
    resource: IdRef
    owner: IdRef
    operator: Literal["present", "absent"] = "present"


class CreateObligationTemplateEffect(_StrictModel):
    kind: Literal["create_obligation"] = "create_obligation"
    obligation_id: str
    deadline_offset: Tick | None = None
    condition: StateConditionTemplate | None = None
    coalesce_key: str | None = None
    priority: StrictInt = 0

    @field_validator("obligation_id")
    @classmethod
    def _validate_obligation_id(cls, value: str) -> str:
        return _non_empty(value, "CreateObligationTemplateEffect.obligation_id")

    @field_validator("coalesce_key")
    @classmethod
    def _validate_coalesce_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, "CreateObligationTemplateEffect.coalesce_key")


class SatisfyObligationTemplateEffect(_StrictModel):
    kind: Literal["satisfy_obligation"] = "satisfy_obligation"
    obligation_id: str

    @field_validator("obligation_id")
    @classmethod
    def _validate_obligation_id(cls, value: str) -> str:
        return _non_empty(value, "SatisfyObligationTemplateEffect.obligation_id")


TemplateEffect: TypeAlias = Annotated[
    AcquireLeaseTemplateEffect
    | ReleaseLeaseTemplateEffect
    | SetStateTemplateEffect
    | IncrementStateTemplateEffect
    | SetCurrentTickTemplateEffect
    | CreateObligationTemplateEffect
    | SatisfyObligationTemplateEffect,
    Field(discriminator="kind"),
]


class IntervalTemplateSpec(_StrictModel):
    id: str
    start_offset: Tick
    duration: PositiveInt
    resource_uses: tuple[ResourceUseTemplate, ...]
    start_effects: tuple[TemplateEffect, ...] = ()
    end_effects: tuple[TemplateEffect, ...] = ()
    audit_kind: str

    @field_validator("id", "audit_kind")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"IntervalTemplateSpec.{info.field_name}")

    @model_validator(mode="after")
    def _validate_resource_uses(self) -> "IntervalTemplateSpec":
        if not self.resource_uses:
            raise ValueError("IntervalTemplateSpec.resource_uses must not be empty")
        return self


class StepDependencySpec(_StrictModel):
    """One precedence edge inside a complete selectable Intent bundle."""

    predecessor_step_id: str
    predecessor_boundary: Literal["start", "end"] = "end"
    successor_step_id: str
    successor_boundary: Literal["start", "end"] = "start"
    minimum_lag: Tick = 0

    @field_validator("predecessor_step_id", "successor_step_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"StepDependencySpec.{info.field_name}")

    @model_validator(mode="after")
    def _validate_distinct_steps(self) -> "StepDependencySpec":
        if self.predecessor_step_id == self.successor_step_id:
            raise ValueError("StepDependencySpec must connect distinct steps")
        return self


class OperatorTemplateSpec(_StrictModel):
    id: str
    origin: Literal["selectable", "automatic"]
    parameters: tuple[ParameterSpec, ...] = ()
    intervals: tuple[IntervalTemplateSpec, ...]
    step_dependencies: tuple[StepDependencySpec, ...] = ()
    decision_policy: Literal["complete_bundle"] = "complete_bundle"

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _non_empty(value, "OperatorTemplateSpec.id")

    @model_validator(mode="after")
    def _validate_template(self) -> "OperatorTemplateSpec":
        if not self.intervals:
            raise ValueError("OperatorTemplateSpec.intervals must not be empty")
        names = [parameter.name for parameter in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError("OperatorTemplateSpec.parameters contains duplicates")
        _unique_ids(self.intervals, "OperatorTemplateSpec.intervals")
        self._validate_step_dependencies()
        for interval in self.intervals:
            self._validate_coalesce_priorities(
                interval.start_effects,
                f"{interval.id}.start",
            )
            self._validate_coalesce_priorities(
                interval.end_effects,
                f"{interval.id}.end",
            )
        return self

    def _validate_step_dependencies(self) -> None:
        steps = {interval.id: interval for interval in self.intervals}
        edges: set[tuple[str, str, str, str]] = set()
        graph: dict[str, set[str]] = {}
        for dependency in self.step_dependencies:
            if dependency.predecessor_step_id not in steps:
                raise ValueError(
                    "step dependency references unknown predecessor "
                    f"{dependency.predecessor_step_id}"
                )
            if dependency.successor_step_id not in steps:
                raise ValueError(
                    "step dependency references unknown successor "
                    f"{dependency.successor_step_id}"
                )
            edge = (
                dependency.predecessor_step_id,
                dependency.predecessor_boundary,
                dependency.successor_step_id,
                dependency.successor_boundary,
            )
            if edge in edges:
                raise ValueError("OperatorTemplateSpec.step_dependencies contains duplicates")
            edges.add(edge)

            predecessor = steps[dependency.predecessor_step_id]
            successor = steps[dependency.successor_step_id]
            predecessor_tick = predecessor.start_offset + (
                predecessor.duration
                if dependency.predecessor_boundary == "end"
                else 0
            )
            successor_tick = successor.start_offset + (
                successor.duration
                if dependency.successor_boundary == "end"
                else 0
            )
            if successor_tick < predecessor_tick + dependency.minimum_lag:
                raise ValueError(
                    "step dependency is not satisfied by the declared offsets: "
                    f"{dependency.predecessor_step_id}."
                    f"{dependency.predecessor_boundary} -> "
                    f"{dependency.successor_step_id}."
                    f"{dependency.successor_boundary}"
                )
            graph.setdefault(dependency.predecessor_step_id, set()).add(
                dependency.successor_step_id
            )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("OperatorTemplateSpec.step_dependencies must be acyclic")
            if step_id in visited:
                return
            visiting.add(step_id)
            for successor_id in graph.get(step_id, ()):
                visit(successor_id)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in graph:
            visit(step_id)

    @staticmethod
    def _validate_coalesce_priorities(
        effects: tuple[TemplateEffect, ...],
        boundary_id: str,
    ) -> None:
        seen: dict[tuple[str, int], str] = {}
        for effect in effects:
            if not isinstance(effect, CreateObligationTemplateEffect):
                continue
            if effect.coalesce_key is None:
                continue
            key = (effect.coalesce_key, effect.priority)
            previous = seen.get(key)
            if previous is not None and previous != effect.obligation_id:
                raise ValueError(
                    "UNDER_SPECIFIED_PRIORITY: obligation requests on "
                    f"{boundary_id} share coalesce key and priority"
                )
            seen[key] = effect.obligation_id


class IntentSeedSpec(_StrictModel):
    id: str
    operator_template_id: str
    bindings: tuple[BindingAssignment, ...]
    alternative_group_id: str | None = None
    earliest_start_offset: Tick = 0
    latest_start_offset: Tick | None = None
    required_obligation_ids: tuple[str, ...] = ()
    guards: tuple[StateCondition | LeaseCondition, ...] = ()
    choice_scope_claims: tuple[ChoiceScopeClaimSpec, ...] = ()

    @field_validator("id", "operator_template_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"IntentSeedSpec.{info.field_name}")

    @field_validator("alternative_group_id")
    @classmethod
    def _validate_alternative_group_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _non_empty(value, "IntentSeedSpec.alternative_group_id")

    @model_validator(mode="after")
    def _validate_bindings(self) -> "IntentSeedSpec":
        names = [binding.parameter for binding in self.bindings]
        if len(set(names)) != len(names):
            raise ValueError("IntentSeedSpec.bindings contains duplicate parameters")
        if len(set(self.required_obligation_ids)) != len(self.required_obligation_ids):
            raise ValueError("IntentSeedSpec.required_obligation_ids contains duplicates")
        scope_keys = [claim.scope_key for claim in self.choice_scope_claims]
        if len(set(scope_keys)) != len(scope_keys):
            raise ValueError("IntentSeedSpec.choice_scope_claims contains duplicates")
        if (
            self.latest_start_offset is not None
            and self.latest_start_offset < self.earliest_start_offset
        ):
            raise ValueError(
                "IntentSeedSpec.latest_start_offset must not precede earliest_start_offset"
            )
        return self


class DynamicIntentSpec(_StrictModel):
    id: str
    operator_template_id: str
    binding_domain_id: str
    choice_scope_templates: tuple[ChoiceScopeTemplateSpec, ...]
    earliest_start_offset: Tick = 0
    latest_start_offset: Tick | None = None
    required_obligation_ids: tuple[str, ...] = ()
    guards: tuple[StateConditionTemplate | LeaseConditionTemplate, ...] = ()

    @field_validator("id", "operator_template_id", "binding_domain_id")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"DynamicIntentSpec.{info.field_name}")

    @model_validator(mode="after")
    def _validate_rule(self) -> "DynamicIntentSpec":
        if not self.choice_scope_templates:
            raise ValueError(
                "DynamicIntentSpec.choice_scope_templates must not be empty"
            )
        if len(set(self.required_obligation_ids)) != len(
            self.required_obligation_ids
        ):
            raise ValueError("DynamicIntentSpec.required_obligation_ids contains duplicates")
        if (
            self.latest_start_offset is not None
            and self.latest_start_offset < self.earliest_start_offset
        ):
            raise ValueError(
                "DynamicIntentSpec.latest_start_offset must not precede earliest_start_offset"
            )
        return self


class BindingForwardSpec(_StrictModel):
    source_parameter: str
    target_parameter: str

    @field_validator("source_parameter", "target_parameter")
    @classmethod
    def _validate_parameters(cls, value: str, info) -> str:
        return _non_empty(value, f"BindingForwardSpec.{info.field_name}")


class AutomaticRuleSpec(_StrictModel):
    id: str
    trigger_operator_template_id: str
    trigger_boundary_id: str
    emit_operator_template_id: str
    binding_forwards: tuple[BindingForwardSpec, ...] = ()

    @field_validator(
        "id",
        "trigger_operator_template_id",
        "trigger_boundary_id",
        "emit_operator_template_id",
    )
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _non_empty(value, f"AutomaticRuleSpec.{info.field_name}")


class TerminalStateSpec(_StrictModel):
    """Required state values (subset) and exact final resource ownership."""

    state_values: tuple[StateAssignment, ...] = ()
    leases: tuple[LeaseSpec, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_facts(self) -> "TerminalStateSpec":
        if len({item.cell_id for item in self.state_values}) != len(self.state_values):
            raise ValueError("terminal state contains duplicate state cells")
        if len({(item.resource_id, item.owner_id) for item in self.leases}) != len(self.leases):
            raise ValueError("terminal state contains duplicate leases")
        return self


class ConstraintIRV1(_StrictModel):
    schema_version: Literal["1.2-reference"] = "1.2-reference"
    semantic_version: Literal["1.2"] = "1.2"
    time_domain: TimeDomain
    resources: tuple[ResourceSpec, ...] = ()
    state_cells: tuple[StateCellSpec, ...] = ()
    initial_state: InitialStateSpec = InitialStateSpec()
    operator_templates: tuple[OperatorTemplateSpec, ...] = ()
    intent_seeds: tuple[IntentSeedSpec, ...] = ()
    automatic_rules: tuple[AutomaticRuleSpec, ...] = ()
    binding_domains: tuple[BindingDomainSpec, ...] = ()
    dynamic_intents: tuple[DynamicIntentSpec, ...] = ()
    terminal_state: TerminalStateSpec | None = None

    def canonical_dict(self) -> dict[str, object]:
        value = _canonicalize(self.model_dump(mode="json"))
        assert isinstance(value, dict)
        # Binding rows are positional, unlike other unordered IR collections.
        # Sort columns WITH their corresponding values, never independently.
        domains = []
        for domain in sorted(self.binding_domains, key=lambda item: item.id):
            columns = sorted(range(len(domain.parameters)), key=lambda i: domain.parameters[i].name)
            rows = [{"values": [row.values[i] for i in columns]} for row in domain.rows]
            domains.append({
                "id": domain.id,
                "parameters": [domain.parameters[i].model_dump(mode="json") for i in columns],
                "rows": sorted(rows, key=lambda row: json.dumps(row, ensure_ascii=False)),
            })
        value["binding_domains"] = domains
        return value

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @property
    def problem_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def _validate_problem(self) -> "ConstraintIRV1":
        _unique_ids(self.resources, "ConstraintIRV1.resources")
        _unique_ids(self.state_cells, "ConstraintIRV1.state_cells")
        _unique_ids(self.operator_templates, "ConstraintIRV1.operator_templates")
        _unique_ids(self.intent_seeds, "ConstraintIRV1.intent_seeds")
        _unique_ids(self.automatic_rules, "ConstraintIRV1.automatic_rules")
        _unique_ids(self.binding_domains, "ConstraintIRV1.binding_domains")
        _unique_ids(self.dynamic_intents, "ConstraintIRV1.dynamic_intents")
        resources = {resource.id: resource for resource in self.resources}
        state_cells = {cell.id: cell for cell in self.state_cells}
        assigned_cells = {
            assignment.cell_id for assignment in self.initial_state.state_values
        }
        missing_cells = sorted(state_cells.keys() - assigned_cells)
        if missing_cells:
            raise ValueError(
                f"initial state is missing values for cells: {missing_cells}"
            )

        for assignment in self.initial_state.state_values:
            cell = state_cells.get(assignment.cell_id)
            if cell is None:
                raise ValueError(
                    f"initial state references unknown cell: {assignment.cell_id}"
                )
            if not cell.accepts(assignment.value):
                raise ValueError(
                    f"initial value for {assignment.cell_id} is outside its domain"
                )

        totals: dict[str, int] = {}
        for lease in self.initial_state.leases:
            resource = resources.get(lease.resource_id)
            if resource is None:
                raise ValueError(
                    f"initial lease references unknown resource: {lease.resource_id}"
                )
            totals[lease.resource_id] = totals.get(lease.resource_id, 0) + lease.amount
            if totals[lease.resource_id] > resource.capacity:
                raise ValueError(
                    f"initial leases exceed resource capacity: {lease.resource_id}"
                )
        self._validate_operators(resources, state_cells)
        self._validate_dynamic_intents(resources, state_cells)
        if self.terminal_state is not None:
            for assignment in self.terminal_state.state_values:
                cell = state_cells.get(assignment.cell_id)
                if cell is None or not cell.accepts(assignment.value):
                    raise ValueError("terminal state references unknown cell or invalid value")
            totals = {}
            for lease in self.terminal_state.leases:
                resource = resources.get(lease.resource_id)
                if resource is None:
                    raise ValueError("terminal lease references unknown resource")
                totals[lease.resource_id] = totals.get(lease.resource_id, 0) + lease.amount
                if totals[lease.resource_id] > resource.capacity:
                    raise ValueError("terminal leases exceed resource capacity")
        return self

    def _validate_dynamic_intents(
        self,
        resources: dict[str, ResourceSpec],
        state_cells: dict[str, StateCellSpec],
    ) -> None:
        templates = {template.id: template for template in self.operator_templates}
        domains = {domain.id: domain for domain in self.binding_domains}
        for domain in self.binding_domains:
            for row in domain.rows:
                for parameter, value in zip(domain.parameters, row.values):
                    if parameter.kind == "resource" and value not in resources:
                        raise ValueError(
                            f"binding domain {domain.id} references unknown resource {value}"
                        )
                    if parameter.kind == "state_cell" and value not in state_cells:
                        raise ValueError(
                            f"binding domain {domain.id} references unknown state cell {value}"
                        )

        for rule in self.dynamic_intents:
            template = templates.get(rule.operator_template_id)
            domain = domains.get(rule.binding_domain_id)
            if template is None or template.origin != "selectable":
                raise ValueError(
                    f"dynamic intent {rule.id} must reference a selectable template"
                )
            if domain is None:
                raise ValueError(
                    f"dynamic intent {rule.id} references unknown binding domain"
                )
            expected = {(item.name, item.kind) for item in template.parameters}
            actual = {(item.name, item.kind) for item in domain.parameters}
            if expected != actual or len(expected) != len(domain.parameters):
                raise ValueError(
                    f"dynamic intent {rule.id} domain must match template parameters"
                )
            parameter_names = {item.name for item in template.parameters}
            boundary_ids = {
                f"{interval.id}.start" for interval in template.intervals
            } | {
                f"{interval.id}.end" for interval in template.intervals
            }
            for scope in rule.choice_scope_templates:
                if not set(scope.identity_parameters).issubset(parameter_names):
                    raise ValueError(
                        f"dynamic intent {rule.id} scope references unknown parameter"
                    )
                if (
                    scope.release_boundary_id is not None
                    and scope.release_boundary_id not in boundary_ids
                ):
                    raise ValueError(
                        f"dynamic intent {rule.id} scope references unknown boundary"
                    )
            parameters = {item.name: item for item in template.parameters}
            for guard in rule.guards:
                if isinstance(guard, LeaseConditionTemplate):
                    self._validate_id_ref(
                        guard.resource, parameters, resources, expected_kind="resource",
                    )
                    self._validate_id_ref(
                        guard.owner, parameters, resources, expected_kind="owner",
                    )
                    continue
                self._validate_state_cell_ref(
                    guard.cell,
                    parameters,
                    state_cells,
                )

    def _validate_operators(
        self,
        resources: dict[str, ResourceSpec],
        state_cells: dict[str, StateCellSpec],
    ) -> None:
        templates = {template.id: template for template in self.operator_templates}
        for template in self.operator_templates:
            parameters = {parameter.name: parameter for parameter in template.parameters}
            for interval in template.intervals:
                for use in interval.resource_uses:
                    self._validate_id_ref(
                        use.resource,
                        parameters,
                        resources,
                        expected_kind="resource",
                    )
                for effect in interval.start_effects + interval.end_effects:
                    if isinstance(
                        effect,
                        (
                            SetStateTemplateEffect,
                            IncrementStateTemplateEffect,
                            SetCurrentTickTemplateEffect,
                        ),
                    ):
                        self._validate_state_cell_ref(
                            effect.cell,
                            parameters,
                            state_cells,
                        )
                        continue
                    if isinstance(effect, CreateObligationTemplateEffect):
                        if effect.condition is not None:
                            self._validate_state_cell_ref(
                                effect.condition.cell,
                                parameters,
                                state_cells,
                            )
                        continue
                    if isinstance(effect, SatisfyObligationTemplateEffect):
                        continue
                    self._validate_id_ref(
                        effect.resource,
                        parameters,
                        resources,
                        expected_kind="resource",
                    )
                    self._validate_id_ref(
                        effect.owner,
                        parameters,
                        resources,
                        expected_kind="owner",
                    )

        for seed in self.intent_seeds:
            template = templates.get(seed.operator_template_id)
            if template is None:
                raise ValueError(
                    f"intent {seed.id} references unknown template {seed.operator_template_id}"
                )
            if template.origin != "selectable":
                raise ValueError(f"intent {seed.id} must reference a selectable template")
            bindings = {binding.parameter: binding.value for binding in seed.bindings}
            parameters = {parameter.name: parameter for parameter in template.parameters}
            if bindings.keys() != parameters.keys():
                raise ValueError(f"intent {seed.id} bindings must match template parameters")
            for name, parameter in parameters.items():
                if parameter.kind == "resource" and bindings[name] not in resources:
                    raise ValueError(
                        f"intent {seed.id} binds unknown resource {bindings[name]}"
                    )
                if parameter.kind == "state_cell" and bindings[name] not in state_cells:
                    raise ValueError(
                        f"intent {seed.id} binds unknown state cell {bindings[name]}"
                    )
            for guard in seed.guards:
                if isinstance(guard, LeaseCondition):
                    if guard.resource_id not in resources:
                        raise ValueError(
                            f"intent {seed.id} guard references unknown resource"
                        )
                elif guard.cell_id not in state_cells:
                    raise ValueError(
                        f"intent {seed.id} guard references unknown state cell"
                    )
            boundary_ids = {
                f"{interval.id}.start" for interval in template.intervals
            } | {
                f"{interval.id}.end" for interval in template.intervals
            }
            for claim in seed.choice_scope_claims:
                if (
                    claim.release_boundary_id is not None
                    and claim.release_boundary_id not in boundary_ids
                ):
                    raise ValueError(
                        f"intent {seed.id} scope references unknown release boundary"
                    )

        for rule in self.automatic_rules:
            trigger = templates.get(rule.trigger_operator_template_id)
            emitted = templates.get(rule.emit_operator_template_id)
            if trigger is None or emitted is None:
                raise ValueError(f"automatic rule {rule.id} references unknown template")
            if emitted.origin != "automatic":
                raise ValueError(
                    f"automatic rule {rule.id} must emit an automatic template"
                )
            boundary_ids = {
                f"{interval.id}.start" for interval in trigger.intervals
            } | {f"{interval.id}.end" for interval in trigger.intervals}
            if rule.trigger_boundary_id not in boundary_ids:
                raise ValueError(
                    f"automatic rule {rule.id} references unknown trigger boundary"
                )
            source_params = {parameter.name: parameter for parameter in trigger.parameters}
            target_params = {parameter.name: parameter for parameter in emitted.parameters}
            forwards = {
                item.target_parameter: item.source_parameter
                for item in rule.binding_forwards
            }
            if forwards.keys() != target_params.keys():
                raise ValueError(
                    f"automatic rule {rule.id} must bind every emitted parameter"
                )
            for target_name, source_name in forwards.items():
                source = source_params.get(source_name)
                if source is None:
                    raise ValueError(
                        f"automatic rule {rule.id} references unknown source parameter"
                    )
                if source.kind != target_params[target_name].kind:
                    raise ValueError(
                        f"automatic rule {rule.id} forwards incompatible parameter kinds"
                    )
        self._validate_automatic_rule_graph()

    def _validate_automatic_rule_graph(self) -> None:
        graph: dict[str, set[str]] = {}
        for rule in self.automatic_rules:
            graph.setdefault(rule.trigger_operator_template_id, set()).add(
                rule.emit_operator_template_id
            )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(template_id: str) -> None:
            if template_id in visiting:
                raise ValueError(
                    "reference automatic rule graph must be acyclic"
                )
            if template_id in visited:
                return
            visiting.add(template_id)
            for emitted_id in graph.get(template_id, ()):
                visit(emitted_id)
            visiting.remove(template_id)
            visited.add(template_id)

        for template_id in graph:
            visit(template_id)

    @staticmethod
    def _validate_id_ref(
        reference: IdRef,
        parameters: dict[str, ParameterSpec],
        resources: dict[str, ResourceSpec],
        *,
        expected_kind: Literal["resource", "owner"],
    ) -> None:
        if isinstance(reference, LiteralIdRef):
            if expected_kind == "resource" and reference.value not in resources:
                raise ValueError(f"template references unknown resource {reference.value}")
            return
        parameter = parameters.get(reference.parameter)
        if parameter is None:
            raise ValueError(f"template references unknown parameter {reference.parameter}")
        if expected_kind == "resource" and parameter.kind != "resource":
            raise ValueError(f"parameter {reference.parameter} must be a resource")
        if expected_kind == "owner" and parameter.kind not in {"owner", "id"}:
            raise ValueError(f"parameter {reference.parameter} must be an owner/id")

    @staticmethod
    def _validate_state_cell_ref(
        reference: IdRef,
        parameters: dict[str, ParameterSpec],
        state_cells: dict[str, StateCellSpec],
    ) -> None:
        if isinstance(reference, LiteralIdRef):
            if reference.value not in state_cells:
                raise ValueError(
                    f"template references unknown state cell {reference.value}"
                )
            return
        parameter = parameters.get(reference.parameter)
        if parameter is None:
            raise ValueError(f"template references unknown parameter {reference.parameter}")
        if parameter.kind != "state_cell":
            raise ValueError(f"parameter {reference.parameter} must be a state cell")


class ActiveObligation(_StrictModel):
    id: str
    deadline_tick: Tick | None = None


class KernelSnapshot(_StrictModel):
    tick: Tick
    state_values: tuple[StateAssignment, ...]
    active_leases: tuple[LeaseSpec, ...]
    active_obligations: tuple[ActiveObligation, ...]
    active_interval_ids: tuple[str, ...]

    def canonical_dict(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "state_values": [
                item.model_dump(mode="json")
                for item in sorted(self.state_values, key=lambda item: item.cell_id)
            ],
            "active_leases": [
                item.model_dump(mode="json")
                for item in sorted(
                    self.active_leases,
                    key=lambda item: (item.resource_id, item.owner_id),
                )
            ],
            "active_obligations": [
                item.model_dump(mode="json")
                for item in sorted(self.active_obligations, key=lambda item: item.id)
            ],
            "active_interval_ids": sorted(self.active_interval_ids),
        }

    @property
    def state_hash(self) -> str:
        return canonical_digest(self.canonical_dict())


class CommittedIntentSpec(_StrictModel):
    source_intent_id: str
    operator_template_id: str
    bindings: tuple[BindingAssignment, ...]
    earliest_start_offset: Tick
    candidate_key: str
    candidate_digest: str
    intent_instance_id: str
    operator_instance_ids: tuple[str, ...]
    choice_scope_claims: tuple[ChoiceScopeClaimSpec, ...]

    @field_validator("source_intent_id", "operator_template_id")
    @classmethod
    def _validate_source_intent_id(cls, value: str) -> str:
        return _non_empty(value, "CommittedIntentSpec.source_intent_id")

    @field_validator("candidate_key", "candidate_digest", "intent_instance_id")
    @classmethod
    def _validate_hash(cls, value: str, info) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(
                f"CommittedIntentSpec.{info.field_name} must be a SHA-256 hex digest"
            )
        return value


class CommitRecordSpec(_StrictModel):
    commit_id: str
    previous_commit_id: str | None
    frame_token: str
    tick: Tick
    selections: tuple[CommittedIntentSpec, ...]
    expanded_schedule_digest: str

    @field_validator(
        "commit_id",
        "previous_commit_id",
        "frame_token",
        "expanded_schedule_digest",
    )
    @classmethod
    def _validate_hash(cls, value: str | None, info) -> str | None:
        if value is None and info.field_name == "previous_commit_id":
            return None
        if value is None or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(
                f"CommitRecordSpec.{info.field_name} must be a SHA-256 hex digest"
            )
        return value

    @model_validator(mode="after")
    def _validate_selections(self) -> "CommitRecordSpec":
        if not self.selections:
            raise ValueError("CommitRecordSpec.selections must not be empty")
        keys = [selection.candidate_key for selection in self.selections]
        if len(set(keys)) != len(keys):
            raise ValueError("CommitRecordSpec.selections contains duplicate candidates")
        return self


class SessionSnapshot(_StrictModel):
    snapshot_version: Literal["1.2-reference"] = "1.2-reference"
    problem_hash: str
    revision: Tick
    tick: Tick
    committed_intent_ids: tuple[str, ...]
    committed_alternative_group_ids: tuple[str, ...]
    schedule: ScheduleV1
    schedule_hash: str
    kernel_snapshot: KernelSnapshot
    kernel_state_hash: str
    commit_log: tuple[CommitRecordSpec, ...] = ()
    active_choice_scope_keys: tuple[str, ...] = ()

    @field_validator("problem_hash", "schedule_hash", "kernel_state_hash")
    @classmethod
    def _validate_hash(cls, value: str, info) -> str:
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"SessionSnapshot.{info.field_name} must be a SHA-256 hex digest")
        return value

    @model_validator(mode="after")
    def _validate_snapshot(self) -> "SessionSnapshot":
        if self.tick != self.kernel_snapshot.tick:
            raise ValueError("SessionSnapshot tick must match KernelSnapshot tick")
        if len(set(self.committed_intent_ids)) != len(self.committed_intent_ids):
            raise ValueError("SessionSnapshot committed intent ids must be unique")
        if len(set(self.committed_alternative_group_ids)) != len(
            self.committed_alternative_group_ids
        ):
            raise ValueError(
                "SessionSnapshot committed alternative group ids must be unique"
            )
        if len(set(self.active_choice_scope_keys)) != len(
            self.active_choice_scope_keys
        ):
            raise ValueError("SessionSnapshot active choice scope keys must be unique")
        return self

    def canonical_dict(self) -> dict[str, object]:
        value = _canonicalize(self.model_dump(mode="json"))
        assert isinstance(value, dict)
        return value

    def canonical_json(self) -> str:
        return canonical_json(self.model_dump(mode="json"))

    @property
    def snapshot_hash(self) -> str:
        return canonical_digest(self.model_dump(mode="json"))

    @property
    def frame_token(self) -> str:
        return canonical_digest(
            {
                "problem_hash": self.problem_hash,
                "revision": self.revision,
                "state_hash": self.kernel_state_hash,
                "commitment_hash": self.commitment_hash,
            }
        )

    @property
    def commitment_hash(self) -> str:
        return canonical_digest(
            {
                "commit_ids": [record.commit_id for record in self.commit_log],
                "active_choice_scope_keys": sorted(self.active_choice_scope_keys),
            }
        )
