"""Finite single-robot PM workload Adapter; no dependency on either Engine."""

from __future__ import annotations

from urllib.parse import quote

from pydantic import ValidationError

from cluster_toolkit.problem import ClusterProblem, ModuleLocation, ModuleType, TMArmType

from .compiler import compile_ticks
from .diagnostics import DiagnosticCode, SemanticError
from .schema import (
    AcquireLeaseTemplateEffect, AutomaticRuleSpec, BindingDomainSpec,
    BindingForwardSpec, BindingRowSpec, ChoiceScopeTemplateSpec, ConstraintIRV1,
    DynamicIntentSpec, InitialStateSpec, IntervalTemplateSpec, LeaseConditionTemplate,
    LeaseSpec, LiteralIdRef, OperatorTemplateSpec, ParameterIdRef, ParameterSpec,
    ReleaseLeaseTemplateEffect, ResourceSpec, ResourceUseTemplate, SetStateTemplateEffect,
    StateAssignment, StateCellSpec, StateConditionTemplate, StepDependencySpec,
    TerminalStateSpec, TimeDomain,
)


def compile_problem(problem: ClusterProblem, time_domain: TimeDomain) -> ConstraintIRV1:
    """Compile IO/LP + capacity-one PMs, positive operations, equal priorities.

    One single/dual-arm robot; every wafer starts unprocessed at step zero in
    IO/LP. Alternatives and repeated visits are supported. Source times are
    seconds. Unsupported semantics fail explicitly, never silently disappear.
    """
    # Source input accepts index expressions but stores integer wafer indexes.
    payload = problem.model_dump(mode="python", by_alias=True)
    for wafer in payload["initial_state"]["wafers"]:
        wafer["wafer_index"] = str(wafer["wafer_index"])
    try:
        problem = ClusterProblem.model_validate(payload)
    except ValidationError as error:
        raise SemanticError(DiagnosticCode.INVALID_PROBLEM, str(error)) from error
    _check_supported(problem, time_domain)
    return _ProblemCompiler(problem, time_domain).compile()


def _unsupported(path: str, reason: str) -> None:
    raise SemanticError(DiagnosticCode.UNSUPPORTED_FEATURE, reason, path=path)


def _positive_ticks(time_domain: TimeDomain, value: float | None, path: str) -> int:
    if value is None:
        _unsupported(path, "an explicit positive duration is required by the first compiler")
    ticks = compile_ticks(time_domain, value, path=path)
    if ticks == 0:
        _unsupported(path, "zero-duration operations are not supported by the first compiler")
    return ticks


def _check_supported(problem: ClusterProblem, time_domain: TimeDomain) -> None:
    if time_domain.unit != "second":
        _unsupported("time_domain.unit", "ClusterProblem durations are in seconds")
    if len(problem.ClusterTool) != 1:
        _unsupported("ClusterTool", "the first compiler supports exactly one robot")
    for name in ("cleaning", "just_in_time"):
        if getattr(problem, name) is not None:
            _unsupported(name, f"{name} is not supported by the first compiler")
    for module_id, module in problem.Modules.items():
        if module.type not in {ModuleType.PM, ModuleType.IO, ModuleType.LP}:
            _unsupported(f"Modules.{module_id}.type", "only PM and IO/LP are supported")
        if module.load_lock is not None:
            _unsupported(f"Modules.{module_id}.load_lock", "pressure transitions are not yet compiled")
        if module.type is ModuleType.PM and module.capacity != 1:
            _unsupported(f"Modules.{module_id}.capacity", "physical PM capacity must be one")
    robot_id, robot = next(iter(problem.ClusterTool.items()))
    reachable = set(robot.module_ids)
    for field in ("pick_time", "place_time"):
        _positive_ticks(time_domain, getattr(robot, field), f"ClusterTool.{robot_id}.{field}")
    compile_ticks(time_domain, robot.travel_times, path=f"ClusterTool.{robot_id}.travel_times")
    if len({wafer.priority for wafer in problem.initial_state.wafers}) > 1:
        _unsupported("initial_state.wafers.priority", "mixed source priorities are not yet compiled")
    for index, wafer in enumerate(problem.initial_state.wafers):
        path = f"initial_state.wafers[{index}]"
        if (wafer.step_index != 0 or wafer.process_end_time is not None
                or not isinstance(wafer.location, ModuleLocation)
                or problem.Modules[wafer.location.module_id].type not in {ModuleType.IO, ModuleType.LP}):
            _unsupported(path, "wafers must start unprocessed at step zero in IO/LP")
        if wafer.location.module_id not in reachable or problem.return_module_id(wafer) not in reachable:
            _unsupported(path, "source and return station must be reachable by the robot")
    for route_id, route in problem.routes.items():
        for index, visit in enumerate(route.visits):
            path = f"routes.{route_id}[{index}]"
            if visit.residency_time is not None:
                _unsupported(f"{path}.residency_time", "residency deadlines are not yet compiled")
            if any(problem.Modules[mid].type is not ModuleType.PM for mid in visit.module_ids):
                _unsupported(path, "internal route visits must use PMs")
            if not reachable.intersection(visit.module_ids):
                _unsupported(path, "visit has no candidate reachable by the robot")
            _positive_ticks(time_domain, visit.process_time, f"{path}.process_time")


def _id(prefix: str, *parts: str | int) -> str:
    """Lossless namespacing; separators in source IDs cannot create aliases."""
    return "/".join((prefix, *(quote(str(part), safe="") for part in parts)))


_PARAMETERS = (
    ParameterSpec(name="hand", kind="resource"),
    ParameterSpec(name="holder", kind="resource"),
    ParameterSpec(name="stage", kind="state_cell"),
    ParameterSpec(name="wafer", kind="owner"),
)


class _ProblemCompiler:
    def __init__(self, problem: ClusterProblem, time_domain: TimeDomain) -> None:
        self.problem, self.time_domain = problem, time_domain
        self.robot_id, self.robot = next(iter(problem.ClusterTool.items()))
        self.motion = _id("motion", self.robot_id)
        self.position = _id("position", self.robot_id)
        self.unknown = _id("unknown")
        initial_robot = problem.initial_state.robots.get(self.robot_id)
        self.initial_position = None if initial_robot is None else initial_robot.position_module_id
        arm_count = 1 if self.robot.arm_type is TMArmType.SINGLE_ARM else 2
        self.hands = tuple(_id("hand", self.robot_id, f"arm{i}") for i in range(arm_count))
        self.templates: dict[str, OperatorTemplateSpec] = {}
        self.automatic: dict[str, AutomaticRuleSpec] = {}
        self.rules: dict[str, DynamicIntentSpec] = {}
        self.rows: dict[str, set[tuple[str, ...]]] = {}

    def compile(self) -> ConstraintIRV1:
        resources = [ResourceSpec(id=self.motion, capacity=1)]
        resources.extend(ResourceSpec(id=hand, capacity=1) for hand in self.hands)
        for mid, module in sorted(self.problem.Modules.items()):
            resources.append(ResourceSpec(id=_id("module", mid), capacity=module.capacity))
            if module.type is ModuleType.PM:
                resources.append(ResourceSpec(id=_id("activity", mid), capacity=1))
        cells = [StateCellSpec(id=self.position, value_type="enum", enum_values=(
            self.unknown, *(_id("at", mid) for mid in sorted(self.robot.module_ids)),
        ))]
        initial_values = [StateAssignment(
            cell_id=self.position,
            value=self.unknown if self.initial_position is None else _id("at", self.initial_position),
        )]
        initial_leases, terminal_leases, terminal_values = [], [], []
        for wafer in sorted(self.problem.initial_state.wafers, key=lambda item: item.wafer_key):
            owner = _id("wafer", wafer.route_id, wafer.wafer_index)
            stage = _id("stage", wafer.route_id, wafer.wafer_index)
            visits = self.problem.routes[wafer.route_id].visits
            final_stage = 2 * (len(visits) + 1)
            cells.append(StateCellSpec(id=stage, value_type="int", minimum=0, maximum=final_stage))
            initial_values.append(StateAssignment(cell_id=stage, value=0))
            terminal_values.append(StateAssignment(cell_id=stage, value=final_stage))
            initial_leases.append(LeaseSpec(resource_id=_id("module", wafer.location.module_id), owner_id=owner))
            return_module = self.problem.return_module_id(wafer)
            terminal_leases.append(LeaseSpec(resource_id=_id("module", return_module), owner_id=owner))
            sources = (wafer.location.module_id,)
            for index in range(len(visits) + 1):
                targets = visits[index].module_ids if index < len(visits) else (return_module,)
                process = (_positive_ticks(self.time_domain, visits[index].process_time,
                                           f"routes.{wafer.route_id}[{index}].process_time")
                           if index < len(visits) else 0)
                for kind, phase, modules, duration in (
                    ("pick", 2 * index, sources, self.robot.pick_time),
                    ("place", 2 * index + 1, targets, self.robot.place_time),
                ):
                    for mid in sorted(set(modules).intersection(self.robot.module_ids)):
                        for hand in self.hands:
                            self._add(kind, phase, mid, duration, process if kind == "place" else 0,
                                      (hand, _id("module", mid), stage, owner))
                sources = targets
        return ConstraintIRV1(
            time_domain=self.time_domain, resources=tuple(resources), state_cells=tuple(cells),
            initial_state=InitialStateSpec(state_values=tuple(initial_values), leases=tuple(initial_leases)),
            terminal_state=TerminalStateSpec(state_values=tuple(terminal_values), leases=tuple(terminal_leases)),
            operator_templates=tuple(self.templates[key] for key in sorted(self.templates)),
            automatic_rules=tuple(self.automatic[key] for key in sorted(self.automatic)),
            dynamic_intents=tuple(self.rules[key] for key in sorted(self.rules)),
            binding_domains=tuple(BindingDomainSpec(
                id=key, parameters=_PARAMETERS,
                rows=tuple(BindingRowSpec(values=row) for row in sorted(self.rows[key])),
            ) for key in sorted(self.rows)),
        )

    def _add(
        self, kind: str, phase: int, mid: str, duration: float, process: int,
        row: tuple[str, ...],
    ) -> None:
        operation_ticks = _positive_ticks(self.time_domain, duration, kind)
        travel_ticks = compile_ticks(self.time_domain, self.robot.travel_times, path="travel_times")
        variants = [("at", 0), ("away", travel_ticks)]
        if self.initial_position is None:
            variants.append(("unknown", 0))
        for variant, travel in variants:
            template_id = _id("operator", kind, phase, mid, travel, operation_ticks, process)
            if template_id not in self.templates:
                self._template(template_id, kind, phase, mid, travel, operation_ticks, process)
            rule_id = _id("candidate", template_id, variant)
            if rule_id not in self.rules:
                position_guards = (StateConditionTemplate(
                    cell=LiteralIdRef(value=self.position),
                    operator="not_equal" if variant == "away" else "equal",
                    value=self.unknown if variant == "unknown" else _id("at", mid),
                ),)
                if variant == "away":
                    position_guards += (StateConditionTemplate(
                        cell=LiteralIdRef(value=self.position), operator="not_equal", value=self.unknown,
                    ),)
                self.rules[rule_id] = DynamicIntentSpec(
                    id=rule_id, operator_template_id=template_id, binding_domain_id=rule_id,
                    guards=(
                        StateConditionTemplate(cell=ParameterIdRef(parameter="stage"), operator="equal", value=phase),
                        LeaseConditionTemplate(resource=ParameterIdRef(parameter="holder" if kind == "pick" else "hand"),
                                               owner=ParameterIdRef(parameter="wafer")),
                        *position_guards,
                    ),
                    choice_scope_templates=(ChoiceScopeTemplateSpec(
                        scope_prefix=_id("work", phase), identity_parameters=("wafer",),
                    ),),
                )
                self.rows[rule_id] = set()
            self.rows[rule_id].add(row)

    def _template(
        self, template_id: str, kind: str, phase: int, mid: str,
        travel: int, duration: int, process: int,
    ) -> None:
        owner = ParameterIdRef(parameter="wafer")
        stage = ParameterIdRef(parameter="stage")
        destination = ParameterIdRef(parameter="hand" if kind == "pick" else "holder")
        source = ParameterIdRef(parameter="holder" if kind == "pick" else "hand")
        end_effects = (ReleaseLeaseTemplateEffect(resource=source, owner=owner),)
        if not process:
            end_effects += (SetStateTemplateEffect(cell=stage, value=phase + 1),)
        action = IntervalTemplateSpec(
            id=kind, start_offset=travel, duration=duration, audit_kind=kind.title(),
            resource_uses=(ResourceUseTemplate(resource=LiteralIdRef(value=self.motion)),),
            start_effects=(
                SetStateTemplateEffect(cell=LiteralIdRef(value=self.position), value=_id("at", mid)),
                AcquireLeaseTemplateEffect(resource=destination, owner=owner),
            ),
            end_effects=end_effects,
        )
        intervals, dependencies = (action,), ()
        if travel:
            intervals = (IntervalTemplateSpec(
                id="travel", start_offset=0, duration=travel, audit_kind="Travel",
                resource_uses=(ResourceUseTemplate(resource=LiteralIdRef(value=self.motion)),),
            ), action)
            dependencies = (StepDependencySpec(predecessor_step_id="travel", successor_step_id=kind),)
        self.templates[template_id] = OperatorTemplateSpec(
            id=template_id, origin="selectable", parameters=_PARAMETERS,
            intervals=intervals, step_dependencies=dependencies,
        )
        if process:
            child_id = _id("automatic", template_id)
            self.templates[child_id] = OperatorTemplateSpec(
                id=child_id, origin="automatic", parameters=_PARAMETERS,
                intervals=(IntervalTemplateSpec(
                    id="process", start_offset=0, duration=process, audit_kind="Process",
                    resource_uses=(ResourceUseTemplate(resource=LiteralIdRef(value=_id("activity", mid))),),
                    end_effects=(SetStateTemplateEffect(cell=stage, value=phase + 1),),
                ),),
            )
            self.automatic[child_id] = AutomaticRuleSpec(
                id=child_id, trigger_operator_template_id=template_id,
                trigger_boundary_id="place.end", emit_operator_template_id=child_id,
                binding_forwards=tuple(BindingForwardSpec(target_parameter=item.name, source_parameter=item.name)
                                       for item in _PARAMETERS),
            )
