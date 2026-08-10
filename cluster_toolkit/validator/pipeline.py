from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Hashable, Mapping, Sequence

from cluster_toolkit.problem import ClusterProblem, InitialSnapshot, ModuleType

from .common import group_actions, parse_actions
from .models import ActionRecord, ValidationReport, WaferKey
from .module_validator import ModuleValidator
from .robot_validator import RobotValidator
from .wafer_validator import WaferValidator


def _module_ids_for_action(action: ActionRecord) -> Iterable[str]:
    return () if action.module_id is None else (action.module_id,)


def _robot_ids_for_action(action: ActionRecord) -> Iterable[str]:
    return () if action.tm_id is None else (action.tm_id,)


def _wafer_keys_for_action(action: ActionRecord) -> Iterable[WaferKey]:
    return () if action.wafer_key is None else (action.wafer_key,)


def _reject_unknown_subjects(
    subject_kind: str,
    observed_ids: Iterable[Hashable],
    configured_ids: Iterable[Hashable],
) -> None:
    unknown_ids = set(observed_ids) - set(configured_ids)
    if unknown_ids:
        rendered = ", ".join(sorted(repr(subject_id) for subject_id in unknown_ids))
        raise ValueError(f"Unknown {subject_kind} referenced by actions: {rendered}")


class ValidatorSuite:
    """Build subject-local validators from one parsed problem and merge their reports."""

    def __init__(self, problem: ClusterProblem) -> None:
        self.problem = problem
        self.module_validators: list[ModuleValidator] = []
        self.robot_validators: list[RobotValidator] = []
        self.wafer_validators: list[WaferValidator] = []

    def validate(self, actions: Sequence[Mapping[str, Any]]) -> ValidationReport:
        parsed_actions = parse_actions(actions)
        initial_snapshot = self.problem.initial_state.to_snapshot()
        self._create_module_validators(parsed_actions, initial_snapshot)
        self._create_robot_validators(parsed_actions, initial_snapshot)
        self._create_wafer_validators(parsed_actions, initial_snapshot)

        report = ValidationReport()
        for validator in self.module_validators:
            report.extend(validator.validate())
        for validator in self.robot_validators:
            report.extend(validator.validate())
        for validator in self.wafer_validators:
            report.extend(validator.validate())
        return report

    def _create_module_validators(
        self,
        actions: Sequence[ActionRecord],
        initial_snapshot: InitialSnapshot,
    ) -> None:
        grouped = group_actions(actions, _module_ids_for_action)
        configured_ids = set(self.problem.Modules)
        _reject_unknown_subjects("Module", grouped, configured_ids)

        self.module_validators = []
        for module_id in sorted(configured_ids):
            module = self.problem.Modules[module_id]
            self.module_validators.append(
                ModuleValidator(
                    module_id=module_id,
                    config=module,
                    actions=grouped.get(module_id, ()),
                    initial_occupants=initial_snapshot.module_occupants.get(
                        module_id,
                        frozenset(),
                    ),
                )
            )

    def _create_robot_validators(
        self,
        actions: Sequence[ActionRecord],
        initial_snapshot: InitialSnapshot,
    ) -> None:
        grouped = group_actions(actions, _robot_ids_for_action)
        configured_ids = set(self.problem.ClusterTool)
        _reject_unknown_subjects("Robot", grouped, configured_ids)

        self.robot_validators = [
            RobotValidator(
                robot_id=robot_id,
                config=self.problem.ClusterTool[robot_id],
                actions=grouped.get(robot_id, ()),
                initial_position_module_id=initial_snapshot.tm_positions.get(robot_id),
                initial_arms=initial_snapshot.tm_arms.get(robot_id),
            )
            for robot_id in sorted(configured_ids)
        ]

    def _create_wafer_validators(
        self,
        actions: Sequence[ActionRecord],
        initial_snapshot: InitialSnapshot,
    ) -> None:
        grouped = group_actions(actions, _wafer_keys_for_action)
        configured_keys = set(initial_snapshot.wafers_by_key)
        source_module_ids = frozenset(
            module_id
            for module_id, module in self.problem.Modules.items()
            if module.type in {ModuleType.IO, ModuleType.LP}
        )
        pm_module_ids = frozenset(
            module_id
            for module_id, module in self.problem.Modules.items()
            if module.type is ModuleType.PM
        )
        _reject_unknown_subjects("Wafer", grouped, configured_keys)

        self.wafer_validators = [
            WaferValidator(
                wafer_key=wafer_key,
                route=self.problem.routes[wafer_key[0]],
                just_in_time=self.problem.just_in_time,
                actions=grouped.get(wafer_key, ()),
                initial_wafer=initial_snapshot.wafers_by_key[wafer_key],
                return_module_id=(
                    self.problem.return_module_id(
                        initial_snapshot.wafers_by_key[wafer_key]
                    )
                    if source_module_ids
                    else None
                ),
                source_module_ids=source_module_ids,
                pm_module_ids=pm_module_ids,
            )
            for wafer_key in sorted(configured_keys)
        ]
