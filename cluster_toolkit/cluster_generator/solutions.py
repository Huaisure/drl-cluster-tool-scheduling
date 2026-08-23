from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import tempfile
from enum import Enum
from pathlib import Path
from typing import Literal, Mapping

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .corpus import _json_bytes, _write_new_atomic


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SolverStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    OPTIMAL = "OPTIMAL"


class ValidationStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    VALID = "VALID"
    INVALID = "INVALID"


class TerminationReason(str, Enum):
    NORMAL = "NORMAL"
    TIME_LIMIT = "TIME_LIMIT"
    ERROR = "ERROR"
    INTERRUPTED = "INTERRUPTED"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    TERMINAL = "TERMINAL"


class GlobalOptimalityStatus(str, Enum):
    UNPROVEN = "UNPROVEN"
    PROVEN_OPTIMAL = "PROVEN_OPTIMAL"


class LabelingStatus(str, Enum):
    PENDING = "PENDING"
    UNVALIDATED = "UNVALIDATED"
    LABELED = "LABELED"


class ComponentResult(_StrictModel):
    status: SolverStatus
    objective: int | None = None
    best_bound: int | None = None
    runtime_seconds: float

    @field_validator("objective", "best_bound", mode="before")
    @classmethod
    def _validate_optional_integer(cls, value: object) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("component objectives and bounds must be non-negative integers")
        return value

    @field_validator("runtime_seconds", mode="before")
    @classmethod
    def _validate_runtime(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("runtime_seconds must be a non-negative finite number")
        result = float(value)
        if result < 0 or not math.isfinite(result):
            raise ValueError("runtime_seconds must be a non-negative finite number")
        return result

    @model_validator(mode="after")
    def _validate_result_semantics(self) -> "ComponentResult":
        if self.status in {SolverStatus.FEASIBLE, SolverStatus.OPTIMAL}:
            if self.objective is None:
                raise ValueError("FEASIBLE and OPTIMAL components require objective")
        if self.status is SolverStatus.INFEASIBLE and self.objective is not None:
            raise ValueError("an INFEASIBLE component must not define objective")
        if (
            self.objective is not None
            and self.best_bound is not None
            and self.best_bound > self.objective
        ):
            raise ValueError("component best_bound must not exceed objective")
        if self.status is SolverStatus.OPTIMAL and self.best_bound != self.objective:
            raise ValueError("an OPTIMAL component requires best_bound equal to objective")
        return self


class SolutionRecord(_StrictModel):
    """One immutable solver attempt; component OPTIMAL is not global optimality."""

    schema_version: Literal[1, 2] = 2
    instance_id: str
    solution_id: str
    solver_name: str
    solver_version: str
    solver_config_hash: str
    seed: int
    solution_status: SolverStatus = Field(
        validation_alias=AliasChoices("solution_status", "status")
    )
    termination_reason: TerminationReason = TerminationReason.NORMAL
    validation_status: ValidationStatus = ValidationStatus.NOT_RUN
    workflow_status: WorkflowStatus = WorkflowStatus.TERMINAL
    validator_name: str | None = None
    validator_version: str | None = None
    global_optimality_status: GlobalOptimalityStatus = (
        GlobalOptimalityStatus.UNPROVEN
    )
    makespan: int | None = None
    best_bound: int | None = None
    best_bound_scope: Literal["full_problem"] | None = None
    runtime_seconds: float
    time_limit_seconds: float | None = None
    components: dict[str, ComponentResult] = Field(default_factory=dict)
    actions_file: str | None = None
    action_count: int | None = None
    actions_sha256: str | None = None
    strong_sample_signals: dict[str, bool | float | int | str | None] = Field(
        default_factory=dict
    )

    @property
    def status(self) -> SolverStatus:
        """Compatibility accessor; persisted schema uses solution_status."""

        return self.solution_status

    @field_validator(
        "instance_id",
        "solution_id",
        "solver_name",
        "solver_version",
        "solver_config_hash",
    )
    @classmethod
    def _validate_identifier(cls, value: str, info) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value

    @field_validator("makespan", "best_bound", mode="before")
    @classmethod
    def _validate_optional_integer(cls, value: object) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("makespan and best_bound must be non-negative integers")
        return value

    @field_validator("runtime_seconds", "time_limit_seconds", mode="before")
    @classmethod
    def _validate_duration(cls, value: object, info) -> float | None:
        if value is None and info.field_name == "time_limit_seconds":
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{info.field_name} must be a non-negative finite number")
        result = float(value)
        if result < 0 or not math.isfinite(result):
            raise ValueError(f"{info.field_name} must be a non-negative finite number")
        return result

    @field_validator("actions_file")
    @classmethod
    def _validate_actions_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not value:
            raise ValueError("actions_file must be a safe relative path")
        return value

    @field_validator("action_count", mode="before")
    @classmethod
    def _validate_action_count(cls, value: object) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("action_count must be a non-negative integer")
        return value

    @field_validator("actions_sha256")
    @classmethod
    def _validate_actions_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("actions_sha256 must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def _validate_result_semantics(self) -> "SolutionRecord":
        if self.status in {SolverStatus.FEASIBLE, SolverStatus.OPTIMAL}:
            if self.makespan is None:
                raise ValueError("FEASIBLE and OPTIMAL results must define makespan")
        if self.validation_status is ValidationStatus.VALID and self.makespan is None:
            raise ValueError("a VALID solution must define makespan")
        if self.validation_status is not ValidationStatus.NOT_RUN:
            if not self.validator_name or not self.validator_version:
                raise ValueError(
                    "completed validation requires validator_name and validator_version"
                )
        if self.status is SolverStatus.INFEASIBLE and self.makespan is not None:
            raise ValueError("an INFEASIBLE result must not define makespan")
        action_metadata = (
            self.actions_file,
            self.action_count,
            self.actions_sha256,
        )
        if any(value is not None for value in action_metadata) and not all(
            value is not None for value in action_metadata
        ):
            raise ValueError(
                "actions_file, action_count, and actions_sha256 must be set together"
            )
        if self.termination_reason is TerminationReason.NOT_ELIGIBLE:
            if self.status is not SolverStatus.UNKNOWN:
                raise ValueError("NOT_ELIGIBLE requires solution status UNKNOWN")
            if self.validation_status is not ValidationStatus.NOT_RUN:
                raise ValueError("NOT_ELIGIBLE attempts must not run validation")
        if self.workflow_status is not WorkflowStatus.TERMINAL:
            if self.status is not SolverStatus.UNKNOWN:
                raise ValueError("non-terminal attempts cannot declare a solver result")
            if self.validation_status is not ValidationStatus.NOT_RUN:
                raise ValueError("non-terminal attempts cannot declare validation")
        if (
            self.makespan is not None
            and self.best_bound is not None
            and self.best_bound > self.makespan
        ):
            raise ValueError("best_bound must not exceed makespan for minimization")
        if (self.best_bound is None) != (self.best_bound_scope is None):
            raise ValueError(
                "best_bound and best_bound_scope=full_problem must be defined together"
            )
        if self.global_optimality_status is GlobalOptimalityStatus.PROVEN_OPTIMAL:
            if self.status is not SolverStatus.OPTIMAL:
                raise ValueError("PROVEN_OPTIMAL requires full-problem status OPTIMAL")
            if self.validation_status is not ValidationStatus.VALID:
                raise ValueError("PROVEN_OPTIMAL requires a VALID schedule")
            if self.makespan is None or self.best_bound != self.makespan:
                raise ValueError("PROVEN_OPTIMAL requires best_bound equal to makespan")
        return self


class SolutionIndex(_StrictModel):
    schema_version: Literal[2] = 2
    instance_id: str
    record_count: int
    valid_solution_count: int
    labeling_status: LabelingStatus
    best_solution_file: str | None = None
    best_makespan: int | None = None
    best_bound: int | None = None
    certified_gap: float | None = None
    global_optimality_status: GlobalOptimalityStatus
    terminal_attempt_count: int = 0
    usable: bool = False
    quarantined: bool = False
    strong_sample_signals: dict[str, bool | float | int | str | None] = Field(
        default_factory=dict
    )


_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class InstanceSolutions:
    """Own per-instance solution writes and deterministic reduction."""

    def __init__(self, instance_dir: str | Path) -> None:
        self.instance_dir = Path(instance_dir)
        problem_path = self.instance_dir / "problem.json"
        if not problem_path.is_file():
            raise FileNotFoundError(f"instance problem does not exist: {problem_path}")
        self.instance_id = json.loads(problem_path.read_text(encoding="utf-8"))[
            "instance_id"
        ]

    def write(
        self,
        record: SolutionRecord,
        actions: list[Mapping[str, object]] | tuple[Mapping[str, object], ...] | None = None,
    ) -> Path:
        if record.instance_id != self.instance_id:
            raise ValueError("SolutionRecord.instance_id does not match problem.json")
        for value, name in (
            (record.solver_name, "solver_name"),
            (record.solution_id, "solution_id"),
        ):
            if not _SAFE_SEGMENT.fullmatch(value):
                raise ValueError(f"{name} contains unsafe path characters: {value!r}")

        directory = self.instance_dir / "solutions" / record.solver_name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{record.solution_id}.solution.json"
        if actions is not None:
            action_values = [dict(action) for action in actions]
            action_bytes = _json_bytes(action_values)
            action_name = f"{record.solution_id}.actions.json.gz"
            relative_action_path = (
                Path("solutions") / record.solver_name / action_name
            ).as_posix()
            record = SolutionRecord.model_validate(
                record.model_copy(
                    update={
                        "actions_file": relative_action_path,
                        "action_count": len(action_values),
                        "actions_sha256": hashlib.sha256(action_bytes).hexdigest(),
                    }
                ).model_dump()
            )
        payload = _json_bytes(record.model_dump(mode="json"))
        if path.exists():
            if path.read_bytes() == payload:
                return path
            raise FileExistsError(f"solution record already exists with other content: {path}")
        if actions is not None:
            action_path = self.instance_dir / record.actions_file
            compressed = gzip.compress(action_bytes, mtime=0)
            if action_path.exists():
                if action_path.read_bytes() != compressed:
                    raise FileExistsError(
                        f"actions already exist with other content: {action_path}"
                    )
            else:
                _write_new_atomic(action_path, compressed)
        _write_new_atomic(path, payload)
        return path

    def reduce(self) -> SolutionIndex:
        records: list[tuple[Path, SolutionRecord]] = []
        solutions_root = self.instance_dir / "solutions"
        for path in sorted(solutions_root.glob("**/*.solution.json")):
            record = SolutionRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if record.instance_id != self.instance_id:
                raise ValueError(f"solution has wrong instance_id: {path}")
            records.append((path, record))

        valid = [
            (path, record)
            for path, record in records
            if record.validation_status is ValidationStatus.VALID
            and record.makespan is not None
        ]
        best = min(
            valid,
            key=lambda item: (
                item[1].makespan,
                item[0].relative_to(self.instance_dir).as_posix(),
            ),
            default=None,
        )
        bounds = [record.best_bound for _, record in records if record.best_bound is not None]
        best_bound = max(bounds, default=None)
        best_makespan = best[1].makespan if best is not None else None
        if (
            best_makespan is not None
            and best_bound is not None
            and best_bound > best_makespan
        ):
            raise ValueError(
                "solution records are inconsistent: full-problem bound exceeds "
                "a validated makespan"
            )
        gap = None
        if best_makespan is not None and best_bound is not None:
            gap = 0.0 if best_makespan == 0 else max(
                0.0,
                (best_makespan - best_bound) / best_makespan,
            )
        proven = any(
            record.global_optimality_status
            is GlobalOptimalityStatus.PROVEN_OPTIMAL
            for _, record in valid
        )
        quarantined = any(
            record.status is SolverStatus.INFEASIBLE
            and record.workflow_status is WorkflowStatus.TERMINAL
            for _, record in records
        )
        no_incumbent = bool(records) and not valid
        gaps = [
            (record.makespan - record.best_bound) / record.makespan
            for _, record in valid
            if record.makespan
            and record.best_bound is not None
        ]
        makespans = {record.makespan for _, record in valid}
        strong_signals: dict[str, bool | float | int | str | None] = {
            "no_incumbent": no_incumbent,
            "unproven": bool(valid) and not proven,
            "max_gap": max(gaps, default=None),
            "solver_disagreement": len(makespans) > 1,
        }
        index = SolutionIndex(
            instance_id=self.instance_id,
            record_count=len(records),
            valid_solution_count=len(valid),
            labeling_status=(
                LabelingStatus.LABELED
                if valid
                else LabelingStatus.UNVALIDATED
                if records
                else LabelingStatus.PENDING
            ),
            best_solution_file=(
                best[0].relative_to(self.instance_dir).as_posix()
                if best is not None
                else None
            ),
            best_makespan=best_makespan,
            best_bound=best_bound,
            certified_gap=gap,
            global_optimality_status=(
                GlobalOptimalityStatus.PROVEN_OPTIMAL
                if proven
                else GlobalOptimalityStatus.UNPROVEN
            ),
            terminal_attempt_count=sum(
                record.workflow_status is WorkflowStatus.TERMINAL
                for _, record in records
            ),
            usable=bool(valid),
            quarantined=quarantined,
            strong_sample_signals=strong_signals,
        )
        _replace_atomic(
            self.instance_dir / "solution_index.json",
            _json_bytes(index.model_dump(mode="json")),
        )
        return index


def _replace_atomic(path: Path, payload: bytes) -> None:
    file_descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
