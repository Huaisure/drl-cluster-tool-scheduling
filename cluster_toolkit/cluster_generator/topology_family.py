from __future__ import annotations

import hashlib
import json
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .pipeline_models import (
    IntInterval,
    ModuleKind,
    ModuleTag,
    RobotArmKind,
    TopologyModule,
    TopologyRobot,
    TopologyTemplate,
)


ATMOSPHERIC_FAMILY_ID = "atmospheric_linear"
ATMOSPHERIC_FAMILY_VERSION = "2.0.0"


class AtmosphericTopologyRequest(BaseModel):
    """One deterministic request for a linear, all-atmosphere topology."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    seed: int
    cell_count: int
    process_chambers_per_cell: IntInterval = Field(
        default_factory=lambda: IntInterval(minimum=2, maximum=6)
    )
    buffers_per_boundary: IntInterval = Field(
        default_factory=lambda: IntInterval(minimum=1, maximum=2)
    )
    include_alignment_chamber: bool = True

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("seed must be an integer")
        return value

    @field_validator("cell_count", mode="before")
    @classmethod
    def _validate_cell_count(cls, value: object) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 3:
            raise ValueError("cell_count must be an integer from 1 to 3")
        return value

    @model_validator(mode="after")
    def _validate_ranges(self) -> "AtmosphericTopologyRequest":
        if self.process_chambers_per_cell.minimum < 2:
            raise ValueError("each Cell requires at least two process CHAMBER Modules")
        if self.buffers_per_boundary.minimum < 1:
            raise ValueError("each Cell boundary requires at least one BUFFER")
        if self.buffers_per_boundary.maximum > 2:
            raise ValueError("v2 supports at most two BUFFER Modules per boundary")
        return self


def generate_atmospheric_topology(
    request: AtmosphericTopologyRequest,
) -> TopologyTemplate:
    """Materialize one immutable 1–3 Cell linear topology snapshot."""

    rng = random.Random(request.seed)
    cell_order = tuple(f"C{index}" for index in range(request.cell_count))
    modules: dict[str, TopologyModule] = {
        "IO": TopologyModule(kind=ModuleKind.IO, cell_id=cell_order[0])
    }

    cell_modules: dict[str, list[str]] = {cell_id: [] for cell_id in cell_order}
    cell_modules[cell_order[0]].append("IO")
    for cell_index, cell_id in enumerate(cell_order):
        chamber_count = rng.randint(
            request.process_chambers_per_cell.minimum,
            request.process_chambers_per_cell.maximum,
        )
        for chamber_index in range(chamber_count):
            module_id = f"{cell_id}_P{chamber_index}"
            modules[module_id] = TopologyModule(
                kind=ModuleKind.CHAMBER,
                cell_id=cell_id,
                tags=(ModuleTag.PROCESS,),
            )
            cell_modules[cell_id].append(module_id)
        if request.include_alignment_chamber and cell_index == 0:
            module_id = f"{cell_id}_AL"
            modules[module_id] = TopologyModule(
                kind=ModuleKind.CHAMBER,
                cell_id=cell_id,
                tags=(ModuleTag.ALIGN,),
            )
            cell_modules[cell_id].append(module_id)

    boundary_buffers: dict[tuple[str, str], tuple[str, ...]] = {}
    for boundary_index, (left, right) in enumerate(
        zip(cell_order, cell_order[1:])
    ):
        buffer_count = rng.randint(
            request.buffers_per_boundary.minimum,
            request.buffers_per_boundary.maximum,
        )
        buffer_ids: list[str] = []
        for buffer_index in range(buffer_count):
            module_id = f"B{boundary_index}_{buffer_index}"
            modules[module_id] = TopologyModule(
                kind=ModuleKind.CHAMBER,
                connected_cell_ids=(left, right),
                tags=(ModuleTag.BUFFER,),
            )
            buffer_ids.append(module_id)
        boundary_buffers[(left, right)] = tuple(buffer_ids)

    robots: dict[str, TopologyRobot] = {}
    for cell_index, cell_id in enumerate(cell_order):
        reachable = list(cell_modules[cell_id])
        if cell_index > 0:
            reachable.extend(boundary_buffers[(cell_order[cell_index - 1], cell_id)])
        if cell_index + 1 < len(cell_order):
            reachable.extend(boundary_buffers[(cell_id, cell_order[cell_index + 1])])
        robots[f"TM{cell_index}"] = TopologyRobot(
            cell_id=cell_id,
            module_ids=tuple(sorted(reachable)),
            arm_kind=rng.choice((RobotArmKind.SINGLE, RobotArmKind.DUAL)),
        )

    content = {
        "family_id": ATMOSPHERIC_FAMILY_ID,
        "family_version": ATMOSPHERIC_FAMILY_VERSION,
        "request": request.model_dump(mode="json"),
        "modules": {
            module_id: module.model_dump(mode="json")
            for module_id, module in sorted(modules.items())
        },
        "robots": {
            robot_id: robot.model_dump(mode="json")
            for robot_id, robot in sorted(robots.items())
        },
    }
    digest = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return TopologyTemplate(
        schema_version=2,
        topology_id=f"atmospheric-{digest[:16]}",
        topology_version=ATMOSPHERIC_FAMILY_VERSION,
        family_id=ATMOSPHERIC_FAMILY_ID,
        cell_order=cell_order,
        modules=modules,
        robots=robots,
    )
