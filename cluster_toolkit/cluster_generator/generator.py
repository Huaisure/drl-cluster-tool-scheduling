from __future__ import annotations

import hashlib
import json
import math
import random
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cluster_toolkit.problem import ClusterProblem, ModuleType, load_problem

from .models import (
    GENERATOR_NAME,
    GENERATOR_VERSION,
    DatasetManifest,
    GenerationConfig,
    ManifestEntry,
)
from .topology import ModuleGraph
from .validation import validate_generated_instance


class DatasetGenerator:
    """Generate reproducible benchmark instances from one fixed topology."""

    def __init__(
        self,
        template: ClusterProblem,
        config: GenerationConfig,
        *,
        template_name: str,
        template_sha256: str,
    ) -> None:
        self.template = template
        self.config = config
        self.template_name = template_name
        self.template_sha256 = template_sha256
        self.graph = ModuleGraph.from_problem(template)
        self._usable_lp_lengths = self._validate_template()

    @classmethod
    def from_template(
        cls,
        path: str | Path,
        config: GenerationConfig,
    ) -> "DatasetGenerator":
        template_path = Path(path)
        template_bytes = template_path.read_bytes()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Explicit Module.capacity overrides the type-based default",
                category=UserWarning,
            )
            template = load_problem(template_path)
        return cls(
            template,
            config,
            template_name=template_path.name,
            template_sha256=hashlib.sha256(template_bytes).hexdigest(),
        )

    def generate_instance(self, index: int = 0) -> dict[str, Any]:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("instance index must be a non-negative integer")

        instance_seed = self.instance_seed(index)
        rng = random.Random(instance_seed)
        route_count_range = self.config.route_count
        wafer_range = self.config.total_wafers
        process_range = self.config.process_time
        assert route_count_range is not None
        assert wafer_range is not None
        assert process_range is not None

        route_count = route_count_range.sample(rng)
        total_wafers = wafer_range.sample(rng)
        wafer_counts = self._distribute_wafers(total_wafers, route_count, rng)
        usable_lps = sorted(self._usable_lp_lengths)
        lp_offset = rng.randrange(len(usable_lps))

        routes: dict[str, list[dict[str, Any]]] = {}
        route_assignments: list[tuple[str, str, int]] = []
        lp_occupancy = {lp_id: 0 for lp_id in self.graph.lp_ids}
        route_metadata: dict[str, dict[str, Any]] = {}

        for route_index in range(route_count):
            route_id = f"R{route_index:03d}"
            start_lp = usable_lps[(route_index + lp_offset) % len(usable_lps)]
            length = rng.choice(self._usable_lp_lengths[start_lp])
            path, end_lp = self.graph.construct_closed_walk(
                start_lp,
                length,
                rng,
                end_lp=start_lp,
            )
            candidates = self.graph.expand_candidates(
                path,
                start_lp,
                end_lp,
                probability=self.config.candidate_probability,
                max_candidates=self.config.max_candidates,
                rng=rng,
            )
            visits: list[dict[str, Any]] = []
            for selected, module_ids in zip(path, candidates, strict=True):
                module_type = self.template.Modules[selected].type
                process_time = (
                    process_range.sample(rng)
                    if module_type is ModuleType.PM
                    else 0
                )
                visit: dict[str, Any] = {
                    "process_time": process_time,
                }
                if len(module_ids) == 1:
                    visit["module_id"] = module_ids[0]
                else:
                    visit["module_ids"] = list(module_ids)
                visits.append(visit)
            routes[route_id] = visits
            wafer_count = wafer_counts[route_index]
            route_assignments.append((route_id, start_lp, wafer_count))
            lp_occupancy[start_lp] += wafer_count
            route_metadata[route_id] = {
                "start_lp": start_lp,
                "end_lp": end_lp,
                "witness_path": list(path),
            }

        modules = self._generate_modules(lp_occupancy, rng)
        cluster_tool = self._generate_cluster_tool(rng)
        initial_robots = {
            robot_id: {
                "position_module_id": robot_state.position_module_id,
            }
            for robot_id, robot_state in sorted(self.template.initial_state.robots.items())
        }
        initial_wafers = [
            {
                "route_id": route_id,
                "wafer_index": "0" if count == 1 else f"0-{count - 1}",
                "priority": 0,
                "step_index": 0,
                "location": {
                    "kind": "module",
                    "module_id": start_lp,
                },
                "process_end_time": None,
                "return_lp_id": start_lp,
            }
            for route_id, start_lp, count in route_assignments
        ]

        raw_instance: dict[str, Any] = {
            "_meta": {
                "generator": {
                    "name": GENERATOR_NAME,
                    "version": GENERATOR_VERSION,
                    "profile": self.config.profile,
                    "master_seed": self.config.seed,
                    "instance_seed": instance_seed,
                    "instance_index": index,
                    "template_sha256": self.template_sha256,
                    "route_witnesses": route_metadata,
                }
            },
            "Modules": modules,
            "ClusterTool": cluster_tool,
            "routes": routes,
            "initial_state": {
                "robots": initial_robots,
                "wafers": initial_wafers,
            },
        }
        validate_generated_instance(raw_instance)
        return raw_instance

    def generate(
        self,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> DatasetManifest:
        output_path = Path(output_dir)
        self._validate_output_directory(output_path, overwrite=overwrite)
        entries: list[ManifestEntry] = []
        generated_instances: list[tuple[str, dict[str, Any]]] = []

        for index in range(self.config.instance_count):
            raw_instance = self.generate_instance(index)
            audit = validate_generated_instance(raw_instance)
            file_name = f"instance-{index:05d}.json"
            generated_instances.append((file_name, raw_instance))
            entries.append(
                ManifestEntry(
                    file=file_name,
                    instance_seed=self.instance_seed(index),
                    audit=audit,
                )
            )

        manifest = DatasetManifest(
            generator={
                "name": GENERATOR_NAME,
                "version": GENERATOR_VERSION,
            },
            template={
                "file": self.template_name,
                "sha256": self.template_sha256,
            },
            config=self.config.model_dump(mode="json"),
            instances=tuple(entries),
        )
        new_files = {file_name for file_name, _ in generated_instances} | {"manifest.json"}
        old_owned = (
            self._owned_files(output_path)
            if output_path.exists() and any(output_path.iterdir()) and overwrite
            else set()
        )
        conflicts = sorted(
            file_name
            for file_name in new_files
            if (output_path / file_name).exists() and file_name not in old_owned
        )
        if conflicts:
            raise ValueError(f"refusing to replace unowned output files: {conflicts}")
        self._prepare_output_directory(output_path, overwrite=overwrite)
        for file_name, raw_instance in generated_instances:
            self._write_json(output_path / file_name, raw_instance)
        self._write_json(
            output_path / "manifest.json",
            manifest.model_dump(mode="json"),
        )
        return manifest

    def instance_seed(self, index: int) -> int:
        digest = hashlib.sha256(f"{self.config.seed}:{index}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def _validate_template(self) -> dict[str, tuple[int, ...]]:
        if not self.graph.lp_ids:
            raise ValueError("template must contain at least one LP")
        if not any(module.type is ModuleType.PM for module in self.template.Modules.values()):
            raise ValueError("template must contain at least one PM")
        if not self.template.ClusterTool:
            raise ValueError("template must contain at least one Robot")

        route_steps = self.config.route_steps
        assert route_steps is not None
        usable: dict[str, tuple[int, ...]] = {}
        for lp_id in self.graph.lp_ids:
            lengths = self.graph.feasible_lengths(
                lp_id,
                route_steps.minimum,
                route_steps.maximum,
                end_lp=lp_id,
            )
            if lengths:
                usable[lp_id] = lengths
        if not usable:
            raise ValueError(
                "template has no LP-to-LP path visiting a PM within the configured route_steps range"
            )
        return usable

    def _generate_modules(
        self,
        lp_occupancy: Mapping[str, int],
        rng: random.Random,
    ) -> dict[str, dict[str, Any]]:
        modules: dict[str, dict[str, Any]] = {}
        for module_id, template_module in sorted(self.template.Modules.items()):
            if template_module.type is ModuleType.LP:
                occupied = lp_occupancy[module_id]
                spare_percent = self.config.lp_spare_percent.sample(rng)
                spare = math.ceil(occupied * spare_percent / 100)
                capacity = max(1, occupied + spare)
                if (
                    self.config.max_lp_capacity is not None
                    and capacity > self.config.max_lp_capacity
                ):
                    raise ValueError(
                        f"LP {module_id} needs capacity {capacity}, above max_lp_capacity "
                        f"{self.config.max_lp_capacity}"
                    )
            elif template_module.type is ModuleType.LL:
                capacity = self.config.ll_capacity.sample(rng)
            else:
                capacity = self.config.pm_capacity.sample(rng)

            raw_module: dict[str, Any] = {
                "type": template_module.type.value,
                "capacity": capacity,
            }
            if template_module.load_lock is not None:
                raw_module["load_lock"] = {
                    "initial_state": template_module.load_lock.initial_state.value,
                    "atmosphere_to_vacuum_time": self.config.pump_time.sample(rng),
                    "vacuum_to_atmosphere_time": self.config.vent_time.sample(rng),
                    "tm_required_states": {
                        robot_id: required_state.value
                        for robot_id, required_state in sorted(
                            template_module.load_lock.tm_required_states.items()
                        )
                    },
                }
            modules[module_id] = raw_module
        return modules

    def _generate_cluster_tool(self, rng: random.Random) -> dict[str, dict[str, Any]]:
        return {
            robot_id: {
                "module_ids": list(robot.module_ids),
                "arm_type": robot.arm_type.value,
                "travel_times": self.config.travel_time.sample(rng),
                "place_time": self.config.place_time.sample(rng),
                "pick_time": self.config.pick_time.sample(rng),
            }
            for robot_id, robot in sorted(self.template.ClusterTool.items())
        }

    @staticmethod
    def _distribute_wafers(
        total_wafers: int,
        route_count: int,
        rng: random.Random,
    ) -> list[int]:
        if total_wafers < route_count:
            raise ValueError("total_wafers must be at least route_count")
        counts = [1] * route_count
        for _ in range(total_wafers - route_count):
            counts[rng.randrange(route_count)] += 1
        return counts

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        ) + "\n"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def _validate_output_directory(cls, output_path: Path, *, overwrite: bool) -> None:
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"output path is not a directory: {output_path}")
        if not output_path.exists():
            return
        contents = list(output_path.iterdir())
        if not contents:
            return
        if not overwrite:
            raise ValueError("output directory is not empty; use --overwrite to replace owned files")

        cls._owned_files(output_path)

    @staticmethod
    def _owned_files(output_path: Path) -> set[str]:
        manifest_path = output_path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("cannot overwrite a non-empty directory without an owned manifest.json")
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("cannot read existing manifest.json") from exc
        if old_manifest.get("generator", {}).get("name") != GENERATOR_NAME:
            raise ValueError("existing manifest.json is not owned by cluster_generator")

        owned_files = {"manifest.json"}
        for entry in old_manifest.get("instances", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
                raise ValueError("existing manifest contains an invalid instance file entry")
            owned_path = output_path / entry["file"]
            if owned_path.parent != output_path:
                raise ValueError("existing manifest contains an unsafe owned file path")
            owned_files.add(entry["file"])
        return owned_files

    @classmethod
    def _prepare_output_directory(cls, output_path: Path, *, overwrite: bool) -> None:
        cls._validate_output_directory(output_path, overwrite=overwrite)
        output_path.mkdir(parents=True, exist_ok=True)
        contents = list(output_path.iterdir())
        if not contents:
            return

        for file_name in sorted(cls._owned_files(output_path)):
            owned_path = output_path / file_name
            if owned_path.parent != output_path:
                raise ValueError("existing manifest contains an unsafe owned file path")
            if owned_path.is_file():
                owned_path.unlink()
