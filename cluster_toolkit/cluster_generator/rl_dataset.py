from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from .models import GENERATOR_NAME, GENERATOR_VERSION, RLGenerationConfig
from .problem_generator import GeneratedBenchmark, ProblemGenerator


class RLDatasetGenerator:
    """Materialize reproducible PPO curriculum instances and reference baselines."""

    def __init__(
        self,
        config: RLGenerationConfig,
        *,
        problem_generator: ProblemGenerator | None = None,
    ) -> None:
        self.config = config
        self.problem_generator = problem_generator or ProblemGenerator()

    def instance_seed(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("instance index must be a non-negative integer")
        digest = hashlib.sha256(
            f"rl:{self.config.seed}:{self.config.split}:{index}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    def generate_instance(self, index: int = 0) -> GeneratedBenchmark:
        instance_seed = self.instance_seed(index)
        if self.config.difficulty is None:
            return self.problem_generator.generate_curriculum(
                instance_seed,
                split=self.config.split,
                weights=self.config.difficulty_weights,
            )
        return self.problem_generator.generate(
            instance_seed,
            difficulty=self.config.difficulty,
            split=self.config.split,
        )

    def generate(
        self,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        output_path = Path(output_dir)
        self._validate_output_directory(output_path, overwrite=overwrite)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, Any]] = []
        file_names: list[str] = []
        with tempfile.TemporaryDirectory(
            prefix=f".{output_path.name}-staging-",
            dir=output_path.parent,
        ) as staging_dir:
            staging_path = Path(staging_dir)
            for index in range(self.config.instance_count):
                instance_id = f"{self.config.split}-{index:05d}"
                instance_seed = self.instance_seed(index)
                if not self.config.materialize_problems:
                    difficulty = self.config.difficulty or self.problem_generator.select_curriculum_difficulty(
                        instance_seed,
                        split=self.config.split,
                        weights=self.config.difficulty_weights,
                    )
                    entries.append(
                        {
                            "instance_id": instance_id,
                            "index": index,
                            "seed": instance_seed,
                            "difficulty": difficulty,
                            "problem_file": None,
                            "actions_file": None,
                            "metadata": None,
                        }
                    )
                    continue

                benchmark = self.generate_instance(index)
                problem_file = f"{instance_id}.json"
                actions_file = (
                    f"{instance_id}.actions.json"
                    if self.config.include_reference_actions
                    else None
                )
                self._write_json(staging_path / problem_file, benchmark.raw_problem)
                file_names.append(problem_file)
                if actions_file is not None:
                    self._write_json(
                        staging_path / actions_file,
                        {
                            "instance_id": instance_id,
                            "problem_file": problem_file,
                            "actions": list(benchmark.actions),
                            "makespan": benchmark.metadata["reference_makespan"],
                            "lower_bound": benchmark.metadata["lower_bound"],
                            "optimality_gap": benchmark.metadata["optimality_gap"],
                            "validator_result": benchmark.metadata["validator_result"],
                            "reference_policy": benchmark.metadata["reference_policy"],
                            "solver_time": None,
                        },
                    )
                    file_names.append(actions_file)
                entries.append(
                    {
                        "instance_id": instance_id,
                        "index": index,
                        "seed": instance_seed,
                        "difficulty": benchmark.metadata["difficulty"],
                        "problem_file": problem_file,
                        "actions_file": actions_file,
                        "metadata": benchmark.metadata,
                    }
                )

            manifest: dict[str, Any] = {
                "schema_version": 1,
                "generator": {
                    "name": GENERATOR_NAME,
                    "version": GENERATOR_VERSION,
                    "mode": "ppo",
                },
                "config": self.config.model_dump(mode="json"),
                "problem_generation_config": self.problem_generator.config.model_dump(
                    mode="json"
                ),
                "instances": entries,
            }
            self._write_json(staging_path / "manifest.json", manifest)
            file_names.append("manifest.json")

            old_owned = (
                self._owned_files(output_path)
                if output_path.exists() and any(output_path.iterdir()) and overwrite
                else set()
            )
            conflicts = sorted(
                file_name
                for file_name in file_names
                if (output_path / file_name).exists() and file_name not in old_owned
            )
            if conflicts:
                raise ValueError(
                    f"refusing to replace unowned output files: {conflicts}"
                )
            self._prepare_output_directory(output_path, overwrite=overwrite)
            for file_name in file_names:
                (staging_path / file_name).replace(output_path / file_name)
            return manifest

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _owned_files(output_path: Path) -> set[str]:
        manifest_path = output_path / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("output directory is not an owned PPO generator dataset") from exc
        generator = manifest.get("generator", {})
        if generator.get("name") != GENERATOR_NAME or generator.get("mode") != "ppo":
            raise ValueError("output directory is not an owned PPO generator dataset")
        owned = {"manifest.json"}
        for entry in manifest.get("instances", []):
            for field in ("problem_file", "actions_file"):
                value = entry.get(field)
                if isinstance(value, str):
                    relative_path = Path(value)
                    if relative_path.is_absolute() or relative_path.parent != Path("."):
                        raise ValueError("owned PPO manifest contains an unsafe file path")
                    owned.add(value)
        return owned

    @classmethod
    def _validate_output_directory(cls, output_path: Path, *, overwrite: bool) -> None:
        if not output_path.exists():
            return
        if not output_path.is_dir():
            raise ValueError(f"output path is not a directory: {output_path}")
        if not any(output_path.iterdir()):
            return
        if not overwrite:
            raise ValueError("output directory is not empty; pass overwrite=True to replace it")
        cls._owned_files(output_path)

    @classmethod
    def _prepare_output_directory(cls, output_path: Path, *, overwrite: bool) -> None:
        if output_path.exists() and any(output_path.iterdir()) and overwrite:
            for relative_name in cls._owned_files(output_path):
                owned_path = output_path / relative_name
                if owned_path.is_file():
                    owned_path.unlink()
        output_path.mkdir(parents=True, exist_ok=True)
