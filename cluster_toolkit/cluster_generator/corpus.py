from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .pipeline import GeneratedInstance


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


class InstanceCorpus:
    """Materialize immutable problems without coupling them to dataset splits."""

    SOLVER_DIRECTORIES = (
        "cpsat_direct",
        "cpsat_periodic",
        "genetic",
        "branch_search",
        "other",
    )

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def materialize(self, generated: GeneratedInstance) -> Path:
        instances_root = self.root / "instances"
        instances_root.mkdir(parents=True, exist_ok=True)
        target = instances_root / generated.instance_id
        problem_bytes = _json_bytes(generated.instance.model_dump(mode="json"))
        metadata_bytes = _json_bytes(generated.metadata)

        if target.exists():
            self._verify_existing(target, problem_bytes, metadata_bytes)
            return target

        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{generated.instance_id}.",
                dir=instances_root,
            )
        )
        try:
            (staging / "problem.json").write_bytes(problem_bytes)
            (staging / "metadata.json").write_bytes(metadata_bytes)
            solutions = staging / "solutions"
            for solver_name in self.SOLVER_DIRECTORIES:
                (solutions / solver_name).mkdir(parents=True)
            try:
                staging.rename(target)
            except FileExistsError:
                self._verify_existing(target, problem_bytes, metadata_bytes)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return target

    @staticmethod
    def _verify_existing(
        target: Path,
        problem_bytes: bytes,
        metadata_bytes: bytes,
    ) -> None:
        existing_problem = target / "problem.json"
        existing_metadata = target / "metadata.json"
        if not existing_problem.is_file() or existing_problem.read_bytes() != problem_bytes:
            raise FileExistsError(
                f"instance directory already exists with different problem content: {target}"
            )
        if not existing_metadata.is_file() or existing_metadata.read_bytes() != metadata_bytes:
            raise FileExistsError(
                f"instance directory already exists with different metadata content: {target}"
            )
