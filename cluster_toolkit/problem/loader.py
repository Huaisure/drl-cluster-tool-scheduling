from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import ClusterProblem


def load_problem(path: str | Path) -> ClusterProblem:
    """Read, normalize, and validate one UTF-8 problem JSON file."""

    json_text = Path(path).read_text(encoding="utf-8")
    return ClusterProblem.model_validate_json(json_text)


def parse_problem(data: Mapping[str, Any]) -> ClusterProblem:
    """Normalize and validate an already decoded problem mapping."""

    return ClusterProblem.model_validate(data)
