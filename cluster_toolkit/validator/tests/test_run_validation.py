from __future__ import annotations

import json
from pathlib import Path

import pytest

from cluster_toolkit.run_validation import load_actions, main, validate_action_sequence


EXAMPLES = Path(__file__).parents[1] / "examples"
PROBLEM_PATH = EXAMPLES / "all_actions_recipe.json"
ACTIONS_PATH = EXAMPLES / "all_actions_actions.json"


def test_validate_action_sequence_connects_problem_and_validator() -> None:
    report = validate_action_sequence(PROBLEM_PATH, ACTIONS_PATH)

    assert report.ok
    assert report.checked_subjects == {"module": 3, "robot": 2, "wafer": 2}


def test_invalid_sequence_returns_combined_report(tmp_path: Path) -> None:
    actions = load_actions(ACTIONS_PATH)
    actions[1] = dict(actions[1], start=0.5, end=1.5)
    actions_path = tmp_path / "invalid_actions.json"
    actions_path.write_text(json.dumps(actions), encoding="utf-8")

    report = validate_action_sequence(PROBLEM_PATH, actions_path)

    constraint_ids = {issue.constraint_id for issue in report.issues}
    assert "robot.action_overlap" in constraint_ids
    assert "wafer.interval_overlap" in constraint_ids


def test_load_actions_rejects_non_list_json(tmp_path: Path) -> None:
    actions_path = tmp_path / "actions.json"
    actions_path.write_text('{"actions": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a list"):
        load_actions(actions_path)


def test_cli_returns_zero_for_valid_sequence(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([str(PROBLEM_PATH), str(ACTIONS_PATH)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Validation passed" in output
