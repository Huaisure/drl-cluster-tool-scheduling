from __future__ import annotations

import json
from pathlib import Path

from cluster_toolkit.run_engine import main


EXAMPLES = Path(__file__).parents[2] / "validator" / "examples"


def test_cli_lists_initial_semantic_actions(capsys) -> None:
    exit_code = main(
        [
            str(EXAMPLES / "all_actions_recipe.json"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["time"] == 0
    assert not payload["complete"]
    assert {action["action_type"] for action in payload["actions"]} == {
        "pick",
        "advance",
    }
