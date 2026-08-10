from __future__ import annotations

import json
from pathlib import Path

from run_generation import load_generation_config, main


EXAMPLES = Path(__file__).parents[2] / "validator" / "examples"


def test_config_precedence(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "small",
                "instance_count": 3,
                "seed": 4,
                "route_count": {"minimum": 2, "maximum": 2},
                "total_wafers": {"minimum": 4, "maximum": 4},
            }
        ),
        encoding="utf-8",
    )

    config = load_generation_config(
        config_path,
        profile="medium",
        instance_count=5,
        seed=6,
    )

    assert config.profile == "medium"
    assert config.instance_count == 5
    assert config.seed == 6
    assert config.route_count.minimum == 2
    assert config.total_wafers.maximum == 4
    assert config.process_time.minimum == 50


def test_cli_generates_manifest_and_instances(tmp_path: Path, capsys) -> None:
    output = tmp_path / "dataset"
    exit_code = main(
        [
            str(EXAMPLES / "all_actions_recipe.json"),
            str(output),
            "--profile",
            "small",
            "--count",
            "2",
            "--seed",
            "42",
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["instance_count"] == 2
    assert len(manifest["instances"]) == 2
    assert (output / "instance-00000.json").is_file()
    assert (output / "instance-00001.json").is_file()


def test_cli_reports_nonempty_output_as_input_error(tmp_path: Path, capsys) -> None:
    output = tmp_path / "dataset"
    output.mkdir()
    (output / "unrelated.txt").write_text("x", encoding="utf-8")

    exit_code = main(
        [
            str(EXAMPLES / "all_actions_recipe.json"),
            str(output),
            "--count",
            "1",
        ]
    )

    assert exit_code == 2
    assert "not empty" in capsys.readouterr().err
