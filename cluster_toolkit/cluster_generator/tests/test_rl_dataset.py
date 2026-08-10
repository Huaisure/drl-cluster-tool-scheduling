from __future__ import annotations

import json
from pathlib import Path

import pytest

from cluster_toolkit.cluster_generator import RLDatasetGenerator, RLGenerationConfig
from cluster_toolkit.run_rl_generation import main


def test_rl_dataset_writes_problems_references_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    config = RLGenerationConfig(
        instance_count=3,
        seed=42,
        split="validation",
        difficulty="easy",
    )
    manifest = RLDatasetGenerator(config).generate(output)

    assert len(manifest["instances"]) == 3
    assert manifest["generator"]["mode"] == "ppo"
    assert manifest["problem_generation_config"]["cooling_probability"] == 0.25
    assert len(list(output.glob("validation-*.json"))) == 6
    entry = manifest["instances"][0]
    problem = json.loads((output / entry["problem_file"]).read_text(encoding="utf-8"))
    reference = json.loads((output / entry["actions_file"]).read_text(encoding="utf-8"))
    assert problem["_meta"]["generator"]["feasible"] is True
    assert reference["validator_result"] is True
    assert reference["makespan"] >= reference["lower_bound"]


def test_rl_dataset_is_byte_reproducible(tmp_path: Path) -> None:
    config = RLGenerationConfig(instance_count=4, seed=7)
    left = tmp_path / "left"
    right = tmp_path / "right"
    RLDatasetGenerator(config).generate(left)
    RLDatasetGenerator(config).generate(right)

    assert {
        path.name: path.read_bytes() for path in left.iterdir()
    } == {
        path.name: path.read_bytes() for path in right.iterdir()
    }


def test_train_seed_only_manifest_has_no_instance_files(tmp_path: Path) -> None:
    config = RLGenerationConfig(
        instance_count=2,
        seed=1,
        split="train",
        materialize_problems=False,
        include_reference_actions=False,
    )
    manifest = RLDatasetGenerator(config).generate(tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == ["manifest.json"]
    assert all(entry["problem_file"] is None for entry in manifest["instances"])


def test_validation_and_test_cannot_be_seed_only() -> None:
    with pytest.raises(ValueError, match="must materialize"):
        RLGenerationConfig(
            split="test",
            materialize_problems=False,
            include_reference_actions=False,
        )


def test_overwrite_removes_only_owned_files(tmp_path: Path) -> None:
    config = RLGenerationConfig(instance_count=2, difficulty="easy")
    generator = RLDatasetGenerator(config)
    generator.generate(tmp_path)
    note = tmp_path / "keep.txt"
    note.write_text("user", encoding="utf-8")

    generator.generate(tmp_path, overwrite=True)

    assert note.read_text(encoding="utf-8") == "user"


def test_nonempty_unowned_output_is_protected(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("user", encoding="utf-8")
    with pytest.raises(ValueError, match="not an owned"):
        RLDatasetGenerator(RLGenerationConfig(instance_count=1)).generate(
            tmp_path,
            overwrite=True,
        )


def test_rl_overwrite_does_not_replace_unowned_future_instance_name(tmp_path: Path) -> None:
    RLDatasetGenerator(
        RLGenerationConfig(instance_count=1, difficulty="easy")
    ).generate(tmp_path)
    collision = tmp_path / "train-00001.json"
    collision.write_text("user", encoding="utf-8")

    with pytest.raises(ValueError, match="unowned output files"):
        RLDatasetGenerator(
            RLGenerationConfig(instance_count=2, difficulty="easy")
        ).generate(tmp_path, overwrite=True)
    assert collision.read_text(encoding="utf-8") == "user"


def test_rl_cli_generates_fixed_test_set(tmp_path: Path, capsys) -> None:
    output = tmp_path / "test-set"
    exit_code = main(
        [
            str(output),
            "--split",
            "test",
            "--difficulty",
            "easy",
            "--count",
            "2",
            "--seed",
            "10",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["instance_count"] == 2
    assert (output / "manifest.json").is_file()


def test_rl_cli_rejects_seed_only_test_set(tmp_path: Path, capsys) -> None:
    exit_code = main([str(tmp_path / "test"), "--split", "test", "--seed-only"])

    assert exit_code == 2
    assert "must materialize" in capsys.readouterr().err
