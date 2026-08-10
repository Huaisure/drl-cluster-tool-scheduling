from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from cluster_toolkit.cluster_generator import (
    DatasetGenerator,
    GenerationConfig,
    IntRange,
    validate_generated_instance,
)
from cluster_toolkit.problem import ModuleType, parse_problem


EXAMPLES = Path(__file__).parents[2] / "validator" / "examples"
SMALL_TEMPLATE = EXAMPLES / "all_actions_recipe.json"
MULTI_LP_TEMPLATE = EXAMPLES / "naura_task1.json"


def _fixed_config(**overrides) -> GenerationConfig:
    raw = {
        "profile": "small",
        "instance_count": 2,
        "seed": 42,
        "route_count": {"minimum": 2, "maximum": 2},
        "total_wafers": {"minimum": 6, "maximum": 6},
        "route_steps": {"minimum": 3, "maximum": 5},
        "process_time": {"minimum": 20, "maximum": 30},
    }
    raw.update(overrides)
    return GenerationConfig.model_validate(raw)


def _parse_without_capacity_warnings(raw):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Explicit Module.capacity overrides the type-based default",
        )
        return parse_problem(raw)


def test_same_seed_produces_byte_identical_dataset(tmp_path: Path) -> None:
    config = _fixed_config()
    generator = DatasetGenerator.from_template(SMALL_TEMPLATE, config)
    left = tmp_path / "left"
    right = tmp_path / "right"

    generator.generate(left)
    generator.generate(right)

    left_files = {path.name: path.read_bytes() for path in left.iterdir()}
    right_files = {path.name: path.read_bytes() for path in right.iterdir()}
    assert left_files == right_files


def test_different_seed_changes_instance() -> None:
    left = DatasetGenerator.from_template(
        SMALL_TEMPLATE,
        _fixed_config(seed=1),
    ).generate_instance(0)
    right = DatasetGenerator.from_template(
        SMALL_TEMPLATE,
        _fixed_config(seed=2),
    ).generate_instance(0)

    assert left != right


def test_generated_topology_matches_template_and_parameters_are_in_range() -> None:
    config = _fixed_config(
        pick_time={"minimum": 2, "maximum": 2},
        place_time={"minimum": 3, "maximum": 3},
        travel_time={"minimum": 4, "maximum": 4},
        pump_time={"minimum": 6, "maximum": 6},
        vent_time={"minimum": 7, "maximum": 7},
        pm_capacity={"minimum": 2, "maximum": 2},
        ll_capacity={"minimum": 2, "maximum": 2},
    )
    generator = DatasetGenerator.from_template(SMALL_TEMPLATE, config)
    raw = generator.generate_instance(0)
    generated = _parse_without_capacity_warnings(raw)
    template = _parse_without_capacity_warnings(
        json.loads(SMALL_TEMPLATE.read_text(encoding="utf-8"))
    )

    assert set(generated.Modules) == set(template.Modules)
    assert set(generated.ClusterTool) == set(template.ClusterTool)
    for module_id in template.Modules:
        assert generated.Modules[module_id].type is template.Modules[module_id].type
    for robot_id in template.ClusterTool:
        generated_robot = generated.ClusterTool[robot_id]
        template_robot = template.ClusterTool[robot_id]
        assert generated_robot.module_ids == template_robot.module_ids
        assert generated_robot.arm_type is template_robot.arm_type
        assert generated_robot.pick_time == 2
        assert generated_robot.place_time == 3
        assert generated_robot.travel_times == 4
    assert generated.Modules["PM1"].capacity == 2
    assert generated.Modules["LLA"].capacity == 2
    assert generated.Modules["LLA"].load_lock.atmosphere_to_vacuum_time == 6
    assert generated.Modules["LLA"].load_lock.vacuum_to_atmosphere_time == 7
    assert generated.just_in_time is None
    assert generated.cleaning is None


def test_routes_use_non_lp_visits_and_pm_process_ranges() -> None:
    generator = DatasetGenerator.from_template(SMALL_TEMPLATE, _fixed_config())
    raw = generator.generate_instance(0)
    problem = _parse_without_capacity_warnings(raw)
    audit = validate_generated_instance(raw)

    for route_id, route in problem.routes.items():
        assert 3 <= len(route.visits) <= 5
        assert route_id in audit.routes
        for visit in route.visits:
            assert all(
                problem.Modules[module_id].type is not ModuleType.LP
                for module_id in visit.module_ids
            )
            selected_type = problem.Modules[visit.module_ids[0]].type
            if selected_type is ModuleType.PM:
                assert 20 <= visit.process_time <= 30
            else:
                assert visit.process_time == 0


def test_wafer_ranges_are_compact_strings_and_capacity_is_sufficient() -> None:
    raw = DatasetGenerator.from_template(SMALL_TEMPLATE, _fixed_config()).generate_instance(0)

    assert all(
        isinstance(wafer["wafer_index"], str)
        for wafer in raw["initial_state"]["wafers"]
    )
    lp_counts: dict[str, int] = {}
    for wafer in raw["initial_state"]["wafers"]:
        expression = wafer["wafer_index"]
        count = 1 if "-" not in expression else int(expression.split("-")[1]) + 1
        lp_id = wafer["location"]["module_id"]
        lp_counts[lp_id] = lp_counts.get(lp_id, 0) + count
    for lp_id, occupied in lp_counts.items():
        assert raw["Modules"][lp_id]["capacity"] >= occupied


def test_multiple_lps_are_used_round_robin() -> None:
    config = _fixed_config(
        route_count={"minimum": 3, "maximum": 3},
        total_wafers={"minimum": 6, "maximum": 6},
        route_steps={"minimum": 3, "maximum": 7},
    )
    raw = DatasetGenerator.from_template(MULTI_LP_TEMPLATE, config).generate_instance(0)

    used_lps = {
        wafer["location"]["module_id"]
        for wafer in raw["initial_state"]["wafers"]
    }
    assert used_lps == {"LP1", "LP2", "LP3"}


def test_candidate_modules_preserve_a_structural_witness() -> None:
    config = _fixed_config(
        route_count={"minimum": 3, "maximum": 3},
        total_wafers={"minimum": 6, "maximum": 6},
        route_steps={"minimum": 3, "maximum": 7},
        candidate_probability=1,
        max_candidates=3,
    )
    raw = DatasetGenerator.from_template(MULTI_LP_TEMPLATE, config).generate_instance(3)
    audit = validate_generated_instance(raw)

    assert audit.candidate_step_count > 0
    assert all(
        route_audit.witnesses
        and all(witness.witness_path for witness in route_audit.witnesses)
        for route_audit in audit.routes.values()
    )


@pytest.mark.parametrize("profile", ["small", "medium", "large"])
def test_profiles_are_structurally_valid_for_seeds_0_through_99(profile: str) -> None:
    for seed in range(100):
        config = GenerationConfig(profile=profile, instance_count=1, seed=seed)
        raw = DatasetGenerator.from_template(SMALL_TEMPLATE, config).generate_instance(0)
        audit = validate_generated_instance(raw)
        assert 1 <= audit.route_count <= 6
        assert 10 <= audit.wafer_count <= 75
        assert audit.wafer_count >= audit.route_count


def test_impossible_topology_is_rejected(tmp_path: Path) -> None:
    template = {
        "Modules": {"LP": {"type": "LP"}, "PM": {"type": "PM"}},
        "ClusterTool": {
            "TM": {
                "module_ids": ["LP"],
                "arm_type": "single_arm",
                "travel_times": 1,
                "place_time": 1,
                "pick_time": 1,
            }
        },
    }
    template_path = tmp_path / "impossible.json"
    template_path.write_text(json.dumps(template), encoding="utf-8")

    with pytest.raises(ValueError, match="no LP-to-LP path"):
        DatasetGenerator.from_template(template_path, _fixed_config())


def test_incompatible_route_step_range_is_rejected() -> None:
    config = _fixed_config(route_steps={"minimum": 2, "maximum": 2})

    with pytest.raises(ValueError, match="route_steps"):
        DatasetGenerator.from_template(SMALL_TEMPLATE, config)


def test_max_lp_capacity_rejects_oversized_workload() -> None:
    config = _fixed_config(
        total_wafers={"minimum": 20, "maximum": 20},
        lp_spare_percent={"minimum": 0, "maximum": 0},
        max_lp_capacity=10,
    )
    generator = DatasetGenerator.from_template(SMALL_TEMPLATE, config)

    with pytest.raises(ValueError, match="max_lp_capacity"):
        generator.generate_instance(0)


def test_nonempty_output_requires_overwrite_and_owned_manifest(tmp_path: Path) -> None:
    generator = DatasetGenerator.from_template(SMALL_TEMPLATE, _fixed_config(instance_count=1))
    output = tmp_path / "dataset"
    generator.generate(output)

    with pytest.raises(ValueError, match="not empty"):
        generator.generate(output)

    unrelated = output / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    generator.generate(output, overwrite=True)
    assert unrelated.read_text(encoding="utf-8") == "keep"

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "file.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="owned manifest"):
        generator.generate(foreign, overwrite=True)


def test_overwrite_does_not_replace_unowned_future_instance_name(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    DatasetGenerator.from_template(
        SMALL_TEMPLATE,
        _fixed_config(instance_count=1),
    ).generate(output)
    collision = output / "instance-00001.json"
    collision.write_text("user", encoding="utf-8")

    larger = DatasetGenerator.from_template(
        SMALL_TEMPLATE,
        _fixed_config(instance_count=2),
    )
    with pytest.raises(ValueError, match="unowned output files"):
        larger.generate(output, overwrite=True)
    assert collision.read_text(encoding="utf-8") == "user"


def test_manifest_contains_audit_and_template_digest(tmp_path: Path) -> None:
    generator = DatasetGenerator.from_template(SMALL_TEMPLATE, _fixed_config(instance_count=1))
    manifest = generator.generate(tmp_path / "dataset")

    assert manifest.generator["name"] == "cluster_generator"
    assert len(manifest.template["sha256"]) == 64
    assert len(manifest.instances) == 1
    assert manifest.instances[0].audit.routes


def test_config_rejects_ranges_that_cannot_give_each_route_a_wafer() -> None:
    with pytest.raises(ValueError, match="every Route"):
        GenerationConfig(
            route_count=IntRange(minimum=5, maximum=10),
            total_wafers=IntRange(minimum=5, maximum=9),
        )


@pytest.mark.parametrize(
    ("profile", "process_bounds"),
    [
        ("small", (50, 100)),
        ("medium", (50, 300)),
        ("large", (100, 600)),
    ],
)
def test_profile_defaults_use_realistic_equipment_time_scale(
    profile: str,
    process_bounds: tuple[int, int],
) -> None:
    config = GenerationConfig(profile=profile)

    assert (config.process_time.minimum, config.process_time.maximum) == process_bounds
    assert (config.pick_time.minimum, config.pick_time.maximum) == (1, 10)
    assert (config.place_time.minimum, config.place_time.maximum) == (1, 10)
    assert (config.travel_time.minimum, config.travel_time.maximum) == (1, 10)
