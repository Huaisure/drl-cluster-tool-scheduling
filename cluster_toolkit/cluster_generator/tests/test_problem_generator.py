from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy

import pytest

from cluster_toolkit.cluster_generator import ProblemGenerationConfig, ProblemGenerator
from cluster_toolkit.problem import ModuleLocation, ModuleType, parse_problem
from cluster_toolkit.validator import ValidatorSuite


def _forced_topology(family: str, **overrides) -> ProblemGenerationConfig:
    weights = {
        split: {
            name: float(name == family)
            for name in ("simple", "single_vacuum", "dual_vacuum")
        }
        for split in ("train", "validation", "test")
    }
    return ProblemGenerationConfig(
        topology_weights_by_split=weights,
        **overrides,
    )


@pytest.mark.parametrize(
    ("difficulty", "wafer_range"),
    [
        ("easy", (10, 25)),
        ("medium", (26, 50)),
        ("hard", (51, 75)),
        ("edge", (51, 75)),
    ],
)
def test_curriculum_profiles_generate_replayable_domain_problems(
    difficulty: str,
    wafer_range: tuple[int, int],
) -> None:
    generator = ProblemGenerator()
    for seed in range(3):
        benchmark = generator.generate(seed, difficulty=difficulty, split="train")
        metadata = benchmark.metadata
        assert metadata["io_count"] == 1
        assert wafer_range[0] <= metadata["wafer_count"] <= wafer_range[1]
        assert 1 <= metadata["route_count"] <= 6
        assert 1 <= metadata["max_process_steps"] <= 8
        assert metadata["lower_bound"] <= metadata["reference_makespan"]
        assert metadata["feasible"] is True
        assert metadata["validator_result"] is True
        assert ValidatorSuite(benchmark.problem).validate(benchmark.actions).ok


def test_same_seed_is_byte_reproducible_and_different_seed_changes_instance() -> None:
    generator = ProblemGenerator()
    first = generator.generate(42, difficulty="medium", split="train")
    repeated = generator.generate(42, difficulty="medium", split="train")
    different = generator.generate(43, difficulty="medium", split="train")

    encoded = lambda benchmark: json.dumps(  # noqa: E731
        {"problem": benchmark.raw_problem, "actions": benchmark.actions},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert encoded(first) == encoded(repeated)
    assert encoded(first) != encoded(different)


def test_simple_topology_is_virtual_io_unified_tm_and_pm_only() -> None:
    benchmark = ProblemGenerator(
        config=_forced_topology("simple")
    ).generate(9, difficulty="easy", split="train")
    problem = benchmark.problem

    assert problem.schema_version == 2
    assert set(problem.ClusterTool) == {"TM1"}
    assert 3 <= benchmark.metadata["pm_count"] <= 6
    assert benchmark.metadata["ll_count"] == 0
    assert benchmark.metadata["buffer_count"] == 0
    assert all(
        problem.Modules[visit.module_ids[0]].type is ModuleType.PM
        for route in problem.routes.values()
        for visit in route.visits
    )


def test_single_vacuum_topology_has_fixed_atmosphere_skeleton() -> None:
    benchmark = ProblemGenerator(
        config=_forced_topology("single_vacuum")
    ).generate(11, difficulty="easy", split="train")
    problem = benchmark.problem
    ll_ids = {
        module_id
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.LL
    }
    pm_ids = {
        module_id
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.PM
    }

    assert set(problem.ClusterTool) == {"ATM1", "VTM1"}
    assert 1 <= len(ll_ids) <= 2
    assert set(problem.ClusterTool["ATM1"].module_ids) == {"IO1", "AL1", *ll_ids}
    assert set(problem.ClusterTool["VTM1"].module_ids) == {*ll_ids, *pm_ids}
    assert 3 <= len(pm_ids) <= 6
    for route in problem.routes.values():
        assert problem.Modules[route.visits[0].module_ids[0]].type is ModuleType.AL
        assert set(route.visits[1].module_ids) == ll_ids
        assert set(route.visits[-1].module_ids) == ll_ids


def test_dual_vacuum_topology_uses_buffer_handoff_and_disjoint_processes() -> None:
    benchmark = ProblemGenerator(
        config=_forced_topology("dual_vacuum")
    ).generate(7, difficulty="easy", split="train")
    problem = benchmark.problem
    buffers = {
        module_id
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.BUFFER
    }
    unit_pms = {
        unit: {
            module_id
            for module_id in problem.ClusterTool[f"VTM{unit}"].module_ids
            if problem.Modules[module_id].type is ModuleType.PM
        }
        for unit in (1, 2)
    }
    unit_processes = {
        unit: {
            process_id
            for pm_id in pm_ids
            for process_id in problem.Modules[pm_id].process_ids
        }
        for unit, pm_ids in unit_pms.items()
    }

    assert set(problem.ClusterTool) == {"ATM1", "VTM1", "VTM2"}
    assert 1 <= len(buffers) <= 2
    assert buffers <= set(problem.ClusterTool["VTM1"].module_ids)
    assert buffers <= set(problem.ClusterTool["VTM2"].module_ids)
    assert all(3 <= len(pm_ids) <= 6 for pm_ids in unit_pms.values())
    assert unit_processes[1].isdisjoint(unit_processes[2])
    assert any(
        set(visit.module_ids) <= unit_pms[2]
        for route in problem.routes.values()
        for visit in route.visits
        if problem.Modules[visit.module_ids[0]].type is ModuleType.PM
    )


def test_every_wafer_starts_at_and_returns_to_virtual_io() -> None:
    benchmark = ProblemGenerator().generate(12, difficulty="hard", split="test")
    initial = {
        wafer.wafer_key: wafer.location.module_id
        for wafer in benchmark.problem.initial_state.wafers
        if isinstance(wafer.location, ModuleLocation)
    }
    final_places = {
        (str(action["route_id"]), int(action["wafer_index"])): str(action["module_id"])
        for action in benchmark.actions
        if action["action_type"] == "place"
        and int(action["step_index"])
        == len(benchmark.problem.routes[str(action["route_id"])].visits) + 1
    }

    assert set(initial.values()) == {"IO1"}
    assert all(final_places[key] == "IO1" for key in initial)
    assert all(wafer.priority >= 0 for wafer in benchmark.problem.initial_state.wafers)
    assert all(
        all(module_id != "IO1" for module_id in visit.module_ids)
        for route in benchmark.problem.routes.values()
        for visit in route.visits
    )


def test_capabilities_are_explicit_and_candidates_are_fully_derived() -> None:
    benchmark = ProblemGenerator(
        config=_forced_topology("dual_vacuum")
    ).generate(31, difficulty="medium", split="train")
    problem = benchmark.problem
    used_pms: set[str] = set()

    for module_id, module in problem.Modules.items():
        if module.type is ModuleType.PM:
            assert len(module.process_ids) in {2, 3}
    for route in problem.routes.values():
        for visit in route.visits:
            module_type = problem.Modules[visit.module_ids[0]].type
            if module_type is not ModuleType.PM:
                assert visit.process_id is None
                continue
            expected = {
                module_id
                for module_id, module in problem.Modules.items()
                if module.type is ModuleType.PM
                and visit.process_id in module.process_ids
            }
            assert set(visit.module_ids) == expected
            used_pms.update(expected)
    assert used_pms == {
        module_id
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.PM
    }


def test_recipe_reentry_and_cooling_obey_domain_bounds() -> None:
    generator = ProblemGenerator(config=_forced_topology("dual_vacuum"))
    for seed in range(12):
        problem = generator.generate(seed, difficulty="medium").problem
        for route in problem.routes.values():
            pm_processes = [
                visit.process_id
                for visit in route.visits
                if problem.Modules[visit.module_ids[0]].type is ModuleType.PM
            ]
            repeated = {
                process_id
                for process_id, count in Counter(pm_processes).items()
                if count > 1
            }
            assert len(repeated) <= 1
            if repeated:
                process_id = next(iter(repeated))
                positions = [
                    index for index, value in enumerate(pm_processes) if value == process_id
                ]
                assert 2 <= len(positions) <= 4
                assert all(right - left > 1 for left, right in zip(positions, positions[1:]))
            cooling = [
                index
                for index, visit in enumerate(route.visits)
                if problem.Modules[visit.module_ids[0]].type is ModuleType.BUFFER
                and float(visit.process_time or 0) > 0
            ]
            assert len(cooling) <= 3
            assert all(right - left > 1 for left, right in zip(cooling, cooling[1:]))


def test_time_generation_uses_agreed_ranges_and_process_anchors() -> None:
    benchmark = ProblemGenerator(
        config=_forced_topology("dual_vacuum")
    ).generate(19, difficulty="medium")
    problem = benchmark.problem
    process_times: dict[str, set[float]] = defaultdict(set)

    for robot in problem.ClusterTool.values():
        assert 8 <= robot.pick_time <= 15
        assert 8 <= robot.place_time <= 15
        assert 8 <= robot.travel_times <= 15
    for module in problem.Modules.values():
        if module.load_lock is not None:
            assert 10 <= module.load_lock.atmosphere_to_vacuum_time <= 30
            assert 10 <= module.load_lock.vacuum_to_atmosphere_time <= 30
    for route in problem.routes.values():
        assert 10 <= float(route.visits[0].process_time or 0) <= 20
        for visit in route.visits:
            module_type = problem.Modules[visit.module_ids[0]].type
            if module_type is ModuleType.PM:
                value = float(visit.process_time or 0)
                assert any(0.9 * anchor <= value <= 1.1 * anchor for anchor in (30, 50, 300, 600))
                assert visit.process_id is not None
                process_times[visit.process_id].add(value)
            elif module_type is ModuleType.BUFFER:
                assert visit.process_time == 0 or 30 <= float(visit.process_time) <= 60
            elif module_type is ModuleType.LL:
                assert visit.process_time == 0
    assert all(len(values) == 1 for values in process_times.values())


@pytest.mark.parametrize("mode", ["none", "recipe", "wave"])
def test_priority_modes_materialize_only_final_per_wafer_values(mode: str) -> None:
    priority_weights = {name: float(name == mode) for name in ("none", "recipe", "wave")}
    config = _forced_topology("single_vacuum", priority_mode_weights=priority_weights)
    benchmark = ProblemGenerator(config=config).generate(23, difficulty="medium")
    by_route: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for wafer in benchmark.problem.initial_state.wafers:
        by_route[wafer.route_id].append((wafer.wafer_index, wafer.priority))

    assert "priority_mode" not in json.dumps(benchmark.raw_problem)
    if mode == "none":
        assert {priority for values in by_route.values() for _, priority in values} == {0}
    elif mode == "recipe":
        assert all(len({priority for _, priority in values}) == 1 for values in by_route.values())
    else:
        assert all(
            [priority for _, priority in sorted(values)]
            == sorted(priority for _, priority in values)
            for values in by_route.values()
        )


def test_config_defaults_encode_agreed_distributions() -> None:
    config = ProblemGenerationConfig()
    assert config.topology_weights_by_split["train"] == {
        "simple": 0.10,
        "single_vacuum": 0.60,
        "dual_vacuum": 0.30,
    }
    assert config.topology_weights_by_split["test"]["dual_vacuum"] == 0.70
    assert config.ll_count_weights == {1: 0.30, 2: 0.70}
    assert config.pm_count_weights == {3: 0.15, 4: 0.35, 5: 0.35, 6: 0.15}
    assert config.reentry_weights == {"none": 0.65, "twice": 0.30, "deep": 0.05}
    assert config.cooling_probability == 0.25


def test_structural_signature_buckets_keep_splits_disjoint() -> None:
    generator = ProblemGenerator()
    buckets: dict[str, set[int]] = {}
    signatures: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        generated = [
            generator.generate(seed, difficulty="easy", split=split)
            for seed in range(4)
        ]
        buckets[split] = {item.metadata["signature_bucket"] for item in generated}
        signatures[split] = {item.metadata["structural_signature"] for item in generated}

    assert buckets["train"] <= set(range(0, 70))
    assert buckets["validation"] <= set(range(70, 85))
    assert buckets["test"] <= set(range(85, 100))
    assert signatures["train"].isdisjoint(signatures["validation"] | signatures["test"])
    assert signatures["validation"].isdisjoint(signatures["test"])


def test_validator_rejects_returning_a_wafer_to_a_non_source_module() -> None:
    benchmark = ProblemGenerator().generate(4, difficulty="hard", split="test")
    problem = benchmark.problem
    actions = [deepcopy(action) for action in benchmark.actions]
    target_index = next(
        index
        for index, action in enumerate(actions)
        if action["action_type"] == "place"
        and int(action["step_index"])
        == len(problem.routes[str(action["route_id"])].visits) + 1
    )
    bad_module = next(
        module_id
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.PM
    )
    actions[target_index]["module_id"] = bad_module

    report = ValidatorSuite(problem).validate(actions)
    assert not report.ok


def test_schema_version_1_json_remains_compatible() -> None:
    legacy = {
        "Modules": {
            "IO1": {"type": "IO", "capacity": 1},
            "PM1": {"type": "PM", "capacity": 1},
        },
        "ClusterTool": {
            "TM1": {
                "module_ids": ["IO1", "PM1"],
                "arm_type": "single_arm",
                "travel_times": 10,
                "pick_time": 10,
                "place_time": 10,
            }
        },
        "routes": {"R0": [{"module_id": "PM1", "process_time": 30}]},
        "initial_state": {
            "robots": {"TM1": {"position_module_id": None}},
            "wafers": [
                {
                    "route_id": "R0",
                    "wafer_index": "0",
                    "priority": 0,
                    "location": {"kind": "module", "module_id": "IO1"},
                }
            ],
        },
    }
    parsed = parse_problem(legacy)
    assert parsed.schema_version == 1
    assert parsed.Modules["PM1"].process_ids == ()
    assert parsed.routes["R0"].visits[0].process_id is None


def test_schema_version_2_rejects_capability_candidate_mismatch() -> None:
    raw = ProblemGenerator(config=_forced_topology("simple")).generate(
        5, difficulty="easy"
    ).raw_problem
    visit = next(iter(raw["routes"].values()))[0]
    if len(visit["module_ids"]) > 1:
        visit["module_ids"] = visit["module_ids"][:-1]
    else:
        replacement = next(
            module_id
            for module_id, module in raw["Modules"].items()
            if module["type"] == "PM" and module_id not in visit["module_ids"]
        )
        visit["module_ids"] = [replacement]

    with pytest.raises(ValueError, match="must equal all configured PM Modules"):
        parse_problem(raw)


def test_problem_schema_rejects_legacy_return_lp_id_for_virtual_io() -> None:
    raw = ProblemGenerator().generate(3, difficulty="easy").raw_problem
    raw["initial_state"]["wafers"][0]["return_lp_id"] = "PM1"
    with pytest.warns(UserWarning, match="Explicit Module.capacity"), pytest.raises(
        ValueError,
        match="must not define return_lp_id in a virtual IO problem",
    ):
        parse_problem(raw)
