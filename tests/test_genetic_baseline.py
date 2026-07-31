from __future__ import annotations

import json

import numpy as np
import pytest

from baseline import solve
from baseline.genetic import main
from cluster_rl.cluster_env import ClusterEnv
from problem import load_problem
from validator import ValidatorSuite

from .test_cluster_env import _problem, _raw_problem


def _crossing_problem():
    return _problem(
        routes={
            "A": [
                {"module_id": "PM1", "process_time": 0},
                {"module_id": "PM2", "process_time": 0},
            ],
            "B": [
                {"module_id": "PM2", "process_time": 0},
                {"module_id": "PM1", "process_time": 0},
            ],
        },
        wafer_routes=("A", "B"),
    )


def _first_legal_makespan(problem) -> float:
    env = ClusterEnv(problem)
    observation, _ = env.reset()
    while True:
        action = int(np.flatnonzero(observation["action_mask"])[0])
        observation, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            assert info["is_success"]
            return float(info["time"])


def test_genetic_algorithm_finds_reproducible_valid_schedule() -> None:
    problem = _crossing_problem()

    first = solve(
        problem,
        population_size=24,
        generations=40,
        patience=10,
        seed=5,
    )
    second = solve(
        problem,
        population_size=24,
        generations=40,
        patience=10,
        seed=5,
    )

    assert first.makespan == second.makespan
    assert [dict(action) for action in first.actions] == [
        dict(action) for action in second.actions
    ]
    assert first.makespan <= _first_legal_makespan(problem)
    assert ValidatorSuite(problem).validate(first.actions).ok
    with pytest.raises(TypeError):
        first.actions[0]["start"] = 0


def test_early_stopping_and_evaluation_count() -> None:
    problem = _problem(
        routes={"A": [{"module_id": "PM1", "process_time": 0}]}
    )

    result = solve(
        problem,
        population_size=4,
        generations=20,
        patience=3,
        seed=0,
    )

    assert result.generations_run == 4
    assert result.evaluations == 16
    assert result.seed == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"population_size": 3}, "population_size"),
        ({"generations": 0}, "generations"),
        ({"patience": 0}, "patience"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_invalid_genetic_parameters_fail_fast(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve(_problem(), **kwargs)


def test_cli_writes_validator_compatible_actions_and_summary(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    problem_path = tmp_path / "problem.json"
    output_path = tmp_path / "actions.json"
    problem_path.write_text(
        json.dumps(_raw_problem(wafer_routes=("A", "A"))),
        encoding="utf-8",
    )

    exit_code = main(
        [
            str(problem_path),
            "--output",
            str(output_path),
            "--population-size",
            "8",
            "--generations",
            "12",
            "--patience",
            "3",
            "--seed",
            "4",
        ]
    )

    actions = json.loads(output_path.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)
    problem = load_problem(problem_path)
    assert exit_code == 0
    assert summary["makespan"] > 0
    assert summary["action_count"] == len(actions)
    assert summary["output"] == str(output_path)
    assert ValidatorSuite(problem).validate(actions).ok


def test_cli_rejects_overwriting_problem_file(tmp_path) -> None:
    problem_path = tmp_path / "problem.json"
    problem_path.write_text(json.dumps(_raw_problem()), encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        main([str(problem_path), "--output", str(problem_path)])

    assert error.value.code == 2
