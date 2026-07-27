from __future__ import annotations

from examples.run_scenarios import run_all


def test_simple_scenarios_finish_with_valid_schedules() -> None:
    results = run_all(seed=0)

    assert len(results) == 6
    assert {result.scenario for result in results} == {
        "mixed_3pm_20w",
        "serial_2pm_10w",
        "serial_3pm_10w",
    }
    assert {
        result.policy for result in results
    } == {"first_legal", "untrained_network_greedy"}
    assert all(result.valid for result in results)
    assert all(result.total_reward == -result.makespan for result in results)
