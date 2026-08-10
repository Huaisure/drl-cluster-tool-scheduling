from __future__ import annotations

from examples.run_scenarios import run_all


def test_simple_scenarios_finish_with_valid_schedules() -> None:
    results = run_all(seed=0)

    assert len(results) == 6
    assert {result.scenario for result in results} == {
        "long_route_1w",
        "mixed_3pm_20w",
        "mixed_5pm_24w",
    }
    assert {
        result.policy for result in results
    } == {"serial_reference", "untrained_network_greedy"}
    assert all(result.total_reward == -result.makespan for result in results)

    references = {
        result.scenario: result
        for result in results
        if result.policy == "serial_reference"
    }
    assert all(result.valid for result in references.values())
    assert references["long_route_1w"].wafer_count == 1
    assert references["long_route_1w"].pm_count == 4
    assert references["long_route_1w"].action_count == 18
    assert references["mixed_5pm_24w"].wafer_count == 24
    assert references["mixed_5pm_24w"].route_count == 3
    assert references["mixed_5pm_24w"].pm_count == 5
    assert references["mixed_5pm_24w"].action_count == 192
