from __future__ import annotations

from cluster_toolkit.problem import ClusterProblem, parse_problem


def load_lock_problem() -> ClusterProblem:
    """Small atmosphere-to-vacuum problem used by LL observation tests."""

    return parse_problem(
        {
            "Modules": {
                "IO1": {"type": "IO"},
                "LL1": {
                    "type": "LL",
                    "load_lock": {
                        "initial_state": "atmosphere",
                        "atmosphere_to_vacuum_time": 5,
                        "vacuum_to_atmosphere_time": 7,
                        "tm_required_states": {
                            "ATM": "atmosphere",
                            "VTM": "vacuum",
                        },
                    },
                },
                "PM1": {"type": "PM"},
            },
            "ClusterTool": {
                "ATM": {
                    "module_ids": ["IO1", "LL1"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                },
                "VTM": {
                    "module_ids": ["LL1", "PM1"],
                    "arm_type": "single_arm",
                    "travel_times": 0,
                    "pick_time": 1,
                    "place_time": 1,
                },
            },
            "routes": {
                "A": [
                    {"module_id": "LL1", "process_time": 0},
                    {"module_id": "PM1", "process_time": 0},
                    # Return through the Load Lock; VTM cannot hand the wafer
                    # directly to ATM or reach IO1.
                    {"module_id": "LL1", "process_time": 0},
                ]
            },
            "initial_state": {
                "wafers": [
                    {
                        "route_id": "A",
                        "wafer_index": "0",
                        "priority": 0,
                        "location": {"kind": "module", "module_id": "IO1"},
                    }
                ]
            },
        }
    )
