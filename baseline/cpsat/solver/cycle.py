import argparse
from pathlib import Path
import sys
from typing import List, Dict, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import (
    load_recipe,
    estimate_cycle_upper_bound,
    _build_recipe_process_times,
    build_solution_result,
    _unscale_schedule,
    build_cycle_nodes,
)
from validate_cycle import validate
from utils.constraints import (
    _enforce_cycle_factory_cycles,
    _enforce_cycle_arm_and_robot_cycles,
    _enforce_cycle_parallel_factory_choice,
    _enforce_cycle_same_truck_unload_load,
    _create_action_time_vars,
    _create_action_active_vars,
    add_cycle_priority,
    _enforce_process_unload_no_overlap_cyclic,
)
import json
import time
import os
from ortools.sat.python import cp_model

def cycle(
    raw,
    requirements,
    time_limit_s=600,
    random_seed=42,
    interleave_search=True,
    num_search_workers=1,
):
    cpu_start = time.time()

    trucks = raw["trucks"]
    factories = raw["factories"]
    recipes = raw["recipes"]

    nodes = build_cycle_nodes(requirements, recipes, trucks)
    n_nodes = len(nodes)

    upper_bound = estimate_cycle_upper_bound(requirements, recipes, trucks)

    model = cp_model.CpModel()
    makespan = model.NewIntVar(0, upper_bound, "makespan")
    action_active = _create_action_active_vars(model, nodes)

    action_times = _create_action_time_vars(model, nodes, upper_bound, action_active, makespan)

    process_times, residency_times = _build_recipe_process_times(recipes)

    _enforce_cycle_parallel_factory_choice(model, nodes, action_active)
    _enforce_cycle_same_truck_unload_load(model, nodes, action_active)
    _enforce_cycle_factory_cycles(model, nodes, makespan, factories, action_times, action_active, process_times, residency_times)
    _enforce_cycle_arm_and_robot_cycles(model, nodes, trucks, action_times, action_active, makespan)
    add_cycle_priority(model, nodes, action_times, action_active, makespan)

    _enforce_process_unload_no_overlap_cyclic(
        model,
        nodes,
        factories,
        action_times,
        action_active,
        makespan,
        upper_bound,
        raw.get("interfere_time", 0),
    )

    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = random_seed
    solver.parameters.interleave_search = interleave_search
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_search_workers

    status = solver.Solve(model)
    elapsed = time.time() - cpu_start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("No solution found. Status:", solver.StatusName(status))
        print("Elapsed time (s):", f"{elapsed:.3f}")
        return None
    result = build_solution_result(
        solver=solver,
        status=status,
        objective_name="makespan",
        objective_value=solver.Value(makespan),
        cpu_time=elapsed,
        nodes=nodes,
        action_times=action_times,
        action_active=action_active,
    )
    return result

def _build_cycle_save_path(save_dir: str, recipe_path: str, requirements: dict) -> str:
    """根据 recipe 名称、requirements 自动构建 cycle 保存路径"""
    recipe_name = Path(recipe_path).stem
    parts = []
    for k in sorted(requirements.keys()):
        parts.append(f"{k}_{requirements[k]}")
    filename = "_".join(parts) + "_cycle.json"
    return str(Path(save_dir) / recipe_name / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-path", default=r"recipes\d_travel.json")
    parser.add_argument("--requirements", default={"C1": 2, "C2": 2, "A5": 1, "A6": 1}, type=str,
                    help="配方需求，格式 k1:v1,k2:v2")
    parser.add_argument("--save-dir", default="results",
                    help="保存根目录，将在该目录下创建 recipe 同名子文件夹")
    parser.add_argument("--save-path", default=None,
                    help="可选，指定完整保存路径（覆盖自动构建）")
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--interleave-search", type=bool, default=True)
    parser.add_argument("--num-search-workers", type=int, default=16)
    args = parser.parse_args()

    recipe_path = args.recipe_path
    time_scale = args.time_scale
    if isinstance(args.requirements, str):
        reqs = {}
        for pair in args.requirements.split(','):
            k, v = pair.split(':')
            reqs[k.strip()] = int(v.strip())
        args.requirements = reqs
    requirements = args.requirements
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = _build_cycle_save_path(args.save_dir, recipe_path, requirements)
    time_limit_s = args.time_limit_s
    raw = load_recipe(recipe_path, time_scale)
    result = cycle(
        raw,
        requirements,
        time_limit_s=time_limit_s,
        random_seed=args.random_seed,
        interleave_search=args.interleave_search,
        num_search_workers=args.num_search_workers,
    )
    result = _unscale_schedule(result, time_scale, "makespan")

    if result is not None:
        validation_report = validate(raw, result, requirements)
        result["validation"] = validation_report
        result["fluc_strategy"] = "max"
        print(f"validation valid={validation_report['valid']}, error_count={validation_report['error_count']}")
        # Ensure target directory exists before writing result
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        print("result has saved to:", save_path)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()