import argparse
from pathlib import Path
import sys
from typing import List, Dict, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import (
    insert_move_entries,
    load_recipe,
    build_nodes,
    estimate_upper_bound,
    _build_recipe_process_times,
    build_solution_result,
    _unscale_schedule,
    _build_goods
)
from utils.constraints import (
    _create_action_active_vars,
    _enforce_parallel_factory_choice,
    _enforce_load_requires_unload,
    _enforce_same_truck_unload_load,
    _enforce_goods_priority_for_actions,
    _enforce_lp_pickup_order,
    _enforce_no_overlap_completion_pm,
    _enforce_truck_idle_for_fluc_pickup,
    _enforce_process_unload_no_overlap,
    _enforce_factory_cycles,
    _create_action_time_vars,
    _enforce_arm_and_robot_cycles
)
from validate import validate
import json
import os
import time
from ortools.sat.python import cp_model


def _build_robot_contexts(trucks):
    contexts = {}
    for truck in trucks:
        truck_id = str(truck.get("id"))
        contexts[truck_id] = {
            "start_time": None,
            "end_time": None,
        }
    return contexts

def noncyclic(
    raw,
    counts,
    time_limit_s=600,
    random_seed=42,
    interleave_search=True,
    num_search_workers=1,
    order=None,
):
    cpu_start = time.time()

    trucks = raw["trucks"]
    factories = raw["factories"]
    recipes = raw["recipes"]

    goods = _build_goods(counts, order)

    nodes, boundry_actions, action_index = build_nodes(goods, recipes, trucks)
    n_nodes = len(nodes)

    upper_bound = estimate_upper_bound(goods, recipes, trucks)

    model = cp_model.CpModel()

    makespan = model.NewIntVar(0, upper_bound, "makespan")

    action_active = _create_action_active_vars(model, nodes)
    _enforce_parallel_factory_choice(model, action_index, action_active)
    _enforce_load_requires_unload(model, nodes, action_active)
    _enforce_same_truck_unload_load(model, nodes, action_active)
    action_times = _create_action_time_vars(model, nodes, upper_bound, action_active, makespan)
    _enforce_goods_priority_for_actions(
        model,
        nodes,
        action_active,
        action_times,
        goods,
    )
    _enforce_lp_pickup_order(model, nodes, action_times, action_active)
    _enforce_no_overlap_completion_pm(model, nodes, action_times, action_active, recipes)

    _enforce_truck_idle_for_fluc_pickup(model, nodes, action_times, action_active, recipes, trucks)

    _enforce_process_unload_no_overlap(
        model,
        nodes,
        factories,
        action_times,
        action_active,
        raw.get("interfere_time", 0),
    )

    process_times, residency_times = _build_recipe_process_times(recipes)
    _enforce_factory_cycles(
        model,
        nodes,
        factories,
        action_active,
        action_times,
        process_times,
        residency_times,
    )

    robot_contexts = _build_robot_contexts(trucks)
    _enforce_arm_and_robot_cycles(
        model,
        nodes,
        trucks,
        action_active,
        action_times,
        makespan,
        robot_contexts,
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


def _build_noncyclic_save_path(save_dir: str, recipe_path: str, counts) -> str:
    """根据 recipe 名称、counts 自动构建 noncyclic 保存路径"""
    recipe_name = Path(recipe_path).stem
    suffix = "_".join(f"{k}{v}" for k, v in sorted(counts.items()))
    filename = f"noncyclic_{suffix}.json"
    return str(Path(save_dir) / recipe_name / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-path", default=r"recipes\1_fluc.json")
    parser.add_argument("--save-dir", default="results",
                    help="保存根目录，将在该目录下创建 recipe 同名子文件夹")
    parser.add_argument("--save-path", default=None,
                    help="可选，指定完整保存路径（覆盖自动构建）")
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument("--counts", type=str, default='{"A": 5}',
                        help="目标产量 JSON，例如 '{\"A\":2,\"B\":2}'")
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--interleave-search", type=bool, default=True)
    parser.add_argument("--num-search-workers", type=int, default=1)
    parser.add_argument("--order", type=str, default=None,
                        help="指定货物顺序，如 'ACAC' 表示 A→C→A→C 的顺序")
    args = parser.parse_args()

    recipe_path = args.recipe_path
    time_scale = args.time_scale
    counts = json.loads(args.counts)
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = _build_noncyclic_save_path(args.save_dir, recipe_path, counts)
    time_limit_s = args.time_limit_s
    raw = load_recipe(recipe_path, time_scale)
    result = noncyclic(
        raw,
        counts,
        time_limit_s=time_limit_s,
        random_seed=args.random_seed,
        interleave_search=args.interleave_search,
        num_search_workers=args.num_search_workers,
        order=args.order,
    )
    result = _unscale_schedule(result, time_scale, "makespan")

    if result is not None:
        validation_report = validate(raw, result, counts)
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