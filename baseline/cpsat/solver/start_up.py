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
    build_nodes,
    estimate_upper_bound,
    _build_recipe_process_times,
    build_solution_result,
    _unscale_schedule
)
from utils.constraints import (
    _create_action_active_vars,
    _enforce_parallel_factory_choice,
    _enforce_load_requires_unload,
    _enforce_same_truck_unload_load,
    _enforce_goods_priority_for_actions,
    _enforce_factory_cycles,
    _create_action_time_vars,
    _enforce_arm_and_robot_cycles,
    _enforce_process_unload_no_overlap,
)
import json
import time
import os
from ortools.sat.python import cp_model


def _build_robot_contexts(trucks, state_trucks):
    state_map = {str(item.get("id")): item for item in state_trucks}
    contexts = {}
    for truck in trucks:
        truck_id = str(truck.get("id"))
        state_info = state_map.get(truck_id, {})
        end_time = state_info.get("bound_time")
        contexts[truck_id] = {
            "start_time": None,
            "end_time": int(end_time) if end_time is not None else None,
            "end_location": state_info.get("location"),
        }
    return contexts

def start_up(
    raw,
    target_state,
    time_limit_s=600,
    random_seed=42,
    interleave_search=True,
    num_search_workers=1,
):
    cpu_start = time.time()

    trucks = raw["trucks"]
    factories = raw["factories"]
    recipes = raw["recipes"]

    state_goods = target_state.get("goods", [])
    state_trucks = target_state.get("trucks", [])

    nodes, boundry_actions, action_index = build_nodes(state_goods, recipes, trucks)
    n_nodes = len(nodes)

    upper_bound = estimate_upper_bound(state_goods, recipes, trucks)

    model = cp_model.CpModel()

    start_up_time = model.NewIntVar(0, upper_bound, "start_up_time")

    action_active = _create_action_active_vars(model, nodes)
    _enforce_parallel_factory_choice(model, action_index, action_active)
    _enforce_load_requires_unload(model, nodes, action_active)
    _enforce_same_truck_unload_load(model, nodes, action_active)
    action_times = _create_action_time_vars(model, nodes, upper_bound, action_active, start_up_time)
    _enforce_goods_priority_for_actions(
        model,
        nodes,
        action_active,
        action_times,
        state_goods,
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
        goods=state_goods,
        start_up_time=start_up_time,
    )

    # 创建额外区间：初态 PM unload 区间 + start_up_time 偏移
    interfere_time = raw.get("interfere_time", 0)
    extra_ivs = []
    for ei in target_state.get("process_unload_intervals") or []:
        orig_start = int(round(ei["start"]))
        orig_end = int(round(ei["end"]))
        raw_dur = orig_end - orig_start
        if raw_dur <= 0:
            continue
        duration = raw_dur + interfere_time
        shifted_start = model.NewIntVar(0, 2 * upper_bound, f"extra_start_{len(extra_ivs)}")
        model.Add(shifted_start == orig_start + start_up_time)
        iv = model.NewFixedSizeIntervalVar(shifted_start, duration, f"extra_iv_{len(extra_ivs)}")
        extra_ivs.append(iv)

    _enforce_process_unload_no_overlap(
        model,
        nodes,
        factories,
        action_times,
        action_active,
        interfere_time,
        extra_ivs or None,
    )

    robot_contexts = _build_robot_contexts(trucks, state_trucks)
    _enforce_arm_and_robot_cycles(
        model,
        nodes,
        trucks,
        action_active,
        action_times,
        start_up_time,
        robot_contexts,
    )

    model.Minimize(start_up_time)

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
        objective_name="start_up_time",
        objective_value=solver.Value(start_up_time),
        cpu_time=elapsed,
        nodes=nodes,
        action_times=action_times,
        action_active=action_active,
    )
    return result


def _build_start_up_save_path(save_dir: str, recipe_path: str) -> str:
    """根据 recipe 名称自动构建 start_up 保存路径"""
    recipe_name = Path(recipe_path).stem
    return str(Path(save_dir) / recipe_name / "start_up.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-path", default=r"easy_travel\recipes\10-a.json")
    parser.add_argument("--state-path", default=r"easy_travel\results\10-a_optimal\initial_state.json")
    parser.add_argument("--save-dir", default="easy_travel\\results",
                    help="保存根目录，将在该目录下创建 recipe 同名子文件夹")
    parser.add_argument("--save-path", default=None,
                    help="可选，指定完整保存路径（覆盖自动构建）")
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--interleave-search", type=bool, default=True)
    parser.add_argument("--num-search-workers", type=int, default=1)
    parser.add_argument("--fluc-strategy", type=str, default="max",
                        choices=["min", "max", "average"],
                        help="波动加工时间 [min,max] 的解析策略: min/max/average")
    args = parser.parse_args()

    recipe_path = args.recipe_path
    time_scale = args.time_scale
    state_path = args.state_path
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = _build_start_up_save_path(args.save_dir, recipe_path)
    time_limit_s = args.time_limit_s
    with open(state_path, "r", encoding="utf-8") as f:
        target_state = json.load(f)
    raw = load_recipe(recipe_path, time_scale)
    result = start_up(
        raw,
        target_state,
        time_limit_s=time_limit_s,
        random_seed=args.random_seed,
        interleave_search=args.interleave_search,
        num_search_workers=args.num_search_workers,
    )
    result = _unscale_schedule(result, time_scale, "start_up_time")

    if result is not None:
        # Ensure target directory exists before writing result
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        print("result has saved to:", save_path)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()