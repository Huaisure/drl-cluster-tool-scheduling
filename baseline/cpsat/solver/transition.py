import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from ortools.sat.python import cp_model

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.constraints import (
    _create_action_active_vars,
    _create_action_time_vars,
    _enforce_arm_and_robot_cycles,
    _enforce_factory_cycles,
    _enforce_goods_priority_for_actions,
    _enforce_load_requires_unload,
    _enforce_same_truck_unload_load,
    _enforce_parallel_factory_choice,
    _enforce_process_unload_no_overlap,
)
from utils.utils import (
    _build_recipe_process_times,
    _unscale_schedule,
    build_nodes,
    build_solution_result,
    estimate_upper_bound,
    load_recipe,
)


def _tag_nodes(nodes: List[Dict], node_type: str) -> List[Dict]:
    tagged_nodes: List[Dict] = []
    for node in nodes:
        tagged_node = dict(node)
        tagged_node["type"] = node_type
        tagged_nodes.append(tagged_node)
    return tagged_nodes


def _build_robot_contexts(trucks, left_trucks, right_trucks):
    left_map = {str(item.get("id")): item for item in left_trucks}
    right_map = {str(item.get("id")): item for item in right_trucks}
    contexts = {}
    for truck in trucks:
        truck_id = str(truck.get("id"))
        left_info = left_map.get(truck_id, {})
        right_info = right_map.get(truck_id, {})
        end_time = right_info.get("bound_time")
        start_time = left_info.get("bound_time")
        contexts[truck_id] = {
            "start_time": int(start_time) if start_time is not None else None,
            "start_location": left_info.get("location"),
            "end_time": int(end_time) if end_time is not None else None,
            "end_location": right_info.get("location"),
        }
    return contexts


def transition(
    raw,
    left_state,
    right_state,
    time_limit_s=600,
    random_seed=42,
    interleave_search=True,
    num_search_workers=1,
):
    cpu_start = time.time()

    trucks = raw["trucks"]
    factories = raw["factories"]
    recipes = raw["recipes"]

    left_goods = [
        g for g in (left_state.get("goods", []) or [])
        if g.get("mode") != "noncyclic"
    ]
    left_trucks = left_state.get("trucks", [])
    right_goods = right_state.get("goods", [])
    # 为避免 left_goods 与 right_goods 的 id 冲突，对 right_goods 重新编号
    max_left_id = 0
    for g in left_goods or []:
        gid = g.get("id")
        gid_int = int(gid)
        if gid_int > max_left_id:
            max_left_id = gid_int

    # 从 max_left_id + 1 开始为 right_goods 赋予新的连续 id
    start_id = max_left_id + 1
    for i, g in enumerate(right_goods or []):
        # 覆盖原 id（保留其它字段不变）
        g["id"] = int(start_id + i)
    right_trucks = right_state.get("trucks", [])

    all_goods = left_goods + right_goods
    nodes, _, action_index = build_nodes(all_goods, recipes, trucks)
    n_nodes = len(nodes)

    # 按 good_id 范围拆回 left_nodes / right_nodes（right_goods 已重新编号）
    left_nodes = [n for n in nodes if int(n["good_id"]) <= max_left_id]
    right_nodes = [n for n in nodes if int(n["good_id"]) > max_left_id]

    upper_bound = (
        estimate_upper_bound(left_goods, recipes, trucks)
        + estimate_upper_bound(right_goods, recipes, trucks)
    )

    model = cp_model.CpModel()
    transition_time = model.NewIntVar(0, upper_bound, "transition_time")

    action_active = _create_action_active_vars(model, nodes)
    action_times = _create_action_time_vars(model, nodes, upper_bound, action_active, transition_time)

    _enforce_parallel_factory_choice(model, action_index, action_active)
    # 合并节点后统一处理 load/unload 依赖，使用合并后的 nodes
    _enforce_load_requires_unload(model, nodes, action_active)
    _enforce_same_truck_unload_load(model, nodes, action_active)

    _enforce_goods_priority_for_actions(
        model,
        nodes,
        action_active,
        action_times,
        all_goods,
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
        goods=all_goods,
        start_up_time=transition_time,
    )

    # 创建额外区间：左状态（StartTime<0，负时间直接使用）+ 右状态（+transition_time 偏移）
    interfere_time = raw.get("interfere_time", 0)
    extra_ivs = []
    for ei in (left_state.get("process_unload_intervals") or []):
        orig_start = int(round(ei["start"]))
        orig_end = int(round(ei["end"]))
        raw_dur = orig_end - orig_start
        if raw_dur <= 0:
            continue
        duration = raw_dur + interfere_time
        iv = model.NewFixedSizeIntervalVar(orig_start, duration, f"extra_left_{len(extra_ivs)}")
        extra_ivs.append(iv)
    for ei in (right_state.get("process_unload_intervals") or []):
        orig_start = int(round(ei["start"]))
        orig_end = int(round(ei["end"]))
        raw_dur = orig_end - orig_start
        if raw_dur <= 0:
            continue
        duration = raw_dur + interfere_time
        shifted_start = model.NewIntVar(0, 2 * upper_bound, f"extra_right_start_{len(extra_ivs)}")
        model.Add(shifted_start == orig_start + transition_time)
        iv = model.NewFixedSizeIntervalVar(shifted_start, duration, f"extra_right_{len(extra_ivs)}")
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

    robot_contexts = _build_robot_contexts(trucks, left_trucks, right_trucks)
    _enforce_arm_and_robot_cycles(
        model,
        nodes,
        trucks,
        action_active,
        action_times,
        transition_time,
        robot_contexts,
    )

    model.Minimize(transition_time)

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
        objective_name="transition_time",
        objective_value=solver.Value(transition_time),
        cpu_time=elapsed,
        nodes=nodes,
        action_times=action_times,
        action_active=action_active,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-path", default=r"recipes\1_ban_fac.json")
    parser.add_argument("--left-state-path", default=r"results\1_to_1_ban_fac\A_320\left_state.json")
    parser.add_argument("--right-state-path", default=r"results\1_ban_fac\A_2\initial_state.json")
    parser.add_argument("--save-path", default=r"results\1_to_1_ban_fac\A_320\transition.json")
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument("--time-limit-s", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--interleave-search", type=bool, default=True)
    parser.add_argument("--num-search-workers", type=int, default=1)
    args = parser.parse_args()

    raw = load_recipe(args.recipe_path, args.time_scale)
    with open(args.left_state_path, "r", encoding="utf-8") as f:
        left_state = json.load(f)
    with open(args.right_state_path, "r", encoding="utf-8") as f:
        right_state = json.load(f)

    result = transition(
        raw,
        left_state,
        right_state,
        time_limit_s=args.time_limit_s,
        random_seed=args.random_seed,
        interleave_search=args.interleave_search,
        num_search_workers=args.num_search_workers,
    )
    result = _unscale_schedule(result, args.time_scale, "transition_time")

    if result is not None:
        dirpath = os.path.dirname(args.save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        print("result has saved to:", args.save_path)
        with open(args.save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()