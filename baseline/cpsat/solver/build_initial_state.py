import argparse
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Any
import json

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import (
    extend_actions,
    pair_load_unload,
    pair_unload_next_load,
    _build_recipe_process_times,
    load_recipe,
)

def build_initial_state(raw, cycle):
    actions_raw = cycle.get("Move_List", [])
    makespan = cycle.get("makespan", 0)
    if not actions_raw:
        raise ValueError("Cycle data must contain 'Move_List' with actions.")

    recipes = raw.get("recipes", [])
    trucks = raw.get("trucks", [])
    factories = raw.get("factories", []) or []
    process_times, residency_times = _build_recipe_process_times(recipes)
    actions = extend_actions(actions_raw, makespan)

    goods = []
    trucks_out = []
    gid_counter = 0

    # Case 1: (load, unload) —— wafer 在 PM 中跨周期边界
    load_unload_pairs = pair_load_unload(actions)
    for load, unload in load_unload_pairs:
        if unload is None:
            continue
        load_start = load.get("StartTime")
        load_end = load.get("EndTime")
        load_duration = load_end - load_start
        unload_start = unload.get("StartTime")
        pr_id = load.get("pr_id")
        step = load.get("step")

        if unload_start >= 0 and load_start < 0:
            process_time = int(process_times.get(pr_id, {}).get(step, 0) or 0)

            left_bound = unload_start - process_time - residency_times.get(pr_id, {}).get(step) - load_duration if residency_times.get(pr_id, {}).get(step) is not None else None
            right_bound = unload_start - process_time - load_duration

            gid = gid_counter
            gid_counter += 1
            good_entry = {
                "id": gid,
                "location": load.get("ModuleName"),
                "pr_id": load.get("pr_id"),
                "step": load.get("step"),
                "left_bound": round(left_bound, 1) if left_bound is not None else None,
                "right_bound": round(right_bound, 1) if right_bound is not None else None,
                "bound": round(unload_start, 1),
                "mode": "enter"
            }
            goods.append(good_entry) 

    # Case 2: (unload, next_load) —— wafer 在工厂之间跨周期边界
    unload_next_pairs = pair_unload_next_load(actions)
    for unload, next_load in unload_next_pairs:
        if next_load is None:
            continue
        unload_start = unload.get("StartTime")
        next_load_start = next_load.get("StartTime")
        if unload_start < 0 <= next_load_start:
            step = unload.get("step")
            pr_id = unload.get("pr_id")
            truck_id = unload.get("truck_id")

            gid = gid_counter
            gid_counter += 1
            goods.append(
                {
                    "id": gid,
                    "location": unload.get("ModuleName"),
                    "pr_id": unload.get("pr_id"),
                    "step": unload.get("step"),
                    "left_bound": None,
                    "right_bound": None,
                    "bound": round(next_load_start, 1),
                    "truck_id": truck_id,
                    "mode": "enter"
                }
            )

    seen = set()
    for truck in trucks:    
        truck_id = str(truck.get("id"))
        if truck_id in seen:
            continue
        seen.add(truck_id)
        first_act = next((a for a in actions if a["truck_id"] == truck_id and a.get("StartTime") >= 0), None)
        if first_act is None:
            continue
        trucks_out.append(
            {
                "id": truck_id,
                "bound_time": round(first_act.get("StartTime", 0), 1),
                "location": first_act.get("ModuleName")
            }
        )  

    # 收集 PM（type=="process"）工厂在 StartTime >= 0 的第一个 unload 区间
    process_factory_ids = {
        str(f.get("id"))
        for f in factories
        if str(f.get("type")) == "process"
    }
    process_unload_intervals = []
    for a in actions:
        fid = str(a.get("ModuleName") or "")
        if fid not in process_factory_ids:
            continue
        if str(a.get("MoveType")) != "unload":
            continue
        start = a.get("StartTime")
        if start is not None and start >= 0:
            process_unload_intervals.append({
                "start": round(start, 1),
                "end": round(a.get("EndTime"), 1),
                "factory_id": fid,
            })

    result = {
        "goods": goods,
        "trucks": trucks_out,
        "process_unload_intervals": process_unload_intervals,
    }
    return result

def _build_initial_state_save_path(save_dir: str, recipe_path: str) -> str:
    """根据 recipe 名称自动构建 initial_state 保存路径"""
    recipe_name = Path(recipe_path).stem
    return str(Path(save_dir) / recipe_name / "initial_state.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle-path", default=r"easy_travel\results\10-a_optimal\cycle1.json")
    parser.add_argument("--recipe-path", default=r"easy_travel\recipes\10-a.json")
    parser.add_argument("--save-dir", default="easy_travel\\results",
                    help="保存根目录，将在该目录下创建 recipe 同名子文件夹")
    parser.add_argument("--save-path", default=None,
                    help="可选，指定完整保存路径（覆盖自动构建）")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--fluc-strategy", type=str, default="max",
                        choices=["min", "max", "average"],
                        help="波动加工时间 [min,max] 的解析策略: min/max/average")
    args = parser.parse_args()

    cycle_path = args.cycle_path
    recipe_path = args.recipe_path
    time_scale = args.time_scale
    if args.save_path:
        save_path = args.save_path
    else:
        save_path = _build_initial_state_save_path(args.save_dir, recipe_path)
    with open(cycle_path, "r") as f:
        cycle = json.load(f)
    raw = load_recipe(recipe_path, time_scale)
    initial_state = build_initial_state(raw, cycle)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(initial_state, f, ensure_ascii=False, indent=2)
        print(f"已保存转换结果到 {save_path}")

if __name__ == "__main__":
    main()