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
    subtract_makespan,
)

def build_end_state_from_action_schedule(raw, schedule: dict) -> dict:
    """Reconstruct a best-effort initial snapshot from a forward action sequence.

    This is intended for startup / enter-steady schedules that already begin at t=0.
    It keeps the same output shape as the cycle-based initializer so downstream code
    can reuse the result without extra adaptation.
    """
    actions_raw = schedule.get("Move_List", [])
    makespan = float(schedule.get("makespan", 0.0))
    recipes = raw.get("recipes", [])
    trucks = raw.get("trucks", [])
    factories = raw.get("factories", []) or []
    pm_factories = {
        str(f.get("id")) for f in factories
        if str(f.get("type", "")).strip() == "process" and f.get("id") is not None
    }
    if actions_raw is None:
        raise ValueError("Schedule JSON must contain 'actions' or 'action_schedule'.")

    actions = subtract_makespan(actions_raw, makespan)
    if not actions:
        return {"goods": [], "trucks": []}

    process_times, residency_times = _build_recipe_process_times(recipes)
    # 按 (pr_id, good_id) 分组，找出每个 good 的最后动作
    groups: Dict[Tuple[str, Any], List[dict]] = {}
    for a in actions:
        key = (str(a.get("pr_id")), a.get("good_id"))
        groups.setdefault(key, []).append(a)

    goods: List[dict] = []
    trucks_out = []
    factory = []
    gid_counter = 0

    # 收集每个工厂最后一个动作的结束时间
    factory_last: Dict[str, float] = {}
    for a in actions:
        mod = a.get("ModuleName")
        if mod is None:
            continue
        end = float(a.get("EndTime", 0))
        cur = factory_last.get(mod)
        if cur is None or end > cur:
            factory_last[mod] = end
    factory = [{"factory_id": fid, "end_time": round(t, 1)} for fid, t in factory_last.items()]

    for (pr_id, good_id), acts in groups.items():
        # 找最后动作（按 StartTime）
        last_act = max(acts, key=lambda a: float(a.get("StartTime", 0)))
        move_type = last_act.get("MoveType")
        module = last_act.get("ModuleName")

        # 如果最后动作是 load 到 LP，说明该 good 已离开系统，跳过
        if move_type == "load" and module == "LP":
            continue

        # 否则记录该 good 的状态
        if move_type == "load":
            # 最后动作是 load，说明 wafer 在某个工厂内正在加工
            location = module
            step = last_act.get("step")
            load_start = float(last_act.get("StartTime", 0))
            load_end = float(last_act.get("EndTime", 0))
            process_time = int(process_times.get(pr_id, {}).get(step, 0) or 0)
            residency_time = residency_times.get(pr_id, {}).get(step)
            
            left_bound = load_end + process_time
            right_bound = left_bound + residency_time if residency_time is not None else None
            goods.append({
                "id": good_id if good_id is not None else gid_counter,
                "location": location,
                "pr_id": pr_id,
                "step": step,
                "left_bound": round(left_bound, 1) if left_bound is not None else None,
                "right_bound": round(right_bound, 1) if right_bound is not None else None,
                "bound": round(load_start, 1),
                "mode": "leave",
            })
        elif move_type == "unload":
            # 最后动作是 unload，说明 wafer 刚被取出，准备下一步
            location = module
            step = last_act.get("step")
            unload_start = float(last_act.get("StartTime", 0))
            truck_id = last_act.get("truck_id")

            goods.append({
                "id": good_id if good_id is not None else gid_counter,
                "location": None,
                "pr_id": pr_id,
                "step": step + 1,
                "left_bound": None,
                "right_bound": None,
                "bound": round(unload_start, 1),
                "truck_id": truck_id,
                "mode": "leave",
            })

        gid_counter += 1

    # 构建 trucks_out：每个 truck 的最后一个动作的结束时间
    truck_ids = sorted({str(a.get("truck_id")) for a in actions if a.get("truck_id") is not None})
    for tid in truck_ids:
        truck_acts = [a for a in actions if str(a.get("truck_id")) == tid]
        if not truck_acts:
            continue
        last_act = max(truck_acts, key=lambda a: float(a.get("StartTime", 0)))
        trucks_out.append({
            "id": tid,
            "bound_time": max(0,round(float(last_act.get("EndTime", 0)), 1)),
        })

    return {"goods": goods, "trucks": trucks_out}

def build_end_state(raw, cycle):
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
    factory = []
    gid_counter = 0

    for a in actions:
        if a["StartTime"] < 0 and a["EndTime"] > 0:
            location = a["ModuleName"]
            factory.append(
                {
                    "factory_id": location,
                    "end_time": round(a["EndTime"], 1)
                }
            )
    
    # Case 1: (load, unload) —— wafer 在 PM 中跨周期边界（leave 视角）
    load_unload_pairs = pair_load_unload(actions)
    for load, unload in load_unload_pairs:
        if unload is None:
            continue
        load_start = load.get("StartTime")
        load_end = load.get("EndTime")
        unload_start = unload.get("StartTime")
        pr_id = load.get("pr_id")
        step = load.get("step")

        if unload_start >= 0 and load_start < 0:
            process_time = int(process_times.get(pr_id, {}).get(step, 0) or 0)

            left_bound = load_end + process_time
            right_bound = load_end + process_time + residency_times.get(pr_id, {}).get(step) if residency_times.get(pr_id, {}).get(step) is not None else None
            gid = gid_counter
            gid_counter += 1
            goods.append(
                {
                    "id": gid,
                    "location": load.get("ModuleName"),
                    "pr_id": load.get("pr_id"),
                    "step": load.get("step"),
                    "left_bound": round(left_bound, 1) if left_bound is not None else None,
                    "right_bound": round(right_bound, 1)if right_bound is not None else None,
                    "bound": round(load_start, 1),
                    "mode": "leave"
                }
            )

    # Case 2: (unload, next_load) —— wafer 在工厂之间跨周期边界（leave 视角）
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
                    "location": next_load.get("ModuleName"),
                    "pr_id": next_load.get("pr_id"),
                    "step": next_load.get("step"),
                    "left_bound": None,
                    "right_bound": None,
                    "bound": round(unload_start, 1),
                    "truck_id": truck_id,
                    "mode": "leave"
                }
            )

    seen = set()
    for truck in trucks:    
        truck_id = str(truck.get("id"))
        if truck_id in seen:
            continue
        seen.add(truck_id)
        valid_actions = [a for a in actions if a.get("truck_id") == truck_id and a.get("StartTime", -1) < 0]
        if not valid_actions:
            continue
        else:
            last_act = max(valid_actions, key=lambda x: x.get("StartTime", -1))
        trucks_out.append(
            {
                "id": truck_id,
                "bound_time": round(last_act.get("EndTime", 0), 1),
                "location": last_act.get("ModuleName")
            }
        )

    # 收集 PM（type=="process"）工厂在 StartTime < 0 的最后一个 unload 区间
    # 这些是周期边界前发生的取货，close_down 调度需要与之保持间隔
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
        if start is not None and start < 0:
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

def _build_end_state_save_path(save_dir: str, recipe_path: str) -> str:
    """根据 recipe 名称自动构建 end_state 保存路径"""
    recipe_name = Path(recipe_path).stem
    return str(Path(save_dir) / recipe_name / "end_state.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle-path", default=r"results\1\C_1_cycle.json")
    parser.add_argument("--recipe-path", default=r"recipes\1.json")
    parser.add_argument("--save-dir", default="results",
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
        save_path = _build_end_state_save_path(args.save_dir, recipe_path)
    with open(cycle_path, "r") as f:
        cycle = json.load(f)
    raw = load_recipe(recipe_path, time_scale)
    end_state = build_end_state(raw, cycle)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(end_state, f, ensure_ascii=False, indent=2)
        print(f"已保存转换结果到 {save_path}")

if __name__ == "__main__":
    main()