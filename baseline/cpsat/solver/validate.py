import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import (
    load_recipe, _build_recipe_process_times,
    as_str, as_int, sort_moves,
    group_by_factory, group_by_truck,
    get_travel_time,
)

def validate(
    raw: Dict[str, Any],
    result: Dict[str, Any],
    counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    moves = sort_moves(list(result.get("Move_List") or []))

    # 过滤掉物理移动条目（MoveType="move"），验证只关注 load/unload
    moves = [m for m in moves if as_str(m.get("MoveType")) != "move"]

    factories = raw.get("factories", []) or []
    trucks = raw.get("trucks", []) or []
    recipes = raw.get("recipes", []) or []
    process_times, residency_times = _build_recipe_process_times(recipes)

    factory_type = {as_str(f.get("id")): as_str(f.get("type")) for f in factories}
    truck_index = {as_str(t.get("id")): t for t in trucks}

    errors: List[Dict[str, Any]] = []

    # 0) 按 pr_id 检查每种货物数量
    if counts is not None:
        goods_by_pr: Dict[str, set] = {}
        for m in moves:
            gid = as_str(m.get("good_id"))
            pid = as_str(m.get("pr_id"))
            if gid and pid:
                goods_by_pr.setdefault(pid, set()).add(gid)
        for pid, expected in counts.items():
            actual = len(goods_by_pr.get(pid, set()))
            if actual != expected:
                errors.append(
                    {
                        "rule": "good_count_by_pr",
                        "pr_id": pid,
                        "expected": expected,
                        "actual": actual,
                    }
                )

    # 1) 手臂序列：取货(unload)后的下一个动作必须是同货物下一步放货(load)
    arm_groups = group_by_truck(moves)
    for truck_id, arm_moves in arm_groups.items():
        for idx in range(len(arm_moves) - 1):
            cur = arm_moves[idx]
            nxt = arm_moves[idx + 1]
            if as_str(cur.get("MoveType")) != "unload":
                continue

            gid = cur.get("good_id")
            pr_id = cur.get("pr_id")
            step = as_int(cur.get("step"))
            if not (
                as_str(nxt.get("MoveType")) == "load"
                and nxt.get("good_id") == gid
                and as_int(nxt.get("step")) == step + 1
                and as_str(nxt.get("pr_id")) == pr_id
            ):
                errors.append(
                    {
                        "rule": "arm_pick_then_next_drop",
                        "truck_id": truck_id,
                        "current_move_id": cur.get("Move_ID"),
                        "next_move_id": nxt.get("Move_ID"),
                        "good_id": gid,
                        "pr_id": pr_id,
                        "step": step,
                    }
                )

    # 2) 工厂序列检查
    factory_groups = group_by_factory(moves)
    for factory_id, factory_moves in factory_groups.items():
        f_type = factory_type.get(factory_id)
        for idx in range(len(factory_moves) - 1):
            left = factory_moves[idx]
            right = factory_moves[idx + 1]

            # 2.1 放货(load)后下一个动作应是同货物同一步取货(unload)，并满足加工/驻留
            if f_type != "LP" and as_str(left.get("MoveType")) == "load":
                same_good = as_str(left.get("good_id")) == as_str(right.get("good_id"))
                same_pr = as_str(left.get("pr_id")) == as_str(right.get("pr_id"))
                same_step = as_int(left.get("step")) == as_int(right.get("step"))
                right_unload = as_str(right.get("MoveType")) == "unload"
                if not (same_good and same_step and right_unload and same_pr):
                    errors.append(
                        {
                            "rule": "factory_drop_then_same_step_pick",
                            "factory_id": factory_id,
                            "left_move_id": left.get("Move_ID"),
                            "right_move_id": right.get("Move_ID"),
                        }
                    )
                else:
                    pr_id = as_str(left.get("pr_id"))
                    step = as_int(left.get("step"))
                    process_time = as_int(process_times.get(pr_id, {}).get(step, 0))
                    residency_time = residency_times.get(pr_id, {}).get(step)
                    load_end = as_int(left.get("EndTime"))
                    unload_start = as_int(right.get("StartTime"))
                    in_module_time = unload_start - load_end

                    # 检查是否有波动加工时间
                    recipes_dict = raw.get("recipes", {})
                    step_info_list = recipes_dict.get(pr_id, [])
                    pt_range = None
                    if isinstance(step, int) and step < len(step_info_list):
                        pt_range = step_info_list[step].get("process_time_range")

                    if pt_range is not None and len(pt_range) == 2:
                        # 波动步骤：p_min ≤ in_module_time ≤ p_max + r
                        p_min, p_max = int(pt_range[0]), int(pt_range[1])
                        if in_module_time < p_min:
                            errors.append(
                                {
                                    "rule": "factory_process_time",
                                    "factory_id": factory_id,
                                    "move_id_load": left.get("Move_ID"),
                                    "move_id_unload": right.get("Move_ID"),
                                    "required_min": p_min,
                                    "actual": in_module_time,
                                    "process_time_range": [p_min, p_max],
                                }
                            )
                        if residency_time is not None and in_module_time > p_max + residency_time:
                            errors.append(
                                {
                                    "rule": "factory_residency",
                                    "factory_id": factory_id,
                                    "move_id_load": left.get("Move_ID"),
                                    "move_id_unload": right.get("Move_ID"),
                                    "required_max": p_max + residency_time,
                                    "actual": in_module_time,
                                    "process_time_range": [p_min, p_max],
                                }
                            )
                    else:
                        # 固定步骤：原逻辑
                        if in_module_time < process_time:
                            errors.append(
                                {
                                    "rule": "factory_process_time",
                                    "factory_id": factory_id,
                                    "move_id_load": left.get("Move_ID"),
                                    "move_id_unload": right.get("Move_ID"),
                                    "required_min": process_time,
                                    "actual": in_module_time,
                                }
                            )
                        if residency_time is not None and in_module_time > process_time + residency_time:
                            errors.append(
                                {
                                    "rule": "factory_residency",
                                "factory_id": factory_id,
                                "move_id_load": left.get("Move_ID"),
                                "move_id_unload": right.get("Move_ID"),
                                "required_max": process_time + residency_time,
                                "actual": in_module_time,
                            }
                        )


    # 3) 卡车序列检查
    truck_groups = group_by_truck(moves)

    for truck_id, truck_moves in truck_groups.items():
        truck = truck_index.get(truck_id)
        for idx in range(len(truck_moves) - 1):
            left = truck_moves[idx]
            right = truck_moves[idx + 1]
            from_loc = as_str(left.get("ModuleName"))
            to_loc = as_str(right.get("ModuleName"))
            move_gap_required = get_travel_time(truck, from_loc, to_loc) if truck else 0
            gap = as_int(right.get("StartTime")) - as_int(left.get("EndTime"))
            if gap < move_gap_required:
                errors.append(
                    {
                        "rule": "truck_travel_gap",
                        "truck_id": truck_id,
                        "left_move_id": left.get("Move_ID"),
                        "right_move_id": right.get("Move_ID"),
                        "from_loc": from_loc,
                        "to_loc": to_loc,
                        "required_min": move_gap_required,
                        "actual": gap,
                    }
                )

    # 4) process 工厂的 unload 区间间隔至少 interfere_time
    process_factory_ids = {
        fid
        for fid, ftype in factory_type.items()
        if ftype == "process"
    }
    interfere_time = int(raw.get("interfere_time", 0) or 0)
    if process_factory_ids:
        process_unloads = [
            m for m in moves
            if as_str(m.get("ModuleName")) in process_factory_ids
            and as_str(m.get("MoveType")) == "unload"
        ]
        process_unloads.sort(key=lambda m: as_int(m.get("StartTime")))
        for i in range(len(process_unloads) - 1):
            a = process_unloads[i]
            b = process_unloads[i + 1]
            a_end = as_int(a.get("EndTime"))
            b_start = as_int(b.get("StartTime"))
            if a_end + interfere_time > b_start:
                errors.append(
                    {
                        "rule": "process_unload_no_overlap",
                        "move_id_a": a.get("Move_ID"),
                        "factory_a": a.get("ModuleName"),
                        "end_a": a_end,
                        "move_id_b": b.get("Move_ID"),
                        "factory_b": b.get("ModuleName"),
                        "start_b": b_start,
                        "interfere_time": interfere_time,
                    }
                )

    return {
        "valid": len(errors) == 0,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate move sequence constraints")
    parser.add_argument("--recipe-path", default=r"easy_travel\recipes\5-b.json", help="Path to recipe json")
    parser.add_argument("--result-path", default=r"easy_travel\results\5-b\5-b.json", help="Path to result json")
    parser.add_argument("--counts", type=int, default=None, help="Expected number of goods")
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path to save validation report json",
    )
    parser.add_argument("--time-scale", type=float, default=1)
    args = parser.parse_args()

    raw = load_recipe(args.recipe_path, args.time_scale)
    with open(args.result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    report = validate(
        raw,
        result,
        args.counts,
    )
    print(f"valid={report['valid']}, error_count={report['error_count']}")

    if report["errors"]:
        preview = report["errors"][:10]
        print(json.dumps(preview, ensure_ascii=False, indent=2))

    if args.report_path:
        save_path = Path(args.report_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
