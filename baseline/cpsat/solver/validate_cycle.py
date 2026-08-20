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
    requirements: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    moves = sort_moves(list(result.get("Move_List") or []))

    factories = raw.get("factories", []) or []
    trucks = raw.get("trucks", []) or []
    recipes = raw.get("recipes", {}) or {}
    process_times, residency_times = _build_recipe_process_times(recipes)

    factory_type = {as_str(f.get("id")): as_str(f.get("type")) for f in factories}
    truck_index = {as_str(t.get("id")): t for t in trucks}

    errors: List[Dict[str, Any]] = []

    # 0) 每个配方的每步动作数量检查
    if requirements is not None:
        # 构建配方步骤数映射（兼容 recipes dict 和 process_recipes list 两种格式）
        recipe_step_count: Dict[str, int] = {}
        raw_recipes = raw.get("recipes")
        if isinstance(raw_recipes, dict):
            for rid, steps in raw_recipes.items():
                recipe_step_count[str(rid)] = len(steps or [])
        else:
            for pr in (raw.get("process_recipes") or []):
                rid = as_str(pr.get("id"))
                recipe_step_count[rid] = len(pr.get("steps") or [])

        # 统计每个 (pr_id, step, MoveType) 的动作数
        action_counts: Dict[Tuple[str, int, str], int] = {}
        for move in moves:
            pr_id = as_str(move.get("pr_id"))
            step = as_int(move.get("step"))
            move_type = as_str(move.get("MoveType"))
            if pr_id and move_type in ("load", "unload"):
                key = (pr_id, step, move_type)
                action_counts[key] = action_counts.get(key, 0) + 1

        for recipe_id, expected_qty in requirements.items():
            n_steps = recipe_step_count.get(recipe_id, 0)
            if n_steps == 0:
                continue
            max_step = n_steps - 1

            # 除了第0步，每步都应有 load，且数量等于 expected_qty
            for step in range(1, n_steps):
                actual = action_counts.get((recipe_id, step, "load"), 0)
                if actual != expected_qty:
                    errors.append({
                        "rule": "recipe_step_load_count",
                        "recipe_id": recipe_id,
                        "step": step,
                        "expected": expected_qty,
                        "actual": actual,
                    })

            # 除了最后一步，每步都应有 unload，且数量等于 expected_qty
            for step in range(0, max_step):
                actual = action_counts.get((recipe_id, step, "unload"), 0)
                if actual != expected_qty:
                    errors.append({
                        "rule": "recipe_step_unload_count",
                        "recipe_id": recipe_id,
                        "step": step,
                        "expected": expected_qty,
                        "actual": actual,
                    })

    # 1) 手臂序列：取货(unload)后的下一个动作必须是同货物下一步放货(load)
    arm_groups = group_by_truck(moves)
    for truck_id, arm_moves in arm_groups.items():
        for idx in range(len(arm_moves) - 1):
            cur = arm_moves[idx]
            nxt = arm_moves[idx + 1]
            if as_str(cur.get("MoveType")) != "unload":
                continue

            pr_id = cur.get("pr_id")
            step = as_int(cur.get("step"))
            if not (
                as_str(nxt.get("MoveType")) == "load"
                and as_int(nxt.get("step")) == step + 1
                and as_str(nxt.get("pr_id")) == pr_id
            ):
                errors.append(
                    {
                        "rule": "arm_pick_then_next_drop",
                        "truck_id": truck_id,
                        "current_move_id": cur.get("Move_ID"),
                        "next_move_id": nxt.get("Move_ID"),
                        "pr_id": pr_id,
                        "step": step,
                    }
                )

        # 循环检查：最后一个动作如果是unload，检查第一个动作是否是下一步load
        if arm_moves and as_str(arm_moves[-1].get("MoveType")) == "unload":
            last = arm_moves[-1]
            first = arm_moves[0]
            last_step = as_int(last.get("step"))
            if not (
                as_str(first.get("MoveType")) == "load"
                and as_int(first.get("step")) == last_step + 1
                and as_str(first.get("pr_id")) == as_str(last.get("pr_id"))
            ):
                errors.append(
                    {
                        "rule": "arm_pick_then_next_drop",
                        "truck_id": truck_id,
                        "current_move_id": last.get("Move_ID"),
                        "next_move_id": first.get("Move_ID"),
                        "pr_id": last.get("pr_id"),
                        "step": last_step,
                    }
                )

    # 2) 工厂序列检查
    factory_groups = group_by_factory(moves)
    for factory_id, factory_moves in factory_groups.items():
        f_type = factory_type.get(factory_id)
        for idx in range(len(factory_moves) - 1):
            left = factory_moves[idx]
            right = factory_moves[idx + 1]

            # 2.1 放货(load)后下一个动作应是同配方同一步取货(unload)，并满足加工/驻留
            if f_type != "LP" and as_str(left.get("MoveType")) == "load":
                same_pr = as_str(left.get("pr_id")) == as_str(right.get("pr_id"))
                same_step = as_int(left.get("step")) == as_int(right.get("step"))
                right_unload = as_str(right.get("MoveType")) == "unload"
                if not (same_pr and same_step and right_unload):
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


        # 循环检查：最后一个动作如果是load，检查第一个动作是否是同配方同一步unload
        if (
            f_type != "LP"
            and factory_moves
            and as_str(factory_moves[-1].get("MoveType")) == "load"
            and as_str(factory_moves[0].get("MoveType")) == "unload"
        ):
            last_load = factory_moves[-1]
            first_unload = factory_moves[0]
            if (
                as_str(last_load.get("pr_id")) == as_str(first_unload.get("pr_id"))
                and as_int(last_load.get("step")) == as_int(first_unload.get("step"))
            ):
                pr_id = as_str(last_load.get("pr_id"))
                step = as_int(last_load.get("step"))
                process_time = as_int(process_times.get(pr_id, {}).get(step, 0))
                residency_time = residency_times.get(pr_id, {}).get(step)
                cycle_time = as_int(result.get("makespan"))
                load_end = as_int(last_load.get("EndTime"))
                unload_start = as_int(first_unload.get("StartTime")) + cycle_time
                in_module_time = unload_start - load_end

                # 检查是否有波动加工时间
                recipes_dict = raw.get("recipes", {})
                step_info_list = recipes_dict.get(pr_id, [])
                pt_range = None
                if isinstance(step, int) and step < len(step_info_list):
                    pt_range = step_info_list[step].get("process_time_range")

                if pt_range is not None and len(pt_range) == 2:
                    p_min, p_max = int(pt_range[0]), int(pt_range[1])
                    if in_module_time < p_min:
                        errors.append(
                            {
                                "rule": "factory_process_time",
                                "factory_id": factory_id,
                                "move_id_load": last_load.get("Move_ID"),
                                "move_id_unload": first_unload.get("Move_ID"),
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
                                "move_id_load": last_load.get("Move_ID"),
                                "move_id_unload": first_unload.get("Move_ID"),
                                "required_max": p_max + residency_time,
                                "actual": in_module_time,
                                "process_time_range": [p_min, p_max],
                            }
                        )
                else:
                    if in_module_time < process_time:
                        errors.append(
                            {
                                "rule": "factory_process_time",
                                "factory_id": factory_id,
                                "move_id_load": last_load.get("Move_ID"),
                                "move_id_unload": first_unload.get("Move_ID"),
                                "required_min": process_time,
                                "actual": in_module_time,
                            }
                        )
                    if residency_time is not None and in_module_time > process_time + residency_time:
                        errors.append(
                            {
                                "rule": "factory_residency",
                                "factory_id": factory_id,
                                "move_id_load": last_load.get("Move_ID"),
                                "move_id_unload": first_unload.get("Move_ID"),
                                "required_max": process_time + residency_time,
                                "actual": in_module_time,
                            }
                        )
            else:
                errors.append(
                    {
                        "rule": "factory_drop_then_same_step_pick",
                        "factory_id": factory_id,
                        "left_move_id": last_load.get("Move_ID"),
                        "right_move_id": first_unload.get("Move_ID"),
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

        # 循环检查：最后一个动作到第一个动作的移动时间
        if len(truck_moves) >= 2:
            last = truck_moves[-1]
            first = truck_moves[0]
            from_loc = as_str(last.get("ModuleName"))
            to_loc = as_str(first.get("ModuleName"))
            move_gap_required = get_travel_time(truck, from_loc, to_loc) if truck else 0
            cycle_time = as_int(result.get("makespan"))
            gap = as_int(first.get("StartTime")) + cycle_time - as_int(last.get("EndTime"))
            if gap < move_gap_required:
                errors.append(
                    {
                        "rule": "truck_travel_gap",
                        "truck_id": truck_id,
                        "left_move_id": last.get("Move_ID"),
                        "right_move_id": first.get("Move_ID"),
                        "from_loc": from_loc,
                        "to_loc": to_loc,
                        "required_min": move_gap_required,
                        "actual": gap,
                    }
                )

    # 4) process 工厂的 unload 区间间隔至少 interfere_time（周期版本）
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

        # 周期检查：最后一个区间与第一个区间的跨周期间隔
        if len(process_unloads) >= 2:
            last = process_unloads[-1]
            first = process_unloads[0]
            last_end = as_int(last.get("EndTime"))
            first_start = as_int(first.get("StartTime"))
            cycle_time = as_int(result.get("makespan"))
            if last_end + interfere_time > first_start + cycle_time:
                errors.append(
                    {
                        "rule": "process_unload_no_overlap_cyclic",
                        "move_id_last": last.get("Move_ID"),
                        "factory_last": last.get("ModuleName"),
                        "end_last": last_end,
                        "move_id_first": first.get("Move_ID"),
                        "factory_first": first.get("ModuleName"),
                        "start_first": first_start,
                        "cycle_time": cycle_time,
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
    parser.add_argument("--recipe-path", default=r"recipes\3.json", help="Path to recipe json")
    parser.add_argument("--result-path", default=r"results\3\F_2_G_2_cycle.json", help="Path to result json")
    parser.add_argument("--requirements", default={"F": 2,"G": 2}, type=str,
                        help="配方需求，格式 k1:v1,k2:v2")
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

    if isinstance(args.requirements, str):
        reqs = {}
        for pair in args.requirements.split(','):
            k, v = pair.split(':')
            reqs[k.strip()] = int(v.strip())
        args.requirements = reqs
    requirements = args.requirements

    report = validate(
        raw,
        result,
        requirements,
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
