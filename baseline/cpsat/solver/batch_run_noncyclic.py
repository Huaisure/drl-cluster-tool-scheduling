from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.concat_sequences import assign_good_ids_by_occurrence
from utils.utils import insert_move_entries, load_recipe, calc_process_residency_stats
from utils.gantt_chart import load_moves, build_gantt_data, draw_gantt
from utils.wafer_gantt_html import load_moves as html_load_moves, build_segments, generate_html
from solver.build_initial_state import build_initial_state
from solver.build_end_state import build_end_state
from solver.start_up import start_up
from solver.close_down import close_down
from solver.validate import validate
from solver.cycle import cycle

RECIPE_PATH = [
    # 格式: (recipe, requirements, counts, title)
    # title 为空字符串时不显示标题
    # (r"recipes\a_no_residency.json", {"A1": 2, "A2": 2}, {"A1": 50, "A2": 50}, ""),
    # (r"recipes\b_no_residency.json", {"B1": 3, "B2": 3}, {"B1": 50, "B2": 50}, ""),
    # (r"recipes\c_no_residency.json", {"C1": 1, "C2": 1}, {"C1": 50, "C2": 50}, ""),
    # (r"recipes\a.json", {"A5": 1, "A6": 1}, {"A5": 50, "A6": 50}, ""),
    # (r"recipes\b.json", {"B1": 2, "B2": 2}, {"B1": 50, "B2": 50}, ""),
    # (r"recipes\c.json", {"C1": 1, "C2": 1}, {"C1": 50, "C2": 50}, ""),
    # (r"recipes\e.json", {"A5": 1, "A6": 1}, {"A5": 50, "A6": 50}, ""),
    # (r"recipes\e.json", {"A1": 1, "A2": 1, "A3": 1, "A4": 1, "A5": 1, "A6": 1}, {"A1": 16, "A2": 16, "A3": 16, "A4": 16, "A5": 16, "A6": 16}, ""),
    # (r"recipes\a_travel.json", {"A5": 1, "A6": 1}, {"A5": 50, "A6": 50}, ""),
    (r"recipes\c_travel.json", {"C1": 1, "C2": 1}, {"C1": 50, "C2": 50}, "C ABC腔室都用"),
    (r"recipes\b_travel.json", {"B1": 2, "B2": 2}, {"B1": 50, "B2": 50}, "B 只用AB腔室"),
    (r"recipes\b_travel.json", {"B1": 2, "B2": 2, "B3": 1, "B4": 1}, {"B1": 34, "B2": 33, "B3": 16, "B4": 17}, "B ABC腔室都用"),
    (r"recipes\e_travel.json", {"A5": 1, "A6": 1}, {"A5": 50, "A6": 50}, "A 只用AB腔室"),
    (r"recipes\e_travel.json", {"A3": 1, "A4": 1, "A5": 1, "A6": 1}, {"A3": 25, "A4": 25, "A5": 25, "A6": 25}, "A ABC腔室都用"),
    # (r"recipes\e_travel.json", {"A3": 1, "A4": 1, "A5": 2, "A6": 2}, {"A3": 17, "A4": 17, "A5": 33, "A6": 33}, "e_travel"),
]
DEFAULT_SAVE_DIR = r"results/new"
TIME_SCALE = 1.0

def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Compose non-cyclic schedule from cycle/enter/leave solvers")
    p.add_argument("--time-limit", type=int, default=600, help="Solver time limit seconds")
    p.add_argument("--out", dest="out_path", default=DEFAULT_SAVE_DIR, help="Output JSON path or directory")
    args = p.parse_args()

    for recipe, requirements, counts, title in RECIPE_PATH:
        scale = TIME_SCALE
        recipe_path = Path(recipe)
        if not recipe_path.exists():
            print("Recipe not found:", recipe_path)
            raise SystemExit(2)

        time_limit = int(args.time_limit)
        start = time.time()

        recipe_stem = recipe_path.stem
        out_dir = Path(DEFAULT_SAVE_DIR) / recipe_stem
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.out_path and args.out_path != DEFAULT_SAVE_DIR:
            candidate = Path(args.out_path)
            solution_dir = candidate if candidate.exists() and candidate.is_dir() else candidate.parent
        else:
            solution_dir = out_dir

        solution_dir.mkdir(parents=True, exist_ok=True)
        req_suffix = "_".join(f"{k}_{v}" for k, v in sorted(requirements.items()))
        solution_dir = solution_dir / req_suffix
        solution_dir.mkdir(parents=True, exist_ok=True)
        counts_suffix = "_".join(f"{k}_{v}" for k, v in sorted(counts.items()))
        out_final = solution_dir / f"noncyclic_{counts_suffix}.json"
        # load recipe once
        print("Loading recipe...")
        raw = load_recipe(str(recipe_path), time_scale=TIME_SCALE)

        cycle_result = cycle(
            raw,
            requirements,
            time_limit_s=time_limit,
            random_seed=42,
            interleave_search=True,
            num_search_workers=16,
        )

        cycle_makespan = cycle_result.get("makespan", 0.0)

        base_actions = sorted(cycle_result.get("Move_List", []), key=lambda x: float(x.get("StartTime", 0.0)))

        init_state = build_initial_state(raw, cycle_result)
        print(f"solving start up...")
        enter_result = start_up(raw, init_state, time_limit_s=time_limit,num_search_workers=16)
        enter_actions = enter_result.get("Move_List", []) or []
        shift_enter = enter_result.get("start_up_time", 0.0)

        goods_list = init_state.get("goods", []) or []
        goods_by_pr: Dict[str, int] = {}
        for g in goods_list:
            pid = g.get("pr_id")
            goods_by_pr[pid] = goods_by_pr.get(pid, 0) + 1
        for pr_id, count in counts.items():
            if goods_by_pr.get(pr_id, 0) > count:
                print(f"pr_id={pr_id} 初始货物 {goods_by_pr.get(pr_id, 0)} > 目标 {count}，无法构建调度")
                return
        k = min((count - goods_by_pr.get(pr_id, 0)) // requirements[pr_id] for pr_id, count in counts.items())
        remaining_by_pr: Dict[str, float] = {}
        for pr_id, count in counts.items():
            remaining = count - goods_by_pr.get(pr_id, 0) - k * requirements[pr_id]
            remaining_by_pr[pr_id] = remaining

        end_state = build_end_state(raw, cycle_result)
        # 将剩余货物加入 end_state，与关闭中的货物一起求解
        end_state_goods = end_state.get("goods", [])
        next_gid = max((g["id"] for g in end_state_goods), default=-1) + 1
        for pr_id, remaining in remaining_by_pr.items():
            remaining_int = int(remaining)
            for _ in range(remaining_int):
                end_state_goods.append({
                    "id": next_gid,
                    "pr_id": pr_id,
                    "mode": "noncyclic",
                })
                next_gid += 1
        print(f"solving close down...")

        leave_result = close_down(raw, end_state, time_limit_s=time_limit, num_search_workers=16)
        leave_actions = leave_result.get("Move_List", []) or []
        leave_makespan = leave_result.get("close_down_time", 0.0)

        combined: List[Dict[str, Any]] = []
        combined.extend(enter_actions)

        steady_duration = k * cycle_makespan

        # Append k repeats of the periodic cycle, each shifted by the cycle makespan
        for cycle_idx in range(k):
            base_offset = cycle_idx * cycle_makespan
            for act in base_actions:
                a = dict(act)
                start_time = base_offset + float(a.get("StartTime", 0.0))
                end_time = base_offset + float(a.get("EndTime", 0.0))
                a["StartTime"] = round(shift_enter + start_time, 3)
                a["EndTime"] = round(shift_enter + end_time, 3)
                combined.append(a)

        leave_base = shift_enter + steady_duration
        for a in leave_actions:
            act_leave = dict(a)
            act_leave["StartTime"] = round(leave_base + float(a.get("StartTime", 0.0)), 3)
            act_leave["EndTime"] = round(leave_base + float(a.get("EndTime", 0.0)), 3)
            combined.append(act_leave)

        combined.sort(key=lambda x: (float(x.get("StartTime", 0.0))))
        actions = assign_good_ids_by_occurrence(combined)

        overall_makespan = leave_base + leave_makespan

        elapsed_traverse = time.time() - start
        cpu_time_val = round(elapsed_traverse, 3)

        # create ordered output with cpu_time placed early (near makespan)
        final_out_obj = {
            "counts": counts,
            "makespan": overall_makespan,
            "cpu_time": cpu_time_val,
            "Move_List": actions,
        }

        # Save optimal intermediates
        save_json(end_state, solution_dir / "end_state.json")
        save_json(init_state, solution_dir / "initial_state.json")
        save_json(enter_result, solution_dir / "start_up.json")
        save_json(leave_result, solution_dir / "close_down.json")
        save_json(cycle_result, solution_dir / "cycle.json")

        # 绘制各部件的甘特图（不绘制完整序列）
        for _fname in ["start_up.json", "close_down.json", "cycle.json"]:
            _json_path = solution_dir / _fname
            _png_path = _json_path.with_suffix(".png")
            try:
                _moves, _mk, _st = load_moves(str(_json_path))
                _data = build_gantt_data(_moves, _mk)
                draw_gantt(_data, title=f"{_json_path.stem} (makespan={_mk})", save_path=str(_png_path))
            except Exception as _e:
                print(f"  绘制甘特图失败 {_fname}: {_e}")

        validation_result = validate(raw, final_out_obj, counts)
        final_out_obj["validation"] = validation_result.get("errors", [])
        if validation_result.get("valid"):
            print("Validation passed!")
        else:
            print(f"Find {validation_result.get('error_count', 0)} validation errors.")

        save_json(final_out_obj, out_final)
        print(f"Overall makespan: {overall_makespan:.3f}")
        print(f"Total cpu_time: {cpu_time_val:.3f}")
        print("Saved final non-cyclic schedule to:", out_final)

        # ---- 计算 process 工厂平均驻留时间 ----
        factory_types: Dict[str, str] = {}
        for factory in raw.get("factories", []):
            fid = factory.get("id", "")
            ftype = factory.get("type", "normal")
            if fid:
                factory_types[fid] = ftype

        recipe_step_times: Dict[Tuple[str, int], int] = {}
        for pr_id, steps in raw.get("recipes", {}).items():
            for step_idx, step in enumerate(steps):
                pt = step.get("process_time")
                if pt is not None:
                    recipe_step_times[(str(pr_id), step_idx)] = int(pt)

        residency_stats = calc_process_residency_stats(
            actions, factory_types, recipe_step_times
        )
        print(f"\n--- Process 工厂驻留时间统计 ---")
        print(f"  总次数: {residency_stats['count']}")
        print(f"  平均驻留: {residency_stats['avg']}")
        print(f"  最小驻留: {residency_stats['min']}")
        print(f"  最大驻留: {residency_stats['max']}")
        if residency_stats.get("by_chamber"):
            print(f"  按 chamber 分:")
            for chamber, stats in residency_stats["by_chamber"].items():
                print(f"    {chamber}: avg={stats['avg']}, count={stats['count']}, "
                      f"min={stats['min']}, max={stats['max']}")

        # ---- 绘制完整序列的交互式 HTML 甘特图 ----
        if title:
            html_path = solution_dir / f"{title}.html"
        else:
            html_path = out_final.with_suffix(".html")
        try:
            html_moves, html_makespan = html_load_moves(str(out_final))
            html_segments = build_segments(html_moves, recipe_step_times, factory_types)
            generate_html(html_segments, html_makespan, title=title, save_path=str(html_path))
        except Exception as _e:
            print(f"  绘制 HTML 甘特图失败: {_e}")



if __name__ == "__main__":
    main()
