"""
从已有的周期结果出发，构建初态/末态，求解启动和关闭，拼接完整调度，并绘制 HTML 甘特图。

通过 TASK_LIST 批量配置，每项指定 recipe 路径和已有的 cycle 结果路径。
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.concat_sequences import assign_good_ids_by_occurrence
from utils.utils import load_recipe
from utils.wafer_gantt_html import (
    build_segments,
    generate_html,
    load_moves as wafer_load_moves,
)
from solver.build_initial_state import build_initial_state
from solver.build_end_state import build_end_state
from solver.start_up import start_up
from solver.close_down import close_down
from solver.validate import validate

# ─────────────────────────────────────────────────────────────
# 任务列表：每项 (recipe 路径, cycle 结果路径)
# ─────────────────────────────────────────────────────────────
TASK_LIST = [
    (r"recipes\b_travel.json", r"results\b_travel\B1_2_B2_2_cycle.json"),
    (r"recipes\b_travel.json", r"results\b_travel\B1_2_B2_2_B3_1_B4_1_cycle.json"),
    (r"recipes\c_travel.json", r"results\c_travel\C1_1_C2_1_cycle.json"),
    (r"recipes\e_travel.json", r"results\e_travel\A5_1_A6_1_cycle.json"),
    (r"recipes\e_travel.json", r"results\e_travel\A3_1_A4_1_A5_1_A6_1_cycle.json"),
]

DEFAULT_SAVE_DIR = r"results"
TIME_SCALE = 1.0
NUM_WORKERS = 16


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_factory_types(raw: Dict[str, Any]) -> Dict[str, str]:
    factory_types: Dict[str, str] = {}
    for f in raw.get("factories", []):
        fid = f.get("id")
        ftype = f.get("type", "normal")
        if fid:
            factory_types[str(fid)] = str(ftype)
    return factory_types


def build_recipe_step_times(raw: Dict[str, Any]):
    recipe_step_times = {}
    for pr_id, steps in raw.get("recipes", {}).items():
        for step_idx, step in enumerate(steps):
            pt = step.get("process_time")
            if pt is not None:
                recipe_step_times[(str(pr_id), int(step_idx))] = int(pt)
    return recipe_step_times


def draw_html_gantt(actions: List[Dict], makespan: float, recipe_path: str,
                    save_path: str, title: str = "Combined Schedule") -> Optional[str]:
    tmp_json = Path(save_path).with_suffix(".tmp.json")
    save_json({"Move_List": actions, "makespan": makespan}, tmp_json)

    try:
        raw = load_recipe(recipe_path, time_scale=TIME_SCALE)
    except Exception:
        raw = {"factories": [], "recipes": {}}

    factory_types = build_factory_types(raw)
    recipe_step_times = build_recipe_step_times(raw)

    moves, mk = wafer_load_moves(str(tmp_json))
    segments = build_segments(moves, recipe_step_times, factory_types)
    html_path = generate_html(segments, int(mk), title=title, save_path=save_path)

    try:
        tmp_json.unlink()
    except OSError:
        pass

    return html_path


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="从周期解批量构建调度并绘制 HTML 甘特图")
    p.add_argument("--time-limit", type=int, default=600, help="求解器时间限制（秒）")
    p.add_argument("--out", dest="out_path", default=DEFAULT_SAVE_DIR, help="输出目录")
    p.add_argument("--no-html", action="store_true", help="跳过 HTML 甘特图生成")
    args = p.parse_args()

    for recipe_path, cycle_path in TASK_LIST:
        recipe_file = Path(recipe_path)
        cycle_file = Path(cycle_path)

        if not recipe_file.exists():
            print(f"Recipe 不存在: {recipe_file}")
            raise SystemExit(2)
        if not cycle_file.exists():
            print(f"Cycle 不存在: {cycle_file}")
            raise SystemExit(2)

        time_limit = args.time_limit
        start = time.time()

        recipe_stem = recipe_file.stem
        cycle_stem = cycle_file.stem

        # 从 cycle 文件名提取 req_suffix（去掉 _cycle 后缀）
        # 如 A5_1_A6_1_cycle.json → A5_1_A6_1
        req_suffix = cycle_stem
        if req_suffix.endswith("_cycle"):
            req_suffix = req_suffix[:-len("_cycle")]

        # 输出目录：results/<recipe>/<req_suffix>/
        out_dir = Path(DEFAULT_SAVE_DIR) / recipe_stem
        solution_dir = out_dir / req_suffix
        solution_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"recipe: {recipe_file}")
        print(f"cycle:  {cycle_file}")
        print(f"输出:   {solution_dir}")

        # 1) 加载 recipe 和 cycle
        print("加载数据 ...")
        raw = load_recipe(str(recipe_file), time_scale=TIME_SCALE)

        with open(str(cycle_file), "r", encoding="utf-8") as f:
            cycle_result = json.load(f)

        cycle_makespan = float(cycle_result.get("makespan", 0.0))
        base_actions = sorted(
            cycle_result.get("Move_List", []),
            key=lambda x: float(x.get("StartTime", 0.0))
        )
        print(f"  cycle makespan = {cycle_makespan:.1f}, 动作数 = {len(base_actions)}")

        # 2) 构建初态 & 求解 start_up
        init_state = build_initial_state(raw, cycle_result)
        save_json(init_state, solution_dir / "initial_state.json")

        print("求解 start_up ...")
        enter_result = start_up(raw, init_state, time_limit_s=time_limit,
                                num_search_workers=NUM_WORKERS)
        if enter_result is None:
            print("  start_up 求解失败，跳过")
            continue
        enter_actions = enter_result.get("Move_List", []) or []
        shift_enter = float(enter_result.get("start_up_time", 0.0))
        print(f"  start_up makespan = {shift_enter:.1f}, 动作数 = {len(enter_actions)}")
        save_json(enter_result, solution_dir / "start_up.json")

        # 3) 构建末态 & 求解 close_down
        end_state = build_end_state(raw, cycle_result)
        save_json(end_state, solution_dir / "end_state.json")
        save_json(cycle_result, solution_dir / "cycle.json")

        print("求解 close_down ...")
        leave_result = close_down(raw, end_state, time_limit_s=time_limit,
                                  num_search_workers=NUM_WORKERS)
        if leave_result is None:
            print("  close_down 求解失败，跳过")
            continue
        leave_actions = leave_result.get("Move_List", []) or []
        leave_makespan = float(leave_result.get("close_down_time", 0.0))
        print(f"  close_down makespan = {leave_makespan:.1f}, 动作数 = {len(leave_actions)}")
        save_json(leave_result, solution_dir / "close_down.json")

        # 4) 拼接：启动 + 1 个周期 + 关闭
        combined: List[Dict[str, Any]] = []

        # 启动段（从 t=0 开始）
        for a in enter_actions:
            combined.append(dict(a))

        # 单个周期（偏移到 start_up 之后）
        for act in base_actions:
            a = dict(act)
            a["StartTime"] = round(shift_enter + float(a.get("StartTime", 0.0)), 3)
            a["EndTime"] = round(shift_enter + float(a.get("EndTime", 0.0)), 3)
            combined.append(a)

        # 关闭段
        leave_base = shift_enter + cycle_makespan
        for a in leave_actions:
            act_leave = dict(a)
            act_leave["StartTime"] = round(leave_base + float(a.get("StartTime", 0.0)), 3)
            act_leave["EndTime"] = round(leave_base + float(a.get("EndTime", 0.0)), 3)
            combined.append(act_leave)

        # 排序并分配 good_id
        combined.sort(key=lambda x: (float(x.get("StartTime", 0.0))))
        actions = assign_good_ids_by_occurrence(combined)

        overall_makespan = leave_base + leave_makespan
        elapsed = round(time.time() - start, 3)

        # 5) 保存最终结果
        final_out_obj = {
            "makespan": overall_makespan,
            "cpu_time": elapsed,
            "fluc_strategy": "max",
            "Move_List": actions,
        }

        # 验证
        validation = validate(raw, final_out_obj)
        final_out_obj["validation"] = validation.get("errors", [])
        if validation.get("valid"):
            print("  Validation passed!")
        else:
            print(f"  发现 {validation.get('error_count', 0)} 个验证错误。")

        out_final = solution_dir / "noncyclic.json"
        save_json(final_out_obj, out_final)
        print(f"  Overall makespan: {overall_makespan:.3f}")
        print(f"  Total cpu_time: {elapsed:.3f}")
        print(f"  已保存: {out_final}")

        # 6) 绘制 HTML 甘特图
        if not args.no_html:
            html_path = solution_dir / "noncyclic.html"
            title = f"{recipe_stem} / {cycle_stem}"
            print("  生成 HTML 甘特图 ...")
            draw_html_gantt(actions, overall_makespan, str(recipe_file),
                            str(html_path), title=title)
            print(f"  已保存 HTML: {html_path}")

    print(f"\n{'='*60}")
    print("全部完成！")


if __name__ == "__main__":
    main()

