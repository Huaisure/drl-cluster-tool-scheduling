"""
运行两次 batch_run_noncyclic，用 transition 连接两个序列。
拼接方式: start_up_A + k*cycle_A + transition(A_end → B_init) + k*cycle_B + close_down_B
"""

from __future__ import annotations

import argparse
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
from utils.gantt_chart import load_moves, build_gantt_data, draw_gantt
from solver.build_initial_state import build_initial_state as build_initial_state_from_cycle
from solver.build_end_state import build_end_state as build_end_state_from_cycle
from solver.start_up import start_up
from solver.close_down import close_down
from solver.validate import validate
from solver.cycle import cycle
from solver.transition import transition

# 两批：分别用各自的 recipe/requirements/counts 求解，然后 transition 连接
BATCH1 = (r"recipes\d.json", {"A1": 1, "A2": 1, "A3": 1, "A4": 1}, {"A1": 9, "A2":8, "A3": 9, "A4": 9})
BATCH2 = (r"recipes\d.json", {"C1": 1, "C2": 1}, {"C1": 33, "C2": 32})
DEFAULT_SAVE_DIR = r"results\new1"
TIME_SCALE = 1.0


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _run_one_batch(raw, requirements, counts, time_limit, remaining_to_startup=False,
                     solve_startup=True, solve_close_down=True):
    """运行一次 batch 逻辑。
    
    remaining_to_startup=True: 剩余货物加入 init_state 由 start_up 处理
    remaining_to_startup=False: 剩余货物加入 end_state 由 close_down 处理
    solve_startup: 是否求解 start_up（过渡场景下第一批需要，第二批不需要）
    solve_close_down: 是否求解 close_down（过渡场景下第二批需要，第一批不需要）
    """
    cycle_result = cycle(raw, requirements, time_limit_s=time_limit, random_seed=42, interleave_search=True, num_search_workers=16)
    cycle_makespan = cycle_result.get("makespan", 0.0)
    base_actions = sorted(cycle_result.get("Move_List", []), key=lambda x: float(x.get("StartTime", 0.0)))

    init_state = build_initial_state_from_cycle(raw, cycle_result)
    init_state_goods = init_state.get("goods", [])
    goods_by_pr: Dict[str, int] = {}
    for g in init_state_goods:
        pid = g.get("pr_id")
        goods_by_pr[pid] = goods_by_pr.get(pid, 0) + 1

    k = min((count - goods_by_pr.get(pr_id, 0)) // requirements[pr_id] for pr_id, count in counts.items())

    # 剩余货物
    remaining_by_pr: Dict[str, int] = {}
    for pr_id, count in counts.items():
        r = int(count - goods_by_pr.get(pr_id, 0) - k * requirements[pr_id])
        remaining_by_pr[pr_id] = r

    if remaining_to_startup:
        # 剩余货物加入 init_state，由 start_up 生产
        next_gid = max((g["id"] for g in init_state_goods), default=-1) + 1
        for pr_id, r in remaining_by_pr.items():
            for _ in range(r):
                init_state_goods.append({"id": next_gid, "pr_id": pr_id, "mode": "noncyclic"})
                next_gid += 1
        init_state["goods"] = init_state_goods

    if solve_startup:
        print("  solving start_up...")
        enter_result = start_up(raw, init_state, time_limit_s=time_limit, num_search_workers=16)
        enter_actions = enter_result.get("Move_List", []) or []
        shift_enter = enter_result.get("start_up_time", 0.0)
    else:
        print("  skipping start_up (handled by transition)...")
        enter_result = {}
        enter_actions = []
        shift_enter = 0.0

    end_state = build_end_state_from_cycle(raw, cycle_result)

    if not remaining_to_startup:
        # 剩余货物加入 end_state，由 close_down 生产
        end_state_goods = end_state.get("goods", [])
        next_gid = max((g["id"] for g in end_state_goods), default=-1) + 1
        for pr_id, r in remaining_by_pr.items():
            for _ in range(r):
                end_state_goods.append({"id": next_gid, "pr_id": pr_id, "mode": "noncyclic"})
                next_gid += 1

    if solve_close_down:
        print("  solving close_down...")
        leave_result = close_down(raw, end_state, time_limit_s=time_limit, num_search_workers=16)
        leave_actions = leave_result.get("Move_List", []) or []
        leave_makespan = leave_result.get("close_down_time", 0.0)
    else:
        print("  skipping close_down (handled by transition)...")
        leave_result = {}
        leave_actions = []
        leave_makespan = 0.0

    # ── 返回各阶段原始结果（不在本函数内拼接）──
    return {
        "enter_actions": enter_actions,
        "cycle_actions": base_actions,
        "leave_actions": leave_actions,
        "init_state": init_state,
        "end_state": end_state,
        "cycle_result": cycle_result,
        "enter_result": enter_result,
        "leave_result": leave_result,
        "k": k,
        "cycle_makespan": cycle_makespan,
        "shift_enter": shift_enter,
        "leave_makespan": leave_makespan,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="两次 batch_run_noncyclic + transition 拼接")
    p.add_argument("--time-limit", type=int, default=600, help="Solver time limit seconds")
    p.add_argument("--out", dest="out_path", default=DEFAULT_SAVE_DIR, help="Output directory")
    p.add_argument("--counts1", type=str, default=None, help="第一批次的 counts, 格式 pr1_n1,pr2_n2")
    p.add_argument("--counts2", type=str, default=None, help="第二批次的 counts, 格式 pr1_n1,pr2_n2")
    args = p.parse_args()

    time_limit = int(args.time_limit)

    recipe1, requirements1, counts1_default = BATCH1
    recipe2, requirements2, counts2_default = BATCH2

    counts1 = dict(counts1_default)
    counts2 = dict(counts2_default)
    if args.counts1:
        for pair in args.counts1.split(","):
            k, v = pair.split("_")
            counts1[k] = int(v)
    if args.counts2:
        for pair in args.counts2.split(","):
            k, v = pair.split("_")
            counts2[k] = int(v)

    raw1 = load_recipe(str(recipe1), time_scale=TIME_SCALE)
    raw2 = load_recipe(str(recipe2), time_scale=TIME_SCALE) if recipe2 != recipe1 else raw1

    t_total_start = time.time()

    # ── 第一批（剩余货物由 start_up 处理，不需要 close_down）──
    print(f"\n===== Batch 1: {counts1} =====")
    t0 = time.time()
    r1 = _run_one_batch(raw1, requirements1, counts1, time_limit, remaining_to_startup=True,
                        solve_startup=True, solve_close_down=False)
    print(f"  Batch 1 done in {time.time() - t0:.1f}s")

    # ── 第二批（剩余货物由 close_down 处理，不需要 start_up）──
    print(f"\n===== Batch 2: {counts2} =====")
    t0 = time.time()
    r2 = _run_one_batch(raw2, requirements2, counts2, time_limit, remaining_to_startup=False,
                        solve_startup=False, solve_close_down=True)
    print(f"  Batch 2 done in {time.time() - t0:.1f}s")

    # ── 构建过渡状态（直接用批次求解的末态和初态）──
    print("\n===== Building transition states =====")
    end_state_A = r1["end_state"]
    init_state_B = r2["init_state"]
    end_goods = [g for g in end_state_A.get("goods", []) if g.get("mode") != "noncyclic"]
    end_state_A["goods"] = end_goods

    print(f"  End state A: {len(end_state_A.get('goods', []))} goods, {len(end_state_A.get('trucks', []))} trucks")
    print(f"  Init state B: {len(init_state_B.get('goods', []))} goods, {len(init_state_B.get('trucks', []))} trucks")

    # ── 求解过渡 ──
    print("\n===== Solving transition =====")
    t0 = time.time()
    trans_result = transition(raw2, end_state_A, init_state_B, time_limit_s=time_limit, num_search_workers=16)
    if trans_result is None:
        print("  Transition FAILED!")
        return
    trans_time = trans_result.get("transition_time", 0.0)
    trans_actions = trans_result.get("Move_List", [])
    print(f"  Transition solved in {time.time() - t0:.1f}s, time={trans_time:.3f}")

    # ── 拼接：A(start_up+cycle) + transition + B(cycle+close_down) ──
    print("\n===== Concatenating =====")
    final: List[Dict[str, Any]] = []

    shift_A = r1["shift_enter"]
    final.extend(r1["enter_actions"])
    for cycle_idx in range(r1["k"]):
        base_offset = cycle_idx * r1["cycle_makespan"]
        for act in r1["cycle_actions"]:
            a = dict(act)
            a["StartTime"] = round(shift_A + base_offset + float(a.get("StartTime", 0.0)), 3)
            a["EndTime"] = round(shift_A + base_offset + float(a.get("EndTime", 0.0)), 3)
            final.append(a)

    trans_base = shift_A + r1["k"] * r1["cycle_makespan"]
    for a in trans_actions:
        at = dict(a)
        at["StartTime"] = round(trans_base + float(a.get("StartTime", 0.0)), 3)
        at["EndTime"] = round(trans_base + float(a.get("EndTime", 0.0)), 3)
        final.append(at)

    # B 的 cycle 紧接 transition（不要 B 的 start_up）
    shift_B = trans_base + trans_time
    for cycle_idx in range(r2["k"]):
        base_offset = cycle_idx * r2["cycle_makespan"]
        for act in r2["cycle_actions"]:
            a = dict(act)
            a["StartTime"] = round(shift_B + base_offset + float(a.get("StartTime", 0.0)), 3)
            a["EndTime"] = round(shift_B + base_offset + float(a.get("EndTime", 0.0)), 3)
            final.append(a)

    leave_base = shift_B + r2["k"] * r2["cycle_makespan"]
    for a in r2["leave_actions"]:
        al = dict(a)
        al["StartTime"] = round(leave_base + float(a.get("StartTime", 0.0)), 3)
        al["EndTime"] = round(leave_base + float(a.get("EndTime", 0.0)), 3)
        final.append(al)

    final.sort(key=lambda x: (float(x.get("StartTime", 0.0))))
    final = assign_good_ids_by_occurrence(final)
    overall_makespan = leave_base + r2["leave_makespan"]

    # ── 只保存过渡结果 ──
    def _name(c): return "_".join(f"{k}_{v}" for k, v in sorted(c.items()))
    stem1, stem2 = Path(recipe1).stem, Path(recipe2).stem
    run_subdir = f"{_name(counts1)}__{_name(counts2)}"
    tag = stem1 if stem1 == stem2 else f"{stem1}_to_{stem2}"
    out_dir = Path(args.out_path) / tag / run_subdir / "transition_two"
    out_dir.mkdir(parents=True, exist_ok=True)

    total_counts = {}
    for k in set(list(counts1.keys()) + list(counts2.keys())):
        total_counts[k] = counts1.get(k, 0) + counts2.get(k, 0)

    final_obj = {
        "counts": total_counts,
        "makespan": overall_makespan,
        "cpu_time": round(time.time() - t_total_start, 3),
        "fluc_strategy": "max",
        "Move_List": final,
    }
    validation_result = validate(raw2, final_obj, total_counts)
    final_obj["validation"] = validation_result.get("errors", [])
    if validation_result.get("valid"):
        print("Validation passed!")
    else:
        print(f"Found {validation_result.get('error_count', 0)} validation errors.")

    out_path_final = out_dir / "final.json"
    save_json(final_obj, out_path_final)
    save_json(trans_result, out_dir / "transition.json")

    # ── 保存各 batch 求解的中间结果（跳过未求解的阶段）──
    def _save_batch(r, sub, save_close):
        d = out_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        save_json(r["initial_state"] if sub == "batch1" else r["init_state"], d / "initial_state.json")
        save_json(r["end_state"], d / "end_state.json")
        save_json(r["cycle_result"], d / "cycle.json")
        files_to_save = ["cycle.json"]
        if sub == "batch1":
            save_json(r["enter_result"], d / "start_up.json")
            files_to_save.append("start_up.json")
        if save_close:
            save_json(r["leave_result"], d / "close_down.json")
            files_to_save.append("close_down.json")
        for fname in files_to_save:
            json_path = d / fname
            png_path = json_path.with_suffix(".png")
            try:
                moves, mk, _ = load_moves(str(json_path))
                data = build_gantt_data(moves, mk)
                draw_gantt(data, title=f"{sub}/{fname} (makespan={mk})", save_path=str(png_path))
            except Exception as e:
                print(f"  绘制甘特图失败 {sub}/{fname}: {e}")

    # batch1 的 initial_state 用 r1["init_state"]（已加入剩余货物）
    r1_with_initial = {**r1, "initial_state": r1["init_state"]}
    _save_batch(r1_with_initial, "batch1", save_close=False)
    _save_batch(r2, "batch2", save_close=True)
    print(f"  Batch results saved to: {out_dir}")

    # 绘制过渡甘特图
    _trans_json = out_dir / "transition.json"
    _trans_png = _trans_json.with_suffix(".png")
    try:
        _moves, _mk, _st = load_moves(str(_trans_json))
        _data = build_gantt_data(_moves, _mk)
        draw_gantt(_data, title=f"transition (makespan={_mk})", save_path=str(_trans_png))
    except Exception as _e:
        print(f"  绘制过渡甘特图失败: {_e}")

    print(f"\nOverall makespan: {overall_makespan:.3f}")
    print(f"Saved to: {out_path_final}")


if __name__ == "__main__":
    main()
