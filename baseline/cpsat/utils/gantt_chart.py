"""
甘特图绘制工具
- 输入：包含 Move_List 的 JSON 文件（start_up / close_down / cycle / transition 等）
- 配对每个工厂内同一 good_id 的 load（放货）与 unload（取货）
- 实际加工时间 = load.EndTime → unload.StartTime
- 无法配对时：
    - 仅有 unload：加工时间 = 0 → unload.StartTime
    - 仅有 load：加工时间 = load.EndTime → makespan
- 机器人行：unload 用一种颜色，load 用另一种颜色，区间为 [StartTime, EndTime]
- 工厂行上叠加机器人动作的阴影（灰色半透明）
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ═══════════════════════════════════════════════════════════════
# 字段名兼容层：处理不同文件中的命名差异
# ═══════════════════════════════════════════════════════════════

# 字段别名映射：规范名 → [可能的别名]
_FIELD_ALIASES: Dict[str, List[str]] = {
    "Move_ID":     ["Move_ID", "move_id", "MoveID", "moveId", "id"],
    "StartTime":   ["StartTime", "start_time", "startTime", "Start", "start"],
    "EndTime":     ["EndTime", "end_time", "endTime", "End", "end"],
    "MoveType":    ["MoveType", "move_type", "moveType", "type", "Type"],
    "ModuleName":  ["ModuleName", "module_name", "moduleName", "Module", "module", "chamber"],
    "good_id":     ["good_id", "goodId", "GoodID", "GoodId", "wafer_id", "waferId"],
    "truck_id":    ["truck_id", "truckId", "TruckID", "TruckId", "robot_id", "robotId", "RobotID"],
    "pr_id":       ["pr_id", "prId", "recipe_id", "recipeId"],
    "step":        ["step", "Step", "step_index", "stepIndex"],
}

# 顶部字段别名
_TOP_FIELD_ALIASES: Dict[str, List[str]] = {
    "Move_List":   ["Move_List", "move_list", "MoveList", "moves", "Moves", "sequence"],
    "makespan":    ["makespan", "Makespan", "total_time", "TotalTime",
                    "start_up_time", "close_down_time", "cycle_time",
                    "transition_time", "noncyclic_time"],
    "status":      ["status", "Status", "result", "Result"],
}


def _build_alias_map(aliases: Dict[str, List[str]]) -> Dict[str, str]:
    """构建 别名→规范名 的反向映射."""
    out: Dict[str, str] = {}
    for canonical, variants in aliases.items():
        for v in variants:
            out[v] = canonical
    return out


_FIELD_MAP = _build_alias_map(_FIELD_ALIASES)
_TOP_MAP = _build_alias_map(_TOP_FIELD_ALIASES)


def _resolve_key(obj: Dict[str, Any], canonical: str,
                 alias_map: Dict[str, str]) -> Optional[Any]:
    """从字典中按规范名或别名取值."""
    if canonical in obj:
        return obj[canonical]
    for k, v in obj.items():
        if alias_map.get(k) == canonical:
            return v
    return None


def _get_str(obj: Dict[str, Any], canonical: str) -> str:
    val = _resolve_key(obj, canonical, _FIELD_MAP)
    return "" if val is None else str(val)


def _get_int(obj: Dict[str, Any], canonical: str, default: int = 0) -> int:
    val = _resolve_key(obj, canonical, _FIELD_MAP)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_moves(filepath: str) -> Tuple[List[Dict[str, Any]], int, str]:
    """
    加载 JSON 文件中的 Move_List，返回 (标准化后的 moves, makespan, status).
    兼容不同的顶层字段命名.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取 Move_List
    raw_moves = _resolve_key(data, "Move_List", _TOP_MAP)
    if raw_moves is None:
        # 尝试遍历顶层找到列表
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                raw_moves = v
                break
    if raw_moves is None:
        raise ValueError(f"无法在文件中找到 Move_List: {filepath}")

    # 获取 makespan：优先从显式字段，否则从 moves 计算
    makespan = None
    for canonical in _TOP_FIELD_ALIASES["makespan"]:
        if canonical in data:
            makespan = data[canonical]
            break
    if makespan is None:
        # 从 moves 推断
        max_end = 0
        for m in raw_moves:
            end = _get_int(m, "EndTime")
            if end > max_end:
                max_end = end
        makespan = max_end
    else:
        makespan = int(makespan)

    status = str(_resolve_key(data, "status", _TOP_MAP) or "UNKNOWN")

    # 标准化每条 move
    normalized: List[Dict[str, Any]] = []
    for i, m in enumerate(raw_moves):
        norm = {
            "Move_ID":    _get_int(m, "Move_ID", i),
            "StartTime":  _get_int(m, "StartTime"),
            "EndTime":    _get_int(m, "EndTime"),
            "MoveType":   _get_str(m, "MoveType").lower(),
            "ModuleName": _get_str(m, "ModuleName"),
            "good_id":    _get_str(m, "good_id"),
            "truck_id":   _get_str(m, "truck_id"),
            "pr_id":      _get_str(m, "pr_id"),
            "step":       _get_int(m, "step"),
        }
        # 过滤掉 move 类型的物理移动条目
        if norm["MoveType"] == "move":
            continue
        normalized.append(norm)

    # 重新计算 makespan（防止显式字段不准确）
    if normalized:
        computed = max(m["EndTime"] for m in normalized)
        makespan = max(makespan, computed)

    return normalized, makespan, status


# ═══════════════════════════════════════════════════════════════
# 配对逻辑
# ═══════════════════════════════════════════════════════════════

def pair_factory_operations(
    moves: List[Dict[str, Any]],
    makespan: int,
) -> Tuple[
    Dict[str, List[Tuple[Dict, Dict]]],   # pairs: factory → [(load, unload), ...]
    Dict[str, List[Dict]],                 # unpaired_unloads
    Dict[str, List[Dict]],                 # unpaired_loads
]:
    """
    对每个工厂，按 good_id 配对 load 与 unload.
    - load 后紧跟同 good_id 的 unload 即配对成功
    - 否则记入 unpaired 列表
    """
    # 按工厂分组
    factory_moves: Dict[str, List[Dict]] = defaultdict(list)
    for m in moves:
        factory_moves[m["ModuleName"]].append(m)

    # 每个工厂内按时间排序
    for f in factory_moves:
        factory_moves[f].sort(key=lambda x: (x["StartTime"], x["EndTime"], x["Move_ID"]))

    pairs: Dict[str, List[Tuple[Dict, Dict]]] = defaultdict(list)
    unpaired_unloads: Dict[str, List[Dict]] = defaultdict(list)
    unpaired_loads: Dict[str, List[Dict]] = defaultdict(list)

    for factory, fmoves in factory_moves.items():
        # 按 good_id 分组
        loads_by_good: Dict[str, List[Dict]] = defaultdict(list)
        unloads_by_good: Dict[str, List[Dict]] = defaultdict(list)

        for m in fmoves:
            gid = m["good_id"]
            if m["MoveType"] == "load":
                loads_by_good[gid].append(m)
            elif m["MoveType"] == "unload":
                unloads_by_good[gid].append(m)

        all_goods = set(loads_by_good.keys()) | set(unloads_by_good.keys())

        for gid in all_goods:
            loads = sorted(loads_by_good.get(gid, []),
                           key=lambda x: (x["StartTime"], x["EndTime"]))
            unloads = sorted(unloads_by_good.get(gid, []),
                             key=lambda x: (x["StartTime"], x["EndTime"]))

            # 贪心配对：每个 load 找之后最早的 unload
            ui = 0
            for load_m in loads:
                # 找第一个 StartTime >= load EndTime 的 unload
                while ui < len(unloads) and unloads[ui]["StartTime"] < load_m["EndTime"]:
                    # 这个 unload 在 load 完成之前就开始了，无法配对 → 作为 unpaired
                    unpaired_unloads[factory].append(unloads[ui])
                    ui += 1
                if ui < len(unloads):
                    pairs[factory].append((load_m, unloads[ui]))
                    ui += 1
                else:
                    unpaired_loads[factory].append(load_m)

            # 剩余无法配对的 unload
            while ui < len(unloads):
                unpaired_unloads[factory].append(unloads[ui])
                ui += 1

    return pairs, unpaired_unloads, unpaired_loads


# ═══════════════════════════════════════════════════════════════
# 构建画图所需的数据
# ═══════════════════════════════════════════════════════════════

def build_gantt_data(
    moves: List[Dict[str, Any]],
    makespan: int,
) -> Dict[str, Any]:
    """从原始 moves 构建所有画图所需的数据结构."""

    # ── 工厂分组 ──
    factory_groups: Dict[str, List[Dict]] = defaultdict(list)
    for m in moves:
        factory_groups[m["ModuleName"]].append(m)

    # ── 机器人分组 ──
    robot_groups: Dict[str, List[Dict]] = defaultdict(list)
    for m in moves:
        if m["truck_id"]:
            robot_groups[m["truck_id"]].append(m)

    # ── 配对 ──
    pairs, unpaired_unloads, unpaired_loads = pair_factory_operations(moves, makespan)

    # ── 每个工厂的加工区间 ──
    factory_process_intervals: Dict[str, List[Tuple[int, int, str, Dict]]] = defaultdict(list)
    # (start, end, type, metadata)  type = "process" | "unpaired_unload" | "unpaired_load"

    for factory, pair_list in pairs.items():
        for load_m, unload_m in pair_list:
            proc_start = load_m["EndTime"]
            proc_end = unload_m["StartTime"]
            if proc_end > proc_start:
                factory_process_intervals[factory].append(
                    (proc_start, proc_end, "process",
                     {"good_id": load_m["good_id"], "pr_id": load_m["pr_id"]})
                )

    for factory, ul_list in unpaired_unloads.items():
        for m in ul_list:
            proc_end = m["StartTime"]
            if proc_end > 0:
                factory_process_intervals[factory].append(
                    (0, proc_end, "unpaired_unload",
                     {"good_id": m["good_id"], "pr_id": m["pr_id"]})
                )

    for factory, l_list in unpaired_loads.items():
        for m in l_list:
            proc_start = m["EndTime"]
            if proc_start < makespan:
                factory_process_intervals[factory].append(
                    (proc_start, makespan, "unpaired_load",
                     {"good_id": m["good_id"], "pr_id": m["pr_id"]})
                )

    # ── 合并工厂动作（load/unload 阴影区间） ──
    factory_action_intervals: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)
    # (start, end, type)  type = "load" | "unload"
    for m in moves:
        factory_action_intervals[m["ModuleName"]].append(
            (m["StartTime"], m["EndTime"], m["MoveType"])
        )

    return {
        "moves": moves,
        "makespan": makespan,
        "factory_groups": dict(factory_groups),
        "robot_groups": dict(robot_groups),
        "pairs": dict(pairs),
        "unpaired_unloads": dict(unpaired_unloads),
        "unpaired_loads": dict(unpaired_loads),
        "factory_process_intervals": dict(factory_process_intervals),
        "factory_action_intervals": dict(factory_action_intervals),
    }


# ═══════════════════════════════════════════════════════════════
# 绘图
# ═══════════════════════════════════════════════════════════════

def _get_factory_color_map(all_factories: List[str]) -> Dict[str, str]:
    """为每个工厂分配唯一颜色，使用 tab20 色表."""
    cmap = plt.colormaps.get_cmap('tab20')
    n = len(all_factories)
    color_map: Dict[str, str] = {}
    for i, f in enumerate(all_factories):
        rgba = cmap(i % 20)
        color_map[f] = matplotlib.colors.to_hex(rgba)
    return color_map


def _get_robot_color(move_type: str) -> str:
    """机器人动作颜色."""
    if move_type == "load":
        return "#4CAF50"   # 绿色 = 放货
    else:
        return "#2196F3"   # 蓝色 = 取货


def draw_gantt(
    data: Dict[str, Any],
    title: str = "Gantt Chart",
    figsize: Tuple[float, float] = None,
    save_path: Optional[str] = None,
):
    """绘制甘特图."""

    makespan = data["makespan"]
    factory_process = data["factory_process_intervals"]
    factory_actions = data["factory_action_intervals"]
    robot_groups = data["robot_groups"]

    # ── 确定行顺序 ──
    # 工厂行按名称排序，跳过 LP (Load Port)
    all_factories = sorted(set(
        list(factory_process.keys()) + list(factory_actions.keys())
    ), key=_factory_sort_key)
    all_factories = [f for f in all_factories if f.upper() != "LP"]

    # 机器人行按名称排序
    all_robots = sorted(robot_groups.keys(), key=lambda x: int(re.sub(r'\D', '', x) or 0))

    # ── 布局：工厂行 + 分隔 + 机器人行 ──
    n_factory_rows = len(all_factories)
    n_robot_rows = len(all_robots)
    n_total_rows = n_factory_rows + n_robot_rows

    # 动态计算图大小
    if figsize is None:
        width = max(14, makespan / 60)
        height = max(6, n_total_rows * 0.8)
        figsize = (width, height)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.08)

    # ── Y轴标签映射 ──
    y_labels = []
    y_positions = []

    # 工厂行 (从上到下)
    for i, factory in enumerate(all_factories):
        y = n_total_rows - 1 - i
        y_labels.append(factory)
        y_positions.append(y)

    # 机器人行
    robot_row_start = n_robot_rows - 1
    for i, robot in enumerate(all_robots):
        y = robot_row_start - i
        y_labels.append(robot)
        y_positions.append(y)

    # ── 工厂颜色映射 ──
    factory_color_map = _get_factory_color_map(all_factories)

    # ── 绘制工厂行 ──
    for i, factory in enumerate(all_factories):
        y = n_total_rows - 1 - i
        bar_height = 0.5
        fac_color = factory_color_map.get(factory, "#BDBDBD")

        # 1) 先绘制阴影（统一灰色+条纹，高度与加工条完全一致）
        for (start, end, mtype) in factory_actions.get(factory, []):
            ax.barh(y, end - start, height=bar_height, left=start,
                    color="#AAAAAA", alpha=0.35, edgecolor="none",
                    hatch="////", linewidth=0, zorder=1)

        # 2) 绘制加工时间色块
        intervals = factory_process.get(factory, [])
        for (start, end, ptype, meta) in intervals:
            duration = end - start
            if duration <= 0:
                continue

            ax.barh(y, duration, height=bar_height, left=start,
                    color=fac_color, edgecolor="#555555", linewidth=0.5,
                    alpha=0.9, zorder=2)

        # 3) 绘制外框
        for (start, end, ptype, meta) in intervals:
            duration = end - start
            if duration <= 0:
                continue
            rect = mpatches.FancyBboxPatch(
                (start, y - bar_height / 2), duration, bar_height,
                boxstyle="round,pad=0.02", facecolor="none",
                edgecolor="#333333", linewidth=0.8, zorder=4
            )
            ax.add_patch(rect)

    # ── 绘制机器人行 ──
    for i, robot in enumerate(all_robots):
        y = robot_row_start - i

        for m in robot_groups.get(robot, []):
            start = m["StartTime"]
            end = m["EndTime"]
            duration = end - start
            if duration <= 0:
                continue
            mtype = m["MoveType"]
            color = _get_robot_color(mtype)

            ax.barh(y, duration, height=0.5, left=start,
                    color=color, edgecolor="#333333", linewidth=0.8,
                    alpha=0.9, zorder=5)

    # ── 刻度和标签 ──
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_xlabel("Time", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # 用虚线分隔工厂和机器人区域
    if n_factory_rows > 0 and n_robot_rows > 0:
        sep_y = n_robot_rows - 0.5
        ax.axhline(y=sep_y, color="#333333", linewidth=1.2, linestyle="-")

        # 添加区域标签
        ax.text(makespan * 0.005, n_total_rows - 0.5, "FACTORIES",
                fontsize=7, fontweight="bold", color="#666666",
                va="bottom", ha="left")
        ax.text(makespan * 0.005, n_robot_rows - 0.5, "ROBOTS",
                fontsize=7, fontweight="bold", color="#666666",
                va="bottom", ha="left")

    ax.set_xlim(0, makespan * 1.02)
    ax.set_ylim(-0.5, n_total_rows - 0.5)
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.invert_yaxis()

    # ── 图例 ──
    legend_elements = [
        mpatches.Patch(facecolor="#4CAF50", edgecolor="#333", label="Robot L (放货)"),
        mpatches.Patch(facecolor="#2196F3", edgecolor="#333", label="Robot U (取货)"),
    ]
    # 为每个工厂添加图例
    for factory in all_factories:
        legend_elements.append(
            mpatches.Patch(facecolor=factory_color_map.get(factory, "#BDBDBD"),
                           edgecolor="#333", label=factory)
        )

    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=7, ncol=2, framealpha=0.9)

    # ── 保存或显示 ──
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"甘特图已保存至: {save_path}")
    else:
        plt.show()

    return fig, ax


def _factory_sort_key(name: str) -> Tuple[int, str, int]:
    """工厂排序键：LP > BA > BB > PA > PB > PC，同类型按数字排序."""
    type_order = {"LP": 0, "BA": 1, "BB": 2, "PA": 3, "PB": 4, "PC": 5}
    # 提取字母前缀和数字
    m = re.match(r"([A-Za-z]+)(\d*)", name)
    if m:
        prefix = m.group(1).upper()
        num = int(m.group(2)) if m.group(2) else 0
        return (type_order.get(prefix[:2], 99), prefix, num)
    return (99, name, 0)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="绘制 IC 优化调度甘特图",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python gantt_chart.py results/a/A1_2_A2_2/start_up.json
  python gantt_chart.py results/d/C1_32_C2_33__A1_18_A2_17/transition_two_max/transition.json -o gantt.png
        """,
    )
    parser.add_argument("input", nargs="?", default=r"results\e_travel\A3_1_A4_1_A5_3_A6_3_cycle.json", help="输入 JSON 文件路径")
    parser.add_argument("-o", "--output", nargs="?", help="输出图片路径", default=r"results\e_travel\cycle.png")
    parser.add_argument("--title", help="自定义标题", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        return

    # 默认保存路径：与输入 JSON 同目录，文件名相同但后缀为 .png
    if args.output is None:
        args.output = str(input_path.with_suffix(".png"))

    # 加载数据
    print(f"加载文件: {input_path}")
    moves, makespan, status = load_moves(str(input_path))
    print(f"  Move 数量: {len(moves)}")
    print(f"  Makespan: {makespan}")
    print(f"  Status: {status}")

    # 构建数据
    data = build_gantt_data(moves, makespan)

    # 统计信息
    n_pairs = sum(len(v) for v in data["pairs"].values())
    n_uul = sum(len(v) for v in data["unpaired_unloads"].values())
    n_ul = sum(len(v) for v in data["unpaired_loads"].values())
    n_factories = len(data["factory_groups"])
    n_robots = len(data["robot_groups"])
    print(f"  工厂数: {n_factories}, 机器人数: {n_robots}")
    print(f"  配对成功: {n_pairs}, 未配对unload: {n_uul}, 未配对load: {n_ul}")

    # 标题
    title = args.title or f"{input_path.stem}  (makespan={makespan}, status={status})"

    # 绘图
    draw_gantt(
        data,
        title=title,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
