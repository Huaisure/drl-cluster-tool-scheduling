"""
晶圆级甘特图 - 交互式 HTML 版
- 使用 Plotly 生成可拖动、缩放的网页
- 每行一个晶圆，横轴为时间
- 仅显示加工段，不同 chamber 类型不同颜色
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ---- 复用 wafer_gantt.py 的数据加载和配对逻辑 ----
# ---- 复用 utils.py 的 load_recipe ----
import sys
_utils_dir = str(Path(__file__).resolve().parent)
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)
from utils.utils import load_recipe, resolve_fluc


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

_TOP_FIELD_ALIASES: Dict[str, List[str]] = {
    "Move_List":   ["Move_List", "move_list", "MoveList", "moves", "Moves", "sequence"],
    "makespan":    ["makespan", "Makespan", "total_time", "TotalTime",
                    "start_up_time", "close_down_time", "cycle_time",
                    "transition_time", "noncyclic_time"],
    "status":      ["status", "Status", "result", "Result"],
}

# 四种段类型的颜色（高饱和度）
_SEGMENT_COLORS = {
    "factory":           "#00E676",   # 亮绿 - 工厂段
    "residency_process": "#FF1744",   # 亮红 - 驻留段(process工厂)
    "residency_normal":  "#FFD600",   # 亮黄 - 驻留段(其他工厂)
    "truck":             "#2979FF",   # 亮蓝 - 卡车段
}

_SEGMENT_LABELS = {
    "factory":           "工厂段 (加工)",
    "residency_process": "驻留段 (process)",
    "residency_normal":  "驻留段 (其他)",
    "truck":             "卡车段 (运输)",
}


def _build_alias_map(aliases):
    out = {}
    for canonical, variants in aliases.items():
        for v in variants:
            out[v] = canonical
    return out

_FIELD_MAP = _build_alias_map(_FIELD_ALIASES)
_TOP_MAP = _build_alias_map(_TOP_FIELD_ALIASES)


def _resolve_key(obj, canonical, alias_map):
    if canonical in obj:
        return obj[canonical]
    for k, v in obj.items():
        if alias_map.get(k) == canonical:
            return v
    return None


def _get_str(obj, canonical):
    val = _resolve_key(obj, canonical, _FIELD_MAP)
    return "" if val is None else str(val)


def _get_int(obj, canonical, default=0):
    val = _resolve_key(obj, canonical, _FIELD_MAP)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def load_moves(filepath: str) -> Tuple[List[Dict[str, Any]], int]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_moves = _resolve_key(data, "Move_List", _TOP_MAP)
    if raw_moves is None:
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                raw_moves = v
                break
    if raw_moves is None:
        raise ValueError(f"无法找到 Move_List: {filepath}")

    makespan = None
    for canonical in _TOP_FIELD_ALIASES["makespan"]:
        if canonical in data:
            makespan = data[canonical]
            break
    if makespan is None:
        max_end = 0
        for m in raw_moves:
            end = _get_int(m, "EndTime")
            if end > max_end:
                max_end = end
        makespan = max_end
    else:
        makespan = int(makespan)

    normalized = []
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
        if norm["MoveType"] == "move":
            continue
        normalized.append(norm)

    if normalized:
        computed = max(m["EndTime"] for m in normalized)
        makespan = max(makespan, computed)

    return normalized, makespan


def pair_wafer_processes(
    moves: List[Dict[str, Any]],
    makespan: int,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """配对 load→unload，返回工厂占用段（load.start → unload.end）。

    每个段 dict:
        start:       load.StartTime   (工厂段开始 = 装载开始)
        end:         unload.EndTime   (工厂段结束 = 卸载结束)
        step:        load.step
        chamber:     chamber name
        pr_id:       recipe id
        load_end:    load.EndTime     (装载完成时刻)
        unload_start: unload.StartTime (卸载开始时刻)
    """
    wafer_moves = defaultdict(list)
    for m in moves:
        wafer_moves[(m["pr_id"], m["good_id"])].append(m)

    wafer_processes: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for wafer_key, wmoves in wafer_moves.items():
        pr_id = wafer_key[0]
        chamber_moves = defaultdict(list)
        for m in wmoves:
            chamber_moves[m["ModuleName"]].append(m)

        for chamber, cmoves in chamber_moves.items():
            if chamber.upper() == "LP":
                continue
            cmoves.sort(key=lambda x: (x["StartTime"], x["EndTime"]))
            loads = [m for m in cmoves if m["MoveType"] == "load"]
            unloads = [m for m in cmoves if m["MoveType"] == "unload"]
            loads.sort(key=lambda x: x["StartTime"])
            unloads.sort(key=lambda x: x["StartTime"])

            ui = 0
            for load_m in loads:
                while ui < len(unloads) and unloads[ui]["StartTime"] < load_m["EndTime"]:
                    ui += 1
                if ui < len(unloads):
                    unload_m = unloads[ui]
                    bar_start = load_m["StartTime"]
                    bar_end = unload_m["EndTime"]
                    if bar_end > bar_start:
                        wafer_processes[wafer_key].append({
                            "start": bar_start,
                            "end": bar_end,
                            "step": load_m["step"],
                            "chamber": chamber,
                            "pr_id": pr_id,
                            "load_end": load_m["EndTime"],
                            "unload_start": unload_m["StartTime"],
                        })
                    ui += 1

        wafer_processes[wafer_key].sort(key=lambda x: x["start"])
    return wafer_processes



# ═══════════════════════════════════════════════════════════════
# 三段拆分：工厂段 / 驻留段 / 卡车段
# ═══════════════════════════════════════════════════════════════

def build_segments(
    moves: List[Dict[str, Any]],
    recipe_step_times: Dict[Tuple[str, int], int],
    factory_types: Dict[str, str],
) -> Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]]:
    """从原始 moves 直接生成四种段。

    对每个货物，将其动作按时间排序后遍历：
    - 放货(load):  找下一次取货(unload) → 工厂段 + 驻留段（LP 除外）
                    工厂段:   load.EndTime → load.EndTime + process_time
                    驻留段:   load.EndTime + process_time → unload.StartTime
                    （按工厂类型分 process / normal）
    - 取货(unload): 找下一次放货(load)   → 卡车段（含 LP）
                    卡车段:   unload.StartTime → next_load.EndTime

    返回: {wafer_key: {"factory": [...], "residency_process": [...],
                       "residency_normal": [...], "truck": [...]}}
    """
    # 按货物分组（包含 LP，卡车段需要 LP 的取货/放货）
    wafer_moves: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for m in moves:
        wafer_moves[(m["pr_id"], m["good_id"])].append(m)

    result: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}
    for wafer_key, wmoves in wafer_moves.items():
        pr_id, good_id = wafer_key
        wmoves.sort(key=lambda x: (x["StartTime"], x["EndTime"]))

        factory_segs: List[Dict[str, Any]] = []
        residency_process_segs: List[Dict[str, Any]] = []
        residency_normal_segs: List[Dict[str, Any]] = []
        truck_segs: List[Dict[str, Any]] = []

        for i, m in enumerate(wmoves):
            if m["MoveType"] == "load":
                # 放货 → 找下一次取货(unload)
                next_unload = None
                for j in range(i + 1, len(wmoves)):
                    if wmoves[j]["MoveType"] == "unload":
                        next_unload = wmoves[j]
                        break
                if next_unload is None:
                    continue

                load_end = m["EndTime"]
                unload_start = next_unload["StartTime"]
                step = m["step"]
                chamber = m["ModuleName"]

                # LP 放货不产生工厂段和驻留段
                if chamber.upper() == "LP":
                    continue

                pt = recipe_step_times.get((pr_id, step), 0)

                # 工厂段: 放货结束 → 放货结束 + 加工时间
                factory_end = load_end + pt
                if factory_end > load_end:
                    factory_segs.append({
                        "start": load_end,
                        "end": factory_end,
                        "step": step,
                        "chamber": chamber,
                        "pr_id": pr_id,
                    })

                # 驻留段: 按工厂类型区分
                residency_start = load_end + pt
                if unload_start > residency_start:
                    resid_seg = {
                        "start": residency_start,
                        "end": unload_start,
                        "step": step,
                        "chamber": chamber,
                        "pr_id": pr_id,
                    }
                    if factory_types.get(chamber, "normal") == "process":
                        residency_process_segs.append(resid_seg)
                    else:
                        residency_normal_segs.append(resid_seg)

            elif m["MoveType"] == "unload":
                # 取货 → 找下一次放货(load)，生成卡车段（含 LP）
                next_load = None
                for j in range(i + 1, len(wmoves)):
                    if wmoves[j]["MoveType"] == "load":
                        next_load = wmoves[j]
                        break
                if next_load is None:
                    continue

                unload_start = m["StartTime"]
                load_end = next_load["EndTime"]
                if load_end > unload_start:
                    truck_segs.append({
                        "start": unload_start,
                        "end": load_end,
                        "step": m["step"],
                        "chamber": "TRUCK",
                        "pr_id": pr_id,
                        "from_chamber": m["ModuleName"],
                        "to_chamber": next_load["ModuleName"],
                    })

        result[wafer_key] = {
            "factory": factory_segs,
            "residency_process": residency_process_segs,
            "residency_normal": residency_normal_segs,
            "truck": truck_segs,
        }

    return result


# ═══════════════════════════════════════════════════════════════
# 生成交互式 HTML
# ═══════════════════════════════════════════════════════════════

def generate_html(
    segments: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]],
    makespan: int,
    title: str = "Wafer Gantt Chart",
    save_path: Optional[str] = None,
):
    """生成带 Plotly 交互式甘特图的 HTML 文件。

    四种段类型，各用一种统一颜色：
    - 工厂段 (factory):           放货结束 → 放货结束 + 加工时间（亮绿）
    - 驻留段 (residency_process):  process工厂驻留（亮红）
    - 驻留段 (residency_normal):   其他工厂驻留（亮黄）
    - 卡车段 (truck):             取货开始 → 放货结束（亮蓝）
    """
    # 按首次出现时间排序
    def sort_key(item):
        all_segs = (item[1].get("factory", []) + item[1].get("truck", [])
                    + item[1].get("residency_process", [])
                    + item[1].get("residency_normal", []))
        if all_segs:
            return min(s["start"] for s in all_segs)
        return float('inf')

    sorted_wafers = sorted(segments.items(), key=sort_key)
    wafer_labels = [f"{pr_id}-{good_id}" for (pr_id, good_id), _ in sorted_wafers]

    import plotly.graph_objects as go

    fig = go.Figure()

    segment_types = ["factory", "residency_process", "residency_normal", "truck"]
    for seg_type in segment_types:
        seg_list: List[Dict[str, Any]] = []
        for (pr_id, good_id), seg_dict in sorted_wafers:
            wafer_label = f"{pr_id}-{good_id}"
            for seg in seg_dict.get(seg_type, []):
                start = seg["start"]
                end = seg["end"]
                duration = end - start
                if duration <= 0:
                    continue
                chamber = seg.get("chamber", "")
                step = seg.get("step", -1)

                # 构建 hover 文本
                if seg_type == "factory":
                    hover_lines = [
                        f"类型: 工厂段（加工）",
                        f"chamber: {chamber}",
                        f"step: {step}",
                        f"加工: {start}→{end} (Δt={duration})",
                    ]
                elif seg_type.startswith("residency"):
                    hover_lines = [
                        f"类型: 驻留段",
                        f"chamber: {chamber}",
                        f"step: {step}",
                        f"驻留: {start}→{end} (Δt={duration})",
                    ]
                else:  # truck
                    from_ch = seg.get("from_chamber", "?")
                    to_ch = seg.get("to_chamber", "?")
                    hover_lines = [
                        f"类型: 卡车段（运输）",
                        f"路径: {from_ch} → {to_ch}",
                        f"运输: {start}→{end} (Δt={duration})",
                    ]

                hover_text = "<br>".join(hover_lines)

                seg_list.append({
                    "wafer": wafer_label,
                    "start": start,
                    "duration": duration,
                    "hover_text": hover_text,
                })

        if seg_list:
            color = _SEGMENT_COLORS[seg_type]
            label = _SEGMENT_LABELS[seg_type]

            fig.add_trace(go.Bar(
                name=label,
                y=[s["wafer"] for s in seg_list],
                x=[s["duration"] for s in seg_list],
                base=[s["start"] for s in seg_list],
                orientation='h',
                marker=dict(
                    color=color,
                    line=dict(color='#333', width=0.5),
                ),
                text=[s["hover_text"] for s in seg_list],
                textposition='none',
                hovertemplate='<b>%{text}</b><extra></extra>',
                showlegend=True,
            ))

    chart_height = max(600, len(wafer_labels) * 22)
    chart_width = max(1200, makespan)

    # 布局
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#333')),
        xaxis=dict(
            title="Time (s)",
            range=[0, makespan],
            rangeslider=dict(visible=False),
            type='linear',
            gridcolor='#e0e0e0',
            fixedrange=False,
        ),
        yaxis=dict(
            title="Wafer",
            categoryorder='array',
            categoryarray=wafer_labels,
            dtick=1,
            gridcolor='#f0f0f0',
            fixedrange=False,
        ),
        dragmode='pan',
        barmode='stack',
        legend=dict(
            title=dict(text="段类型"),
            orientation='v',
            x=1.02,
            y=1,
        ),
        height=chart_height,
        margin=dict(l=60, r=100, t=50, b=30),
        hovermode='closest',
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(color='#333', size=12),
    )

    # 保存
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ background: #fff; font-family: 'Segoe UI', sans-serif; }}
        #container {{
            width: 100vw;
            height: 100vh;
            overflow: auto;
        }}
        #chart {{ width: {chart_width}px; height: {chart_height}px; }}
    </style>
</head>
<body>
    <div id="container">
        <div id="chart"></div>
    </div>
    <script>
        var figData = {fig.to_json()};
        var layout = figData.layout;

        var config = {{
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false,
            scrollZoom: false,
            responsive: true,
        }};
        Plotly.newPlot('chart', figData.data, layout, config);
    </script>
</body>
</html>"""

    if save_path is None:
        save_path = "wafer_gantt.html"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"交互式甘特图已保存至: {save_path}")
    return save_path


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="生成交互式晶圆甘特图 HTML")
    parser.add_argument("input", nargs="?", default=r"results\new5\e\A1_1_A2_1_A3_1_A4_1_A5_1_A6_1\noncyclic_A1_16_A2_16_A3_16_A4_16_A5_16_A6_16.json", help="输入 JSON 文件路径")
    parser.add_argument("-o", "--output", nargs="?", help="输出 HTML 路径", default=r"results\new5\e\A1_1_A2_1_A3_1_A4_1_A5_1_A6_1\e.html")
    parser.add_argument("--title", help="自定义标题", default="e")
    parser.add_argument("--recipes", help="recipes 目录路径",
                        default=r"recipes\e.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        return

    # 确定 recipes 目录
    if args.recipes:
        recipes_dir = args.recipes
    else:
        # 默认: 脚本所在目录的 ../recipes
        recipes_dir = str(Path(__file__).resolve().parent.parent / "recipes")

    # 加载 recipe 信息
    print(f"加载 recipe 信息: {recipes_dir}")
    factory_types: Dict[str, str] = {}
    recipe_step_times: Dict[Tuple[str, int], int] = {}
    recipe_step_ranges: Dict[Tuple[str, int], str] = {}

    recipes_path = Path(recipes_dir)
    if recipes_path.is_file():
        recipe_files = [recipes_path]
    elif recipes_path.is_dir():
        recipe_files = sorted(recipes_path.glob("*.json"))
    else:
        recipe_files = []
        print(f"  警告: recipes 路径不存在: {recipes_dir}")

    if recipe_files:
        for recipe_file in recipe_files:
            try:
                raw = load_recipe(str(recipe_file), time_scale=1)
            except Exception as e:
                print(f"  警告: 无法加载 {recipe_file}: {e}")
                continue

            # 提取工厂类型
            for factory in raw.get("factories", []):
                fid = factory.get("id", "")
                ftype = factory.get("type", "normal")
                if fid:
                    factory_types[fid] = ftype

            # 提取配方步骤的 process_time（已由 resolve_fluc 解析为上限值）
            for pr_id, steps in raw.get("recipes", {}).items():
                for step_idx, step in enumerate(steps):
                    pt = step.get("process_time")
                    if pt is not None:
                        recipe_step_times[(pr_id, step_idx)] = int(pt)
                    pr_range = step.get("process_time_range")
                    if pr_range and len(pr_range) == 2:
                        recipe_step_ranges[(pr_id, step_idx)] = f"{pr_range[0]}~{pr_range[1]}"
                    elif pt is not None:
                        recipe_step_ranges[(pr_id, step_idx)] = str(int(pt))
    else:
        print(f"  警告: recipes 目录不存在: {recipes_dir}")

    n_process = sum(1 for ft in factory_types.values() if ft == "process")
    print(f"  工厂类型: {len(factory_types)} 个 (其中 process: {n_process})")
    print(f"  配方步骤: {len(recipe_step_times)} 条")

    if args.output is None:
        args.output = str(input_path.with_suffix(".html"))

    print(f"加载文件: {input_path}")
    moves, makespan = load_moves(str(input_path))
    print(f"  Move 数量: {len(moves)}, Makespan: {makespan}")

    # 统计晶圆数
    wafer_ids: set = set()
    for m in moves:
        wafer_ids.add((m["pr_id"], m["good_id"]))
    print(f"  晶圆数: {len(wafer_ids)}")

    # 拆分为四种段：工厂段 / 驻留段(process) / 驻留段(其他) / 卡车段
    segments = build_segments(moves, recipe_step_times, factory_types)
    n_factory = sum(len(v["factory"]) for v in segments.values())
    n_res_p = sum(len(v["residency_process"]) for v in segments.values())
    n_res_n = sum(len(v["residency_normal"]) for v in segments.values())
    n_truck = sum(len(v["truck"]) for v in segments.values())
    print(f"  工厂段: {n_factory}, 驻留段(process): {n_res_p}, 驻留段(其他): {n_res_n}, 卡车段: {n_truck}")

    title = args.title or input_path.stem
    generate_html(segments, makespan, title=title, save_path=args.output)


if __name__ == "__main__":
    main()
