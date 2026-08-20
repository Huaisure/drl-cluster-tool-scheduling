import json
import math
from typing import Dict, List, Optional, Union, Tuple, Any

def resolve_fluc(value):
    """将波动加工时间 [min, max] 统一取上限解析为单一值。

    Args:
        value: 原始 process_time，可以是 None / int / float / [min, max]

    Returns:
        解析后的单一数值，或 None
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list) and len(value) == 2:
        return float(value[1])
    raise TypeError(f"Unexpected process_time type: {type(value)} value={value!r}")


def scale_up(value, time_scale=1):
    if value is None:
        return None
    return int(math.ceil(value * time_scale))


def get_travel_time(truck: Dict, from_loc: str, to_loc: str) -> int:
    """从卡车的 travel_times 矩阵中查询两个位置之间的移动时间。"""
    travel_times = truck.get("travel_times", {})
    if not isinstance(travel_times, dict):
        return 0
    return int(travel_times.get(from_loc, {}).get(to_loc, 0) or 0)


def get_max_travel_time(truck: Dict) -> int:
    """获取卡车 travel_times 矩阵中的最大移动时间。"""
    travel_times = truck.get("travel_times", {})
    if not isinstance(travel_times, dict):
        return 0
    max_val = 0
    for dests in travel_times.values():
        if isinstance(dests, dict):
            for t in dests.values():
                max_val = max(max_val, int(t or 0))
    return max_val


def load_recipe(path, time_scale=1):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    raw_trucks = data.get("trucks", [])
    factories = data.get("factories", [])
    recipes = data.get("process_recipes", [])
    interfere_time = data.get("interfere_time", 0)
    interfere_time = scale_up(interfere_time, time_scale)

    normalized_recipes = {}   # id -> steps

    for recipe in recipes:
        normalized_steps = []
        for step in recipe.get("steps", []):
            s = step.copy()
            raw = step.get("process_time")
            resolved = resolve_fluc(raw)
            s["process_time"] = scale_up(resolved, time_scale)
            # 保留原始波动信息，供后续分析使用
            if isinstance(raw, list) and len(raw) == 2:
                s["process_time_range"] = [
                    scale_up(raw[0], time_scale),
                    scale_up(raw[1], time_scale),
                ]
            normalized_steps.append(s)
        normalized_recipes[recipe["id"]] = normalized_steps
    
    # 缩放卡车时间（单臂机器人，不拆分容量）
    scaled_trucks = []
    for truck in raw_trucks:
        t = truck.copy()
        t["load_time"] = scale_up(truck.get("load_time"), time_scale)
        t["unload_time"] = scale_up(truck.get("unload_time"), time_scale)

        # 处理移动时间：矩阵直接缩放，标量自动转换为矩阵
        travel_times = truck.get("travel_times")
        if travel_times and isinstance(travel_times, dict):
            scaled_matrix = {}
            for from_loc, dests in travel_times.items():
                scaled_matrix[from_loc] = {
                    to_loc: scale_up(val, time_scale)
                    for to_loc, val in dests.items()
                }
            t["travel_times"] = scaled_matrix
        else:
            scalar = truck.get("travel_time")
            locations = truck.get("available_location", [])
            if scalar is not None and locations:
                scaled_val = scale_up(scalar, time_scale)
                matrix = {}
                for loc_a in locations:
                    matrix[loc_a] = {}
                    for loc_b in locations:
                        matrix[loc_a][loc_b] = 0 if loc_a == loc_b else scaled_val
                t["travel_times"] = matrix
            else:
                t["travel_times"] = {}

        scaled_trucks.append(t)
    return {
        "trucks": scaled_trucks,
        "factories": factories,
        "recipes": normalized_recipes,
        "interfere_time": interfere_time,
    }


def _build_recipe_process_times(recipes: Dict[str, List[Dict]]) -> Dict[str, Dict[int, int]]:
    process_times: Dict[str, Dict[int, int]] = {}
    residency_times: Dict[str, Dict[int, int]] = {}
    for pr_id, steps in recipes.items():
        step_times1 = {}
        step_times2 = {}
        for idx, step in enumerate(steps):
            step_times1[idx] = int(step.get("process_time") or 0)
            step_times2[idx] = int(step.get("residency_time")) if step.get("residency_time") is not None else None
        process_times[str(pr_id)] = step_times1
        residency_times[str(pr_id)] = step_times2
    return process_times, residency_times


def calc_process_residency_stats(
    moves: List[Dict[str, Any]],
    factory_types: Dict[str, str],
    recipe_step_times: Dict[Tuple[str, int], int],
) -> Dict[str, Any]:
    """计算所有 process 类型工厂的驻留时间统计。

    驻留时间 = unload.StartTime - (load.EndTime + process_time)
    仅统计工厂类型为 "process" 的 chamber（不含 LP）。

    Args:
        moves: 标准化后的 Move_List，每个元素含 StartTime, EndTime, MoveType,
               ModuleName, good_id, pr_id, step
        factory_types: {工厂ID: 类型}，类型为 "process" / "normal" / "LP"
        recipe_step_times: {(pr_id, step): process_time}

    Returns:
        {
            "avg": 平均驻留时间,
            "count": 统计次数,
            "max": 最大驻留时间,
            "min": 最小驻留时间,
            "by_chamber": {chamber: {"avg": ..., "count": ..., "max": ..., "min": ...}},
            "all_values": [所有驻留时间值],
        }
    """
    from collections import defaultdict

    # 按 wafer 分组
    wafer_moves = defaultdict(list)
    for m in moves:
        wafer_moves[(str(m.get("pr_id", "")), str(m.get("good_id", "")))].append(m)

    all_residencies: List[float] = []
    by_chamber: Dict[str, List[float]] = defaultdict(list)

    for wafer_key, wmoves in wafer_moves.items():
        pr_id = wafer_key[0]
        wmoves.sort(key=lambda x: (float(x.get("StartTime", 0)), float(x.get("EndTime", 0))))

        for i, m in enumerate(wmoves):
            if str(m.get("MoveType", "")).lower() != "load":
                continue

            chamber = str(m.get("ModuleName", ""))
            if chamber.upper() == "LP":
                continue

            ftype = factory_types.get(chamber, "normal")
            if ftype != "process":
                continue

            # 找下一次 unload
            next_unload = None
            for j in range(i + 1, len(wmoves)):
                if str(wmoves[j].get("MoveType", "")).lower() == "unload":
                    next_unload = wmoves[j]
                    break

            if next_unload is None:
                continue

            load_end = float(m.get("EndTime", 0))
            unload_start = float(next_unload.get("StartTime", 0))
            step = int(m.get("step", 0))

            pt = recipe_step_times.get((pr_id, step), 0)

            residency = unload_start - (load_end + pt)

            all_residencies.append(residency)
            by_chamber[chamber].append(residency)

    if not all_residencies:
        return {"avg": 0, "count": 0, "max": 0, "min": 0, "by_chamber": {}, "all_values": []}

    result: Dict[str, Any] = {
        "avg": round(sum(all_residencies) / len(all_residencies), 2),
        "count": len(all_residencies),
        "max": round(max(all_residencies), 2),
        "min": round(min(all_residencies), 2),
        "by_chamber": {},
        "all_values": [round(v, 2) for v in all_residencies],
    }

    for chamber, values in sorted(by_chamber.items()):
        result["by_chamber"][chamber] = {
            "avg": round(sum(values) / len(values), 2),
            "count": len(values),
            "max": round(max(values), 2),
            "min": round(min(values), 2),
        }

    return result

def build_nodes(goods, recipes, trucks):
    nodes: List[Dict] = []
    boundary_actions: Dict[str, Optional[int]] = {}
    action_index: Dict[str, Dict[int, Dict[str, List[int]]]] = {}

    truck_index: Dict[str, Dict] = {}
    for t in trucks:
        tid = t.get("id")
        if tid and tid not in truck_index:
            truck_index[tid] = t

    def _truck_can_reach(truck: Dict, factory_id: str) -> bool:
        if factory_id in truck.get("available_location", []):
            return True
        return factory_id in truck.get("travel_times", {})

    def _pick_trucks(
        action_type: str,
        step_idx: int,
        factory_id: str,
        steps: List[Dict],
    ) -> List[Tuple[str, Dict]]:
        """返回所有兼容的卡车列表。"""
        prev_factory_options: List[str] = []
        next_factory_options: List[str] = []

        if step_idx > 0:
            prev_factory_options = steps[step_idx - 1].get("factory_options", [])
        if step_idx < len(steps) - 1:
            next_factory_options = steps[step_idx + 1].get("factory_options", [])

        results: List[Tuple[str, Dict]] = []
        for tid, truck in truck_index.items():
            if not _truck_can_reach(truck, factory_id):
                continue
            if action_type == "unload":
                if not next_factory_options:
                    results.append((tid, truck))
                elif any(_truck_can_reach(truck, n) for n in next_factory_options):
                    results.append((tid, truck))
            else:
                if not prev_factory_options:
                    results.append((tid, truck))
                elif any(_truck_can_reach(truck, p) for p in prev_factory_options):
                    results.append((tid, truck))
        return results

    def _add_action(
        good: Dict,
        step_idx: int,
        factory_id: str,
        action_type: str,
        tid: Optional[str],
        truck: Optional[Dict],
    ) -> int:
        node_id = len(nodes)
        action_duration = None
        if truck is not None:
            if action_type == "load":
                action_duration = truck.get("load_time")
            else:
                action_duration = truck.get("unload_time")
        node = {
            "id": node_id,
            "good_id": good.get("id"),
            "pr_id": good.get("pr_id"),
            "step": step_idx,
            "factory_id": factory_id,
            "action_type": action_type,
            "truck_id": tid,
            "action_duration": action_duration,
        }
        nodes.append(node)

        gid = str(good.get("id"))
        action_index.setdefault(gid, {}).setdefault(step_idx, {"load": [], "unload": []})
        action_index[gid][step_idx][action_type].append(node_id)
        return node_id

    for good in goods:
        pr_id = good.get("pr_id")
        if pr_id not in recipes:
            continue
        steps = recipes[pr_id]
        if not steps:
            continue

        good_mode = str(good.get("mode"))

        if good_mode in ("enter", "leave"):
            target_step = int(good.get("step"))

        last_step = len(steps) - 1
        last_action_id: Optional[int] = None

        if good_mode == "enter":
            # 对于目标步骤之前的每一步：生成取货(unload) 和 放货(load)，
            # 如果是第一步(step_idx==0)，则不生成放货(load)
            for step_idx in range(0, max(target_step, 0)):
                step = steps[step_idx]
                for factory_id in step.get("factory_options", []):
                    # 先生成取货(unload) —— 为每个兼容卡车生成节点
                    for tid, truck in _pick_trucks("unload", step_idx, factory_id, steps):
                        last_action_id = _add_action(
                            good, step_idx, factory_id, "unload", tid, truck
                        )
                    # 非第一步再生成放货(load) —— 为每个兼容卡车生成节点
                    if step_idx != 0:
                        for tid, truck in _pick_trucks("load", step_idx, factory_id, steps):
                            last_action_id = _add_action(
                                good, step_idx, factory_id, "load", tid, truck
                            )

            # 目标步：如果状态中给出了具体工厂，只为该工厂生成动作；
            # 目标步（不是第一步）要生成放货(load)，如果货物给出了 truck_id，则该步也要生成取货(unload)
            if target_step <= last_step:
                step = steps[target_step]
                factory_list = [good.get("location")] if good.get("location") else step.get("factory_options", [])
                for factory_id in factory_list:
                    if target_step != 0:
                        for tid, truck in _pick_trucks("load", target_step, factory_id, steps):
                            last_action_id = _add_action(
                                good, target_step, factory_id, "load", tid, truck
                            )
                    # 只有当状态中给出了卡车信息时，生成取货(unload)
                    if good.get("truck_id"):
                        tid = good.get("truck_id")
                        truck = truck_index.get(tid)
                        last_action_id = _add_action(
                            good, target_step, factory_id, "unload", tid, truck
                        )

        elif good_mode == "leave":
            # 对于目标步骤之后的每一步：生成取货(unload) 和 放货(load)，
            # 如果是最后一步(step_idx==last_step)，则不生成取货(unload)
            for step_idx in range(target_step + 1, last_step + 1):
                step = steps[step_idx]
                for factory_id in step.get("factory_options", []):
                    if step_idx != last_step:
                        for tid, truck in _pick_trucks("unload", step_idx, factory_id, steps):
                            last_action_id = _add_action(
                                good, step_idx, factory_id, "unload", tid, truck
                            )
                    # 始终生成放货(load) —— 为每个兼容卡车生成节点
                    for tid, truck in _pick_trucks("load", step_idx, factory_id, steps):
                        last_action_id = _add_action(
                            good, step_idx, factory_id, "load", tid, truck
                        )

            # 目标步：如果状态中给出了具体工厂，只为该工厂生成动作；
            # 目标步（不是最后一步）要生成取货(unload)，如果货物给出了 truck_id，则该步也要生成放货(load)
            if target_step <= last_step:
                step = steps[target_step]
                factory_list = [good.get("location")] if good.get("location") else step.get("factory_options", [])
                for factory_id in factory_list:
                    if target_step != last_step:
                        for tid, truck in _pick_trucks("unload", target_step, factory_id, steps):
                            last_action_id = _add_action(
                                good, target_step, factory_id, "unload", tid, truck
                            )
                    if good.get("truck_id"):
                        tid = good.get("truck_id")
                        truck = truck_index.get(tid)
                        last_action_id = _add_action(
                            good, target_step, factory_id, "load", tid, truck
                        )
        elif good_mode == "noncyclic":
            for step_idx in range(0, last_step + 1):
                step = steps[step_idx]
                for factory_id in step.get("factory_options", []):
                    if step_idx != last_step:
                        for tid, truck in _pick_trucks("unload", step_idx, factory_id, steps):
                            last_action_id = _add_action(
                                good, step_idx, factory_id, "unload", tid, truck
                            )
                    if step_idx != 0:
                        for tid, truck in _pick_trucks("load", step_idx, factory_id, steps):
                            last_action_id = _add_action(
                                good, step_idx, factory_id, "load", tid, truck
                            )

        boundary_actions[str(good.get("id"))] = last_action_id

    return nodes, boundary_actions, action_index


def build_solution_result(
    solver,
    status,
    objective_name: str,
    objective_value: int,
    cpu_time: float,
    nodes: List[Dict],
    action_times: Dict[int, Dict[str, Any]],
    action_active: Dict[int, Any],
) -> Dict[str, Any]:
    def _safe_value(var) -> int:
        return int(solver.Value(var))

    move_rows: List[Dict[str, Any]] = []
    for node in nodes:
        node_id = int(node["id"])
        if action_active and _safe_value(action_active[node_id]) != 1:
            continue

        start_time = _safe_value(action_times[node_id]["start"])
        end_time = _safe_value(action_times[node_id]["end"])

        move_rows.append(
            {
                "StartTime": start_time,
                "EndTime": end_time,
                "MoveType": node.get("action_type"),
                "ModuleName": node.get("factory_id"),
                "good_id": str(node.get("good_id")),
                "truck_id": None if node.get("truck_id") is None else str(node.get("truck_id")),
                "pr_id": str(node.get("pr_id")),
                "step": int(node.get("step") or 0),
                "_node_id": node_id,
            }
        )

    move_rows.sort(
        key=lambda item: (
            item["StartTime"],
            item["EndTime"],
        )
    )

    move_list: List[Dict[str, Any]] = []
    for move_id, move in enumerate(move_rows):
        move_list.append(
            {
                "Move_ID": move_id,
                "StartTime": move["StartTime"],
                "EndTime": move["EndTime"],
                "MoveType": move["MoveType"],
                "ModuleName": move["ModuleName"],
                "good_id": move["good_id"],
                "truck_id": move["truck_id"],
                "pr_id": move.get("pr_id"),
                "step": move["step"],
            }
        )

    result: Dict[str, Any] = {
        "status": solver.StatusName(status),
        objective_name: objective_value,
        "cpu_time": round(float(cpu_time), 3),
        "Move_List": move_list,
    }
    return result

def estimate_upper_bound(
    goods: List[Dict],
    recipes: Dict[str, List[Dict]],
    trucks: List[Dict],
) -> int:

    max_load_time = 0
    max_unload_time = 0
    max_travel_time = 0

    for truck in trucks:
        load_time = truck.get("load_time")
        unload_time = truck.get("unload_time")
        if load_time is not None:
            max_load_time = max(max_load_time, int(load_time))
        if unload_time is not None:
            max_unload_time = max(max_unload_time, int(unload_time))
        max_travel_time = max(max_travel_time, get_max_travel_time(truck))

    total = 0
    
    for good in goods:
        pr_id = good.get("pr_id")
        if pr_id not in recipes:
            continue
        steps = recipes[pr_id]
        if not steps:
            continue

        good_mode = str(good.get("mode"))
        last_step = len(steps) - 1

        target_step = None
        if good_mode in ("enter", "leave"):
            try:
                target_step = int(good.get("step"))
            except (TypeError, ValueError):
                continue
            if target_step < 0 or target_step > last_step:
                continue

        if good_mode == "enter":
            step_range = range(0, target_step + 1)
        elif good_mode == "leave":
            step_range = range(target_step, last_step + 1)
        elif good_mode == "noncyclic":
            step_range = range(0, last_step + 1)

        step_count = len(list(step_range))
        transition_count = max(0, step_count - 1)

        # Worst-case serial time for this good: process + load/unload + travel.
        for step_idx in step_range:
            step = steps[step_idx]
            process_time = step.get("process_time") or 0
            total += int(process_time)

            total += max_unload_time
            total += max_load_time

        total += transition_count * max_travel_time

    return int(total) + 100

def estimate_cycle_upper_bound(requirements: Dict[str, int], recipes: Dict[str, List[Dict]], trucks: List[Dict]) -> int:
    max_load_time = 0
    max_unload_time = 0
    max_travel_time = 0

    for truck in trucks:
        load_time = truck.get("load_time")
        unload_time = truck.get("unload_time")
        if load_time is not None:
            max_load_time = max(max_load_time, int(load_time))
        if unload_time is not None:
            max_unload_time = max(max_unload_time, int(unload_time))
        max_travel_time = max(max_travel_time, get_max_travel_time(truck))

    total = 0

    for pr_id, cycles in requirements.items():
        if pr_id not in recipes:
            continue
        steps = recipes[pr_id]
        if not steps:
            continue

        last_step = len(steps) - 1

        step_count = last_step + 1
        transition_count = max(0, step_count - 1)

        # Worst-case serial time for this good: process + load/unload + travel.
        for step_idx in range(step_count):
            step = steps[step_idx]
            process_time = step.get("process_time") or 0
            total += int(process_time) * cycles

            total += (max_unload_time + max_load_time) * cycles

        total += transition_count * max_travel_time * cycles

    return int(total) + 100

def _unscale_schedule(result: Optional[Dict[str, Any]], time_scale: float, time_key: str) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    if time_scale == 1:
        return result
    scale = float(time_scale) if float(time_scale) != 0 else 1.0
    out = dict(result)
    out["Move_List"] = _unscale_actions(result.get("Move_List", []) or [], scale)
    if time_key in out and out[time_key] is not None:
        out[time_key] = round(float(out[time_key]) / scale, 3)
    return out

def _unscale_actions(actions: List[Dict[str, Any]], time_scale: float) -> List[Dict[str, Any]]:
    scale = float(time_scale) if float(time_scale) != 0 else 1.0
    out: List[Dict[str, Any]] = []
    for a in actions or []:
        na = dict(a)
        na["StartTime"] = round(float(na["StartTime"]) / scale, 3)
        na["EndTime"] = round(float(na["EndTime"]) / scale, 3)
        out.append(na)
    return out

def _build_goods(counts, order=None):
    """
    构建 goods 列表。
    Args:
        counts: {"A": 2, "C": 2}
        order: 可选，例如 "ACAC" 指定顺序。长度必须等于总货物数。
    """
    goods = []
    gid = 1

    if order is not None:
        # 指定顺序
        from collections import Counter
        order_counts = Counter(order)
        counts_verify = dict(order_counts)
        if counts_verify != counts:
            raise ValueError(
                f"order '{order}' 中的各配方数量 {dict(order_counts)} "
                f"与需求 {counts} 不匹配"
            )
        for pr_id in order:
            goods.append({
                "id": gid,
                "pr_id": pr_id,
                "mode": "noncyclic",
            })
            gid += 1
    else:
        for pr_id, n in counts.items():
            for i in range(n):
                goods.append({
                    "id": gid,
                    "pr_id": pr_id,
                    "mode": "noncyclic",
                })
                gid += 1
    return goods

def subtract_makespan(
    actions: List[dict],
    makespan: float,
    fields: Tuple[str, ...] = ("StartTime", "EndTime"),
) -> List[dict]:
    """将动作序列的时间减去 makespan，用于循环调度中获取负周期的偏移副本。"""
    result: List[dict] = []
    for a in actions:
        item = {**a}
        for f in fields:
            if f in a:
                item[f] = a[f] - makespan
        result.append(item)
    return result


def extend_actions(actions: List[dict], makespan: float) -> List[dict]:
    prev = [{**a, "StartTime": a["StartTime"] - makespan, "EndTime": a["EndTime"] - makespan} for a in actions]
    cur = [{**a} for a in actions]
    all_actions = prev + cur
    all_actions.sort(key=lambda x: x["StartTime"])
    return all_actions


def pair_load_unload(actions: List[dict]) -> List[Tuple[dict, Optional[dict]]]:
    """为每个 load 动作找到紧随其后的匹配 unload（同 pr_id、同 step、同工厂）。"""
    res: List[Tuple[dict, Optional[dict]]] = []
    n = len(actions)
    for i, a in enumerate(actions):
        if a["MoveType"] != "load":
            continue
        pr_id = a.get("pr_id")
        step = a.get("step")
        next_unload = None
        for j in range(i + 1, n):
            b = actions[j]
            if (b["MoveType"] == "unload"
                    and b.get("pr_id") == pr_id
                    and b.get("step") == step
                    and b.get("ModuleName") == a.get("ModuleName")):
                next_unload = b
                break
        res.append((a, next_unload))
    return res


def pair_unload_next_load(actions: List[dict]) -> List[Tuple[dict, Optional[dict]]]:
    """为每个 unload 动作找到紧随其后的下一步 load（同 pr_id、step+1、同卡车）。"""
    res: List[Tuple[dict, Optional[dict]]] = []
    n = len(actions)
    for i, a in enumerate(actions):
        if a["MoveType"] != "unload":
            continue
        pr_id = a.get("pr_id")
        step = a.get("step")
        truck_id = a.get("truck_id")
        next_load = None
        for j in range(i + 1, n):
            b = actions[j]
            if (b["MoveType"] == "load"
                    and b.get("pr_id") == pr_id
                    and b.get("step") == step + 1
                    and b.get("truck_id") == truck_id):
                next_load = b
                break
        res.append((a, next_load))
    return res


def build_cycle_nodes(requirements: Dict[str, int], recipes: List[Dict], trucks: List[Dict]) -> List[Dict]:
    """
    根据需求数量生成基于“周期”的动作节点。
    每个周期的每一步都会为所有并行工厂生成节点。
    :param requirements: 字典，键为配方 id（如 "L"），值为需要生产的周期数
    :param recipes: 原始配方列表（包含 id 和 steps）
    :param trucks: 卡车信息列表
    :return: 动作节点列表，每个节点代表某个周期在某一步某个工厂的装卸动作
    :raises ValueError: 当某一步的并行工厂数不能整除需求数量时
    """
    # 1. 构建索引
    truck_index = {t['id']: t for t in trucks if 'id' in t}

    def _truck_can_reach(truck: Dict, factory_id: str) -> bool:
        """检查卡车能否到达指定工厂"""
        if factory_id in truck.get("available_location", []):
            return True
        return factory_id in truck.get("travel_times", {})

    def _pick_trucks(
        action_type: str,
        step_idx: int,
        factory_id: str,
        steps: List[Dict],
    ) -> List[Tuple[str, Dict]]:
        """
        为指定的工厂选择所有兼容的卡车。
        - 取货（unload）：要求卡车能到达当前工厂，且能到达下一步的任一工厂（若存在下一步）
        - 放货（load）：要求卡车能到达当前工厂，且能到达上一步的任一工厂（若存在上一步）
        """
        prev_factory_options: List[str] = []
        next_factory_options: List[str] = []

        if step_idx > 0:
            prev_factory_options = steps[step_idx - 1].get("factory_options", [])
        if step_idx < len(steps) - 1:
            next_factory_options = steps[step_idx + 1].get("factory_options", [])

        results: List[Tuple[str, Dict]] = []
        for tid, truck in truck_index.items():
            if not _truck_can_reach(truck, factory_id):
                continue
            if action_type == "unload":
                if not next_factory_options:
                    results.append((tid, truck))
                elif any(_truck_can_reach(truck, n) for n in next_factory_options):
                    results.append((tid, truck))
            else:  # load —— 放货需能到当前工厂及上一步工厂
                if not prev_factory_options:
                    results.append((tid, truck))
                elif any(_truck_can_reach(truck, p) for p in prev_factory_options):
                    results.append((tid, truck))
        return results

    # 2. 开始生成节点
    nodes: List[Dict] = []
    global_cycle_id = 0  # 全局周期编号，从 0 开始

    for pr_id, quantity in requirements.items():
        if pr_id not in recipes:
            print(f"警告：配方 {pr_id} 不存在，已跳过")
            continue

        steps = recipes[pr_id]
        if not steps:
            continue

        num_steps = len(steps)

        # ---------- 验证整除性 ----------
        for step in steps:
            factory_options = step.get("factory_options", [])
            if not factory_options:
                raise ValueError(f"配方 {pr_id} 的某个步骤没有提供工厂选项")

            num_factories = len(factory_options)
            if num_factories > 1:
                if quantity % num_factories != 0:
                    raise ValueError(
                        f"配方 {pr_id} 步骤中有 {num_factories} 个并行工厂，"
                        f"但需求数量 {quantity} 不能被整除"
                    )

        # ---------- 为每个周期、每个并行工厂生成所有步骤的节点 ----------
        for cycle_idx in range(quantity):
            # 分配一个全局唯一的周期 id
            current_cycle_id = global_cycle_id
            global_cycle_id += 1

            for step_idx in range(num_steps):
                step = steps[step_idx]
                factory_options = step.get("factory_options", [])

                for factory_id in factory_options:
                    # 卸载动作（最后一步不生成卸载）—— 为每个兼容卡车生成节点
                    if step_idx != num_steps - 1:
                        for tid, truck in _pick_trucks(
                            "unload", step_idx, factory_id, steps
                        ):
                            nodes.append({
                                "id": len(nodes),
                                "cycle_id": current_cycle_id,
                                "pr_id": pr_id,
                                "step": step_idx,
                                "factory_id": factory_id,
                                "action_type": "unload",
                                "truck_id": tid,
                                "action_duration": truck["unload_time"] if truck else None,
                            })

                    # 装载动作（第一步不生成装载）—— 为每个兼容卡车生成节点
                    if step_idx != 0:
                        for tid, truck in _pick_trucks(
                            "load", step_idx, factory_id, steps
                        ):
                            nodes.append({
                                "id": len(nodes),
                                "cycle_id": current_cycle_id,
                                "pr_id": pr_id,
                                "step": step_idx,
                                "factory_id": factory_id,
                                "action_type": "load",
                                "truck_id": tid,
                                "action_duration": truck["load_time"] if truck else None,
                            })

    return nodes


def truncate_sequence(
    result: Dict[str, Any],
    time_threshold: float,
) -> Dict[str, Any]:
    """保留 StartTime < time_threshold 的动作，更新 makespan 为阈值。不重编号 Move_ID。"""
    filtered = [
        m for m in result.get("Move_List", [])
        if float(m.get("StartTime", 0)) < time_threshold
    ]
    truncated = dict(result)
    truncated["Move_List"] = filtered
    for key in ("makespan", "transition_time", "start_up_time", "close_down_time"):
        if key in truncated:
            truncated[key] = time_threshold
    return truncated


def shift_move_times(
    moves: List[Dict[str, Any]],
    offset: float,
) -> List[Dict[str, Any]]:
    """将所有动作的 StartTime / EndTime 增加 offset（原地修改并返回）。"""
    for a in moves:
        a["StartTime"] = round(float(a.get("StartTime", 0)) + offset, 3)
        a["EndTime"] = round(float(a.get("EndTime", 0)) + offset, 3)
    return moves


# ── 验证辅助函数 ──────────────────────────────────────────

def as_str(value: Any) -> str:
    return "" if value is None else str(value)


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def sort_moves(moves: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        moves,
        key=lambda item: (
            as_int(item.get("StartTime")),
            as_int(item.get("EndTime")),
            as_int(item.get("Move_ID")),
        ),
    )


def group_by_factory(moves: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for move in moves:
        factory_id = as_str(move.get("ModuleName"))
        groups.setdefault(factory_id, []).append(move)
    for key in groups:
        groups[key] = sort_moves(groups[key])
    return groups


def group_by_truck(moves: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for move in moves:
        truck_id = as_str(move.get("truck_id"))
        if not truck_id:
            continue
        groups.setdefault(truck_id, []).append(move)
    for key in groups:
        groups[key] = sort_moves(groups[key])
    return groups


def insert_move_entries(
    moves: List[Dict[str, Any]],
    truck_info: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """在 Move_List 中插入 Move 类型的条目，表示机械手在连续动作之间的物理移动。

    对每个 (truck_id) 分组内的连续 load/unload 动作之间，插入一条
    MoveType="move" 的条目，记录 From_loc → To_loc 的移动。
    EndTime = 下一动作的 StartTime，StartTime = EndTime - travel_time。

    Args:
        moves: 原始的 Move_List（load/unload 动作列表）
        truck_info: 卡车信息字典 {truck_id: {"travel_time": ...}}

    Returns:
        插入 Move 条目后的新 Move_List，Move_ID 已重新编号
    """
    if not moves:
        return []

    # 1. 按时间排序
    sorted_moves = sorted(
        moves,
        key=lambda m: (
            float(m.get("StartTime", 0) or 0),
            float(m.get("EndTime", 0) or 0),
            int(m.get("Move_ID", 0) or 0),
        ),
    )

    # 2. 按 truck_id 分组
    groups: Dict[str, List[int]] = {}
    for idx, move in enumerate(sorted_moves):
        tid = str(move.get("truck_id", ""))
        if not tid:
            continue
        groups.setdefault(tid, []).append(idx)

    # 3. 为每组内的连续动作之间插入 move 条目
    inserts: List[Tuple[int, Dict[str, Any]]] = []

    for tid, indices in groups.items():
        tinfo = truck_info.get(tid, {})
        travel_times = tinfo.get("travel_times", {})
        if not travel_times or not isinstance(travel_times, dict):
            continue  # 没有矩阵则跳过该 truck

        for i in range(len(indices) - 1):
            left_idx = indices[i]
            right_idx = indices[i + 1]
            left = sorted_moves[left_idx]
            right = sorted_moves[right_idx]

            left_end = float(left.get("EndTime", 0) or 0)
            right_start = float(right.get("StartTime", 0) or 0)

            if right_start <= left_end + 1e-9:
                continue  # 无间隙，跳过

            from_loc = str(left.get("ModuleName", ""))
            to_loc = str(right.get("ModuleName", ""))
            travel_time = get_travel_time(tinfo, from_loc, to_loc)

            end_time = round(right_start, 3)
            start_time = round(end_time - float(travel_time), 3)

            inserts.append((
                left_idx,
                {
                    "MoveType": "move",
                    "StartTime": start_time,
                    "EndTime": end_time,
                    "From_loc": left.get("ModuleName"),
                    "To_loc": right.get("ModuleName"),
                    "truck_id": tid,
                },
            ))

    # 4. 按插入位置从后往前插入，避免索引偏移
    inserts.sort(key=lambda x: x[0], reverse=True)
    for insert_after_idx, move_entry in inserts:
        sorted_moves.insert(insert_after_idx + 1, move_entry)

    # 5. 重新编号 Move_ID
    for new_id, move in enumerate(sorted_moves):
        move["Move_ID"] = new_id

    return sorted_moves