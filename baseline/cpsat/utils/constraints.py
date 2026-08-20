from typing import Dict, List, Tuple, Any, Optional
from ortools.sat.python import cp_model
from utils.utils import get_travel_time


def _create_action_active_vars(
    model: cp_model.CpModel, nodes: List[Dict]
) -> Dict[int, cp_model.IntVar]:
    active: Dict[int, cp_model.IntVar] = {}
    for node in nodes:
        node_id = node["id"]
        active[node_id] = model.NewBoolVar(f"action_active_{node_id}")
    return active


def _enforce_parallel_factory_choice(
    model: cp_model.CpModel,
    action_index: Dict[str, Dict[int, Dict[str, List[int]]]],
    action_active: Dict[int, cp_model.IntVar],
) -> None:
    for _, step_map in action_index.items():
        for _, type_map in step_map.items():
            for _, action_ids in type_map.items():
                if not action_ids:
                    continue
                model.Add(sum(action_active[i] for i in action_ids) == 1)
                # Enforce ordered priority: later factories can be active only if all earlier are inactive.
                for idx in range(1, len(action_ids)):
                    for prev_idx in range(idx):
                        model.Add(action_active[action_ids[idx]] <= action_active[action_ids[prev_idx]].Not())


def _enforce_cycle_parallel_factory_choice(
    model: cp_model.CpModel,
    nodes: List[Dict],
    action_active: Dict[int, cp_model.IntVar],
) -> None:
    """对于每个 (cycle_id, step, pr_id)，在并行工厂中恰好选择一个，且 load/unload 必须选同一工厂。"""
    # 1. 同一工厂的 load 和 unload 激活状态绑定
    by_factory: Dict[Tuple[str, int, str, str], Dict[str, List[int]]] = {}
    for nd in nodes:
        key = (str(nd.get("cycle_id")), int(nd.get("step")), str(nd.get("factory_id")), str(nd.get("pr_id", "")))
        by_factory.setdefault(key, {}).setdefault(str(nd.get("action_type")), []).append(nd["id"])

    for key, pair in by_factory.items():
        load_ids = pair.get("load", [])
        unload_ids = pair.get("unload", [])
        if load_ids and unload_ids:
            model.Add(sum(action_active[l] for l in load_ids) == sum(action_active[u] for u in unload_ids))

    # 2. 仿照非周期版 action_index：按 (cycle_id, step, action_type, pr_id) 收集所有节点
    action_index: Dict[Tuple[str, int, str, str], List[int]] = {}
    for nd in nodes:
        key = (str(nd.get("cycle_id")), int(nd.get("step")), str(nd.get("action_type")), str(nd.get("pr_id", "")))
        action_index.setdefault(key, []).append(nd["id"])

    # 3. 每个 (cycle_id, step, action_type, pr_id) 恰好激活一个节点
    for (cycle_id, step, action_type, pr_id), action_ids in action_index.items():
        if not action_ids:
            continue
        model.Add(sum(action_active[i] for i in action_ids) == 1)
        # Enforce ordered priority: later factories can be active only if all earlier are inactive.
        for idx in range(1, len(action_ids)):
            for prev_idx in range(idx):
                model.Add(action_active[action_ids[idx]] <= action_active[action_ids[prev_idx]].Not())


def _enforce_load_requires_unload(
    model: cp_model.CpModel,
    nodes: List[Dict],
    action_active: Dict[int, cp_model.IntVar],
) -> None:
    by_key: Dict[Tuple[str, int, str, str], Dict[str, List[int]]] = {}
    for node in nodes:
        key = (
            str(node.get("good_id")),
            int(node.get("step")),
            str(node.get("factory_id")),
            str(node.get("pr_id", "")),
        )
        by_key.setdefault(key, {}).setdefault(node.get("action_type"), []).append(node["id"])

    for _, pair in by_key.items():
        load_ids = pair.get("load", [])
        unload_ids = pair.get("unload", [])
        if not load_ids or not unload_ids:
            continue
        # 若有任何 load 激活，则至少一个 unload 激活
        model.Add(sum(action_active[u] for u in unload_ids) == sum(action_active[l] for l in load_ids))


def _enforce_just_in_time_between_loads(
    model: cp_model.CpModel,
    nodes: List[Dict],
    action_index: Dict[str, Dict[int, Dict[str, List[int]]]],
    action_active: Dict[int, cp_model.IntVar],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    recipe_process_times: Dict[str, Dict[int, int]],
    just_in_time: Any,
) -> None:
    if just_in_time is None:
        return
    node_map = {nd["id"]: nd for nd in nodes}
    for good_id, step_map in action_index.items():
        for step, type_map in step_map.items():
            load_ids = type_map.get("load", [])
            next_load_ids = step_map.get(step + 1, {}).get("load", [])
            if not load_ids or not next_load_ids:
                continue

            pr_id = None
            for node in nodes:
                if str(node.get("good_id")) == str(good_id):
                    pr_id = str(node.get("pr_id"))
                    break
            if pr_id is None:
                continue

            step_process_time = int(recipe_process_times.get(pr_id, {}).get(step, 0) or 0)

            for load_id in load_ids:
                for next_load_id in next_load_ids:
                    model.Add(
                        action_times[next_load_id]["start"]
                        <= action_times[load_id]["start"]
                        + int(node_map[load_id].get("action_duration") or 0)
                        + step_process_time
                        + int(just_in_time)
                    ).OnlyEnforceIf([action_active[load_id], action_active[next_load_id]])


def _enforce_goods_priority_for_actions(
    model: cp_model.CpModel,
    nodes: List[Dict],
    action_active: Dict[int, cp_model.IntVar],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    goods: List[Dict],
) -> None:
    goods_priority: Dict[str, Tuple] = {}
    for good in goods:
        good_id = str(good.get("id"))
        good_mode = str(good.get("mode") or "")
        step = int(good.get("step") or 0)
        has_truck = 1 if good.get("truck_id") else 0
        bound_time = good.get("bound")
        bound_time_val = int(bound_time) if bound_time is not None else 10**9

        if good_mode == "noncyclic":
            goods_priority[good_id] = (1, -int(good.get("id", 0)))
        elif good_mode == "enter":
            goods_priority[good_id] = (0, step, has_truck, -bound_time_val)
        else:
            # leave 或无 mode -> closedown 风格优先级
            goods_priority[good_id] = (2, step, -has_truck, -bound_time_val)

    action_groups: Dict[Tuple[int, str], List[int]] = {}
    for nd in nodes:
        gid = str(nd.get("good_id"))
        step = int(nd.get("step"))
        action_type = str(nd.get("action_type"))
        action_groups.setdefault((step, action_type), []).append(nd["id"])
    node_map = {nd["id"]: nd for nd in nodes}

    for (_, _), node_ids in action_groups.items():
        for i in range(len(node_ids)):
            left_id = node_ids[i]
            left_good = str(node_map[left_id].get("good_id"))
            left_pr_id = str(node_map[left_id].get("pr_id", ""))
            for j in range(i + 1, len(node_ids)):
                right_id = node_ids[j]
                right_good = str(node_map[right_id].get("good_id"))
                right_pr_id = str(node_map[right_id].get("pr_id", ""))
                if left_good == right_good:
                    continue

                if left_pr_id != right_pr_id:
                    # 不同配方的动作不比较优先级
                    continue

                left_pri = goods_priority.get(left_good)
                right_pri = goods_priority.get(right_good)
                if left_pri is None or right_pri is None:
                    continue

                if left_pri > right_pri:
                    model.Add(
                        action_times[left_id]["start"] <= action_times[right_id]["start"]
                    ).OnlyEnforceIf([action_active[left_id], action_active[right_id]])
                elif right_pri > left_pri:
                    model.Add(
                        action_times[right_id]["start"] <= action_times[left_id]["start"]
                    ).OnlyEnforceIf([action_active[left_id], action_active[right_id]])


def _enforce_lp_pickup_order(
    model: cp_model.CpModel,
    nodes: List[Dict],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    action_active: Dict[int, cp_model.IntVar],
) -> None:
    """
    强制 LP 第 0 步的取货（unload）顺序与 good_id 一致。
    good_id 小的动作优先开始。
    """
    # 筛选 LP 第 0 步的 unload 动作
    lp_unload_nodes = [
        nd for nd in nodes
        if int(nd.get("step", -1)) == 0
        and str(nd.get("action_type")) == "unload"
    ]
    # 按 good_id 升序排序
    lp_unload_nodes.sort(key=lambda nd: int(nd.get("good_id", 0)))

    for i in range(len(lp_unload_nodes) - 1):
        left = lp_unload_nodes[i]
        right = lp_unload_nodes[i + 1]
        left_id = left["id"]
        right_id = right["id"]
        model.Add(
            action_times[left_id]["start"] <= action_times[right_id]["start"]
        ).OnlyEnforceIf([action_active[left_id], action_active[right_id]])


def _enforce_no_overlap_completion_pm(
    model: cp_model.CpModel,
    nodes: List[Dict],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    action_active: Dict[int, cp_model.IntVar],
    recipes: Dict[str, List[Dict]],
) -> None:
    """
    不同 PM 工厂的完工窗口 [load.start+dur+p_min, load.start+dur+p_max] 不能重叠。
    直接从 recipes 中读取 process_time_range。
    """
    fluc_loads: List[Dict] = []
    for nd in nodes:
        if str(nd.get("action_type")) != "load":
            continue
        pr_id = str(nd.get("pr_id", ""))
        step = int(nd.get("step", -1))
        steps = recipes.get(pr_id, [])
        if step >= len(steps):
            continue
        pt_range = steps[step].get("process_time_range")
        if not pt_range or len(pt_range) != 2:
            continue
        p_min, p_max = int(pt_range[0]), int(pt_range[1])
        if p_min == p_max:
            continue
        nd_copy = dict(nd)
        nd_copy["_p_min"] = p_min
        nd_copy["_p_max"] = p_max
        nd_copy["_fid"] = str(nd.get("factory_id", ""))
        fluc_loads.append(nd_copy)

    for i in range(len(fluc_loads)):
        a = fluc_loads[i]
        a_id = a["id"]
        a_pmin = a["_p_min"]
        a_pmax = a["_p_max"]
        a_fid = a["_fid"]
        a_end_var = model.NewIntVar(0, 10**9, f"pm_cw_end_{a_id}")
        model.Add(a_end_var == action_times[a_id]["start"] + a_pmax)

        for j in range(i + 1, len(fluc_loads)):
            b = fluc_loads[j]
            if b["_fid"] == a_fid:
                continue
            b_id = b["id"]
            b_pmin = b["_p_min"]
            b_pmax = b["_p_max"]
            b_end_var = model.NewIntVar(0, 10**9, f"pm_cw_end_{b_id}")
            model.Add(b_end_var == action_times[b_id]["start"] + b_pmax)

            a_before_b = model.NewBoolVar(f"pm_cw_{a_id}_before_{b_id}")
            b_before_a = model.NewBoolVar(f"pm_cw_{b_id}_before_{a_id}")

            # 仅当两个节点都激活时，才强制窗口不重叠
            model.Add(a_before_b + b_before_a == 1).OnlyEnforceIf([action_active[a_id], action_active[b_id]])

            model.Add(
                a_end_var <= action_times[b_id]["start"] + b_pmin
            ).OnlyEnforceIf([a_before_b, action_active[a_id], action_active[b_id]])

            model.Add(
                b_end_var <= action_times[a_id]["start"] + a_pmin
            ).OnlyEnforceIf([b_before_a, action_active[a_id], action_active[b_id]])

def add_cycle_priority(
    model: cp_model.CpModel,
    nodes: List[Dict],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    action_active: Dict[int, cp_model.IntVar],
    makespan: cp_model.IntVar,
) -> None:
    """
    同一工序 (step, action_type, pr_id) 内的动作，按 cycle_id 升序在有效时间轴上排列，
    cycle_id 小的在有效时间上不晚于 cycle_id 大的（FIFO）。
    只有同一配方的动作才参与排序，不同配方的动作互不影响。

    为每个相邻 pair 引入 wrap 变量：
    - wrap=0 → 按时间顺序（start_left <= start_right）
    - wrap=1 → +makespan 后按时间顺序（start_right + makespan >= start_left）
    - 每组至多一个 wrap，不同组之间独立。
    """
    node_map = {nd["id"]: nd for nd in nodes}

    # 按 (step, action_type, pr_id) 分组，只有同一配方的动作才参与排序
    action_groups: Dict[Tuple[int, str, str], List[int]] = {}
    for nd in nodes:
        step = int(nd.get("step"))
        action_type = str(nd.get("action_type"))
        pr_id = str(nd.get("pr_id", ""))
        action_groups.setdefault((step, action_type, pr_id), []).append(nd["id"])

    for _, node_ids in action_groups.items():
        if len(node_ids) < 2:
            continue

        # 按 cycle_id 升序排序
        sorted_ids = sorted(node_ids, key=lambda nid: int(node_map[nid].get("cycle_id", 0)))

        wrap_vars: List[cp_model.IntVar] = []
        for i in range(len(sorted_ids)):
            left_id = sorted_ids[i]
            for j in range(i + 1, len(sorted_ids)):
                right_id = sorted_ids[j]
                wrap = model.NewBoolVar(f"cycle_pri_wrap_{left_id}_{right_id}")
                wrap_vars.append(wrap)

                # wrap=0：按时间顺序
                model.Add(
                    action_times[left_id]["start"] <= action_times[right_id]["start"]
                ).OnlyEnforceIf([action_active[left_id], action_active[right_id], wrap.Not()])

                # wrap=1：+makespan 后按时间顺序
                model.Add(
                    action_times[right_id]["start"] + makespan >= action_times[left_id]["start"]
                ).OnlyEnforceIf([action_active[left_id], action_active[right_id], wrap])

        # 每组至多一个 wrap
        if wrap_vars:
            model.Add(sum(wrap_vars) <= 1)

def _enforce_factory_cycles(
    model: cp_model.CpModel,
    nodes: List[Dict],
    factories: List[Dict],
    action_active: Dict[int, cp_model.IntVar],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    process_times: Dict[str, Dict[int, int]],
    residency_times: Dict[str, Dict[int, int]],
    goods: List[Dict] = None,
    start_up_time: cp_model.IntVar = None,
) -> None:
    factory_type = {str(f.get("id")): str(f.get("type")) for f in factories}

    node_map = {nd["id"]: nd for nd in nodes}

    factory_nodes: Dict[str, List[int]] = {}
    for nd in nodes:
        factory_id = str(nd.get("factory_id"))
        if factory_type.get(factory_id) == "LP":
            continue
        factory_nodes.setdefault(factory_id, []).append(nd["id"])

    # 构建货物查找表，用于边界约束：(good_id, step, factory_id) → good
    good_by_key: Dict[Tuple[str, int, str], Dict] = {}
    if goods:
        for g in goods:
            gid = str(g.get("id"))
            step = g.get("step")
            loc = str(g.get("location") or "")
            if step is not None and loc:
                good_by_key[(gid, int(step), loc)] = g

    def _apply_startup_boundary(node_id: int, good: Dict) -> None:
        """对 unmatched load 施加启动边界约束（含 start_up_time 偏移）。"""
        left_bound = good.get("left_bound")
        right_bound = good.get("right_bound")
        offset = start_up_time if start_up_time is not None else 0
        if left_bound is not None:
            model.Add(action_times[node_id]["start"] >= int(left_bound) + offset).OnlyEnforceIf(action_active[node_id])
        if right_bound is not None:
            model.Add(action_times[node_id]["start"] <= int(right_bound) + offset).OnlyEnforceIf(action_active[node_id])

    def _apply_closedown_boundary(node_id: int, good: Dict) -> None:
        """对 unmatched unload 施加关闭边界约束（无时间偏移）。"""
        left_bound = good.get("left_bound")
        right_bound = good.get("right_bound")
        if left_bound is not None:
            model.Add(action_times[node_id]["start"] >= int(left_bound)).OnlyEnforceIf(action_active[node_id])
        if right_bound is not None:
            model.Add(action_times[node_id]["start"] <= int(right_bound)).OnlyEnforceIf(action_active[node_id])

    def _build_cycle_with_dummy(
        name_prefix: str,
        node_ids: List[int],
        assigned_vars: Dict[int, cp_model.IntVar],
        dummy_id: int,
    ) -> Dict[Tuple[int, int], cp_model.IntVar]:
        arcs = []
        arc_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}

        dummy_loop = model.NewBoolVar(f"{name_prefix}_arc_{dummy_id}_{dummy_id}")
        arc_vars[(dummy_id, dummy_id)] = dummy_loop
        arcs.append([dummy_id, dummy_id, dummy_loop])

        for i in node_ids:
            a_di = model.NewBoolVar(f"{name_prefix}_arc_{dummy_id}_{i}")
            arc_vars[(dummy_id, i)] = a_di
            arcs.append([dummy_id, i, a_di])
            model.Add(a_di <= assigned_vars[i])

            a_id = model.NewBoolVar(f"{name_prefix}_arc_{i}_{dummy_id}")
            arc_vars[(i, dummy_id)] = a_id
            arcs.append([i, dummy_id, a_id])
            model.Add(a_id <= assigned_vars[i])

        for i in node_ids:
            a_ii = model.NewBoolVar(f"{name_prefix}_arc_{i}_{i}")
            arc_vars[(i, i)] = a_ii
            arcs.append([i, i, a_ii])
            model.Add(a_ii + assigned_vars[i] == 1)

            for j in node_ids:
                if i == j:
                    continue
                a_ij = model.NewBoolVar(f"{name_prefix}_arc_{i}_{j}")
                arc_vars[(i, j)] = a_ij
                arcs.append([i, j, a_ij])
                model.Add(a_ij <= assigned_vars[i])
                model.Add(a_ij <= assigned_vars[j])

        model.AddCircuit(arcs)
        return arc_vars

    by_key: Dict[Tuple[str, int, str, str], Dict[str, List[int]]] = {}
    for nd in nodes:
        key = (
            str(nd.get("good_id")),
            int(nd.get("step")),
            str(nd.get("factory_id")),
            str(nd.get("pr_id", "")),
        )
        by_key.setdefault(key, {}).setdefault(str(nd.get("action_type")), []).append(nd["id"])

    dummy_seed = len(nodes)
    for idx, (factory_id, node_ids) in enumerate(factory_nodes.items()):
        if not node_ids:
            continue
        dummy_id = dummy_seed + idx
        assigned_vars = {i: action_active[i] for i in node_ids}
        arc_vars = _build_cycle_with_dummy(f"factory_{factory_id}", node_ids, assigned_vars, dummy_id)
        # Enforce ordering: if an arc (i->j) is chosen on the factory ring,
        # then the successor action j must start no earlier than the end of i.
        for (i, j), arc in arc_vars.items():
            # skip dummy and self-loop arcs
            if i == dummy_id or j == dummy_id or i == j:
                continue
            model.Add(action_times[j]["start"] >= action_times[i]["end"]).OnlyEnforceIf(arc)
        
        # Pair load/unload for same good+step at this factory.
        for (gid, step, fid, _pr_id), pair in by_key.items():
            if fid != factory_id:
                continue
            load_ids = pair.get("load", [])
            unload_ids = pair.get("unload", [])
            if not load_ids or not unload_ids:
                continue

            for load_id in load_ids:
                for unload_id in unload_ids:
                    arc_lu = arc_vars.get((load_id, unload_id))
                    if arc_lu is not None:
                        model.Add(arc_lu == 1).OnlyEnforceIf([action_active[load_id], action_active[unload_id]])

                    pr_id = str(node_map[load_id].get("pr_id"))
                    step_process_time = int(process_times.get(pr_id, {}).get(step, 0) or 0)
                    load_duration = int(node_map[load_id].get("action_duration") or 0)
                    model.Add(
                        action_times[unload_id]["start"] - action_times[load_id]["start"] - load_duration
                        >= step_process_time
                    ).OnlyEnforceIf([action_active[load_id], action_active[unload_id]])
                    if residency_times.get(pr_id, {}).get(step) is not None:
                        model.Add(
                            action_times[unload_id]["start"] - action_times[load_id]["start"] - load_duration
                            <= step_process_time + int(residency_times.get(pr_id, {}).get(step))
                        ).OnlyEnforceIf([action_active[load_id], action_active[unload_id]])

        # Unmatched unload -> first, unmatched load -> last within factory ring.
        for (gid, step, fid, _pr_id), pair in by_key.items():
            if fid != factory_id:
                continue
            load_ids = pair.get("load", [])
            unload_ids = pair.get("unload", [])
            if not load_ids:
                for unload_id in unload_ids:
                    arc_du = arc_vars.get((dummy_id, unload_id))
                    if arc_du is not None:
                        model.Add(arc_du == 1).OnlyEnforceIf(action_active[unload_id])
                    # closedown 边界约束：仅取货、无放货 → 末态边界
                    good = good_by_key.get((gid, step, fid))
                    if good:
                        _apply_closedown_boundary(unload_id, good)
            if not unload_ids:
                for load_id in load_ids:
                    arc_ld = arc_vars.get((load_id, dummy_id))
                    if arc_ld is not None:
                        model.Add(arc_ld == 1).OnlyEnforceIf(action_active[load_id])
                    # startup 边界约束：仅放货、无取货 → 初态边界
                    good = good_by_key.get((gid, step, fid))
                    if good:
                        _apply_startup_boundary(load_id, good)

def _enforce_cycle_factory_cycles(
    model: cp_model.CpModel,
    nodes: List[Dict],
    makespan: cp_model.IntVar,           # 周期长度变量（或常量）
    factories: List[Dict],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    action_active: Dict[int, cp_model.IntVar],
    process_times: Dict[str, Dict[int, int]],
    residency_times: Dict[str, Dict[int, int]],
) -> None:
    factory_type = {str(f.get("id")): str(f.get("type")) for f in factories}
    node_map = {nd["id"]: nd for nd in nodes}

    factory_nodes: Dict[str, List[int]] = {}
    for nd in nodes:
        fid = str(nd.get("factory_id"))
        if factory_type.get(fid) == "LP":
            continue
        factory_nodes.setdefault(fid, []).append(nd["id"])

    def _build_cycle(name: str, node_ids: List[int], assigned_vars: Dict[int, cp_model.IntVar]) -> Dict[Tuple[int,int], cp_model.IntVar]:
        arcs = []
        arc_vars = {}
        for i in node_ids:
            a_ii = model.NewBoolVar(f"{name}_arc_{i}_{i}")
            arc_vars[(i, i)] = a_ii
            arcs.append([i, i, a_ii])
            model.Add(a_ii + assigned_vars[i] == 1)
            for j in node_ids:
                if i == j:
                    continue
                v = model.NewBoolVar(f"{name}_arc_{i}_{j}")
                arc_vars[(i, j)] = v
                arcs.append([i, j, v])
                model.Add(v <= assigned_vars[i])
                model.Add(v <= assigned_vars[j])
        model.AddCircuit(arcs)
        return arc_vars

    # 索引： (cycle_id, step, factory_id, pr_id) -> {"load": [ids], "unload": [ids]}
    by_key: Dict[Tuple[str, int, str, str], Dict[str, List[int]]] = {}
    for nd in nodes:
        key = (str(nd.get("cycle_id")), int(nd.get("step")), str(nd.get("factory_id")), str(nd.get("pr_id", "")))
        by_key.setdefault(key, {}).setdefault(str(nd.get("action_type")), []).append(nd["id"])

    all_wrap_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}  # 全局 wrap，用于跨工厂 FIFO

    for factory_id, node_ids in factory_nodes.items():
        if not node_ids:
            continue
        
        assigned_vars = {i: action_active[i] for i in node_ids}
        arc_vars = _build_cycle(f"factory_{factory_id}", node_ids, assigned_vars)

        # ---------- 为每条弧创建 wrap 变量，并加入顺序约束 ----------
        wrap_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}  # 存放该工厂所有弧的 wrap 布尔变量
        for (i, j), arc in arc_vars.items():
            # 跳过自环弧（自环不参与跨周期）
            if i == j:
                continue
            wrap = model.NewBoolVar(f"wrap_{factory_id}_{i}_{j}")
            wrap_vars[(i, j)] = wrap
            all_wrap_vars[(i, j)] = wrap
            model.Add(wrap <= arc)  # wrap 只能在该弧被选中时为 1

            # 顺序约束（分 wrap 两种情况）
            model.Add(
                action_times[j]["start"] >= action_times[i]["end"]
            ).OnlyEnforceIf([arc, wrap.Not()])

            model.Add(
                action_times[j]["start"] + makespan >= action_times[i]["end"]
            ).OnlyEnforceIf([arc, wrap])

        # 限制工厂内跨周期弧的总数 ≤ 1
        if wrap_vars:
            model.Add(sum(wrap_vars.values()) <= 1)

        # ---------- 处理 load-unload 紧邻对及工艺时间 ----------
        for (gid, step, fid, _pr_id), pair in by_key.items():
            if fid != factory_id:
                continue
            load_ids = pair.get("load", [])
            unload_ids = pair.get("unload", [])
            if not load_ids or not unload_ids:
                continue

            for load_id in load_ids:
                for unload_id in unload_ids:
                    arc_lu = arc_vars.get((load_id, unload_id))
                    if arc_lu is not None:
                        model.Add(arc_lu == 1).OnlyEnforceIf([action_active[load_id], action_active[unload_id]])

                    pr_id = str(node_map[load_id].get("pr_id"))
                    proc_time = int(process_times.get(pr_id, {}).get(step, 0) or 0)
                    load_dur = int(node_map[load_id].get("action_duration") or 0)

                    wrap_lu = wrap_vars.get((load_id, unload_id))

                    model.Add(
                        action_times[unload_id]["start"] - action_times[load_id]["start"] - load_dur >= proc_time
                    ).OnlyEnforceIf([action_active[load_id], action_active[unload_id], wrap_lu.Not()])

                    model.Add(
                        action_times[unload_id]["start"] - action_times[load_id]["start"] - load_dur + makespan >= proc_time
                    ).OnlyEnforceIf([action_active[load_id], action_active[unload_id], wrap_lu])
                    if residency_times.get(pr_id, {}).get(step) is not None:
                        res_time = int(residency_times[pr_id][step])
                        model.Add(
                            action_times[unload_id]["start"] - action_times[load_id]["start"] - load_dur <= proc_time + res_time
                        ).OnlyEnforceIf([action_active[load_id], action_active[unload_id], wrap_lu.Not()])

                        model.Add(
                            action_times[unload_id]["start"] - action_times[load_id]["start"] - load_dur + makespan <= proc_time + res_time
                        ).OnlyEnforceIf([action_active[load_id], action_active[unload_id], wrap_lu])


    # ---------- 同步骤 FIFO（跨工厂）：先放先取，取货用等效时间 ----------
    # 收集所有工厂同一步骤的 load/unload 对
    step_pairs: Dict[int, List[Tuple[int, int, cp_model.IntVar]]] = {}
    for (gid, step, fid, _pr_id), pair in by_key.items():
        load_ids = pair.get("load", [])
        unload_ids = pair.get("unload", [])
        if not load_ids or not unload_ids:
            continue
        for load_id in load_ids:
            for unload_id in unload_ids:
                wrap_lu = all_wrap_vars.get((load_id, unload_id))
                if wrap_lu is None:
                    continue
                step_pairs.setdefault(step, []).append((load_id, unload_id, wrap_lu))

    for step, pairs in step_pairs.items():
        if len(pairs) < 2:
            continue

        for a in range(len(pairs)):
            load_a, unload_a, wrap_a = pairs[a]
            active_a = [action_active[load_a], action_active[unload_a]]
            for b in range(a + 1, len(pairs)):
                load_b, unload_b, wrap_b = pairs[b]
                active_b = [action_active[load_b], action_active[unload_b]]
                active_both = active_a + active_b

                # 1. load 顺序：order=1 表示 A 先于 B
                order = model.NewBoolVar(f"fifo_order_s{step}_{load_a}_{load_b}")
                model.Add(action_times[load_a]["start"] <= action_times[load_b]["start"]
                    ).OnlyEnforceIf(active_both + [order])
                model.Add(action_times[load_b]["start"] <= action_times[load_a]["start"]
                    ).OnlyEnforceIf(active_both + [order.Not()])

                # 2. wrap 兼容：先放的不能 wrap 而后放的不 wrap
                #    A 先 B → wrap_a ≤ wrap_b；B 先 A → wrap_b ≤ wrap_a
                model.Add(wrap_a <= wrap_b).OnlyEnforceIf(active_both + [order])
                model.Add(wrap_b <= wrap_a).OnlyEnforceIf(active_both + [order.Not()])

                # 3. same_wrap = (wrap_a == wrap_b)
                same_wrap = model.NewBoolVar(f"fifo_sw_s{step}_{load_a}_{load_b}")
                model.Add(same_wrap <= 1 - wrap_a + wrap_b)
                model.Add(same_wrap <= 1 + wrap_a - wrap_b)
                model.Add(same_wrap >= wrap_a + wrap_b - 1)
                model.Add(same_wrap >= 1 - wrap_a - wrap_b)

                # 4. unload FIFO
                #    same-wrap：unload 顺序 = load 顺序（物理到达时间偏移相同）
                model.Add(action_times[unload_a]["start"] <= action_times[unload_b]["start"]
                    ).OnlyEnforceIf(active_both + [order, same_wrap])
                model.Add(action_times[unload_b]["start"] <= action_times[unload_a]["start"]
                    ).OnlyEnforceIf(active_both + [order.Not(), same_wrap])
                #    diff-wrap：wrap=1 的一方物理到达更早（t−M < 0），必须优先取货
                model.Add(action_times[unload_a]["start"] <= action_times[unload_b]["start"]
                    ).OnlyEnforceIf(active_both + [same_wrap.Not(), wrap_a])
                model.Add(action_times[unload_b]["start"] <= action_times[unload_a]["start"]
                    ).OnlyEnforceIf(active_both + [same_wrap.Not(), wrap_b])

def _create_action_time_vars(
    model: cp_model.CpModel,
    nodes: List[Dict],
    upper_bound: int,
    action_active: Dict[int, cp_model.IntVar],
    horizon_time: cp_model.IntVar
) -> Dict[int, Dict[str, cp_model.IntVar]]:
    action_times: Dict[int, Dict[str, cp_model.IntVar]] = {}
    for node in nodes:
        node_id = node["id"]
        start = model.NewIntVar(0, upper_bound, f"action_{node_id}_start")
        end = model.NewIntVar(0, upper_bound, f"action_{node_id}_end")
        duration = int(node.get("action_duration") or 0)
        if duration < 0:
            raise ValueError(f"Negative action duration for node {node_id}: {duration}")

        model.Add(start + duration == end).OnlyEnforceIf(action_active[node_id])
        model.Add(end <= horizon_time).OnlyEnforceIf(action_active[node_id])
        model.Add(start == 0).OnlyEnforceIf(action_active[node_id].Not())
        model.Add(end == 0).OnlyEnforceIf(action_active[node_id].Not())
        action_times[node_id] = {"start": start, "end": end}
    return action_times

def _enforce_arm_and_robot_cycles(
    model: cp_model.CpModel,
    nodes: List[Dict],
    trucks: List[Dict],
    action_active: Dict[int, cp_model.IntVar],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    horizon_time: cp_model.IntVar,
    robot_contexts: Dict[str, Dict[str, Any]],
) -> None:
    """单臂机器人，按 truck_id 建带 dummy 的环，含移动时间。

    环结构：dummy → … 实际动作 … → dummy
    - 未配对 load（wafer 在卡车上，前一步 unload 不在本环中）→ dummy → load（环开头）
    - 未配对 unload（取货后无同车下一步 load）→ unload → dummy（环末尾）
    - 配对 unload_k → load_{k+1} → 紧邻弧 arc_uv == 1
    """
    node_map = {nd["id"]: nd for nd in nodes}
    truck_dict = {str(t.get("id")): t for t in trucks}

    # 按 truck_id 分组节点
    truck_nodes: Dict[str, List[int]] = {}
    for nd in nodes:
        tid = str(nd.get("truck_id"))
        if tid and tid != "None":
            truck_nodes.setdefault(tid, []).append(nd["id"])

    def _build_cycle_with_dummy(name_prefix, node_ids, assigned_vars, dummy_id):
        """建带 dummy 节点的环，dummy 自环 + dummy↔节点 全连接。"""
        arcs = []
        arc_vars: Dict[Tuple[int, int], cp_model.IntVar] = {}

        dummy_loop = model.NewBoolVar(f"{name_prefix}_arc_{dummy_id}_{dummy_id}")
        arc_vars[(dummy_id, dummy_id)] = dummy_loop
        arcs.append([dummy_id, dummy_id, dummy_loop])

        for i in node_ids:
            a_di = model.NewBoolVar(f"{name_prefix}_arc_{dummy_id}_{i}")
            arc_vars[(dummy_id, i)] = a_di
            arcs.append([dummy_id, i, a_di])
            model.Add(a_di <= assigned_vars[i])

            a_id = model.NewBoolVar(f"{name_prefix}_arc_{i}_{dummy_id}")
            arc_vars[(i, dummy_id)] = a_id
            arcs.append([i, dummy_id, a_id])
            model.Add(a_id <= assigned_vars[i])

        for i in node_ids:
            a_ii = model.NewBoolVar(f"{name_prefix}_arc_{i}_{i}")
            arc_vars[(i, i)] = a_ii
            arcs.append([i, i, a_ii])
            model.Add(a_ii + assigned_vars[i] == 1)

            for j in node_ids:
                if i == j:
                    continue
                a_ij = model.NewBoolVar(f"{name_prefix}_arc_{i}_{j}")
                arc_vars[(i, j)] = a_ij
                arcs.append([i, j, a_ij])
                model.Add(a_ij <= assigned_vars[i])
                model.Add(a_ij <= assigned_vars[j])

        model.AddCircuit(arcs)
        return arc_vars

    # ── 按 (good_id, step, action_type, pr_id) 索引 ──
    by_key: Dict[Tuple[str, int, str, str], List[int]] = {}
    for nd in nodes:
        gid = str(nd.get("good_id"))
        step = int(nd.get("step"))
        action_type = str(nd.get("action_type"))
        pr_id = str(nd.get("pr_id") or "")
        by_key.setdefault((gid, step, action_type, pr_id), []).append(nd["id"])

    truck_arcs: Dict[str, Dict[Tuple[int, int], cp_model.IntVar]] = {}
    dummy_seed = len(nodes)

    for idx, (truck_id, node_ids) in enumerate(truck_nodes.items()):
        if len(node_ids) < 2:
            continue
        truck = truck_dict.get(truck_id)
        context = robot_contexts.get(truck_id, {})
        start_time = context.get("start_time")
        end_time = context.get("end_time")

        dummy_id = dummy_seed + idx
        assigned_vars = {i: action_active[i] for i in node_ids}
        ring = _build_cycle_with_dummy(f"truck_{truck_id}", node_ids, assigned_vars, dummy_id)
        truck_arcs[truck_id] = ring

        # ── 弧上的顺序 + travel_time 约束（只对非 dummy 弧，无需 wrap）──
        for (i, j), arc in ring.items():
            if i == dummy_id or j == dummy_id or i == j:
                continue
            # 从矩阵查询 i → j 的移动时间
            from_factory = str(node_map[i].get("factory_id") or "")
            to_factory = str(node_map[j].get("factory_id") or "")
            travel_time = get_travel_time(truck, from_factory, to_factory) if truck else 0

            action_type_i = str(node_map[i].get("action_type") or "")
            if action_type_i == "unload":
                model.Add(
                    action_times[j]["start"] == action_times[i]["end"] + travel_time
                ).OnlyEnforceIf(arc)
            else:
                model.Add(
                    action_times[j]["start"] >= action_times[i]["end"] + travel_time
                ).OnlyEnforceIf(arc)

        # ── 边界时间约束：dummy→首动作 和 末动作→dummy ──
        for (i, j), arc in ring.items():
            # dummy → j：首个动作，受 start_time 约束
            if i == dummy_id and j != dummy_id and start_time is not None:
                to_factory = str(node_map[j].get("factory_id") or "")
                start_loc = str(context.get("start_location") or "")
                travel_time = get_travel_time(truck, start_loc, to_factory) if truck else 0
                action_type_j = str(node_map[j].get("action_type") or "")
                if action_type_j == "load":
                    model.Add(
                        action_times[j]["start"] == int(start_time) + travel_time
                    ).OnlyEnforceIf(arc)
                else:
                    model.Add(
                        action_times[j]["start"] >= int(start_time) + travel_time
                    ).OnlyEnforceIf(arc)
            # i → dummy：末尾动作，受 end_time 约束
            if j == dummy_id and i != dummy_id and end_time is not None:
                from_factory = str(node_map[i].get("factory_id") or "")
                end_loc = str(context.get("end_location") or "")
                travel_time = get_travel_time(truck, from_factory, end_loc) if truck else 0
                action_type_i = str(node_map[i].get("action_type") or "")
                if action_type_i == "unload":
                    model.Add(
                        action_times[i]["end"] + travel_time == horizon_time + int(end_time)
                    ).OnlyEnforceIf(arc)
                else:
                    model.Add(
                        action_times[i]["end"] + travel_time <= horizon_time + int(end_time)
                    ).OnlyEnforceIf(arc)                    

        # ── unload_k → load_{k+1} 同卡车紧邻 ──
        for (gid, step, action_type, pr_id), unload_ids in list(by_key.items()):
            if action_type != "unload":
                continue
            load_ids = by_key.get((gid, step + 1, "load", pr_id), [])
            if not load_ids:
                continue
            for u in unload_ids:
                u_truck = str(node_map[u].get("truck_id"))
                if u_truck != truck_id:
                    continue
                for v in load_ids:
                    v_truck = str(node_map[v].get("truck_id"))
                    if v_truck != truck_id or v not in node_ids:
                        continue
                    arc_uv = ring.get((u, v))
                    if arc_uv is not None:
                        model.Add(arc_uv == 1).OnlyEnforceIf(
                            [action_active[u], action_active[v]]
                        )
                    model.Add(
                        action_times[u]["end"] <= action_times[v]["start"]
                    ).OnlyEnforceIf([action_active[u], action_active[v]])

        # ── 未配对 load → 环开头（dummy → load）──
        # 前一步 unload 不在本环中（wafer 已在卡车上）
        for (gid, step, action_type, pr_id), load_ids in list(by_key.items()):
            if action_type != "load":
                continue
            if (gid, step - 1, "unload", pr_id) in by_key:
                continue  # 有配对 unload，跳过
            for l in load_ids:
                if l not in node_ids:
                    continue
                arc_dl = ring.get((dummy_id, l))
                if arc_dl is not None:
                    model.Add(arc_dl == 1).OnlyEnforceIf(action_active[l])

        # ── 未配对 unload → 环末尾（unload → dummy）──
        # 下一步 load 不在本环中（由其他卡车接或 wafer 退出）
        for (gid, step, action_type, pr_id), unload_ids in list(by_key.items()):
            if action_type != "unload":
                continue
            if (gid, step + 1, "load", pr_id) in by_key:
                continue  # 有配对 load，跳过
            for u in unload_ids:
                if u not in node_ids:
                    continue
                arc_ud = ring.get((u, dummy_id))
                if arc_ud is not None:
                    model.Add(arc_ud == 1).OnlyEnforceIf(action_active[u])

def _enforce_cycle_arm_and_robot_cycles(
    model: cp_model.CpModel,
    nodes: List[Dict],
    trucks: List[Dict],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    action_active: Dict[int, cp_model.IntVar],
    makespan: cp_model.IntVar,
) -> None:
    """简化版：单臂机器人，按 truck_id 直接建环，含移动时间，无 x_assign。"""
    node_map = {nd["id"]: nd for nd in nodes}
    truck_dict = {str(t.get("id")): t for t in trucks}

    # 按 truck_id 分组节点
    truck_nodes: Dict[str, List[int]] = {}
    for nd in nodes:
        tid = str(nd.get("truck_id"))
        if tid and tid != "None":
            truck_nodes.setdefault(tid, []).append(nd["id"])

    def _build_cycle(name_prefix, node_ids, assigned):
        arcs = []
        arc_vars = {}
        for i in node_ids:
            for j in node_ids:
                if i == j:
                    continue
                a_ij = model.NewBoolVar(f"{name_prefix}_{i}_{j}")
                arc_vars[(i, j)] = a_ij
                arcs.append([i, j, a_ij])
                model.Add(a_ij <= assigned[i])
                model.Add(a_ij <= assigned[j])
        for i in node_ids:
            a_ii = model.NewBoolVar(f"{name_prefix}_self_{i}")
            arc_vars[(i, i)] = a_ii
            arcs.append([i, i, a_ii])
            model.Add(a_ii == 1 - assigned[i])
        model.AddCircuit(arcs)
        return arc_vars

    truck_arcs: Dict[str, Dict[Tuple[int, int], cp_model.IntVar]] = {}
    truck_wrap_vars: Dict[str, Dict[Tuple[int, int], cp_model.IntVar]] = {}

    for truck_id, node_ids in truck_nodes.items():
        if len(node_ids) < 2:
            continue
        truck = truck_dict.get(truck_id)

        assigned_vars = {i: action_active[i] for i in node_ids}
        truck_arcs[truck_id] = _build_cycle(f"truck_{truck_id}", node_ids, assigned_vars)

        truck_wrap_vars[truck_id] = {}
        truck_wraps = []
        for (i, j), arc in truck_arcs[truck_id].items():
            wrap = model.NewBoolVar(f"wrap_{truck_id}_{i}_{j}")
            truck_wrap_vars[truck_id][(i, j)] = wrap
            model.Add(wrap <= arc)
            if i == j:
                continue
            # 从矩阵查询 i → j 的移动时间
            from_factory = str(node_map[i].get("factory_id") or "")
            to_factory = str(node_map[j].get("factory_id") or "")
            travel_time = get_travel_time(truck, from_factory, to_factory) if truck else 0

            action_type_i = str(node_map[i].get("action_type") or "")
            if action_type_i == "unload":
                model.Add(action_times[j]["start"] == action_times[i]["end"] + travel_time).OnlyEnforceIf([arc, wrap.Not()])
                model.Add(action_times[j]["start"] + makespan == action_times[i]["end"] + travel_time).OnlyEnforceIf([arc, wrap])
            else:
                model.Add(action_times[j]["start"] >= action_times[i]["end"] + travel_time).OnlyEnforceIf([arc, wrap.Not()])
                model.Add(action_times[j]["start"] + makespan >= action_times[i]["end"] + travel_time).OnlyEnforceIf([arc, wrap])                

            if i != j:
                truck_wraps.append(wrap)

        if truck_wraps:
            model.Add(sum(truck_wraps) <= 1)

    # unload_k → load_{k+1} 同卡车紧邻 + 不跨周期
    by_key: Dict[Tuple[str, int, str, str], List[int]] = {}
    for nd in nodes:
        gid = str(nd.get("cycle_id"))
        step = int(nd.get("step"))
        action_type = str(nd.get("action_type"))
        pr_id = str(nd.get("pr_id", ""))
        by_key.setdefault((gid, step, action_type, pr_id), []).append(nd["id"])

    for (gid, step, action_type, pr_id), unload_ids in list(by_key.items()):
        if action_type != "unload":
            continue
        load_ids = by_key.get((gid, step + 1, "load", pr_id), [])
        if not load_ids:
            continue

        for u in unload_ids:
            u_truck = str(node_map[u].get("truck_id"))
            for v in load_ids:
                v_truck = str(node_map[v].get("truck_id"))
                if u_truck != v_truck:
                    continue
                ring = truck_arcs.get(u_truck)
                if ring is None:
                    continue
                arc_uv = ring.get((u, v))
                if arc_uv is not None:
                    model.Add(arc_uv == 1).OnlyEnforceIf([action_active[u], action_active[v]])
                wrap_uv = truck_wrap_vars.get(u_truck, {}).get((u, v))
                if wrap_uv is not None:
                    model.Add(wrap_uv == 0).OnlyEnforceIf([action_active[u], action_active[v]])


def _enforce_cycle_same_truck_unload_load(
    model: cp_model.CpModel,
    nodes: List[Dict],
    action_active: Dict[int, cp_model.IntVar],
) -> None:
    """周期版本：同一 cycle_id 的取货(unload_k)和下一步放货(load_{k+1})必须使用同一卡车。"""
    by_key: Dict[Tuple[str, int, str, str], List[Dict]] = {}
    for nd in nodes:
        key = (str(nd.get("cycle_id")), int(nd.get("step")), str(nd.get("action_type")), str(nd.get("pr_id", "")))
        by_key.setdefault(key, []).append(nd)

    for (gid, step, action_type, pr_id), unload_nds in by_key.items():
        if action_type != "unload":
            continue
        load_nds = by_key.get((gid, step + 1, "load", pr_id), [])
        if not load_nds:
            continue
        for u_nd in unload_nds:
            for v_nd in load_nds:
                if str(u_nd.get("truck_id")) != str(v_nd.get("truck_id")):
                    model.Add(action_active[u_nd["id"]] + action_active[v_nd["id"]] <= 1)


def _enforce_same_truck_unload_load(
    model: cp_model.CpModel,
    nodes: List[Dict],
    action_active: Dict[int, cp_model.IntVar],
) -> None:
    """非周期版本：同一 good_id 的取货(unload_k)和下一步放货(load_{k+1})必须使用同一卡车。"""
    by_key: Dict[Tuple[str, int, str, str], List[Dict]] = {}
    for nd in nodes:
        key = (str(nd.get("good_id")), int(nd.get("step")), str(nd.get("action_type")), str(nd.get("pr_id", "")))
        by_key.setdefault(key, []).append(nd)

    for (gid, step, action_type, pr_id), unload_nds in by_key.items():
        if action_type != "unload":
            continue
        load_nds = by_key.get((gid, step + 1, "load", pr_id), [])
        if not load_nds:
            continue
        for u_nd in unload_nds:
            for v_nd in load_nds:
                if str(u_nd.get("truck_id")) != str(v_nd.get("truck_id")):
                    model.Add(action_active[u_nd["id"]] + action_active[v_nd["id"]] <= 1)


def _enforce_process_unload_no_overlap(
    model: cp_model.CpModel,
    nodes: List[Dict],
    factories: List[Dict],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    action_active: Dict[int, cp_model.IntVar],
    interfere_time: int = 0,
    extra_intervals: Optional[List[cp_model.IntervalVar]] = None,
) -> None:
    """type 为 process 的工厂，不同工厂的 unload 区间间隔至少 interfere_time（非周期版本）。
    
    将 interval 长度扩展为 duration + interfere_time，
    然后用 AddNoOverlap 保证扩展后的区间不重叠，自然等价于原始区间间隔 ≥ interfere_time。
    
    extra_intervals: 从初态/末态传入的已创建好的 PM unload 区间变量列表，
        由调用方负责创建（含必要的 start_up_time 偏移等），直接加入 NoOverlap。
    """
    process_factory_ids = {
        str(f.get("id"))
        for f in factories
        if str(f.get("type")) == "process"
    }
    if not process_factory_ids:
        return

    intervals = []
    for nd in nodes:
        fid = str(nd.get("factory_id"))
        if fid not in process_factory_ids:
            continue
        if str(nd.get("action_type")) != "unload":
            continue
        nid = nd["id"]
        duration = int(nd.get("action_duration") or 0) + interfere_time
        interval = model.NewOptionalFixedSizeIntervalVar(
            action_times[nid]["start"],
            duration,
            action_active[nid],
            f"proc_nolap_{fid}_{nid}",
        )
        intervals.append(interval)

    # 额外区间（由调用方创建并传入，已含必要的偏移和 interfere_time）
    if extra_intervals:
        intervals.extend(extra_intervals)

    if intervals:
        model.AddNoOverlap(intervals)


def _enforce_process_unload_no_overlap_cyclic(
    model: cp_model.CpModel,
    nodes: List[Dict],
    factories: List[Dict],
    action_times: Dict[int, Dict[str, cp_model.IntVar]],
    action_active: Dict[int, cp_model.IntVar],
    makespan: cp_model.IntVar,
    upper_bound: int,
    interfere_time: int = 0,
) -> None:
    """type 为 process 的工厂，不同工厂的 unload 区间间隔至少 interfere_time（周期版本）。
    
    使用"双副本"技巧处理循环时间轴：
    对每个区间 i，创建原始区间 [s_i, s_i + d_i + g) 和幽灵区间 [s_i + M, s_i + M + d_i + g)，
    其中 M = makespan。然后用 AddNoOverlap 对所有区间（原始 + 幽灵）做非周期不重叠约束。
    
    幽灵 vs 原始 的跨周期约束等价于：原始区间 i 的结束 + interfere_time 不侵入
    下一周期区间 j 的开始位置。
    """
    process_factory_ids = {
        str(f.get("id"))
        for f in factories
        if str(f.get("type")) == "process"
    }
    if not process_factory_ids:
        return

    all_intervals = []
    for nd in nodes:
        fid = str(nd.get("factory_id"))
        if fid not in process_factory_ids:
            continue
        if str(nd.get("action_type")) != "unload":
            continue
        nid = nd["id"]
        start = action_times[nid]["start"]
        active = action_active[nid]
        duration = int(nd.get("action_duration") or 0) + interfere_time

        # 原始区间
        all_intervals.append(
            model.NewOptionalFixedSizeIntervalVar(
                start, duration, active, f"proc_cyc_{fid}_{nid}",
            )
        )
        # 幽灵区间：start + makespan（跨周期副本）
        ghost_start = model.NewIntVar(0, 2 * upper_bound, f"proc_cyc_ghost_start_{nid}")
        model.Add(ghost_start == start + makespan)
        all_intervals.append(
            model.NewOptionalFixedSizeIntervalVar(
                ghost_start, duration, active, f"proc_cyc_ghost_{fid}_{nid}",
            )
        )

    if all_intervals:
        model.AddNoOverlap(all_intervals)
