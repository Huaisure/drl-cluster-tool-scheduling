from __future__ import annotations

import warnings
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from cluster_toolkit.cluster_engine import ClusterEngine, PickAction
from cluster_toolkit.problem import ModuleLocation, ModuleType, parse_problem

from .models import GenerationAudit, RouteAudit, RouteWitness
from .topology import ModuleGraph


def validate_generated_instance(raw_instance: Mapping[str, Any]) -> GenerationAudit:
    """Prove schema validity and source-to-source feasibility for every Route."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Explicit Module.capacity overrides the type-based default",
            category=UserWarning,
        )
        problem = parse_problem(raw_instance)

    if problem.just_in_time is not None:
        raise ValueError("generated benchmark instances must not contain just_in_time")
    if problem.cleaning is not None:
        raise ValueError("generated benchmark instances must not contain cleaning")

    if problem.schema_version >= 2:
        _validate_domain_instance(problem)

    graph = ModuleGraph.from_problem(problem)
    if not graph.lp_ids:
        raise ValueError("generated instance has no IO or legacy LP source")

    route_lps: dict[str, set[str]] = defaultdict(set)
    route_return_lps: dict[str, set[str]] = defaultdict(set)
    route_wafer_counts: dict[str, int] = defaultdict(int)
    for wafer in problem.initial_state.wafers:
        if wafer.step_index != 0 or not isinstance(wafer.location, ModuleLocation):
            raise ValueError("generated wafers must start at step 0 in a source Module")
        if problem.Modules[wafer.location.module_id].type not in {
            ModuleType.IO,
            ModuleType.LP,
        }:
            raise ValueError("generated wafers must start in IO or a legacy LP")
        route_lps[wafer.route_id].add(wafer.location.module_id)
        route_return_lps[wafer.route_id].add(problem.return_module_id(wafer))
        route_wafer_counts[wafer.route_id] += 1

    witnesses: dict[str, RouteAudit] = {}
    candidate_step_count = 0
    route_lengths: list[int] = []
    for route_id, route in sorted(problem.routes.items()):
        if route_wafer_counts[route_id] == 0:
            raise ValueError(f"Route {route_id} has no generated wafers")
        start_lps = sorted(route_lps[route_id])
        visits = tuple(visit.module_ids for visit in route.visits)
        if any(
            problem.Modules[module_id].type in {ModuleType.IO, ModuleType.LP}
            for candidates in visits
            for module_id in candidates
        ):
            raise ValueError(f"Route {route_id} contains a source as an internal visit")
        route_witnesses: list[RouteWitness] = []
        for start_lp in start_lps:
            possible_returns = sorted(route_return_lps[route_id])
            for return_lp in possible_returns:
                has_pair = any(
                    wafer.route_id == route_id
                    and isinstance(wafer.location, ModuleLocation)
                    and wafer.location.module_id == start_lp
                    and problem.return_module_id(wafer) == return_lp
                    for wafer in problem.initial_state.wafers
                )
                if not has_pair:
                    continue
                witness = graph.candidate_witness(start_lp, visits, end_lp=return_lp)
                if witness is None:
                    raise ValueError(
                        f"Route {route_id} has no materialization from {start_lp} to {return_lp}"
                    )
                path, end_lp = witness
                if not any(problem.Modules[module_id].type is ModuleType.PM for module_id in path):
                    raise ValueError(f"Route {route_id} has no PM visit")
                route_witnesses.append(
                    RouteWitness(
                        start_lp=start_lp,
                        end_lp=end_lp,
                        witness_path=path,
                    )
                )
        witnesses[route_id] = RouteAudit(
            witnesses=tuple(route_witnesses),
        )
        route_lengths.append(len(route.visits))
        candidate_step_count += sum(len(visit.module_ids) > 1 for visit in route.visits)

    engine = ClusterEngine(problem)
    engine.reset()
    legal_route_ids = {
        action.wafer_key[0]
        for action in engine.available_actions()
        if isinstance(action, PickAction)
    }
    if not legal_route_ids:
        raise ValueError("generated instance has no legal initial Pick")
    if engine.is_deadlocked():
        raise ValueError("generated initial state is deadlocked")

    module_counts = {
        module_type.value: sum(
            module.type is module_type
            for module in problem.Modules.values()
        )
        for module_type in ModuleType
    }
    return GenerationAudit(
        module_counts=module_counts,
        robot_count=len(problem.ClusterTool),
        route_count=len(problem.routes),
        wafer_count=len(problem.initial_state.wafers),
        min_route_steps=min(route_lengths),
        max_route_steps=max(route_lengths),
        candidate_step_count=candidate_step_count,
        routes=witnesses,
    )


def _validate_domain_instance(problem) -> None:
    io_ids = problem.io_module_ids
    if len(io_ids) != 1:
        raise ValueError("schema version 2 generated problems require one virtual IO")
    io_id = io_ids[0]
    if any(
        not isinstance(wafer.location, ModuleLocation)
        or wafer.location.module_id != io_id
        or wafer.step_index != 0
        for wafer in problem.initial_state.wafers
    ):
        raise ValueError("schema version 2 generated wafers must start in virtual IO")
    wafer_count = len(problem.initial_state.wafers)
    if problem.Modules[io_id].capacity < wafer_count:
        raise ValueError("virtual IO capacity must cover every generated wafer")
    if any(
        module.capacity != 1
        for module_id, module in problem.Modules.items()
        if module_id != io_id
    ):
        raise ValueError("every physical generated Module must have capacity 1")

    al_ids = tuple(
        module_id
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.AL
    )
    ll_ids = tuple(
        module_id
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.LL
    )
    buffer_ids = tuple(
        module_id
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.BUFFER
    )
    pm_ids = {
        module_id
        for module_id, module in problem.Modules.items()
        if module.type is ModuleType.PM
    }
    used_pms = {
        module_id
        for route in problem.routes.values()
        for visit in route.visits
        for module_id in visit.module_ids
        if problem.Modules[module_id].type is ModuleType.PM
    }
    if used_pms != pm_ids:
        raise ValueError(f"generated problem contains unused PMs: {sorted(pm_ids - used_pms)}")
    if any(len(problem.Modules[pm_id].process_ids) not in {2, 3} for pm_id in pm_ids):
        raise ValueError("every generated PM must configure two or three processes")

    simple = not al_ids and not ll_ids and not buffer_ids
    if simple:
        if set(problem.ClusterTool) != {"TM1"}:
            raise ValueError("simple topology must contain exactly one unified TM1")
        if not 3 <= len(pm_ids) <= 6:
            raise ValueError("simple topology must retain three to six PMs")
        for route_id, route in problem.routes.items():
            if not 1 <= len(route.visits) <= 8:
                raise ValueError(f"simple Route {route_id} must contain one to eight PM steps")
            if any(
                problem.Modules[visit.module_ids[0]].type is not ModuleType.PM
                for visit in route.visits
            ):
                raise ValueError(f"simple Route {route_id} may contain only PM visits")
            _validate_reentry(problem, route_id, route)
        return

    if len(al_ids) != 1 or not 1 <= len(ll_ids) <= 2:
        raise ValueError("complex topology requires AL x1 and LL x1-2")
    if "ATM1" not in problem.ClusterTool or "VTM1" not in problem.ClusterTool:
        raise ValueError("complex topology requires ATM1 and VTM1")
    expected_atm = {io_id, al_ids[0], *ll_ids}
    if set(problem.ClusterTool["ATM1"].module_ids) != expected_atm:
        raise ValueError("ATM1 reachability must equal IO + AL + LLs")

    dual = bool(buffer_ids)
    if dual:
        if not 1 <= len(buffer_ids) <= 2 or "VTM2" not in problem.ClusterTool:
            raise ValueError("dual topology requires Buffer x1-2 and VTM2")
    elif "VTM2" in problem.ClusterTool:
        raise ValueError("single-vacuum topology must not contain VTM2")

    unit_pms: dict[int, set[str]] = {
        1: {
            module_id
            for module_id in problem.ClusterTool["VTM1"].module_ids
            if problem.Modules[module_id].type is ModuleType.PM
        }
    }
    if dual:
        unit_pms[2] = {
            module_id
            for module_id in problem.ClusterTool["VTM2"].module_ids
            if problem.Modules[module_id].type is ModuleType.PM
        }
    if any(not 3 <= len(unit) <= 6 for unit in unit_pms.values()):
        raise ValueError("every vacuum unit must retain three to six PMs")
    if dual:
        unit1_processes = {
            process_id
            for pm_id in unit_pms[1]
            for process_id in problem.Modules[pm_id].process_ids
        }
        unit2_processes = {
            process_id
            for pm_id in unit_pms[2]
            for process_id in problem.Modules[pm_id].process_ids
        }
        if unit1_processes & unit2_processes:
            raise ValueError("process capabilities must not overlap across vacuum units")

    for route_id, route in problem.routes.items():
        if problem.Modules[route.visits[0].module_ids[0]].type is not ModuleType.AL:
            raise ValueError(f"complex Route {route_id} must start at AL")
        if set(route.visits[1].module_ids) != set(ll_ids):
            raise ValueError(f"complex Route {route_id} second visit must be all LL candidates")
        if set(route.visits[-1].module_ids) != set(ll_ids):
            raise ValueError(f"complex Route {route_id} must end at all LL candidates")
        if route.visits[0].process_time is None or not 10 <= route.visits[0].process_time <= 20:
            raise ValueError(f"complex Route {route_id} AL time must be 10-20 seconds")
        if route.visits[1].process_time != 0 or route.visits[-1].process_time != 0:
            raise ValueError(f"complex Route {route_id} LL Route times must be zero")
        pm_visits = [
            visit
            for visit in route.visits
            if problem.Modules[visit.module_ids[0]].type is ModuleType.PM
        ]
        if not 1 <= len(pm_visits) <= 8:
            raise ValueError(f"complex Route {route_id} must contain one to eight PM steps")
        unit_pattern: list[int] = []
        for visit in pm_visits:
            selected = set(visit.module_ids)
            matching = [index for index, modules in unit_pms.items() if selected <= modules]
            if len(matching) != 1:
                raise ValueError(f"Route {route_id} PM candidates do not belong to one unit")
            unit_pattern.append(matching[0])
        compact = [unit_pattern[0]]
        for unit_index in unit_pattern[1:]:
            if unit_index != compact[-1]:
                compact.append(unit_index)
        if compact not in ([1], [2], [1, 2], [2, 1], [1, 2, 1]):
            raise ValueError(f"Route {route_id} enters VTM2 more than once")
        if not dual and compact != [1]:
            raise ValueError(f"single-vacuum Route {route_id} references another unit")
        cooling_visits = [
            index
            for index, visit in enumerate(route.visits)
            if problem.Modules[visit.module_ids[0]].type is ModuleType.BUFFER
            and float(visit.process_time or 0) > 0
        ]
        if len(cooling_visits) > 3:
            raise ValueError(f"Route {route_id} has more than three cooling visits")
        if any(right == left + 1 for left, right in zip(cooling_visits, cooling_visits[1:])):
            raise ValueError(f"Route {route_id} contains consecutive cooling visits")
        _validate_reentry(problem, route_id, route)


def _validate_reentry(problem, route_id: str, route) -> None:
    process_ids = [
        visit.process_id
        for visit in route.visits
        if problem.Modules[visit.module_ids[0]].type is ModuleType.PM
    ]
    repeated = {
        process_id
        for process_id in process_ids
        if process_ids.count(process_id) > 1
    }
    if len(repeated) > 1:
        raise ValueError(f"Route {route_id} repeats more than one process")
    if repeated:
        process_id = next(iter(repeated))
        positions = [index for index, value in enumerate(process_ids) if value == process_id]
        if len(positions) not in {2, 3, 4}:
            raise ValueError(f"Route {route_id} has unsupported reentry depth")
        if any(right == left + 1 for left, right in zip(positions, positions[1:])):
            raise ValueError(f"Route {route_id} has adjacent repeated process steps")
