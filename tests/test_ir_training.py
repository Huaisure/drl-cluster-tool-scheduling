from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from cluster_toolkit.constraint_ir import (
    ConstraintIRV1, InitialStateSpec, IntentSeedSpec,
    IntervalTemplateSpec, LiteralIdRef,
    OperatorTemplateSpec, ReferenceValidator, ResourceSpec, ResourceUseTemplate,
    SetStateTemplateEffect, StateAssignment,
    StateCellSpec, StateCondition, TerminalStateSpec, TimeDomain, compile_problem,
)
from cluster_toolkit.problem import parse_problem
from cluster_rl.ir.data import generate_dataset, load_cases, load_ir
from cluster_rl.ir.env import IRSchedulingEnv
from cluster_rl.ir.graph import (
    EDGE_INDEX, FEATURE_VERSION, IRGraphEncoder, NODE_INDEX, NUMERIC_INDEX,
)
from cluster_rl.ir.network import IRActorCritic, collate_graphs
from cluster_rl.ir.migrate_checkpoint import migrate_graph2_checkpoint
from cluster_rl.ir.sample_search import _portfolio_attempts
from cluster_rl.ir.sft import _policy_score
from cluster_rl.ir.sft_data import replay_expert, verify_expert_coverage
from cluster_rl.ir.safety_head import select_safe_action
from cluster_rl.ir.train import IRTrainConfig, evaluate, load_checkpoint, train
from cluster_rl.ir.wait_control import select_wait_control_action


@pytest.fixture(autouse=True)
def _torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _choice_problem(*, duration: int = 3, label: str = "Pick", name: str = "done") -> ConstraintIRV1:
    templates, seeds = [], []
    for key, ticks in (("short", 1), ("long", duration)):
        templates.append(OperatorTemplateSpec(
            id=key, origin="selectable", intervals=(IntervalTemplateSpec(
                id="activity", start_offset=0, duration=ticks, audit_kind=label,
                resource_uses=(ResourceUseTemplate(resource=LiteralIdRef(value="shared")),),
                end_effects=(SetStateTemplateEffect(cell=LiteralIdRef(value=name), value=True),),
            ),),
        ))
        seeds.append(IntentSeedSpec(id=f"choose/{key}", operator_template_id=key, bindings=(),
                                    alternative_group_id="one", guards=(StateCondition(cell_id=name, operator="equal", value=False),)))
    return ConstraintIRV1(
        time_domain=TimeDomain(unit="second", ticks_per_unit=1),
        resources=(ResourceSpec(id="shared", capacity=1),),
        state_cells=(StateCellSpec(id=name, value_type="bool"),),
        initial_state=InitialStateSpec(state_values=(StateAssignment(cell_id=name, value=False),)),
        terminal_state=TerminalStateSpec(state_values=(StateAssignment(cell_id=name, value=True),)),
        operator_templates=tuple(templates), intent_seeds=tuple(seeds),
    )


def _pm_problem(*, wafers: int = 1, process: int = 5) -> ConstraintIRV1:
    return compile_problem(parse_problem({
        "Modules": {"io": {"type": "IO", "capacity": wafers}, "p": {"type": "PM"}, "q": {"type": "PM"}},
        "ClusterTool": {"r": {"module_ids": ["io", "p", "q"], "arm_type": "single_arm",
                              "pick_time": 1, "place_time": 1, "travel_times": 1}},
        "routes": {"route": [{"module_ids": ["p", "q"], "process_time": process}]},
        "initial_state": {"wafers": [{"route_id": "route", "wafer_index": f"0-{wafers-1}",
                                      "priority": 0, "location": {"kind": "module", "module_id": "io"}}]},
    }), TimeDomain(unit="second", ticks_per_unit=1))


def _graph_equal(a, b) -> bool:
    return all(np.array_equal(getattr(a, name), getattr(b, name)) for name in
               ("node_types", "node_features", "edge_index", "edge_types", "action_nodes"))


def test_safety_filter_preserves_actor_order_and_has_actor_fallback() -> None:
    actor = torch.tensor([2.0, 4.0, 3.0])
    safety = torch.tensor([1.0, -2.0, 0.5])
    assert select_safe_action(actor, safety, 0.0) == 2
    assert select_safe_action(actor, safety, 2.0) == 1
    assert select_safe_action(actor, safety, None) == 1
    assert select_safe_action(actor, safety, None, 0.0) == 0
    assert select_safe_action(actor, safety, None, 0.75) == 2


def test_solver_actions_replay_as_ir_supervision() -> None:
    problem = _pm_problem()
    actions = [
        {"action_type": "pick", "module_id": "io", "tm_id": "r",
         "route_id": "route", "wafer_index": 0, "step_index": 0},
        {"action_type": "place", "module_id": "p", "tm_id": "r",
         "route_id": "route", "wafer_index": 0, "step_index": 1},
        {"action_type": "pick", "module_id": "p", "tm_id": "r",
         "route_id": "route", "wafer_index": 0, "step_index": 1},
        {"action_type": "place", "module_id": "io", "tm_id": "r",
         "route_id": "route", "wafer_index": 0, "step_index": 2},
    ]
    verify_expert_coverage(problem, actions)
    report = replay_expert(problem, actions)
    assert report["success"]
    assert report["expert_action_count"] == len(actions)
    assert report["supervised_choice_count"] > 0


def test_simultaneous_multi_robot_expert_actions_form_a_label_set() -> None:
    problem = compile_problem(parse_problem({
        "Modules": {
            "io": {"type": "IO", "capacity": 2},
            "p": {"type": "PM"},
            "q": {"type": "PM"},
        },
        "ClusterTool": {
            robot: {
                "module_ids": ["io", "p", "q"],
                "arm_type": "single_arm",
                "pick_time": 1,
                "place_time": 1,
                "travel_times": 1,
            }
            for robot in ("r0", "r1")
        },
        "routes": {"route": [{"module_ids": ["p", "q"], "process_time": 5}]},
        "initial_state": {"wafers": [{
            "route_id": "route", "wafer_index": "0-1", "priority": 0,
            "location": {"kind": "module", "module_id": "io"},
        }]},
    }), TimeDomain(unit="second", ticks_per_unit=1))
    actions = []
    for wafer, robot, process_module in ((0, "r0", "p"), (1, "r1", "q")):
        actions.extend((
            {"action_type": "pick", "module_id": "io", "tm_id": robot,
             "route_id": "route", "wafer_index": wafer, "step_index": 0,
             "start": 0},
            {"action_type": "place", "module_id": process_module, "tm_id": robot,
             "route_id": "route", "wafer_index": wafer, "step_index": 1,
             "start": 2},
            {"action_type": "pick", "module_id": process_module, "tm_id": robot,
             "route_id": "route", "wafer_index": wafer, "step_index": 1,
             "start": 8},
            {"action_type": "place", "module_id": "io", "tm_id": robot,
             "route_id": "route", "wafer_index": wafer, "step_index": 2,
             "start": 10},
        ))
    actions.sort(key=lambda action: action["start"])
    label_sets = []
    report = replay_expert(
        problem,
        actions,
        on_choice_set=lambda _graph, choices: label_sets.append(choices),
    )
    assert report["success"]
    assert any(len(choices) == 2 for choices in label_sets)


def test_expert_replay_stably_orders_unsorted_start_times() -> None:
    problem = _pm_problem()
    actions = [
        {"action_type": "pick", "module_id": "io", "tm_id": "r",
         "route_id": "route", "wafer_index": 0, "step_index": 0, "start": 0},
        {"action_type": "place", "module_id": "p", "tm_id": "r",
         "route_id": "route", "wafer_index": 0, "step_index": 1, "start": 2},
        {"action_type": "pick", "module_id": "p", "tm_id": "r",
         "route_id": "route", "wafer_index": 0, "step_index": 1, "start": 8},
        {"action_type": "place", "module_id": "io", "tm_id": "r",
         "route_id": "route", "wafer_index": 0, "step_index": 2, "start": 10},
    ]
    report = replay_expert(problem, [actions[0], actions[2], actions[1], actions[3]])
    assert report["success"]
    assert report["ir_makespan"] == 11


def test_candidate_links_to_immediately_enabled_successor_plans() -> None:
    graph, _ = IRSchedulingEnv(_pm_problem()).reset()
    forward_remaining = EDGE_INDEX["remaining"]
    for action_node in graph.action_nodes:
        outgoing = graph.edge_types[graph.edge_index[0] == action_node]
        assert forward_remaining in outgoing


def test_graph3_exposes_generic_action_statistics_on_candidate_nodes() -> None:
    env = IRSchedulingEnv(_choice_problem(duration=5))
    graph, _ = env.reset()
    duration_feature = NUMERIC_INDEX["action_duration_seconds"]
    encoded = graph.node_features[graph.action_nodes, duration_feature]
    expected = [
        np.arcsinh(candidate.duration_ticks)
        for candidate in env.frame.intents
    ]
    assert FEATURE_VERSION == "ir-graph-3"
    np.testing.assert_allclose(encoded, expected)


def test_graph2_checkpoint_migration_preserves_scalar_projection(tmp_path) -> None:
    model = IRActorCritic(width=16, layers=2)
    old_weight = torch.randn(16, 1)
    state = model.state_dict()
    state["numeric.weight"] = old_weight
    source, destination = tmp_path / "old.pt", tmp_path / "new.pt"
    torch.save({"feature_version": "ir-graph-2", "model": state}, source)
    migrate_graph2_checkpoint(source, destination)
    migrated = torch.load(destination, weights_only=True)
    assert migrated["feature_version"] == "ir-graph-3"
    torch.testing.assert_close(migrated["model"]["numeric.weight"][:, :1], old_weight)
    assert torch.count_nonzero(migrated["model"]["numeric.weight"][:, 1:]) == 0


def test_sample_search_portfolio_keeps_greedy_as_fallback() -> None:
    greedy = {"success": True, "makespan": 10.0}
    sampled = [
        {"success": False, "makespan": None},
        {"success": True, "makespan": 8.0},
    ]
    attempts = _portfolio_attempts(greedy, sampled)
    assert attempts[0] is greedy
    assert attempts[1:] == sampled


def test_wait_control_penalizes_only_optional_wait() -> None:
    logits = torch.tensor([0.1, 0.4, 1.0])
    assert select_wait_control_action(logits, intent_count=2, penalty=0.0) == 2
    assert select_wait_control_action(logits, intent_count=2, penalty=0.7) == 1
    assert select_wait_control_action(logits, intent_count=2, penalty=math.inf) == 1
    assert select_wait_control_action(logits[:2], intent_count=2, penalty=math.inf) == 1


def test_sft_validation_score_prioritizes_completion_before_makespan() -> None:
    reliable = {
        "success_rate": 1.0, "deadlock_rate": 0.0,
        "mean_ratio_to_branch_search": 2.0, "mean_ratio_to_genetic": 2.0,
    }
    fast_subset = {
        "success_rate": 0.5, "deadlock_rate": 0.5,
        "mean_ratio_to_branch_search": 0.5, "mean_ratio_to_genetic": 0.5,
    }
    assert _policy_score(reliable) > _policy_score(fast_subset)


def test_environment_complete_audit_and_telescoping_time_reward():
    env = IRSchedulingEnv(_pm_problem())
    graph, info = env.reset(seed=2)
    assert env.observation_space.contains(graph)
    total = 0.0
    while env.reason is None:
        graph, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(graph)
        total += reward
    assert terminated and not truncated and info["success"]
    assert env.audit().ok and graph.action_count == 0
    assert total == pytest.approx(1 - 0.5 * info["tick"] / (100 + info["tick"]))
    assert len(env.snapshot.kernel_snapshot.active_leases) == 1
    with pytest.raises(RuntimeError):
        env.step(0)


def test_one_decision_budget_allows_its_automatic_completion():
    env = IRSchedulingEnv(_choice_problem(), max_decisions=1)
    env.reset()
    _, _, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and info["success"]


@pytest.mark.parametrize("action", [-1, 10, True, 0.5])
def test_invalid_action_does_not_mutate_session(action):
    env = IRSchedulingEnv(_choice_problem())
    env.reset()
    before = env.snapshot.snapshot_hash
    with pytest.raises(ValueError):
        env.step(action)
    assert env.session.snapshot().snapshot_hash == before


def test_wait_remains_available_beside_other_actions():
    problem = _choice_problem(duration=5)
    # A second independent resource keeps the optional short task committable.
    payload = problem.model_dump(mode="json")
    payload["resources"].append({"id": "other", "capacity": 1})
    payload["operator_templates"][0]["intervals"][0]["resource_uses"][0]["resource"]["value"] = "other"
    for seed in payload["intent_seeds"]:
        seed["alternative_group_id"] = None
    env = IRSchedulingEnv(ConstraintIRV1.model_validate(payload))
    env.reset()
    index = next(i for i, item in enumerate(env.frame.intents) if item.operator_template_id == "long")
    graph, _, t, tr, _ = env.step(index)
    assert not t and not tr and len(env.frame.intents) == 1
    assert graph.action_count == 2 and env.wait_tick == 5
    _, _, t, tr, info = env.step(1)
    assert t and not tr and info["success"] and info["tick"] == 5
    assert env.audit().ok


def test_elapsed_guard_creates_wakeup_and_reset_wait_cost_is_not_lost():
    payload = _choice_problem().model_dump(mode="json")
    payload["state_cells"].append({"id": "timestamp", "value_type": "int"})
    payload["initial_state"]["state_values"].append({"cell_id": "timestamp", "value": 0})
    for seed in payload["intent_seeds"]:
        seed["guards"].append({"cell_id": "timestamp", "operator": "elapsed_at_least", "value": 5})
    env = IRSchedulingEnv(ConstraintIRV1.model_validate(payload))
    _, info = env.reset()
    assert info["tick"] == 5 and env.reason is None
    index = next(i for i, candidate in enumerate(env.frame.intents) if candidate.duration_ticks == 1)
    _, reward, _, _, info = env.step(index)
    assert info["tick"] == 6 and reward == pytest.approx(1 - 0.5 * 6 / 106)
    assert env.audit().ok
    # The same timer through a typed binding row, not a legacy seed.
    payload["binding_domains"] = [{"id": "rows", "parameters": [{"name": "mark", "kind": "state_cell"}],
                                   "rows": [{"values": ["done"]}]}]
    for template in payload["operator_templates"]:
        template["parameters"] = [{"name": "mark", "kind": "state_cell"}]
        template["intervals"][0]["end_effects"][0]["cell"] = {"kind": "parameter", "parameter": "mark"}
    payload["dynamic_intents"] = [{
        "id": seed["id"], "operator_template_id": seed["operator_template_id"], "binding_domain_id": "rows",
        "choice_scope_templates": [{"scope_prefix": "once", "identity_parameters": ["mark"]}],
        "guards": [{"cell": {"kind": "literal", "value": "timestamp"}, "operator": "elapsed_at_least", "value": 5}],
    } for seed in payload["intent_seeds"]]
    payload["intent_seeds"] = []
    dynamic = IRSchedulingEnv(ConstraintIRV1.model_validate(payload))
    assert dynamic.reset()[1]["tick"] == 5
    dynamic.step(0)
    assert dynamic.reason == "success" and dynamic.audit().ok


def test_deadlock_is_not_success_or_infinite_wait():
    payload = _choice_problem().model_dump(mode="json")
    for seed in payload["intent_seeds"]:
        seed["guards"][0]["value"] = True
    env = IRSchedulingEnv(ConstraintIRV1.model_validate(payload))
    graph, info = env.reset()
    assert info["termination_reason"] == "deadlock" and not info["success"]
    assert graph.action_count == 0 and env.audit().ok


@pytest.mark.parametrize("limit", ["time", "decisions"])
def test_episode_caps_are_explicit_failures(limit):
    env = (IRSchedulingEnv(_choice_problem(duration=5), max_time_seconds=2) if limit == "time"
           else IRSchedulingEnv(_pm_problem(), max_decisions=1))
    env.reset()
    index = (next(i for i, c in enumerate(env.frame.intents) if c.duration_ticks == 5)
             if limit == "time" else 0)
    _, reward, terminated, truncated, info = env.step(index)
    assert not terminated and truncated and not info["success"] and reward <= -1
    assert info["termination_reason"] == ("time_limit" if limit == "time" else "decision_limit")
    assert env.audit().ok


def test_deadline_failure_does_not_get_a_success_bonus():
    payload = _choice_problem(duration=5).model_dump(mode="json")
    payload["initial_state"]["obligations"] = [{"id": "pending", "deadline_tick": 3}]
    env = IRSchedulingEnv(ConstraintIRV1.model_validate(payload))
    env.reset()
    index = next(i for i, c in enumerate(env.frame.intents) if c.duration_ticks == 5)
    _, reward, terminated, truncated, info = env.step(index)
    assert terminated and not truncated and reward <= -1
    assert info["termination_reason"] == "deadline_missed" and env.audit().ok


def test_generic_lease_guards_and_unbounded_obligations_without_pm_semantics():
    payload = _choice_problem(label="SomethingUnrelated").model_dump(mode="json")
    payload["resources"][0]["capacity"] = 2
    payload["initial_state"]["leases"] = [{"resource_id": "shared", "owner_id": "token", "amount": 1}]
    payload["terminal_state"]["leases"] = payload["initial_state"]["leases"]
    payload["initial_state"]["obligations"] = [{"id": "pending", "deadline_tick": None}]
    for seed in payload["intent_seeds"]:
        seed["guards"].append({"resource_id": "shared", "owner_id": "token", "operator": "present"})
        seed["required_obligation_ids"] = ["pending"]
    for template in payload["operator_templates"]:
        template["intervals"][0]["start_effects"] = [{
            "kind": "create_obligation", "obligation_id": "generated", "deadline_offset": None,
            "coalesce_key": "request", "priority": 1,
            "condition": {"cell": {"kind": "literal", "value": "done"},
                          "operator": "equal", "value": False, "view": "before"},
        }]
        template["intervals"][0]["end_effects"].append({"kind": "satisfy_obligation", "obligation_id": "pending"})
        template["intervals"][0]["end_effects"].append({"kind": "satisfy_obligation", "obligation_id": "generated"})
    env = IRSchedulingEnv(ConstraintIRV1.model_validate(payload))
    graph, _ = env.reset()
    assert NODE_INDEX["present"] in graph.node_types and NODE_INDEX["obligation"] in graph.node_types
    assert NODE_INDEX["create_obligation"] in graph.node_types
    env.step(0)
    assert env.reason == "success" and env.audit().ok


def test_business_labels_and_identity_spellings_never_enter_numeric_graph():
    original = _choice_problem(label="Pick", name="wafer_progress")
    renamed = _choice_problem(label="CoolingOrCleaning", name="opaque_z17")
    a, b = IRSchedulingEnv(original).reset()[0], IRSchedulingEnv(renamed).reset()[0]
    assert _graph_equal(a, b)
    assert all(isinstance(value, np.ndarray) for value in vars(a).values())
    torch.manual_seed(3)
    model = IRActorCritic(width=16, layers=2)
    result = model(collate_graphs([a, b]))
    torch.testing.assert_close(result.logits[0], result.logits[1])
    torch.testing.assert_close(result.value[0], result.value[1])
    # Rename resources, state cells, symbolic values, parameters, binding rows,
    # templates and rule identities in a compiled dynamic program as well.
    original = _pm_problem(wafers=2)
    raw = original.model_dump(mode="json")
    identities = {item.id for collection in (original.resources, original.state_cells, original.operator_templates,
                                             original.binding_domains, original.dynamic_intents, original.automatic_rules)
                  for item in collection}
    identities.update(value for cell in original.state_cells for value in cell.enum_values)
    identities.update(p.name for domain in original.binding_domains for p in domain.parameters)
    identities.update(value for domain in original.binding_domains for row in domain.rows for value in row.values)
    identities.update(scope.scope_prefix for rule in original.dynamic_intents for scope in rule.choice_scope_templates)
    mapping = {name: f"anonymous_{len(identities)-i}" for i, name in enumerate(sorted(identities))}

    def rename(value):
        if isinstance(value, dict):
            return {key: ("not_a_business_label" if key == "audit_kind" else rename(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        return mapping.get(value, value) if isinstance(value, str) else value

    a = IRSchedulingEnv(original).reset()[0]
    b = IRSchedulingEnv(ConstraintIRV1.model_validate(rename(raw))).reset()[0]
    result = model(collate_graphs([a, b]))
    torch.testing.assert_close(result.logits[0].sort().values, result.logits[1].sort().values, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(result.value[0], result.value[1], atol=2e-6, rtol=2e-6)


def test_future_plan_goal_and_guard_changes_are_observable_before_local_delta_changes():
    first, second = IRSchedulingEnv(_pm_problem(process=3)), IRSchedulingEnv(_pm_problem(process=9))
    a, _ = first.reset()
    b, _ = second.reset()
    assert first.frame.intents[0].state_delta == second.frame.intents[0].state_delta
    assert not _graph_equal(a, b)  # The future Process duration is still visible.
    payload = _choice_problem().model_dump(mode="json")
    baseline = IRSchedulingEnv(ConstraintIRV1.model_validate(payload)).reset()[0]
    payload["terminal_state"]["leases"] = [{"resource_id": "shared", "owner_id": "missing", "amount": 1}]
    changed_goal = IRSchedulingEnv(ConstraintIRV1.model_validate(payload)).reset()[0]
    assert not _graph_equal(baseline, changed_goal)
    payload["intent_seeds"][0]["guards"][0]["operator"] = "not_equal"
    assert not _graph_equal(changed_goal, IRSchedulingEnv(ConstraintIRV1.model_validate(payload)).reset()[0])


def test_node_edge_action_permutations_and_variable_size_batching():
    graph = IRSchedulingEnv(_choice_problem()).reset()[0]
    rng = np.random.default_rng(7)
    order = rng.permutation(len(graph.node_types))
    inverse = np.argsort(order)
    edges = rng.permutation(len(graph.edge_types))
    permuted = replace(graph, node_types=graph.node_types[order], node_features=graph.node_features[order],
                       edge_index=inverse[graph.edge_index[:, edges]], edge_types=graph.edge_types[edges],
                       action_nodes=inverse[graph.action_nodes[::-1]])
    torch.manual_seed(12)
    model = IRActorCritic(width=16, layers=3)
    other = IRSchedulingEnv(_pm_problem()).reset()[0]
    output = model(collate_graphs([graph, permuted, other]))
    torch.testing.assert_close(output.logits[0], output.logits[1].flip(0), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(output.value[0], output.value[1], atol=2e-6, rtol=2e-6)
    assert torch.isneginf(output.logits[2, other.action_count:]).all()
    solo = model(collate_graphs([other]))
    torch.testing.assert_close(output.logits[2, :other.action_count], solo.logits[0], atol=2e-6, rtol=2e-6)
    (output.value.sum() + output.logits[torch.isfinite(output.logits)].sum()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_missing_goal_and_unsupported_time_units_fail_explicitly():
    problem = _choice_problem()
    with pytest.raises(ValueError, match="terminal_state"):
        IRSchedulingEnv(problem.model_copy(update={"terminal_state": None}))
    with pytest.raises(ValueError, match="second"):
        IRGraphEncoder(problem.model_copy(update={"time_domain": TimeDomain(unit="minute", ticks_per_unit=1)}))


def test_dataset_is_materialized_disjoint_repeatable_and_non_destructive(tmp_path):
    first = generate_dataset(tmp_path / "a", train_count=2, validation_count=1, test_count=1)
    second = generate_dataset(tmp_path / "b", train_count=2, validation_count=1, test_count=1)
    hashes = []
    for split in first:
        a = load_cases([first[split]], expected_split=split)
        b = load_cases([second[split]], expected_split=split)
        assert [ir.problem_hash for _, ir in a] == [ir.problem_hash for _, ir in b]
        hashes.extend(ir.problem_hash for _, ir in a)
    assert len(set(hashes)) == len(hashes)
    with pytest.raises(FileExistsError):
        generate_dataset(tmp_path / "a")
    with pytest.raises(ValueError, match="expected validation"):
        load_cases([first["train"]], expected_split="validation")


def test_manifest_detects_tampering_and_path_escape(tmp_path):
    manifests = generate_dataset(tmp_path / "data", train_count=1, validation_count=1, test_count=1)
    path = manifests["train"]
    raw = json.loads(path.read_text())
    raw["instances"][0]["problem_hash"] = "0" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cases([path])
    raw["instances"][0]["ir_file"] = "../../outside.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="directory"):
        load_cases([path])


def test_ppo_updates_actor_saves_loadable_checkpoint_and_audits_evaluation(tmp_path):
    paths = []
    for index, duration in enumerate((3, 4, 5)):
        path = tmp_path / f"case{index}.json"
        path.write_text(_choice_problem(duration=duration).canonical_json())
        paths.append(path)
    config = IRTrainConfig(total_steps=16, rollout_steps=8, minibatch_size=4,
                           epochs=2, width=16, layers=2, seed=13)
    run_dir = tmp_path / "run"
    result = train([paths[0]], [paths[1]], run_dir, config, test_paths=[paths[2]])
    assert result["steps"] == 16 and result["updates"] == 2
    assert result["parameter_change_l2"] > 0 and result["actor_change_l2"] > 0
    assert result["validation"]["success_rate"] == result["test"]["success_rate"] == 1
    for report in (result["validation"], result["test"]):
        assert all(row["audit_ok"] for row in report["cases"])
    saved, loaded_config = load_checkpoint(run_dir / "best.pt")
    again = evaluate(saved, [(paths[1], load_ir(paths[1]))], loaded_config)
    assert again == result["validation"]
    snapshot = (run_dir / "validation_traces/0000.snapshot.json").read_text()
    assert ReferenceValidator.validate_session(load_ir(paths[1]), snapshot, require_terminal=True).ok
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert all(math_value >= 0 for row in metrics for math_value in [row["grad_norm"], row["steps_per_second"]])
    with pytest.raises(ValueError, match="disjoint"):
        train([paths[0]], [paths[0]], tmp_path / "bad", config)
    assert not (tmp_path / "bad").exists()
    with pytest.raises(FileExistsError):
        train([paths[0]], [paths[1]], run_dir, config)


def test_legacy_checkpoints_are_not_misread_as_ir_checkpoints(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model": {}}, path)
    with pytest.raises(ValueError, match="feature protocol"):
        load_checkpoint(path)
