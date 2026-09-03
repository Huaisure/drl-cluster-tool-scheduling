# Constraint IR v1 Reference Slice

This package is the executable semantic reference for golden cases G01-G10 and the prerequisite Composite Intent contract. It is intentionally isolated from the existing Cluster Engine and Validator so the new semantics can be tested before a production migration.

The first business-label-free training Adapter now lives in [cluster_rl/ir](../../cluster_rl/ir/README.md): a Session environment, complete-program/state/goal graph encoder, shared candidate scorer and PPO entry point. It does not replace this reference implementation or the legacy RL entry point.

The first real-input Adapter is now available:

```python
from cluster_toolkit.problem import load_problem
from cluster_toolkit.constraint_ir import compile_problem, TimeDomain

ir = compile_problem(load_problem("examples/scenarios/long_route_1w.json"),
                     TimeDomain(unit="second", ticks_per_unit=1000))
```

It accepts one single/dual-arm robot, IO/LP plus capacity-one PMs, positive Pick/Place/Process durations, equal initial priorities, and wafers initially unprocessed in IO/LP. Finite routes, repeated PM visits, reachable alternatives, schema-v2 process capabilities, configured travel, and prescribed returns are lowered to ordinary resources, typed binding rows, state/Lease guards, effects, and automatic activities. The Adapter never calls the old Engine; the separate `cluster_rl/ir` training Adapter consumes its output.

LL/AL/BUFFER, cleaning, JIT/residency, multiple robots, multi-slot PMs, mixed priorities, running initial states and zero-duration operations are explicitly rejected with `UNSUPPORTED_FEATURE` and a source path. Zero travel is supported. Times must be exactly representable; loss of precision is an error. See [the first compiler contract](../../dev/problem-to-ir.md) for details and test boundaries.

To export canonical JSON (refuses to overwrite an existing output):

```bash
.venv/bin/python -m cluster_toolkit.run_ir_compilation \
  examples/scenarios/long_route_1w.json --output /tmp/long-route.ir.json
```

The public interface is deliberately small:

```python
ticks = compile_ticks(time_domain, external_value, path="...")
result = ReferenceKernel.execute(problem, schedule)
report = ReferenceValidator.validate(problem, schedule)

session = ReferenceKernel.start(problem)
frame = session.frame()
commit = session.commit(frame.frame_token, (frame.intents[0].candidate_key,))
# commit.execution ends at frame.tick, not at the Schedule horizon.
execution = session.advance_next()  # earliest future boundary, or None
frame = session.frame()
report = ReferenceValidator.validate_session(problem, session.snapshot())
# Also accepts canonical JSON; require_terminal=True checks declared goals and work closure.
```

For migration compatibility, `commit()` also accepts the legacy seed `id`.

`ReferenceValidator` does not import or call `ReferenceKernel`; the two implementations share only immutable schema types and diagnostic codes.

Current scope:

- G01 exact time conversion;
- G02 half-open interval resource capacity;
- G03 inclusive deadline satisfaction;
- G04 permutation-independent same-tick effects.
- G05 selectable Transport Intent, multi-boundary expansion, commit-time reservations, and automatic Process.
- G06 a persistent pressure value plus independent activity intervals, concurrent automatic operators, and resource-configured overlap policy. Cooling/processing and pressure-transition running phases are derived, not stored as additional StateCells.
- G07 conditional obligation creation, priority-based coalescing, obligation-gated Intent generation, explicit Cleaning, and multiple decision epochs.
- G08 independent hand leases, whole-robot motion capacity, compiler-lowered geometry exclusion resources, and selectable Operator conformance replay.
- G09 alternative bindings, exclusive choice groups, deterministic earliest placement, candidate time windows, and future target reservations.
- G10 canonical serialization, restorable Session snapshots, state-bound frame tokens, effect digests, and terminal audit.
- Composite Intent prerequisite: explicit step dependencies, complete-bundle commit, generic involved-entity projection, predicted State Delta, dual-arm PickOut/PlaceIn exchange, and parallel orthogonal activity.

G07 adds a deliberately small condition algebra (`equal`, `not_equal`, `greater_equal`, and `elapsed_at_least`) with explicit before/after boundary views. Conditional requests sharing a `coalesce_key` retain the highest priority request. A highest-priority tie between different obligation IDs is rejected as `UNDER_SPECIFIED_PRIORITY`; ties for the same obligation use the earliest finite deadline, or `None` if every winning request is unbounded.

Obligation deadlines are optional throughout initial state, event/template effects, snapshots, Session expansion, and independent validation. `None` means a requirement without a timeout, not an absent requirement. It does not schedule a deadline event, but still prevents terminal validation until satisfied. Existing finite deadline semantics remain inclusive.

Candidate `guards` can mix `StateCondition` and `LeaseCondition`, combined with AND. Dynamic rules use their template forms and the same typed parameter bindings:

```python
LeaseCondition(resource_id="hand0", owner_id="wafer.A", operator="present")
LeaseConditionTemplate(
    resource=ParameterIdRef(parameter="holder"),
    owner=ParameterIdRef(parameter="wafer"),
    operator="present",
)
```

`present` / `absent` test membership of the exact resource-owner pair in the current Stable State. An absent pair does not imply an empty resource or free capacity; a present pair does not imply exclusive ownership or an exact amount. These are admission guards, evaluated before any effects of the new commit, not invariants over its duration or guards at a delayed start. Future reservations cannot satisfy them. Capacity, Lease effects, and scope checks remain separate. No `wafer.location` state, business-specific predicate, or new Kernel execution primitive is introduced. General holder queries, continuous possession and cross-resource owner invariants remain unfinished. Conditional obligation effects still accept only StateCondition.

Both legacy and dynamic generation paths filter these guards, and `validate_session` checks them independently against each replayed decision's prior state. Standalone `validate(schedule)` does not certify admission conditions. The guard extension itself was additive; the later 1.2 protocol change below requires regenerating older reference artifacts.

The reference session supports multiple decision epochs over finite one-shot Intent seeds or declared typed binding rows, including repeated instances of the same dynamic binding after an explicitly releasable Choice Scope becomes available. Instance ordinals are derived from the CommitLog, not another mutable state counter. A permanent scope still prevents repetition; current guards and resources still decide eligibility.

`commit()` reserves and checks the complete future bundle but settles only the current tick. `CommitResult.execution` is actual replay through that tick, not a forecast. `advance_next()` advances to the earliest future Event, interval boundary, or active finite deadline. It returns `None` without mutation when none exists; this does not imply terminal or deadlock. `advance_to(tick)` remains an explicit diagnostic fast-forward, not a policy action. General Wait/Deadlock/Terminal classification, future guard timers, indexed candidate generation, and flexible temporal placement are not implemented.

Events introduced at the current tick carry a positive `decision_round`, after already scheduled round-0 consequences. Each round is atomic and checked for capacity/deadlines; future precommitted boundaries stay in round 0. This prevents later decisions from changing the meaning of an earlier conditional effect. Reservations also order same-tick Lease release/acquire by decision round.

Future feasibility projection uses `allow_open_obligations=True` only to defer as-yet-unscheduled requirements: a shorter intermediate action can satisfy them before a long background interval ends. Current-tick replay and actual advancement still enforce deadlines, and explicitly scheduled late satisfaction is rejected even in projection. This is not a certificate of eventual schedule completion.

G08 does not add a dual-arm action or a geometry predicate to the Kernel interface. A source-level geometry compatibility table is compiled into ordinary capacity-1 exclusion resources for incompatible motion variants. The independent Validator now reconstructs selectable Operator intervals, resources, bindings, boundaries, and effects from their templates, so removing an exclusion claim from a Schedule is auditable.

G09 keeps PM alternatives as bindings of one Operator Template. Each candidate exposes its alternative group, earliest/latest start ticks, duration, and generic resource footprint; `latest_start_tick` remains `None` when no upper bound exists. Commit uses deterministic earliest placement in this reference slice, rejects more than one member of an alternative group, and immediately returns future target Lease reservations. The Validator independently enforces the same at-most-one group contract.

G10 separates a point-in-time `KernelSnapshot` from a restorable `SessionSnapshot`. The latter binds the problem hash, revision, committed choices, complete future Schedule, Schedule hash, and Kernel state hash into canonical JSON. Frame tokens are derived from `problem_hash + revision + state_hash + commitment_hash`. Restore now requires an independent CommitLog audit before creating runtime state. Expanded Events carry SHA-256 effect digests.

`OperatorTemplateSpec` is the one deep module for both primitive and composite selectable behaviour. Multiple intervals are the steps of one Intent; `step_dependencies` state their partial order, and `decision_policy="complete_bundle"` means the Session validates and commits the entire expansion before changing its Schedule. This is commit atomicity, not simultaneous physical execution. The Kernel still executes and audits the underlying interval boundaries.

Every `IntentCandidate` exposes `involved_entity_ids`, `resource_footprint`, and a deterministic `state_delta` containing changed StateCells and Lease ownership. These partial features are now used by the separate full-program/state/goal graph encoder; the delta alone still excludes obligations and is not the entire model input. The encoder includes obligation guards/effects and current obligations explicitly. The Exchange case is expressed as ordinary `PickOut -> PlaceIn` data and remains available beside the independent `PickOut` candidate.

G01-G10 still use the finite `IntentSeedSpec` adapter. New cases declare a typed finite `BindingDomainSpec` and `DynamicIntentSpec`; the exhaustive generator re-evaluates their guards against every current Stable State. Reusing a released scope creates a new operation occurrence, not a second commit of the old instance. The real-input compiler lowers finite visit identity to per-wafer progress and phase-specific permanent scopes; general route relations, per-trigger obligation identity, and the full G11-G13 scenarios remain unfinished.

The dynamic-lifecycle seam is executable: `ReferenceSession` receives a `CandidateGenerator`; `ExhaustiveReferenceCandidateGenerator` enumerates typed rows and composes `LegacyIntentSeedCandidateGenerator` for compatibility. Both paths reuse the same planning, full-bundle validation, canonical sorting, Choice Scope, State Delta, and commit logic. Successful atomic commits append a hash-chained `CommitRecordSpec`; snapshots preserve and verify the CommitLog and active scope set.

`ReferenceValidator.validate_session(problem, snapshot_or_json)` independently audits both declared dynamic candidates and legacy seeds. It reconstructs each decision's prior state and frame token; checks typed row membership, guards, required obligations, temporal variant, occurrence identity, and active/batch Choice Scopes; rederives candidate keys, projected effects, and resource footprints; expands the selected templates and automatic children; and compares the exact complete Schedule and final state. It checks batch resource feasibility separately from individual candidates. It does not import or call the Session generator/expander or Kernel. Existing `validate(problem, schedule)` remains the low-level replay/legacy conformance entry; use `validate_session` for dynamic provenance.

CommitLog causality is defined by `previous_commit_id`, not array order: canonical JSON can reorder arrays. Audit and restore follow the validated chain, reject missing/disconnected/branched records, and restore chronological history before further commits. Recomputed outer hashes do not bypass semantic checks. Hashes are integrity links, not signatures: a completely different but internally consistent legal history cannot be distinguished without an external trusted commitment.

This reference audit proves legality of the selected declared candidates, not completeness of the generated candidate set, the internal behaviour of a custom generator, policy optimality, or eventual schedulability. Forecast replay can defer unresolved obligations; actual decision/checkpoint replay cannot. `require_terminal=True` rejects active obligations, intervals and future committed boundaries. Optional `TerminalStateSpec` requires a subset of exact state values and the exact final Lease set (including amounts); without it, all Leases must close as before. The real-input compiler declares every wafer's final progress and prescribed return Lease, preserving real final occupancy. This is not a general terminal-expression language. Repeated prefix replay intentionally favours independent correctness over throughput; this is an offline reference auditor, not a training-step fast path.

Compatibility: the compiled reference schema and SessionSnapshot versions are now `1.2-reference` (semantic version `1.2`). Terminal goals and a binding canonicalization fix invalidate older reference hashes/snapshots; regenerate them from source data. Binding columns and their row values are now reordered together; positional values must never be independently sorted. Production Problem, Engine, RL inputs, and checkpoints are unchanged. Snapshots cannot predate the latest recorded decision.

The minimal vocabulary groups existing effects into state updates, Lease changes, and Obligation changes. No wrapper classes or business-specific execution branches are added. See [semantic-foundation.md](../../dev/semantic-foundation.md#3-最少事实与通用活动收敛后的合同) for the contract and remaining boundaries. G06 tests exercise the same activity with `Cooling`, `Process`, and neutral audit labels; renaming the label does not change execution semantics. This does not yet prove label-invariant model behaviour.

Run the focused tests with:

```bash
.venv/bin/python -m pytest cluster_toolkit/constraint_ir/tests -q
```
