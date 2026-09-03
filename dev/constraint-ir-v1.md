# Constraint IR v1 设计规格

状态：设计提案，等待 golden cases 验证后冻结
依赖：[语义基础 v1](./semantic-foundation.md)
目标：把现有 Cluster Tool 问题和未来复杂约束编译为一套封闭、可类型检查、可执行、可回放的通用中间表示

当前可执行增量：已提供普通单机械手 PM 场景的 `compile_problem(ClusterProblem, TimeDomain)`，支持有限路线、重复访问、候选 PM、通用终态声明。具体支持/拒绝范围见[首版转换合同](./problem-to-ir.md)。下文仍包含未实现的目标设计，不能整体当作当前接口。

## 1. 设计结论

Constraint IR v1 不应是一份“字段更多的机台 JSON”，也不应是一门允许注入 Python 的脚本语言。它应当是一个 **Closed Typed IR（封闭的强类型中间表示）**：

- **封闭**：Kernel 只实现有限、版本化的原语；未知节点立即拒绝；
- **强类型**：实体引用、枚举、Tick、整数、布尔值和资源容量不能隐式混用；
- **声明式**：表达约束成立的条件、资源占用和状态效果，不嵌入执行代码；
- **可执行**：同一 IR 同时驱动 Kernel 候选生成和事件推进；
- **可回放**：独立 Validator 根据 IR 复核 Schedule，并根据 CommitLog 复核动态选择的来源和前态；
- **模型无关**：神经网络只消费从 IR 投影出的通用残余约束图，不消费 `LL`、`Clean` 等业务分支。

推荐的数据流：

```text
ClusterProblem / future domain inputs
                │
                ▼
       Domain Compiler Adapter
                │
                ▼
        Constraint IR source
                │
        type-check + static audit
                │
                ▼
         CompiledProblem
        ┌───────┴────────┐
        ▼                ▼
 Semantic Kernel    Independent Validator
        │                ▲
        ├─ DecisionFrame │
        ├─ Snapshot      │
        └─ Schedule ─────┘
```

现有 `ClusterProblem` 继续承担业务输入兼容和领域校验；Compiler Adapter 负责把 PM、LL、Robot、Route、Cleaning 和 JIT 等语义翻译成 IR。这样未来可以增加另一类设备输入 adapter，而不用复制 Kernel、Validator 或模型接口。

## 2. 模块与接口

外部只保留三个深模块，复杂度留在实现内部。

### 2.1 Compiler 模块

```python
compile_problem(domain_problem, compiler_options) -> CompiledProblem
```

接口合同：

- 成功时返回经过类型检查和静态审计的不可变产物；
- 失败时返回带稳定错误码、source path 和相关 ID 的诊断；
- 相同规范化输入、compiler 版本和 options 必须产生相同 `problem_hash`；
- 不修复含糊输入，不对无法精确表示的时间静默取整。

### 2.2 Semantic Kernel 模块

```python
start(compiled_problem) -> KernelSession

KernelSession.frame() -> DecisionFrame
KernelSession.commit(frame_token, intent_ids) -> CommitResult
KernelSession.snapshot() -> KernelSnapshot
KernelSession.schedule() -> Schedule
restore(compiled_problem, session_snapshot) -> KernelSession
```

`commit` 一次接收同一 Decision Epoch 的兼容 Intent 子集。空子集表示通用 Wait：只有存在未来事件、Trigger 或 Deadline 边界，而且推进到下一个稳定态不会确定性地违反 hard obligation 时才可提交。Kernel 在返回前自动推进确定性事件并达到下一个稳定态；模型不再选择 `AdvanceAction`。

当满足事件恰好位于 Deadline Tick 时，该 Event 必须已由更早的 commit 建立 reservation。同 Tick 的处理顺序是“结束边界 → 已承诺的开始边界 → 自动后果 fixed point → Deadline/Invariant 检查”；Kernel 不会在已经宣告 Deadline 失败后向模型开放一个补救决策。

`frame_token` 绑定 `problem_hash + revision + state_hash`。过期 frame 必须返回 `STALE_FRAME`，Kernel 不能按旧 action mask 继续执行。

### 2.3 Validator 模块

```python
validate(compiled_problem, schedule) -> ValidationReport
validate_session(compiled_problem, session_snapshot_or_json) -> ValidationReport
```

Validator 不调用 Kernel 的状态推进实现。它可以复用 IR 数据类型和表达式规范，但必须拥有独立的回放器、资源扫描和义务检查逻辑；否则“Kernel 验证自己”无法发现共同实现错误。

当前 reference 的动态轨迹使用 `validate_session`：快照携带完整 CommitLog/Schedule/状态，审计按提交链独立验证每次选择的绑定域、Guard、scope、预测影响、自动展开和最终状态；恢复前必须通过。单独 `validate(schedule)` 不具备动态选择的历史证据。审计不证明候选全集、最优性或最终可完成性。`TerminalStateSpec` 已支持状态值子集和精确最终 Lease 集合；首版 Compiler 用它检查有限路线全部完成且返回指定站点。通用终态表达式语言仍未实现。

## 3. 顶层产物

```text
ConstraintIR:
  schema_version
  semantic_version
  time_domain
  entity_types
  entities
  relation_tables
  resources
  state_variables
  operator_templates
  automatic_rules
  invariants
  objective
  initial_state
  audit_metadata
```

### 3.1 版本与规范化

```yaml
schema_version: "1.2-reference"
semantic_version: "1.2"
time_domain:
  unit: millisecond
  ticks_per_unit: 1
```

- `schema_version` 决定字段和 AST 节点是否合法；
- `semantic_version` 决定同 Tick 顺序、区间、Deadline 等执行含义；
- `time_domain` 记录外部单位到整数 Tick 的精确映射；
- 规范化产物按稳定 ID 排序，移除仅供展示的 metadata 后计算 `problem_hash`；
- dataset、checkpoint、Schedule 和 Snapshot 都必须保存 `problem_hash` 与两个版本号。

Reference canonicalization 递归排序所有无序声明集合和映射，使用 UTF-8、无多余空白的 JSON，并以 SHA-256 生成 `problem_hash`、`schedule_hash` 和 `snapshot_hash`。因此声明顺序或 Schedule 中 Event/Interval 输入顺序变化不会改变规范化内容。

## 4. 类型系统

v1 只支持以下值类型：

```text
Bool
Int[min, max]
Tick
Enum[finite symbols]
EntityRef[entity_type]
Optional[T]
Set[EntityRef[T]]       # 只读有限集合
```

明确不支持：Float、任意字符串值、动态对象、用户函数、递归类型和运行时创建新类型。

### 4.1 Entity

Entity 是具有稳定身份的静态对象，例如 wafer、module、robot、hand、route_visit。类型名和 ID 用于审计，但字符串本身不得隐式携带语义。

```yaml
entity_types:
  - id: wafer
  - id: module
  - id: robot
  - id: holder

entities:
  - {type: module, id: PM1, attributes: {module_kind: pm}}
  - {type: robot, id: TM1}
  - {type: holder, id: PM1.slot0, attributes: {holder_kind: module_slot}}
  - {type: holder, id: TM1.arm0, attributes: {holder_kind: robot_hand}}
```

`attributes` 必须在 entity type 中声明类型；它们是静态只读数据，不得由 Effect 修改。

### 4.2 Relation Table

Relation Table 是有限的、带类型的静态事实表，用于表达 reachability、process compatibility、route alternatives 和 LL interface side 等关系。

```yaml
relation_tables:
  - id: robot_reaches_module
    columns: [EntityRef[robot], EntityRef[module]]
    rows:
      - [TM1, PM1]
      - [TM1, LL1]
```

Compiler 必须把业务侧的集合、映射和候选列表规范化为 relation rows。表达式通过 `exists_row` 或 `lookup_unique` 读取表；v1 不允许在运行时执行无界 join。

## 5. 动态状态

### 5.1 State Variable

State Variable 是按零个或多个 EntityRef 建索引的类型化动态值。

```yaml
state_variables:
  - id: ll.pressure_level
    keys: [EntityRef[load_lock]]
    value: Enum[atmosphere, vacuum]

  - id: wafer.route_progress
    keys: [EntityRef[wafer]]
    value: Int
```

State Variable 声明可选默认值，`initial_state` 保存例外项。编译后必须能为每个有限 key tuple 得到唯一初值。

只保存约束确实需要且不能从其他记录推导的事实。晶圆位置从 Lease 读取，不再同时写 `wafer.holder`；冷却、加工、压力转换的运行阶段从 Activity interval 推导，完成与否检查对应实例的结束边界。`thermal_phase` 和 `pressure_transition` 不再是基础状态要求。实际温度只有在真实物理约束要求时才作为普通 State Variable 出现。

### 5.2 Kernel Snapshot

Snapshot 是完整、可恢复的动态事实，不是第二种问题输入：

```text
KernelSnapshot:
  problem_hash
  tick
  revision
  state_values
  active_intervals
  active_leases
  future_reservations
  active_obligations
  objective_accumulators
  next_ids
  state_hash
```

Schedule 历史不必全部复制进 Snapshot；恢复后产生的新 Event ID 必须保持确定性且不碰撞。

Reference 将两个层次分开：`KernelSnapshot` 只描述某个 Tick 的动态事实，`SessionSnapshot` 才是可恢复 checkpoint。后者额外保存 `problem_hash`、revision、已提交 Intent/alternative groups、包含未来 reservation 的完整 Schedule、`schedule_hash` 和 `kernel_state_hash`。恢复时不直接信任保存的动态状态，而是独立从 Schedule 回放到 checkpoint Tick，并要求重建 state hash 完全一致。

## 6. 资源模型

所有有限容量约束都统一成 Resource：

```yaml
resources:
  - {id: robot.TM1.motion, capacity: 1}
  - {id: hand.TM1.arm0, capacity: 1}
  - {id: module.PM1.slot, capacity: 1}
  - {id: module.LL1.interface, capacity: 1}
```

v1 有三种运行时占用形式，但共享同一容量账本：

1. **Claim**：Operator 内一个已知 `[start,end)` 区间的占用；
2. **Lease**：在某个 boundary acquire、在未来另一个 boundary release 的跨 Operator 占用；
3. **Reservation**：Intent commit 后、Claim/Lease 生效前对未来容量的承诺。

```text
ResourceUse:
  resource_ref
  amount: positive Int
  owner_key
  start_tick
  end_tick | OPEN
  mode: claim | lease | reservation
```

同一资源任意 Tick 上 `sum(amount) <= capacity`。Lease 的 `OPEN` 只允许存在于 Snapshot，不允许出现在最终 Schedule；终态仍有非白名单 open lease 时验证失败。

一个具体设备若禁止 Pump 与 cooling 重叠，Compiler 让两个 interval claim 同一个 capacity-1 resource。若允许重叠，它们使用不同 resource；不改变 Operator 类型和模型分支。

双臂 Robot 使用同一套表达：`arm0`、`arm1` 分别是 capacity-1 hand resource，可独立持有 wafer Lease；整机运动能力是另一个 `robot.motion` resource。不同 hand 不会隐式绕过整机容量。Source-level geometry compatibility relation 由 Compiler 静态求值：不兼容的并行 variant claim 同一个 capacity-1 exclusion resource，兼容 variant 不 claim。Kernel 只执行统一容量账本，不读取 Robot、arm 或 geometry 名称。

## 7. 受限表达式 AST

所有 Guard、duration、Effect value、Deadline 和 Objective 都使用纯函数 AST。节点必须有静态输入输出类型，不允许副作用。

### 7.1 v1 节点

```text
value:
  literal | parameter | static_attribute | state_value | current_tick
  event_binding | relation_lookup

numeric:
  add | subtract | multiply_by_int | min | max | clamp

boolean:
  equal | not_equal | less | less_equal | greater | greater_equal
  and | or | not | in_set | exists_row

selection:
  if_then_else
```

限制：

- 不支持循环、递归、任意代码、随机数、I/O 和系统时间；
- `multiply` 至少一侧必须是编译期整数常量；
- `relation_lookup` 必须由 schema 证明唯一，否则编译失败；
- Compiler 展开业务侧的 `for all wafers/modules`，运行时 AST 不含量词；
- 每棵 AST 有节点数和深度上限，防止恶意或意外的表达式爆炸；
- Audit metadata 中的人类标签不进入表达式计算。

### 7.2 示例 Guard

```yaml
op: equal
left: {op: state_value, variable: ll.pressure_level, keys: [$ll]}
right: {op: parameter, name: required_side}
```

压力值匹配本身不足以允许取片，还必须通过压力转换结束的边界依赖或相应资源互斥阻止转换期间取片。上述 AST 是目标合同；reference 目前仅实现有限的 StateCondition 和 LeaseCondition 提交前条件，不表示通用表达式和跨 Intent 完成关系已经实现。

`IntentSeedSpec.guards` 接受两种条件的混合列表，`DynamicIntentSpec.guards` 接受对应模板；列表按 AND 求值。`LeaseCondition(resource_id, owner_id, operator="present"|"absent")` 只判断当前资源-owner 对是否存在。`absent` 不等于资源空闲，`present` 不等于唯一持有。它在 commit 前的 Stable State 求值，不是延迟开始或活动全程的不变量；未来预约和新 commit 自身效果不能满足它。独立 Session 审计从前态验证该条件，单独 Schedule 回放不提供此保证。

## 8. Operator Template

Operator Template 是一个有限参数集合加一个确定性的 Boundary Graph（边界图）。

```text
OperatorTemplate:
  id
  origin: selectable | automatic
  parameters
  binding_domain
  commit_guard
  temporal_variables
  temporal_constraints
  intervals
  instant_boundaries
  effects
  lease_changes
  obligation_changes
  step_dependencies
  decision_policy: complete_bundle
  audit_kind
```

同一个 Template 可以只有一个 interval，也可以是由多个基础 interval 组成的复合 Intent。核心语义不区分名为 `Exchange`、`Transport` 或 `CleanThenProcess` 的特殊类型；它只读取步骤、边界依赖、Effect 和资源 Claim。

`step_dependencies` 是步骤边界之间的偏序约束，例如：

```yaml
- predecessor_step_id: pick_out
  predecessor_boundary: end
  successor_step_id: place_in
  successor_boundary: start
  minimum_lag: 0
```

没有依赖边连接、且资源兼容的步骤允许并行，因此 LL 的 cooling 和 pressure transitioning 不需要合成一个枚举状态。`decision_policy: complete_bundle` 表示整个展开先通过 Guard、Lease 和资源回放，再一次性写入 Session；它不表示所有物理 Effect 同 Tick 发生。

### 8.1 参数与 Binding Domain

参数只能绑定有限 Entity、Enum、Int 或 Tick。Binding Domain 由类型集合、relation rows 和静态属性过滤组成。动态 State Guard 不属于 Binding Domain。

```yaml
parameters:
  - {name: wafer, type: EntityRef[wafer]}
  - {name: robot, type: EntityRef[robot]}
  - {name: source, type: EntityRef[module]}
  - {name: target, type: EntityRef[module]}

binding_domain:
  relations:
    - {table: robot_reaches_module, values: [$robot, $source]}
    - {table: robot_reaches_module, values: [$robot, $target]}
```

Candidate Generator 对静态 domain 预建索引，对动态 State Guard 增量求值，不应在每个 Decision Epoch 全量做笛卡尔积。

### 8.2 Boundary Graph

一个 interval 有 `start_boundary`、`end_boundary`、正 duration 和 Resource Claim。Instant boundary 允许零时长，但不得持有区间 Claim。

v1 temporal constraints 只接受可化为 difference constraints（差分约束）的形式：

```text
t_b - t_a >= min_lag
t_b - t_a <= max_lag
t_end - t_start == duration
t_anchor == current_tick
```

边界图必须有限且无正时长因果环。Kernel 使用确定性最早时刻规则求解；若存在多个等价解，按规范化 boundary ID 排序打破平局。v1 不在单个 Intent 内运行通用排程搜索。

### 8.3 Transport bundle 示例

```text
MoveToSource [b0,b1)
Pick         [b1,b2)
MoveToTarget [b2,b3)
Place        [b3,b4)
```

- `b0 == current_tick`；
- 每个 interval duration 从静态表或 State 读取；
- robot motion 对 `[b0,b4)` 连续 reservation，避免中途被其他 Intent 抢占；
- hand lease 在 `Pick.start` acquire、`Place.end` release；
- source slot lease 在 `Pick.end` release；
- target slot reservation 在 commit 时建立，`Place.start` 转成 lease；
- `Place.end` 更新 wafer location，并可能触发 automatic Process/cooling。

如果需要 Pump/Vent，Compiler 生成另一个满足相同高层 transport 选择的合法 Operator variant，其中包含显式 pressure transition interval。模型看到的是 variant 的通用 footprint、duration、slack 和冲突关系，而不是专用 LL 动作分支。

### 8.4 Effect

v1 Effect 仅允许：

```text
set(variable[keys], value)
increment(variable[keys], bounded_int_delta)
set_current_tick(variable[keys])
acquire_lease(resource, owner, amount)
release_lease(resource, owner)
create_obligation(template, bindings, deadline, condition, coalesce_key, priority)
satisfy_obligation(instance_id)
```

`set_current_tick` 是 Operator Template 层的确定性便利原语，实例化后降级为当前边界 Tick 的普通 `set`。同一 fixed-point round 内对同一 state cell 的多个 `set` 必须写入相同值，否则为 `CONFLICTING_EFFECTS`。所有 Effect 先验证、后原子提交，不能出现执行一半失败的状态。

概念上只有三类修改：状态更新（前三项）、持有关系变化（acquire/release）、待办变化（create/satisfy）。保留具体形式是为了明确写入和同 Tick 冲突规则，不给每种业务新增 Effect，也不增加重复的统一包装接口。模型投影尚待实现，不能把这份归类当成现有模型输入协议。

## 9. Automatic Rule

Automatic Rule 只响应具体 Event Boundary，不扫描任意 Python predicate：

```text
AutomaticRule:
  id
  on: event pattern
  when: BoolExpr
  emit:
    operator_template + bindings
    or obligation_template + bindings + deadline
  multiplicity: once_per_event | once_per_binding
```

示例：`Place.end` 到 PM 后自动发出 Process Operator；`Process.end` 后创建 residency obligation；`Place.end` 到 LL 后可同时发出 cooling Operator。Process 和 cooling 可以与之后的其他独立 interval 并行。

静态审计为 automatic rule 建立 emission graph。任何环都必须经过一个正 duration interval、从而推进到未来 Tick；纯同 Tick emission cycle 一律返回 `NON_TERMINATING_AUTO_RULE`。v1 不尝试证明用户自定义的“单调递减度量”。

## 10. Obligation 与 Invariant

### 10.1 Obligation

```text
ObligationTemplate:
  id
  parameters
  satisfy_on: event pattern
  satisfy_when: BoolExpr
  deadline: TickExpr | none
  priority: Int
  violation_kind: hard | soft
```

有 Deadline 的 hard obligation 在 Deadline Tick 完成 fixed point 后仍未满足即非法；等号满足合法。`deadline_tick=None` / `deadline_offset=None` 表示无截止时间，仍是必须满足的待办，不产生 Deadline 边界，不使用大数代替。部分 Schedule 可以保留它，`require_terminal=True` 时不能遗留它；没有要求时则不创建 Obligation。多个业务 trigger 如果产生同一逻辑要求，必须通过显式 `coalesce_key` 合并，或保留为多个独立 obligation；Compiler 不自行猜测。

G07 reference 先冻结一个最小条件代数：`equal`、`not_equal`、`greater_equal`、`elapsed_at_least`，并要求条件明确读取同一边界 Effect 提交前的 `before` view 或提交后的 `after` view。同一 Tick、同一 `coalesce_key` 下只保留最高 priority 请求；最高 priority 若仍对应不同 obligation ID，则静态或回放返回 `UNDER_SPECIFIED_PRIORITY`。若最高 priority 请求对应同一 obligation，则取最早的有限 Deadline，全部无期限时为 `None`，保证结果不依赖 Effect 排列。

### 10.2 Invariant

v1 Invariant 是一个已展开到具体有限 bindings 的 BoolExpr，检查时点为：

- 初始状态；
- 每轮同 Tick Effect 原子提交之后；
- 稳定态对模型可见之前；
- 终态。

持续区间的容量和 no-overlap 不重复写成高频 Invariant，由 Resource ledger 直接保证。

## 11. Intent 与 DecisionFrame

```text
DecisionFrame:
  frame_token
  tick
  revision
  state_projection
  obligations
  intents
  global_features

Intent:
  ephemeral_id
  operator_template_ref
  bindings
  earliest_start
  latest_start
  boundary_plan
  read_set
  write_set
  resource_footprint
  obligation_footprint
  static_objective_delta
```

Intent 分三步过滤：

1. 静态 Binding Domain 成立；
2. commit/start Guard 成立，得到 Enabled Intent；
3. 完整 boundary plan 可建立 reservation 且不立即违反 obligation，得到 Committable Intent。

模型候选集只包含 Committable Intent。`commit` 仍对整批 Intent 重新检查：

- frame token 未过期；
- Intent 之间 read/write 不产生未定义顺序；
- Resource reservations 兼容；
- 同一对象没有互斥 lease/effect；
- Deadline 与已知 temporal constraints 未被破坏。

G09 reference 为候选接口增加 `alternative_group_id`、`earliest_start_tick`、`latest_start_tick`、`duration_ticks` 和 `resource_footprint`。同一 route visit 的多个目标继续引用同一 Operator Template，只改变 bindings 与通用数值特征。一次 commit 对每个 alternative group 最多接受一个 Intent；一旦接受，该 group 的其他 one-shot seed 不再出现在后续 DecisionFrame。Reference 采用确定性 earliest placement；存在 latest tick 时可据此计算 slack，不存在上界时该字段为 `None`，尚不允许策略直接选择连续开始时刻。

Target reservation 不增加新 Effect：未来 `Place.start` 的 `acquire_lease(target, wafer)` 在 Intent 展开时即成为 open-ended Reservation，并参与 batch compatibility 检查。另一个 Intent 若在重叠未来区间预留同一 capacity-1 target，会在 commit 阶段被统一资源账本拒绝。

人类可读的 `audit_kind=Pick/Clean/Pump` 保存在 Schedule，但默认不进入模型特征。模型使用 AST 类型、图角色、资源类型、duration、slack、读写 footprint 和关系结构。

## 12. Schedule

Schedule 是 Validator 的唯一动态输入：

```text
Schedule:
  problem_hash
  schema_version
  semantic_version
  terminal_tick
  operator_instances
  events
  intervals

Event:
  event_id
  tick
  operator_instance_id
  boundary_id
  origin: intent_id | automatic_rule_id
  bindings
  effect_digest
  audit_kind

Interval:
  interval_id
  operator_instance_id
  start_event_id
  end_event_id
  start_tick
  end_tick
  resource_uses
  audit_kind
```

Schedule 必须包含 automatic Process、cooling 以及显式 Pump、Vent、Clean。可以额外保存 trace/debug metadata，但 Validator 不依赖它们判定合法性。

## 13. Objective

v1 支持可组合但受限的确定性目标：

```text
Objective:
  mode: lexicographic | weighted_sum
  terms:
    - makespan
    - sum_completion_time
    - tardiness
    - event_count(filter)
    - interval_duration(filter)
    - soft_obligation_violation
```

权重必须是整数有理缩放后的 Int，不能使用运行时 Float。Feasibility 永远先于 soft objective；hard violation 不允许通过负 reward 抵消。

## 14. 编译与静态审计

Compiler 产出 `CompiledProblem` 前按顺序执行：

1. 输入规范化与 ID 唯一性；
2. 时间单位精确转换；
3. 引用解析和类型检查；
4. finite binding domain 检查；
5. expression AST 类型与复杂度检查；
6. temporal graph 可满足性和因果环检查；
7. Effect 冲突与资源容量基本检查；
8. automatic rule 同 Tick 终止性检查；
9. 初始 State、Lease、Obligation 与 Invariant 检查；
10. canonical serialization 与 hash。

建议的稳定诊断码：

```text
UNKNOWN_REFERENCE
TYPE_MISMATCH
TIME_PRECISION_LOSS
UNBOUNDED_BINDING_DOMAIN
NON_UNIQUE_LOOKUP
UNSAT_TEMPORAL_GRAPH
CONFLICTING_EFFECTS
RESOURCE_OVER_CAPACITY
NON_TERMINATING_AUTO_RULE
UNDER_SPECIFIED_PRIORITY
INVALID_INITIAL_STATE
```

## 15. 现有代码的迁移边界

当前实现可以作为行为样本和 adapter 输入，但不能直接成为 v1 Kernel：

| 当前结构 | v1 去向 | 原因 |
|---|---|---|
| `ClusterProblem` Pydantic schema | Domain Compiler 输入 | 保留已有输入兼容与领域校验 |
| float time + `EPS` | Compiler 转整数 Tick | 消除边界歧义 |
| `PickAction` / `PlaceAction` | Transport Operator variant | 一次承诺完整边界和 reservation |
| `AdvanceAction` | `commit([])` + Kernel 自动推进 | 时间推进不应是模型物理动作 |
| `PendingOperation` | OperatorInstance + Interval + Reservation | 统一可审计运行时对象 |
| wafer `ready_at` | 对应 Activity 的结束边界 + 后续步骤依赖 | 不重复存储运行阶段；跨 Intent 的动态完成关系仍待实现 |
| LL `occupied_ready_at` | Pressure Transition 与 Thermal State | 允许二者独立结束和并行 |
| 现有 action list | v1 expanded Schedule adapter | 兼容旧 Validator 输出 |
| 分主体 Validator | 独立 IR replay Validator | 覆盖 Clean/Pump/Vent/跨主体约束 |

迁移期间可保留旧 Engine 作为 differential oracle（差分参照），但只在两套语义真正重叠的简单案例上比较。

## 16. 明确不进入 v1 的能力

- 用户提供的 Python/DSL 函数；
- 连续浮点时间和概率持续时间；
- 随机故障、随机 arrival 和部分可观测状态；
- Operator 内部通用 CP-SAT 排程；
- 无界量词或运行时创建实体；
- 软化物理合法性的 reward penalty；
- 模型直接修改 State 或发出 raw Event；
- 仅由人类字符串名称推导语义。

这些能力不是永远禁止，而是在 golden cases、Kernel 和 Validator 稳定前不扩大语义面。

## 17. 冻结 v1 前需要验证的设计门

1. 所有首批复杂约束都能只用现有原语表达；
2. LL cooling 与 Pump/Vent 并行不需要组合状态；
3. 清洗三类 trigger 不需要 Kernel 专用分支；
4. Pick/Place bundle 能准确表达双臂、travel 和 target reservation；
5. Candidate Generator 不需要读取 audit name 才能生成候选；
6. Validator 能从 Schedule 独立重建全部状态；
7. 同 Tick permutation tests 得到同一 state hash；
8. 每种 AST 节点至少有一个正例和一个静态拒绝例；
9. `CompiledProblem` 与 `KernelSnapshot` 可 canonical serialize/restore；
10. 10 个 golden cases 全部通过第二实现或手工状态表复核。

通过这些门后，再把本提案转为完整生产 schema 和正式 ADR；现在不应先写大量生产实现。

## 18. Reference implementation 状态

[`cluster_toolkit/constraint_ir`](../cluster_toolkit/constraint_ir/README.md) 已实现 G01—G10 和 Composite Intent 前置合同的可执行纵向切片：

- G01—G04：整数 Tick、左闭右开资源区间、包含边界的 Deadline、同 Tick 原子 Effect；
- G05：参数化 Operator Template、Transport Intent、四段 boundary bundle、commit-time Reservation 和 automatic Process；
- G06：仅保留必要的压力值；cooling/Pump 的运行状态由独立 interval 推导，共享资源控制 overlap policy；
- G07：条件 State Effect、counter/timer 更新、priority coalescing、obligation-gated Clean Intent、显式 Clean interval，以及多 Decision Epoch commit；
- G08：独立 hand Lease、整机 motion capacity、geometry exclusion resource 和 batch Intent compatibility；
- G09：alternative bindings、候选时间窗与 footprint、alternative group at-most-one 合同，以及 future target reservation；
- G10：canonical problem/Schedule/Snapshot、Session restore、state-bound frame token、effect digest 和 terminal closure；
- Composite Intent 前置合同：显式 step dependency、`complete_bundle` 原子提交、候选 involved entities/State Delta、双臂 `PickOut → PlaceIn`，以及正交活动并行；
- 动态生命周期基础：可注入 `CandidateGenerator` seam、Legacy seed Adapter、canonical candidate key/digest、Choice Scope Claim、hash-chain CommitRecord，以及 Snapshot 中的 CommitLog/scope 校验；
- Exhaustive typed Binding Domain：参数列携带 `resource/state_cell/owner/id` 类型，有限行在每个 Stable State 重新求值 Guard；同一 Operator Template 可由不同 bindings 形成多个 OperatorInstance；
- 最小 Lease 查询：State/Lease 条件共用 guards 列表，复用已有 Lease 事实和参数引用，候选生成与独立审计分别求值；支持 present/absent，不新增位置状态或 Kernel 业务分支；
- 最小语义收敛：G06 去除冗余运行阶段状态；初始、事件、模板和快照中的 Obligation 支持 `None` 期限，Kernel/Validator/Session 一致处理；未新增业务 Effect 或统一包装类；
- Validator 独立检查 automatic rule 以及 selectable Operator 的 interval、resource uses、bindings、start/end boundary、effect digest 和 effects，不调用 Kernel 展开或推进实现。
- 动态 CommitLog 审计：从前态重建 frame，检查有限绑定域/Guard/义务门槛、实例身份、单候选和 batch scope/资源，重算候选预测及展开，核对完整 Schedule 和快照事实；恢复使用此入口，替换原先 Session 自检。规范 JSON 的记录数组可重排，因果顺序由 hash chain 决定。

Reference session 已支持相同动态绑定在 scope 释放后重复实例化，以及 commit 停留当前 Tick、`advance_next()` 到下一事件/区间/Deadline。当前 Tick 新提交使用更高 decision round，未来预检查允许待中途决策满足的 open obligation，真实推进仍检查截止时间。连续运行测试可跨多个决策点恢复 Snapshot；snapshot 不能早于最近记录的决策。

首版真实输入 Compiler 已把单机械手 PM 的有限路线降级为每片进度 StateCell、分阶段 scope、State/Lease Guard 和有限绑定行；支持重复 PM 访问、候选站点、自动加工以及真实返回占位。终态是 `TerminalStateSpec` 声明的状态值子集和精确 Lease 集合，并要求无运行活动、未满足义务或未来已承诺事件；未声明终态时保留关闭全部 Lease 的旧规则。

Compiled reference schema 与 Snapshot 为 `1.2-reference`（semantic version `1.2`）。新增终态声明并修正绑定表规范化：参数列与对应值一起排序，不能把行值单独排序。旧参考 IR/快照须从源数据重新生成。完整 RouteVisit 关系、trigger-scoped Obligation、自动 Wait、索引化候选和灵活时间放置仍未实现；Composite/连续运行与有限路线测试不代表完整 G11—G13 已验收。旧 Engine、RL 输入和训练 checkpoint 未变。
