# 动态 Intent 生命周期设计

状态：设计冻结候选稿  
更新时间：2026-09-03  
覆盖矩阵：[constraint-coverage-matrix.md](./constraint-coverage-matrix.md)

实现进度：有限 typed Binding Domain、State/Lease admission guards、canonical candidate key/digest、Choice Scope 和 hash-chain CommitLog 已实现。连续运行基础支持同 bindings 在 scope 释放后形成新实例、commit 停留当前 Tick、`advance_next()` 到下一显式边界，以及跨中途决策恢复。动态 CommitLog 独立审计已接入恢复：逐次验证已选候选合法性、精确展开、资源/义务和最终状态。完整 G11—G13、typed Relation/通用 holder expressions、持续持有不变量、逐次 obligation identity 和 Indexed Generator 尚未实现。以下仍包含目标合同，不应整体视为已实现。

真实输入接入：普通单机械手 PM Compiler 已通过有限进度值和 visit-phase scope 表达重复访问、候选 PM 与返回，并声明状态/占位终态；不依赖 one-shot seeds。支持范围和验收见[首版转换合同](./problem-to-ir.md)。这不是完整通用 RouteVisit/触发身份实现。

## 1. 结论

下一版语义保留现有外部 Session Interface：

```python
frame = session.frame()
result = session.commit(frame.frame_token, selected_candidate_keys)
execution = session.advance_next()  # 显式推进一步；可能返回 None
snapshot = session.snapshot()
```

但 `frame()` 不再遍历预先写入 Problem 的 one-shot `IntentSeedSpec`。它调用动态 Candidate Generator，根据当前 Stable State、未完成承诺、Relation 和 Reservation 重新生成有限的 Committable Intent Candidate。

核心原则：

1. Operator Template 可被重复实例化，不能通过“template/seed 曾提交过”永久屏蔽；
2. Intent Candidate 是当前 DecisionFrame 中的临时选择，不是长期状态；
3. commit 创建不可变的 Committed Intent、Choice Scope Claim 和 OperatorInstance；
4. 候选是否再次出现由新状态和未完成 Claim 决定，不由全局 `_committed_ids` 决定；
5. 一个 Intent 可以承诺多个 owner transition 和多个内部边界；“完整”指完整提交声明的 boundary bundle，不要求每片 wafer 都在该 Intent 结束时到达 Module；
6. Candidate Generator、Kernel 和 Validator 不读取 `audit_kind=Pick/Clean/Exchange` 来决定合法性；
7. 相同 CompiledProblem、Stable State 和未完成承诺必须生成字节级相同的 DecisionFrame。

## 2. 本轮范围

本设计解决覆盖矩阵中的：

- D02：Enabled 与 Committable 分离；
- D03：动态候选生成；
- D04：重复 OperatorInstance；
- P03/P08：visit-scoped alternative 和重复 RouteVisit；
- R04/R07：holder Guard 和 Exchange；
- C06：重复 Cleaning；
- A04：CommitLog 可审计性。

本轮不同时实现：

- 通用 Wait/Deadlock/Terminal；
- flexible temporal placement；
- Objective/reward；
- 随机 duration、故障、外源动态到片；
- commit 后取消、抢占或回滚已开始 Operator。

这些内容分别依赖动态候选稳定后再设计。第一版仍采用 deterministic earliest placement，`latest_start_tick` 没有上界时保持 `None`。

## 3. 术语与对象

### 3.1 Operator Template

静态、可重复使用的声明规则。它描述参数、Binding Domain、Guard、Choice Scope、边界图、Claim、Lease、Effect、Trigger 和 Obligation，但不代表一次具体执行。

### 3.2 Intent Candidate

Candidate Generator 在一个 Decision Epoch 为 Operator Template 产生的一组具体 bindings 和通用 footprint。它只在所属 DecisionFrame 有效。

Candidate 不是持久状态。Snapshot 恢复后应重新生成，而不是保存一份可变 Candidate 列表。

### 3.3 Choice Scope

Choice Scope 是候选之间“正在推进同一份当前工作”的稳定身份。共享同一 scope 的候选互斥；一个复合候选可以同时 claim 多个 scope。

示例：

```text
route-visit/wafer-A/3/place       # wafer A 第 3 个 visit 的落位选择
holder-stage/wafer-B/location-7  # wafer B 当前 holder 阶段的移出选择
obligation/clean-PM1/instance-4  # 第 4 次具体清洗要求
```

Choice Scope 不编码优先级，也不表示资源。它解决的是选择身份；Resource Claim/Lease/Reservation 解决的是物理容量。

### 3.4 Committed Intent

从当前 DecisionFrame 接受 Candidate 后形成的不可变调度承诺。它保存 Candidate 的规范化内容、frame token、scope claims、时间绑定和展开摘要。

Committed Intent 不因执行完成而删除。它属于 CommitLog，是审计事实。

### 3.5 OperatorInstance

Committed Intent 或 automatic rule 对 Operator Template 的一次具体实例化。其 planned/active/completed 状态由当前 Tick 和事件边界推导，不另存一份容易失真的可变状态枚举。

### 3.6 CommitRecord

一次原子 commit 的审计记录。一个 CommitRecord 可以包含多个彼此兼容的 Committed Intent。

## 4. “完整 bundle”的语义修订

“模型不逐个控制 Pick/Place 边界”仍然成立，但不能进一步推导为“每个 Intent 必须把一片 wafer 从 source Module 完整送到 target Module”。后者会排除双臂预取和 Exchange。

允许的通用 holder-transition bundle 包括：

| Bundle 形态 | 初始 holder | 结束 holder | 用途 |
|---|---|---|---|
| Direct transfer | source Module | target Module | 常规搬运 |
| Prefetch | source Module | robot hand | 在目标可用前提前取片 |
| Place continuation | robot hand | target Module | 完成已预取 wafer 的后续放片 |
| Exchange | PM＋incoming hand | outgoing hand＋PM | 同位置先取出、再放入 |

这些不是 Kernel 中的四种硬编码 action。Compiler 使用同一套 boundary、Guard、Lease、Reservation 和 Effect 原语生成不同 Operator Template/variant。模型看到的是绑定、边界图角色、duration、slack、scope 和 resource footprint；人类可读名字只用于 Schedule 审计。

“完整 bundle”准确含义如下：

> commit 一旦接受，Kernel 必须同时验证并预留该 Candidate 声明的全部边界、资源和状态转换；模型不能在 bundle 的内部边界之间改变、删减或替换该承诺。

## 5. Module 与 seam

### 5.1 外部 Session Module

外部 Interface 继续保持小而稳定：

```python
class KernelSession:
    def frame(self) -> DecisionFrame: ...
    def commit(
        self,
        frame_token: str,
        selected_candidate_keys: tuple[str, ...],
    ) -> CommitResult: ...
    def snapshot(self) -> SessionSnapshot: ...
```

调用者不需要知道 Candidate 如何枚举、索引或排除。

### 5.2 Candidate Generator 内部 seam

Candidate Generator 使用独立 Interface：

```python
class CandidateGenerator:
    def generate(
        self,
        problem: CompiledProblem,
        state: KernelSnapshot,
        commitments: ActiveCommitmentView,
    ) -> CandidateGenerationResult: ...
```

需要这个 seam 的理由是存在两个真实实现：

1. `ExhaustiveReferenceCandidateGenerator`：枚举有限 Binding Domain，用于 golden case 和正确性基准；
2. `IndexedCandidateGenerator`：按关系、状态读集和资源变化增量更新，用于生产运行。

两者必须通过同一 conformance suite，返回相同 canonical Candidate 集合。

### 5.3 Kernel 的职责

Kernel 只负责：

- 校验 frame token；
- 重新确认被选 Candidate 仍属于当前 canonical frame；
- 对 batch 进行 scope 与资源联合检查；
- 实例化 Operator Template；
- 原子追加 CommitRecord、Reservation、Event 和 Interval；
- 应用当前 Tick fixed point；
- 产生新 Stable State。

Kernel 不枚举业务候选，也不判断 PM、LL、Clean、Swap 等名称。

### 5.4 Validator 的职责

Validator 独立重放 CommitLog 和 Schedule，检查：

- 每次 commit 的 frame token 与前态一致；
- Candidate 的 bindings、scope、时间和 footprint 合法；
- 每个 Committed Intent 恰好展开其 OperatorInstance；
- automatic Event 不缺失、不重复；
- 不存在没有 commit/trigger 来源的孤立 Event；
- scope claim、Reservation、Lease、Effect 和 terminal 条件一致。

Validator 不调用生产 Candidate Generator 或 Kernel 实现。

当前可执行入口：

```python
report = ReferenceValidator.validate_session(problem, session_snapshot_or_json)
report = ReferenceValidator.validate_session(problem, session_snapshot_or_json, require_terminal=True)
```

`SessionSnapshot` 已携带完整 Schedule、CommitLog 和 KernelSnapshot，不增加另一套审计输入对象。独立实现位于 `commit_audit.py`，复用 Validator 自己的状态回放，不复用 Session 的生成/展开或 Kernel 的执行。`restore()` 先通过该审计，再构造运行时状态，并比较 Kernel 回放结果。

审计只证明“声明规则下，这些已选操作在当时合法且恰好产生所存排程”。它不证明候选集没有遗漏、不审计自定义 Generator 的内部筛选策略，也不保证排程最优或必能完成。`require_terminal=True` 检查没有未满足 Obligation、运行区间和未来已承诺边界；若声明 `TerminalStateSpec`，还检查状态值子集和精确最终 Lease 集合，否则要求所有 Lease 关闭。首版 Compiler 已据此验证有限 Route 完成及返回指定 holder，不删除终点占位；通用终态表达式仍未实现。当前采用多次前缀回放，是小规模离线正确性基准，不应直接放进训练热路径。

## 6. 建议的数据合同

以下是语义字段，不是最终 Pydantic 命名承诺。

### 6.1 IntentCandidate

```yaml
candidate_key: sha256(...)
operator_template_id: transfer.exchange
bindings:
  outgoing: wafer.1
  incoming: wafer.2
  robot: VTM1
  outgoing_hand: VTM1.arm1
  incoming_hand: VTM1.arm0
  module: PM1
choice_scope_claims:
  - scope_key: holder-stage/wafer.1/location-version.8
    release_boundary: pick_out.end
  - scope_key: route-visit/wafer.2/visit.3/place
    release_boundary: place_in.end
earliest_start_tick: 100
latest_start_tick: null
duration_ticks: 2
resource_footprint: [...]
involved_entity_ids: [...]
state_delta:
  completion_tick: 102
  state_values: [...]
  leases: [...]
read_footprint: [...]
write_footprint: [...]
candidate_digest: sha256(...)
```

`candidate_key` 用于稳定排序和选择；`candidate_digest` 绑定 Candidate 的全部规范化内容，防止只保留相同 key 却篡改 footprint。

当前 reference slice 已先验证 `involved_entity_ids + state_delta` 的确定性投影，以及多 interval 的 `complete_bundle` 提交。该验证仍经过 Legacy `IntentSeedSpec` Adapter，只是 G11—G13 的前置条件，不替代本文件定义的动态 Candidate 生命周期。

### 6.2 DecisionFrame

```yaml
frame_token: sha256(problem_hash, revision, state_hash, commitment_hash)
tick: 100
candidates: [...]        # 只包含 individually committable candidates
batch_conflict_data: ... # 可由通用 scope/footprint 计算或压缩表示
```

生成过程中发现的 Enabled-but-not-Committable 选项可以进入 audit diagnostics，但默认不进入模型动作集合。

### 6.3 CommitRecord

```yaml
commit_id: sha256(previous_commit_hash, frame_token, sorted selections)
previous_commit_hash: sha256(...)
frame_token: sha256(...)
tick: 100
selections:
  - candidate_key: sha256(...)
    candidate_digest: sha256(...)
    intent_instance_id: sha256(frame_token, candidate_key)
    operator_instance_ids: [...]
    choice_scope_claims: [...]
expanded_schedule_digest: sha256(...)
```

CommitLog 使用 hash chain，由 `previous_commit_id` 表达因果顺序。规范 JSON 允许重排记录数组，审计和恢复按链重建顺序，不把数组排列当作决策顺序。缺失、分叉、断链和内容不符会被拒绝；即使重算外层 hash，仍须通过候选/展开/状态语义检查。数字签名不属于 v1；没有外部可信摘要时，不能鉴别另一条完全自洽的合法历史是否真实发生过。

## 7. 确定性身份规则

### 7.1 Candidate Key

```text
candidate_key = hash(
    semantic_version,
    source_intent_id,
    operator_template_id,
    canonical bindings,
    canonical choice_scope_claims,
    temporal_variant
)
```

不包含：

- audit name；
- Python object address；
- Candidate 枚举顺序；
- 随机数；
- 模型分数。

### 7.2 Intent Instance ID

```text
intent_instance_id = hash(frame_token, candidate_key)
```

相同逻辑 Candidate 在后续新状态再次出现时，由于 revision/state/commitment 已变化，会得到新的 Intent Instance ID。

当前 reference 用提交记录中相同 `(operator_template_id, canonical bindings)` 的历史次数构造动态 source ID 的 occurrence 后缀，再纳入 Candidate key。它是执行发生编号，不是业务 RouteVisit 编号；没有额外可写计数器。Legacy seed 仍保持一次性。相同模板/绑定在不同规则下的编号共享这段历史，但 source ID 还包含 rule ID，不会混淆事件来源。

### 7.3 OperatorInstance ID

Selectable Operator：

```text
operator_instance_id = hash(intent_instance_id, expansion_ordinal, template_id)
```

Automatic Operator：

```text
operator_instance_id = hash(trigger_event_id, rule_id, emission_ordinal)
```

### 7.4 Choice Scope Key

Scope key 必须包含具体 occurrence 身份，而不能只包含业务类型：

```text
正确：route-visit/wafer.A/visit.3/place
错误：route.visit

正确：obligation/PM1/clean/instance.4
错误：clean.PM1
```

因此同一 wafer 第二次访问 PM1、同一 PM 第二次触发清洗时会得到新的 scope，而不是被第一次 commit 永久屏蔽。

## 8. Candidate 生成流程

### 8.1 进入条件

Candidate Generator 只能在 Stable State 上运行：

1. 当前 Tick 的 end/start/instant boundary 已原子应用；
2. automatic rules 已达到 fixed point；
3. deadline 已检查；
4. 当前 Lease、Reservation 和 active interval 已规范化；
5. State hash 和 active commitment hash 已确定。

### 8.2 生成管线

```text
Operator Templates
    ↓
Binding Domain 的有限关系连接
    ↓
去除不满足静态属性与 Relation 的 bindings
    ↓
计算 choice scopes
    ↓
去除已被 active commit claim 的 scopes
    ↓
求值当前 State/Lease Guards
    ↓
Enabled Candidates
    ↓
确定性 earliest temporal placement
    ↓
展开完整 Resource/Read/Write footprint
    ↓
与当前 Lease/Reservation/Obligation 联合检查
    ↓
Individually Committable Candidates
    ↓
canonical sort + digest
    ↓
DecisionFrame
```

### 8.3 最小表达式能力

第一轮只增加覆盖矩阵已经证明需要的通用节点：

- parameter/value reference；
- StateCell read 与相等/不等/整数比较；
- `relation_contains(table, tuple)`；
- `lease_owner(resource)` 与 `lease_is_free(resource)`；
- entity equality/distinct；
- boolean `and/or/not`；
- Tick/duration 的有界整数表达式。

不加入任意 Python predicate，也不加入 `is_swap`、`is_clean`、`is_load_lock` 等业务节点。

当前已实现的最小子集：`guards` 列表中的条件全部同时成立（AND），每项可为原有 StateCondition 或新增 LeaseCondition。动态规则分别使用对应模板，参数校验复用现有 resource/owner 引用类型。

```python
LeaseCondition(resource_id="robot.arm0", owner_id="wafer.A", operator="present")
LeaseCondition(resource_id="robot.arm0", owner_id="wafer.B", operator="absent")
```

含义分别是“arm0 当前持有 A”“arm0 当前没有持有 B”。第二句不代表 arm0 空闲，第一句不代表它只持有 A；容量仍由资源检查保证。判断只读取已有 Lease，不另存位置状态。

这些是 **Admission Guard（提交前条件）**：每个候选在本次 commit 之前的 Stable State 求值。不能用自身开始效果、同批其他候选的效果或未来 Reservation 证明当前持有。它们也不承诺延迟开始时或整个活动期间持续成立；连续持有和跨资源 owner 唯一性仍待后续不变量合同。条件 Obligation Effect 仍只接受 StateCondition，未扩展任意布尔表达式。

Legacy seed、动态候选、恢复与独立 `validate_session` 已接入这一子集。已有取片/Exchange 参考案例使用相同 LeaseCondition 数据声明；无需修改 Kernel。

### 8.4 Candidate 排序

排序键固定为：

```text
(
  operator_template_id,
  canonical bindings,
  canonical choice_scope_claims,
  temporal_variant,
  candidate_key,
)
```

模型 action index 只是在该 canonical 序列中的位置，不是持久身份。

## 9. Enabled、Committable 与 batch compatibility

### 9.1 Enabled

静态关系、当前 boundary Guard、holder identity 和 choice scope 可用性满足。

### 9.2 Individually Committable

在当前已有承诺不变的前提下，Candidate 的完整 bundle 可以放置并预约，且不会造成已知的 capacity、Lease、deadline 或 invariant 冲突。

这不是“保证一定存在最终完整 Schedule”。全局可完成性属于搜索层或 certificate，不由 Kernel 假装证明。

### 9.3 Batch Committable

每个 Candidate 单独 Committable，不代表它们可以一起提交。batch 必须再次检查：

- choice scope 交集为空；
- Resource Claim/Reservation 联合容量合法；
- 同 Tick Effect 不冲突；
- 同一 owner 不产生多个物理 holder；
- obligation satisfy/create 组合不矛盾；
- alternative bindings 没有同时被接受。

batch 检查失败必须原子返回，CommitLog、Schedule、State 和 revision 均不变化。

## 10. commit 与时间推进

ReferenceSession 已移除 commit 后直接推进到完整 Schedule horizon 的行为。

正确流程：

1. 在当前 Tick 原子追加完整未来 bundle 和 Reservation；
2. 应用当前 Tick 已开始的 boundary 并达到 fixed point；
3. revision 和 commitment hash 更新；
4. 如果当前 Tick 仍有 Committable Candidate，立即返回新的 DecisionFrame；
5. 如果没有候选但存在未来 event/deadline/timer boundary，后续 Wait 语义将自动推进到最早边界；
6. 不允许模型指定任意 `advance_to(t)` 绕过中间边界。

当前确定性推进方法为 `advance_next()`，取最早未来 Event、interval boundary 或 active finite Deadline；无未来边界则无副作用地返回 `None`，不据此判断终态。第 5 步中的自动 Wait、未来 Guard timer 和完整 Waiting/Deadlock/Terminal 状态仍未实现。`advance_to(tick)` 保留给调试/显式回放，不能让模型用它跳过决策点。

新提交只把当前 Tick 的新增事件放入更高 `decision_round`，未来事件仍在 round 0 到时结算。Kernel 与独立 Validator 都按轮次执行，轮内效果顺序无关；`validate_session` 从逐次合法 commit 独立推导轮次，不能通过篡改轮次隐藏较早的资源或 Deadline 违规。

未来预检查允许尚未安排满足动作的 Obligation 保持 open，避免后台长任务把本可在中途处理的 deadline 提前判死。真实推进仍严格检查 Deadline，已经排定却迟到的满足效果也不能通过预检查。该能力不等于全局可完成性证明。

## 11. Choice Scope Claim 生命周期

每个 Candidate 声明一个或多个 scope claim：

```text
available
    ↓ commit
claimed by intent_instance_id
    ↓ declared release boundary
released/consumed
```

第一版不支持 commit cancellation，因此 Claim 不存在“执行失败后自动释放”。commit 前任何失败都必须原子回滚；commit 后的 bundle 被视为确定性可执行承诺。

release boundary 到达后：

- 旧 scope 不再 active；
- State Effect 必须让同一旧工作不再生成，或产生带新 occurrence identity 的下一阶段 scope；
- Validator 检查 scope 的 release boundary 确实属于该 OperatorInstance。

当前 reference 已支持固定 scope key 在显式 release 后复用，每次接受产生新的 source/Operator instance ID。若没有 release，则始终保持占用。这只证明通用活动可重复；业务“某次访问已经消费、不能再做”仍需 visit-specific bindings/Guard，不能把复用 scope 当作完整 RouteVisit 身份机制。重复动作也不是自动执行，只有模型再次选中才会提交。

对于永久 alternative，例如一个 RouteVisit 在 PM1/PM2 二选一，Place.end 后旧 visit scope 被消费，后续 route progress 产生新 visit scope。

## 12. G11：重复 RouteVisit

场景：

```text
Route A:
  visit.1 -> PM1
  visit.2 -> PM2
  visit.3 -> PM1
```

期望：

1. visit.1 和 visit.3 都可以使用同一个 Transport/Process Operator Template；
2. 两次 PM1 访问具有不同 Choice Scope；
3. 第一次提交不把 template 或 `wafer.A→PM1` 永久加入 consumed set；
4. visit.1 Place.end 后 route progress 只前进到 visit.1，不会跳到 visit.3；
5. Snapshot 恢复后得到相同的下一 visit Candidate；
6. Validator 能区分两个不同 OperatorInstance。

关键身份示例：

```text
route-visit/wafer.A/1/place
route-visit/wafer.A/3/place
```

## 13. G12：双臂 Exchange

### 13.1 初始状态

```text
PM1         holds wafer1
arm0        holds wafer2
arm1        empty
wafer1      process completed
wafer2.next allows PM1
robot       can reach PM1
```

### 13.2 Candidate bindings

```text
outgoing       = wafer1
incoming       = wafer2
module         = PM1
robot          = VTM1
outgoing_hand  = arm1
incoming_hand  = arm0
```

### 13.3 Guards

```text
lease_owner(PM1.slot) == wafer1
lease_owner(VTM1.arm0) == wafer2
lease_is_free(VTM1.arm1)
wafer1 != wafer2
arm0 != arm1
state(wafer1.process_phase) == completed
relation_contains(robot_reaches_module, [VTM1, PM1])
relation_contains(route_visit_allows_module, [wafer2.current_visit, PM1])
```

### 13.4 Boundary bundle

```text
b0                     b1                     b2
|------ PickOut -------|------ PlaceIn --------|

robot.motion Claim: [b0,b2)
```

Effects：

```text
b0: acquire arm1 Lease(owner=wafer1)
b1: release PM1.slot Lease(owner=wafer1)
b1: acquire PM1.slot Lease(owner=wafer2)
b2: release arm0 Lease(owner=wafer2)
b2: advance wafer2 route progress
b2: automatic Process(wafer2, PM1)
```

PM1 对 wafer2 的 Reservation 在 commit 时建立，从 `b1` 开始；它与 wafer1 在 `b1` 释放的 Lease 按 `[start,end)` 衔接。

### 13.5 Scope

Exchange 同时 claim：

```text
holder-stage/wafer1/current-location/extract
route-visit/wafer2/current-visit/place
```

因此同一个 batch 中不能再选择另一个 PickOut(wafer1)，也不能同时选择 wafer2→PM2 的 alternative。

### 13.6 Prefetch

Exchange 的 incoming wafer 已在 hand 上，因此系统还必须允许一个较早的 Prefetch bundle：

```text
MoveToSource -> Pick -> end held in hand
```

Prefetch 是一个完整 Operator bundle，但不是完整的 Module-to-Module transfer。后续 Place continuation 或 Exchange 会推进其 holder stage。

### 13.7 负例

- incoming hand 不持有 incoming wafer；
- outgoing hand 已占用；
- PM occupant identity 错误；
- outgoing process 未完成；
- incoming route 不允许 PM1；
- 两个 bindings 使用同一 hand；
- robot.motion 已预约；
- PM1 在 `b1` 之后已被其他 Intent 预约；
- Schedule 删除任一 PickOut/PlaceIn boundary；
- Effect 把错误 wafer 放入 PM1。

## 14. G13：重复 Cleaning

固定业务名不进入 Kernel，但 golden case 可用 Clean 作为代表场景：

```text
Process.end
  → create obligation instance clean/PM1/1
  → generate Clean candidate scoped to instance 1
  → commit/claim instance 1
  → Clean.end satisfy instance 1
  → later Process.end
  → create obligation instance clean/PM1/2
  → generate a new Clean candidate
```

要求：

1. obligation instance id 从 trigger event identity 确定性生成，不能固定为 `clean.PM1`；
2. Clean commit 时 claim obligation scope，防止 Clean 尚未结束时重复提交；
3. Clean.end satisfy 对应具体 instance；
4. 后续同类 trigger 生成不同 instance id；
5. coalesce 只合并声明属于同一 logical trigger window 的请求，不跨清洗周期永久合并；
6. active hard cleaning obligation 必须阻止声明为 incompatible 的 Place Candidate，而不只负责开放 Clean Candidate。

## 15. 与 one-shot IntentSeed 的迁移

G01—G10 不应一次性重写。提供临时 Adapter：

```text
IntentSeedSpec
    ↓ LegacyIntentSeedAdapter
ExhaustiveReferenceCandidateGenerator Interface
```

Adapter 为每个 seed 建立：

```text
choice_scope = legacy-seed/<seed-id>
```

并保持一次性消费语义，只服务已有 golden cases。G11 之后的新案例不得继续依赖 `IntentSeedSpec` 作为真实候选来源。

迁移完成条件：

1. G01—G10 通过 Adapter 保持 canonical 行为；
2. G11—G13 只使用动态 Binding Domain/State；
3. Reference 与 Indexed Candidate Generator 输出一致；
4. 生产 RL Adapter 不再读取 `IntentSeedSpec`；
5. 最终在下一个 schema major version 删除 legacy seed。

## 16. Snapshot 与 CommitLog

动态版本的 SessionSnapshot 不再以 `committed_intent_ids`/`committed_alternative_group_ids` 作为主要真相。建议保存：

```text
problem_hash
revision
tick
commit_log / commit_log_hash
active_choice_scope_claims
schedule / schedule_hash
kernel_snapshot / state_hash
active_commitment_hash
```

恢复时：

1. 验证 Problem hash；
2. 验证 CommitLog hash chain；
3. 独立从 commit 和 automatic trigger 重建 Schedule；
4. 回放到 snapshot Tick；
5. 比较 State、Lease、Obligation、Reservation 和 active scope claim；
6. 重新生成 DecisionFrame 并得到相同 frame token。

连续运行切片曾将 compiled reference schema 与 SessionSnapshot 升级为 `1.1-reference`。首版真实输入 Compiler 又增加通用终态声明并修正绑定表列/值成对规范化，当前为 `1.2-reference`、semantic version `1.2`；旧参考 IR/快照须从源数据重新生成，不影响旧 Engine 或 RL checkpoint。Snapshot 的时刻不能早于最近一次已经记录的决策，避免携带“未来才作出的选择”回到过去。

## 17. 失败原子性与诊断

建议增加稳定诊断类别：

| Code | 含义 |
|---|---|
| `UNKNOWN_CANDIDATE` | selection 不属于声明的 frame |
| `CANDIDATE_DIGEST_MISMATCH` | key 相同但 Candidate 内容不一致 |
| `CHOICE_SCOPE_CONFLICT` | batch 或 active commit 重复 claim scope |
| `HOLDER_MISMATCH` | Lease owner 与 Guard/Effect 声明不一致 |
| `OWNER_LOCATION_CONFLICT` | 同一 wafer 同时存在于不允许并存的 holder |
| `COMMIT_LOG_MISMATCH` | commit hash chain 或 Schedule 展开不一致 |

诊断码不携带业务名字。具体 entity、resource、scope 和 Tick 放在结构化 details 中。

## 18. Conformance 与验收标准

### 18.1 确定性

- 打乱 Entity、Relation、Template、State 和 Lease 的声明顺序，DecisionFrame canonical bytes 不变；
- 同一 Snapshot 重复生成 Candidate，key、digest、顺序和 footprint 完全一致；
- Snapshot round-trip 后一致；
- Reference 与 Indexed Generator 一致。

### 18.2 生命周期

- commit 后旧 frame token 失效；
- active scope 不会重复生成可提交 Candidate；
- release boundary 后根据新状态生成下一阶段 Candidate；
- 同一 Template 可产生多个不同 OperatorInstance；
- batch scope/resource 冲突失败不改变任何状态。

### 18.3 通用性

- G11 重复 RouteVisit、G12 Exchange、G13 重复 Cleaning 共用同一 candidate/claim/commit 实现；
- Kernel 和 Candidate Generator 不出现 `if clean`、`if swap`、`if load_lock`；
- audit name 从 Candidate 特征删除后，合法性和排序不变；
- 每个新增 AST node 至少有正例、静态类型拒绝例和 Validator 篡改负例。

### 18.4 性能接口

第一版正确性基准允许 exhaustive enumeration，但 Interface 需要承诺：

- Binding Domain 有限；
- Candidate 数有配置上限和结构化超限诊断；
- Indexed Generator 可以依据 changed StateCell/resource/relation 增量失效缓存；
- 不要求调用者了解索引策略。

## 19. 实现顺序

1. 增加 Choice Scope、Candidate digest、CommitRecord 的 schema；
2. 抽取 `CandidateGenerator` Interface，并把现有 seed 遍历放入 Legacy Adapter；
3. 实现 Exhaustive Reference Generator；
4. 增加最小 typed Entity、Relation、Binding Domain 和 holder expression；
5. 实现 commit-time scope claim 与 CommitLog；
6. 修改 Session，使 commit 不再自动跳到完整 Schedule horizon；
7. 完成 G11 重复 RouteVisit；
8. 完成 G12 Prefetch＋Exchange；
9. 完成 G13 重复 Cleaning；
10. 独立 Validator 重放 CommitLog（已提前完成当前声明式 reference 协议，为 G11—G13 提供审计基础）；
11. 实现 Indexed Generator，并与 Reference 做差分测试；
12. 再进入 Wait/Deadlock/Terminal 设计。

## 20. 仍需业务确认的事项

这些问题不阻塞 Candidate Generator 的基础 Interface，但会影响 Compiler 和数据：

1. Exchange 是额外的可选复合 Candidate，还是设备要求一旦满足条件就只能 Exchange；推荐默认：作为额外候选，普通合法 continuation 仍保留；
2. source priority 是硬 admission 还是 Objective；
3. 同 recipe wafer-index FIFO 是硬 admission 还是启发式；
4. 是否允许 preventive cleaning；
5. 多 slot LL 是否有 slot directionality 和 batch transition；
6. terminal 是否要求所有 wafer 返回原 IO/LP；
7. flexible temporal variants 应包含哪些有限时间锚点。

在这些规则未明确前，Compiler 必须显式拒绝相关配置或使用问题声明的默认值，不能从 audit name 或设备类型字符串猜测。
