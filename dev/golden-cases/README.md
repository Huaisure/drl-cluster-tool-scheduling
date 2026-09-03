# Constraint IR v1 Golden Cases

这些案例不是普通单元测试样例，而是 IR、Kernel、Validator 和后续模型数据的共同语义基准。每个案例最终应包含：

```text
case.yaml                 # 最小 Constraint IR source
commits.jsonl             # 每个 Decision Epoch 提交的 Intent bindings
expected-events.jsonl     # 规范化展开事件
expected-snapshots.jsonl  # 每个稳定态的 canonical snapshot
expected-validation.json  # 合法性或稳定错误码
```

在 Pydantic/JSON Schema 冻结前，本文件先固定场景和可观察结果，避免 schema 细节反过来绑架语义。

## G01：整数 Tick 精度拒绝

输入单位为秒，`ticks_per_unit=1000`。

- `pick_time=0.125` 秒必须编译为 125 Tick；
- `process_time=0.0005` 秒无法精确表示，Compiler 返回 `TIME_PRECISION_LOSS`；
- 禁止四舍五入为 1 Tick。

验证目的：输入 adapter 可以使用物理单位，CompiledProblem 只能包含整数 Tick。

## G02：左闭右开容量衔接

capacity-1 的 PM 上：wafer A 的 Process 为 `[0,10)`，wafer B 的 Place/Lease 从 Tick 10 开始。

- A 在 Tick 10 释放资源；
- B 在 Tick 10 获得资源；
- 两者不冲突；
- 若 B 从 Tick 9 开始，返回 `RESOURCE_OVER_CAPACITY`。

验证目的：Kernel、Validator 和 batch selector 使用同一个 `[start,end)` 定义。

## G03：Deadline 等号成立

Process 在 Tick 10 结束并创建 residency obligation：`Pick.start <= 15`。

- 已在更早 commit 中预留的 Pick.start=15：合法并满足 obligation；
- Pick.start=16：在 Tick 15 fixed point 后得到 hard violation；
- Pick.end=15 但 Pick.start=14：仍按声明的 `Pick.start` 满足。

验证目的：Deadline 的边界、满足事件和检查阶段必须显式；Kernel 不在 Deadline 已失败后开放同 Tick 补救决策。

## G04：同 Tick 原子顺序无关

Tick 20 同时发生：

- Process.end 释放 PM operation resource；
- Pick.end 释放 source slot lease；
- Place.start 获取刚释放的另一 slot；
- 一个 Trigger 创建 obligation；
- 另一个 Event 在同 Tick 满足该 obligation。

将输入边界的排列全部打乱，最终必须得到相同：

- state values；
- active leases；
- active obligations；
- resource ledger；
- state hash。

若同轮两个 Effect 向同一 cell 写不同值，应统一返回 `CONFLICTING_EFFECTS`，而不是依赖列表顺序。

## G05：Transport multi-boundary bundle

单臂 Robot 从 IO 取 wafer 并送入 PM：

```text
MoveToSource [0,2)
Pick         [2,3)
MoveToTarget [3,7)
Place        [7,8)
Process      [8,18)   # automatic
```

期望：

- DecisionFrame 只出现一个 Transport Intent，不出现四个内部动作；
- commit 时一次性预留 robot `[0,8)` 和 PM target capacity；
- Schedule 包含全部边界以及 automatic Process；
- Process 不出现在任何可选 Intent 集合中；
- Tick 4 不能插入另一个使用同一 Robot 的 Intent。

## G06：LL pressure 与 cooling 并行

wafer 在 Tick 0 Place.end 到 LL：

```text
cooling: [0,3)
Pump:    [1,5)
```

期望稳定态：

| Tick | 唯一持久状态 pressure_level | 从区间推导的运行活动 |
|---:|---|---|
| 0 | atmosphere | cooling |
| 1 | atmosphere | cooling、Pump |
| 3 | atmosphere | Pump |
| 5 | vacuum | 无 |

- cooling.end 不结束 Pump；
- Pump.end 不重复完成 cooling；
- 目标接口规则要求 vacuum-side Pick 不早于 Tick 5；完整 Pick 门控不在本例验证范围；
- Schedule 显式包含 cooling 与 Pump 两个 interval。

同一 IR 只增加一个共享 capacity-1 resource claim，即可得到“设备禁止二者重叠”的 variant；Kernel 不增加 LL 特判。

不再额外保存 `thermal_phase`、`pressure_transition` 或冷却结束时间字段。活动完成看对应实例的 end boundary；没有活动不意味着已经完成。加工与冷却共用此表示，实际温度只有在约束确实需要时才建模。

## G07：Cleaning trigger 合并与优先级

PM 在同一个 Process.end 同时达到 wafer-count threshold、elapsed-since-clean threshold 和 process-switch clean 条件。

建立三个子例：

1. 三个条件请求使用相同 `coalesce_key`，只创建显式 priority 最高的一个 Clean obligation；
2. obligation 出现后，下一 Decision Epoch 才开放与之匹配的 Clean Intent；
3. 两个不同 obligation 使用相同 `coalesce_key` 和 priority 时，Compiler 返回 `UNDER_SPECIFIED_PRIORITY`。

Clean 必须：

- 要求 PM empty；
- 显式占用 PM interval；
- 在 Schedule 中出现 start/end；
- 只重置其声明负责的 counter/timer；
- 不是 reward penalty 或隐藏 downtime。

## G08：双臂与整机资源

Robot 有 `arm0`、`arm1` 两个 capacity-1 hand resource，同时有 capacity-1 `robot.motion`。

- 两片 wafer 可以同时拥有不同 hand lease；
- 两个 Pick interval 不能仅因为使用不同 hand 就并行占用 `robot.motion`；
- 如果设备 variant 把 `robot.motion` capacity 改为 2，且 geometry compatibility table 允许，两者才可并行；
- 不为“双臂”增加专用 Kernel action。

Reference case 将 geometry compatibility table 视为 Compiler 输入：不兼容的 binding variant 在 Compiled IR 中共同 claim 一个 capacity-1 `robot.geometry.exclusion` resource，兼容 variant 则不 claim。这样 compatibility table 的变化只改变编译产物，不改变 Kernel 或模型动作类型。

验收子例：

1. capacity-1 motion 下两个 Pick 串行执行，最终 `arm0/wafer.A` 与 `arm1/wafer.B` Lease 同时存在；
2. 两个 Intent 分别使用不同 hand，但同 batch 并行提交仍因 `robot.motion` 超容量失败；
3. motion capacity 为 2 且 geometry compatible 时，两个 `[0,2)` Pick 合法并行；
4. motion capacity 为 2 但 geometry incompatible 时，共享 exclusion resource 阻止并行；若从 Schedule 删除该 claim，独立 Validator 返回 `OPERATOR_CONFORMANCE_MISMATCH`。

## G09：Alternative binding 与 target reservation

一个 route visit 可去 PM1 或 PM2。两台都支持同一 process，但 PM1 将更早空闲。

- DecisionFrame 生成两个 bindings 不同、结构相同的 Intent；
- 两者属于 alternative group，同一 wafer 只能提交一个；
- 选择 PM1 时立即建立其 target reservation；
- 同 batch 中另一个 wafer 若导致 PM1 超容量，两个 Intent 不兼容；
- 模型可从 duration、slack、resource footprint 判断差异，不依赖 `PM1` 字符串。

Reference case 使用同一个 `transport.to.target` Operator Template 产生不同 binding：

- wafer A → PM1：earliest/latest start 为 `[1,4]`，target Lease 从 Tick 3 开始；
- wafer A → PM2：earliest/latest start 为 `[4,8]`，target Lease 从 Tick 6 开始；
- 两者共享 `route.visit.A` alternative group，duration 均为 3 Tick；
- wafer B → PM1 属于另一个 group，但与 wafer A → PM1 同时提交会竞争同一 capacity-1 target。

DecisionFrame 不依赖 PM 业务名称判断优劣，而是显式返回 `earliest_start_tick`、`latest_start_tick`、`duration_ticks` 和 `resource_footprint`。不存在最晚开始上界时，`latest_start_tick` 必须保持 `None`，不能用 earliest tick 代替。Reference commit 当前采用 deterministic earliest placement；完整 flexible temporal placement 不在 G09 范围内。

## G10：Snapshot 恢复、stale frame 与独立回放

在 G05 的 Tick 3 保存 Snapshot 并恢复新 KernelSession。

- 恢复前后 `frame_token`、候选规范化内容和后续 Schedule 完全一致；
- 用 Tick 0 的 frame token 在 Tick 3 commit，返回 `STALE_FRAME` 且状态不变；
- 删除 Schedule 中 automatic Process.start，独立 Validator 必须失败；
- 修改一个 Event 的 `effect_digest` 或 boundary binding，Validator 必须失败；
- 完整 Schedule 验证成功，且终态无未关闭 Lease 或 hard obligation。

Reference case 在 G05 Transport 基础上加入独立 `Inspect` 候选和显式 `Unload`：

- Transport commit 后从完整未来 Schedule 截取 Tick 3 checkpoint，此时 `MoveToTarget` 仍活跃；
- `SessionSnapshot` 保存 problem hash、revision、committed Intent/group、完整 Schedule、Schedule hash、KernelSnapshot 和 state hash；
- canonical JSON decode/encode 后字节内容、snapshot hash、frame token 和 Tick 3 候选保持一致；
- Tick 0 token 在恢复后的 Tick 3 commit 返回 `STALE_FRAME`，失败前后 snapshot hash 不变；
- 恢复 Session 自动推进到 Tick 18 后提交 Unload，与未中断 Session 产生相同 canonical Schedule 和终态 state hash；
- `require_terminal=True` 时，Process 后仍持有 target Lease 的中间 Schedule 返回 `NON_TERMINAL_STATE`，Unload 后完整 Schedule 合法；
- expanded Event 的 Effect 与 `effect_digest` 不一致时返回 `EFFECT_DIGEST_MISMATCH`；即使重新计算 digest，错误 boundary binding 或 template Effect 仍由独立 conformance/replay 拒绝。

## 首批覆盖矩阵

| 语义能力 | Cases |
|---|---|
| integer Tick / precision | G01 |
| half-open interval / capacity | G02, G05, G08 |
| deadline / obligation | G03, G04, G07 |
| same-tick fixed point | G04 |
| multi-boundary / reservation | G05, G09 |
| automatic operator | G05, G06, G07 |
| LL orthogonal state | G06 |
| cleaning composition | G07 |
| dual-arm geometry | G08 |
| alternative binding | G09 |
| persistence / audit | G10 |

## 实现顺序

1. 先实现 G01—G04，只包含时间、状态、Effect、资源与 obligation；
2. 再实现 G05，打通 selectable Intent 到 automatic Process；
3. 用 G06 验证正交状态和并发；
4. 用 G07—G09 验证约束组合，而不是增加专用分支；
5. 最后实现 G10，冻结 canonical serialization 和独立 Validator 合同。

每个 case 必须在 Kernel 和 Validator 两边分别断言，且 expected artifact 经人工审查后才能更新；测试失败时不得自动重写 golden output。

## 当前实现状态

G01—G10 的第一版可执行参考位于 [`cluster_toolkit/constraint_ir`](../../cluster_toolkit/constraint_ir/README.md)：

- Pydantic compiled schema：`schema.py`；
- 外部时间精确转换：`compiler.py`；
- Reference Kernel：`reference_kernel.py`；
- 多 Decision Epoch 的 Intent/Reservation 展开：`reference_session.py`；
- 独立回放 Validator：`reference_validator.py`；
- golden tests：`tests/test_golden_cases.py`。

G05 已覆盖 selectable Transport Operator、四段 boundary bundle、commit-time Reservation、`Place.end` 自动 Process，以及 Validator 对缺失 automatic boundary 的独立检查。

G06 已收敛为一个 `pressure_level` StateCell 加两个独立活动区间。Reference case 使用一 Tick Place，因此文中的示意时刻整体平移为 Tick 1/2/4/6；cooling 与 Pump 仍分别结束。测试分别使用 Cooling、Process 和 Activity 审计标签，验证执行语义不依赖业务名称。允许重叠时二者 claim 不同资源；设备配置加入共享 capacity-1 exclusion resource 后，重叠 Intent 自动变为 non-committable。当前 reference 不负责把 Pump 自动延后到 cooling 之后，flexible temporal placement 留给后续阶段。

G07 已覆盖 `Process.end` 对 counter、last-process-type 和 timer 的通用状态更新，四种受限条件运算、条件的 before/after view、同一 `coalesce_key` 的最高优先级归并，以及由 active obligation 和普通 state guard 共同开放下一轮 Clean Intent。Clean 以显式 `[2,7)` interval 占用 PM，结束时重置声明的状态并清偿 obligation；删除清偿 Effect 后，独立 Validator 在 Tick 22 返回 `DEADLINE_MISSED`。这些行为只组合 State、Condition、Effect、Resource、Obligation 和 Intent，没有加入 Clean 专用 Kernel 分支。

G08 已覆盖两只 hand 的独立 Lease、整机 motion Claim、batch commit 容量冲突，以及 geometry incompatibility 到共享 exclusion resource 的编译结果。Validator 现在除 automatic Operator 外，也会独立重建 selectable Operator 的 interval、resource uses、bindings、boundaries 和 effects；因此 Schedule 不能通过删除 geometry claim 绕过约束。

G09 已覆盖同一 Operator Template 的 alternative bindings、最早/最晚开始时间窗、候选 duration 与 resource footprint、alternative group 的跨 Decision Epoch at-most-one 合同，以及未来 target Lease reservation。同组双选返回 `ALTERNATIVE_GROUP_CONFLICT`；不同 wafer 在同 batch 预留同一 target 时返回 `RESOURCE_OVER_CAPACITY`；独立 Validator 也会拒绝拼接出的双选 Schedule。

G10 已覆盖 problem/Schedule/Snapshot canonical hash、Tick 3 SessionSnapshot JSON round-trip、Schedule replay 恢复、state-bound frame token、stale commit 原子失败、effect digest、boundary binding 审计和 terminal closure。恢复路径与未中断路径最终得到相同 canonical Schedule。

收敛补充测试 `tests/test_optional_obligations.py` 覆盖无 Deadline 待办：空 Schedule 下保留、同 Tick/后续满足、与有限 Deadline 共存、合并时只取获胜请求中的最早有限期限、模板展开、候选门控、快照恢复以及终态不能遗留待办。它们补充既有语义，不占用 G11—G13 动态生命周期编号。

连续运行补充测试 `tests/test_continuous_session.py` 覆盖：同一动态绑定连续三个实例、scope 不释放仍不可重复、5 秒短活动结束后在 20 秒后台活动期间重新决策、同 Tick Lease 释放/再获取的 Reservation、8 秒截止前通过中途操作满足待办、未满足时真实推进拒绝、轮次因果顺序、延期开始事件、旧 frame 拒绝及跨决策恢复。G01—G10 中依赖完成结果的测试已改为显式推进，不再将 commit 等同执行结束。

这些测试完成通用 Activity 重复与显式 `advance_next()` 基础，不代替完整 G11—G13。动态来源的独立 Validator 审计、RouteVisit/trigger-scoped Obligation、通用 Wait/Deadlock/Terminal 及灵活时间放置仍留给后续阶段。参考协议现为 `1.1-reference`，旧参考快照不兼容。
