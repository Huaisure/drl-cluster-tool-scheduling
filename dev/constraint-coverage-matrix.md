# 约束覆盖矩阵（首版）

更新时间：2026-09-03

动态候选的后续设计见 [dynamic-intent-lifecycle.md](./dynamic-intent-lifecycle.md)。

本文把真实 Cluster Tool 调度约束逐条映射到统一语义，目的是回答三个问题：

1. 现有 Constraint IR v1 是否已经能够完整表达该约束；
2. 如果不能，约束应在 Compiler、Candidate Generator、Kernel、Validator 还是 Objective 中解决；
3. 下一步应补通用语义，还是只补编译规则、动态候选或数据案例。

这不是最终需求清单。首版优先覆盖当前 Problem Schema、旧 ClusterEngine、旧 Validator、G01—G10 以及项目内领域参考中已经出现的约束。后续应使用真实设备配置、生产 recipe 和失败轨迹继续补充。

## 1. 覆盖状态

| 状态 | 定义 |
|---|---|
| `IR-COVERED` | Constraint IR 参考实现已有可执行语义和直接测试；不表示生产 Adapter 已完成 |
| `IR-PARTIAL` | 已有核心原语，但缺少通用 Guard、动态实例、Compiler 或完整 golden case |
| `COMPILER` | 不应增加 Kernel 分支；应由 Compiler 转成关系、资源、状态或 Operator variant |
| `GAP` | 当前通用语义或运行时生命周期缺失，无法可靠端到端表达 |
| `DECISION` | 约束本身存在，但尚未决定它是硬合法性、admission policy 还是 Objective |
| `DEFERRED` | 明确不进入确定性语义 v1，后续通过版本化扩展处理 |

“旧实现”只作为真实需求和回归证据。旧 ClusterEngine 或 Validator 已实现，不代表新 Constraint IR 已覆盖；反之，G01—G10 已覆盖的语义也尚未接入旧的 RL 环境。

## 2. 时间与区间

| ID | 真实约束 | 性质 | 通用表达 | 当前状态 | 证据与缺口 |
|---|---|---|---|---|---|
| T01 | 外部物理时间必须无损转换为整数时间 | 硬约束 | `TimeDomain`、integer `Tick`、精度拒绝 | `IR-COVERED` | G01 拒绝不可精确表示的时间，不允许静默四舍五入 |
| T02 | Tick `t` 释放的容量可被 Tick `t` 的新操作使用 | 硬约束 | 左闭右开区间 `[start,end)`；同 Tick release-before-acquire | `IR-COVERED` | G02、G04；旧 `ModuleValidator` 也有同 Tick Pick.end/Place.start 测试 |
| T03 | Pick、Place、Process、Clean、Pump、Vent 满足声明 duration | 硬约束 | `end_tick = start_tick + duration` | `IR-COVERED` | `IntervalTemplateSpec.duration`；G05—G07；不同绑定的 duration 仍主要由 Compiler 物化 |
| T04 | 同一 Robot 的连续操作必须留出移动/旋转时间 | 硬约束 | Boundary difference constraint 或显式 Move interval | `IR-PARTIAL` | 旧 `RobotValidator._validate_movement_times` 已检查；G05 有 Move interval，但 IR 尚无 module-pair relation duration lookup |
| T05 | Deadline 等号满足合法，检查发生在该 Tick fixed point 后 | 硬约束 | inclusive deadline、显式 satisfaction boundary | `IR-COVERED` | G03 |
| T06 | 没有最晚开始时间时必须保持 `None` | 接口语义 | nullable `latest_start_tick` | `IR-COVERED` | G09；不能用 earliest 或大数哨兵替代 |
| T07 | 模型可能需要主动延迟某个合法 Intent | 优化相关 | 有限 temporal variants 或通用时间边界候选 | `GAP` | Reference 只做 deterministic earliest placement；不能表达 JIT、凑批等主动等待决策 |

## 3. 资源、占用与身份

| ID | 真实约束 | 性质 | 通用表达 | 当前状态 | 证据与缺口 |
|---|---|---|---|---|---|
| R01 | Module/slot 同时占用数不能超过 capacity | 硬约束 | capacity resource＋wafer Lease | `IR-COVERED` | G02、G05、G09；旧 `ModuleValidator` 检查 capacity |
| R02 | Pick 只能释放实际持有该 wafer 的 source | 硬约束 | owner-aware `Resource Lease` | `IR-COVERED` | Kernel 拒绝释放不存在的 `(resource, owner)` Lease；旧 Validator 仅在 `require_complete` 时检查 occupant identity |
| R03 | 同一 wafer 不能同时位于两个 holder，也不能被两个 Robot 同时操作 | 不变量 | wafer location uniqueness / mutually-exclusive owner invariant | `GAP` | 当前 Kernel 逐 resource 检查容量，但没有跨多个 resource 检查同一 owner 的唯一物理位置；依赖模板正确性不足以防恶意 Schedule |
| R04 | 每只 hand capacity=1；Place 必须使用持有该 wafer 的 hand | 硬约束 | 每 hand 一个 owner-aware Lease＋admission guard | `IR-PARTIAL` | LeaseCondition 已检查提交时指定 hand/wafer 持有关系；测试覆盖错误 hand/owner、Exchange 缺 incoming wafer、容量仍独立检查，以及重算 hash 的非法提交审计。延迟开始和全程持有不变量仍缺 |
| R05 | 双臂可同时持片，但整机通常同一时刻只能做一个机械动作 | 硬约束 | 独立 hand Lease＋共享 `robot.motion` capacity-1 Claim | `IR-COVERED` | G08；没有双臂专用 Kernel action |
| R06 | 双臂姿态或几何关系禁止某些并发组合 | 设备约束 | compatibility relation 编译为 capacity-1 exclusion resource | `COMPILER` | G08 验证编译结果和 Validator 审计；生产 geometry table 与 Compiler 尚未实现 |
| R07 | Robot 可提前持有 incoming wafer，然后对 PM 执行 Pick-out/Place-in Swap | 复合硬约束 | 单个 Exchange Intent 展开多边界 Operator | `IR-PARTIAL` | 已有完整 bundle 的 Exchange template、独立审计和提交前 LeaseCondition；缺包含前序 Prefetch、动态 visit/holder 阶段的完整 G12 场景 |
| R08 | 同一 valve/interface 不能服务重叠 Pick/Place | 可选设备约束 | interface resource Claim / exclusion group | `COMPILER` | 领域参考已明确；当前 Problem Schema 未声明 valve/interface 数据 |

## 4. Route、Process 与绑定

| ID | 真实约束 | 性质 | 通用表达 | 当前状态 | 证据与缺口 |
|---|---|---|---|---|---|
| P01 | wafer 必须按 RouteVisit 顺序流转，不能跳步 | 硬约束 | route-visit state＋Guard＋Effect | `IR-PARTIAL` | 首版 Compiler 用进度值和阶段 scope 表达普通 PM 有限路线，真实 JSON 已端到端验收；通用 typed Entity/Relation 和动态路线未实现 |
| P02 | wafer 在 Place.end 后必须完成 process/align/cool/hold 才能 Pick | 硬约束 | 通用 Activity 结束边界＋后续步骤依赖 | `IR-PARTIAL` | Compiler 测试已证明 Process 结束前没有该片 Pick 候选，后台加工可与其他工作并发；G06 cooling 共用活动语义。AL/LL 等源输入尚未接入 |
| P03 | 同一 RouteVisit 可以选择多个兼容 Module，但只能选择一个 | 硬约束 | binding alternatives＋visit-scoped alternative group | `IR-PARTIAL` | 首版 Compiler 为同片同阶段绑定使用永久 scope，候选 PM 互斥且重复访问可重新选择；通用 RouteVisit 关系未实现 |
| P04 | Robot 只能访问拓扑可达的 Module | 硬约束 | `robot_reaches_module` Relation 过滤 Binding Domain | `COMPILER` | 首版 Compiler 将可达性编译到有限 Binding Domain，并拒绝无可达候选的访问；通用 RelationTable 和多机械手尚未实现 |
| P05 | wafer 的 process_id 必须与 PM 能力兼容 | 硬约束 | `module_supports_process` Relation | `COMPILER` | 首版 Compiler 复用 schema v2 的工艺能力校验，再物化可达候选绑定；Kernel 无 PM 特判，通用 Relation 未实现 |
| P06 | 某些 Module 只允许同一 Robot 取回；BUFFER/LL 可作为 Robot handoff seam | 设备约束 | `handoff_allowed(module, from_robot, to_robot)` Relation＋Guard | `COMPILER` | 旧 Engine/WaferValidator 有 ModuleType 特判；最终应由设备关系数据决定，当前关系表达未实现 |
| P07 | wafer 完成最后 RouteVisit 后必须回到指定 IO/LP | 终态硬约束 | declarative terminal predicate＋return binding | `IR-PARTIAL` | 首版 Compiler 声明最终进度与精确返回 Lease，独立终态审计及不同返回站点测试通过；限已支持的源问题子集 |
| P08 | 同一 wafer 或 Route 可以重复访问同一 PM/Operator | 硬约束 | 唯一 RouteVisit instance＋重复 OperatorInstance | `IR-PARTIAL` | 首版 Compiler 不使用 one-shot seeds；真实 long_route_1w 重复访问已端到端通过，进度＋阶段 scope 区分工作；完整 G11 合同仍待完成 |
| P09 | target 在 Place 真正开始前就必须防止被其他 Intent 抢占 | 硬约束 | commit-time future Reservation | `IR-COVERED` | G05、G09 |

## 5. Load Lock

| ID | 真实约束 | 性质 | 通用表达 | 当前状态 | 证据与缺口 |
|---|---|---|---|---|---|
| L01 | LL 的 Pick/Place side 必须与 Robot interface side 和稳定压力相符 | 硬约束 | pressure State＋`robot_uses_interface_side` Relation＋Guard | `IR-PARTIAL` | G06 覆盖压力值转换，尚非完整取放侧检查；Problem Schema 有 `tm_required_states`；typed Relation 和生产 Compiler 未实现 |
| L02 | Pump/Vent 是有 duration 的显式压力转换，结束时才更新 pressure level | 硬约束 | 通用 Activity interval＋压力值 Effect | `IR-COVERED` | G06；Schedule 中显式可审计，运行阶段无需额外状态 |
| L03 | 压力值与压力转换、晶圆活动的运行阶段相互独立 | 状态语义 | 必要 StateCell＋独立 Activity intervals | `IR-COVERED` | G06 去除 thermal/transition 阶段字段，运行状态从区间推导；不使用组合枚举 |
| L04 | wafer cooling/processing 可以与 Pump/Vent 同时进行 | 设备 variant | 独立资源允许重叠；共享 exclusion resource 禁止重叠 | `IR-COVERED` | G06 已验证两种配置；LL 中 `processing` 与 `transitioning` 可以共存 |
| L05 | 压力转换期间是否允许 Pick/Place 由设备资源配置决定 | 设备 variant | transition 与 transfer claim 相同 interface/exclusion resource | `COMPILER` | 原语足够；尚无包含 Pick/Place 冲突 variant 的 LL golden case |
| L06 | LL slot 可能有方向限制，例如只允许 atmosphere→vacuum | 可选设备约束 | slot-direction Relation＋Binding Domain | `COMPILER` | 领域参考已有需求；当前 Problem Schema 与 IR 均未物化方向关系 |
| L07 | 多 slot LL 的每个 slot 独立持片，同时共享压力状态 | 硬约束 | per-slot Lease＋LL-shared pressure State | `IR-PARTIAL` | capacity/Lease 原语可以表达；G06 仅验证单一 wafer/slot，没有多 slot identity 与批量边界案例 |

## 6. Cleaning

| ID | 真实约束 | 性质 | 通用表达 | 当前状态 | 证据与缺口 |
|---|---|---|---|---|---|
| C01 | Clean.start 时 PM 必须为空，Clean interval 与 wafer/process 互斥 | 硬约束 | Guard＋PM operation resource Claim | `IR-COVERED` | G07 使用 PM occupancy Guard 和共享 operation resource |
| C02 | process type 切换触发清洗，并在不同类型的下一片进入前完成 | 硬约束 | incoming binding 与 last-process state 比较→Obligation→阻断 Place.start | `IR-PARTIAL` | G07 验证 state comparison 和 obligation 请求，但示例在 Process.end 触发，尚未验证真实的“下一片进入前清洗”时序 |
| C03 | 完成 wafer 数达到阈值触发清洗，并在下一片进入前完成 | 硬约束 | bounded counter＋threshold condition→Obligation→阻断 Place.start | `IR-PARTIAL` | G07 验证 counter、threshold 和 Clean 候选；动态版本还需证明 active obligation 会阻断所有不兼容的下一次 Place |
| C04 | PM 连续空闲达到阈值时触发清洗 | 硬约束 | timer boundary→conditional Obligation | `IR-PARTIAL` | G07 有 `elapsed_at_least` 条件，但没有在“未来刚达到 idle threshold”时自动形成新的 Decision Epoch；依赖通用 Wait/timer event |
| C05 | 多个清洗触发同时出现时必须按显式 priority/coalesce 规则处理 | 硬约束 | obligation priority＋coalesce key | `IR-COVERED` | G07；同优先级不同义务返回 `UNDER_SPECIFIED_PRIORITY` |
| C06 | 同一 PM 清洗后继续加工，之后可以再次触发同类清洗 | 硬约束 | 动态 ObligationInstance＋重复 Clean OperatorInstance | `IR-PARTIAL` | 相同动态绑定重复 Activity 已通过；逐次 trigger/obligation 身份及完整 Process→Clean 循环未完成，不能把固定 obligation ID 当作完整 G13 |
| C07 | 是否允许未触发时 preventive clean | 业务规则 | 普通可选 Intent 或 obligation-gated Intent | `DECISION` | 项目参考允许由问题显式决定，但当前 Problem Schema 没有该字段 |

## 7. JIT、Residency 与跨操作约束

| ID | 真实约束 | 性质 | 通用表达 | 当前状态 | 证据与缺口 |
|---|---|---|---|---|---|
| J01 | Process 完成后必须在最大 residency 内开始 Pick | 硬约束 | deadline Obligation，满足边界为 Pick.start | `IR-COVERED` | G03 直接验证 deadline 等号；生产 Problem→IR Compiler 尚未接入 |
| J02 | Process 完成后必须在最大 transfer time 内完成下一次 Place | 硬约束 | deadline Obligation，满足边界为 next Place.end | `IR-PARTIAL` | Effect/Obligation 原语可组合，但当前 automatic rule 只能简单 forward bindings，缺 visit-scoped 动态 obligation satisfaction |
| J03 | RouteVisit 局部 residency、PM/LL 类型默认和全局默认的优先级 | 输入语义 | deterministic precedence rule | `DECISION` | HeteroGraph builder 当前采用 local→module-type→global；尚未在 Compiler/语义文档冻结为统一规则 |
| J04 | 没有 residency 上限时不产生 deadline | 接口语义 | 有后续要求时 Obligation deadline=`None`；无要求时不创建 | `IR-COVERED` | 可选 Deadline 补充测试覆盖初始/事件/模板、混合期限、终态及恢复；生产 Compiler 尚未接入 |

## 8. 决策、动态候选与可完成性

| ID | 真实约束 | 性质 | 通用表达 | 当前状态 | 证据与缺口 |
|---|---|---|---|---|---|
| D01 | 同一 Decision Epoch 可批量提交多个互不冲突 Intent | 接口语义 | batch commit＋联合资源账本检查 | `IR-COVERED` | G05、G08、G09 |
| D02 | Enabled 不等于 Committable；完整 bundle 无法预约时不能提交 | 接口语义 | Guard evaluation＋full-footprint reservation check | `IR-PARTIAL` | Composite prerequisite 已验证 Pick 合法而 Place 资源冲突时完整候选被隐藏，失败 commit 不留部分 Lease/Schedule；最终 Candidate Generator 不应为每个候选全量回放 |
| D03 | 候选必须由当前 Stable State 动态生成，而不是预置一次性列表 | 运行时语义 | indexed Candidate Generator | `IR-PARTIAL` | Exhaustive Generator 已从 typed Binding Domain 逐 Stable State 枚举行并求值 Guard；尚未实现 Relation join、增量索引和完整读集 |
| D04 | 同一 Operator Template 必须支持任意多个 OperatorInstance | 运行时语义 | candidate/commit/instance identity lifecycle | `IR-COVERED` | 连续运行测试验证相同 bindings 连续生成三个不同实例，并跨快照恢复；scope 需显式释放，完整 RouteVisit 身份仍另属 P08/G11 |
| D05 | 没有当前候选但存在未来事件时应自动等待到下一 Decision Epoch | 运行时语义 | Wait/next-boundary semantics | `IR-PARTIAL` | commit 停留当前 Tick；显式 `advance_next()` 到下一 Event/interval/Deadline 已通过，自动 Wait 与未来 Guard timer 未实现；None 不等于完成 |
| D06 | Waiting、Deadlock、Terminal 必须可区分 | 运行时语义 | future-event test＋terminal predicates＋deadlock diagnostics | `GAP` | 旧 Engine 有基础 `is_complete/is_deadlocked`；新 IR 未定义完整决策活性合同 |
| D07 | Candidate 在旧 Snapshot 或旧 DecisionFrame 上不能提交 | 一致性 | state-bound frame token | `IR-COVERED` | G10 |
| D08 | automatic rule fixed point 必须确定性终止 | 安全性 | emission graph＋multiplicity＋cycle rules | `IR-PARTIAL` | 当前按 rule/event 去重并有 1000 次安全上限；未覆盖动态重复实例和所有合法正 duration cycle |
| D09 | action safety 不应把合法双臂交换误判为死锁 | 搜索正确性 | Completable State approximation / continuation proof | `IR-PARTIAL` | 旧 `ActionSafetyFilter` 有 exchange 回归测试；新语义仅定义术语，尚无通用 completable certificate/interface |

## 9. 终态、审计与目标

| ID | 真实约束 | 性质 | 通用表达 | 当前状态 | 证据与缺口 |
|---|---|---|---|---|---|
| A01 | Schedule 中 selectable/automatic Operator 必须与模板展开一致 | 审计 | independent conformance replay | `IR-COVERED` | `validate_session` 从已审计的 legacy/dynamic commit 独立重建全部 Operator、自动子操作及 decision round，与完整 Schedule 精确对照；篡改后重算 hash 仍拒绝 |
| A02 | Problem、Schedule、Snapshot 必须规范化序列化并可恢复 | 一致性 | canonical serialization/hash | `IR-COVERED` | G10 |
| A03 | 完整终态必须包括所有 wafer 完成 Route、返回允许 holder、无运行 interval、无硬义务和非法开放 Lease | 终态硬约束 | declarative terminal predicates | `IR-PARTIAL` | TerminalStateSpec 已支持状态子集＋精确 Lease 集合；Compiler 的有限 PM 路线通过独立完整终态审计。无运行活动/未满足义务/未来边界仍必需；通用终态表达式未完成 |
| A04 | Validator 必须知道每个 Schedule bundle 来自哪次 commit，拒绝孤立或额外事件 | 审计 | first-class CommitLog | `IR-COVERED` | `test_commit_audit.py` 验证链、前态/frame、有限绑定域、Guard/义务、scope、candidate digest 和“恰好展开”；独立于生成器/Kernel，恢复已接入。限已声明 reference 规则，不证明候选集完整性 |
| O01 | 最小 makespan、tardiness、空闲、切换和搬运次数只决定优劣，不改变合法性 | Objective | replayable objective terms | `GAP` | 当前 Constraint IR reference 没有可执行 Objective；RL reward 和合法性必须保持分离 |
| O02 | 初始 wafer priority 是否是必须严格服从的 admission 约束 | 未决 | Guard/admission policy 或 Objective | `DECISION` | 旧 Engine 把较小 priority 作为 source Pick 硬过滤；需要确认业务语义 |
| O03 | 同 priority、同 recipe 的 wafer 是否必须按 wafer_index FIFO | 未决 | admission policy 或 Objective | `DECISION` | 旧 Env 的 ActionSafetyFilter 执行 FIFO，但基础 Engine 只按 priority；当前语义未冻结 |

## 10. 当前数据覆盖情况

现有生产数据计划覆盖了多种 wafer 规模、1—3 个 cell、1—3 个 recipe、单臂/双臂和不同 route pattern，这对学习资源竞争、路径选择和规模泛化有价值。

但 `cluster_toolkit/cluster_generator/validation.py` 当前明确拒绝：

```text
just_in_time != None
cleaning != None
```

因此需要区分：

- **语义已覆盖**：例如 G07 已验证 Cleaning 的部分通用语义；
- **训练数据已覆盖**：当前 production corpus 没有 Cleaning/JIT 实例；
- **旧系统字段存在**：Problem Schema 有字段，但旧 Validator 的测试明确说明 JIT 尚未检查。

仅凭当前训练数据无法证明以下能力：

- residency deadline；
- completion-to-next-load；
- idle/process-switch/wafer-count cleaning；
- repeated cleaning；
- 显式 Pump/Vent 与 cooling overlap；
- Exchange Intent；
- 多 slot LL directionality；
- 故障或外源事件。

## 11. 首版结论

### 11.1 已证明足够通用的语义

- integer Tick 与 `[start,end)`；
- capacity Claim、owner-aware Lease 和 future Reservation；
- multi-boundary Operator；
- automatic Operator；
- 同 Tick 原子 Effect；
- 必要压力值＋独立 Activity intervals，不另存 thermal/transition 运行阶段；
- obligation、deadline、priority 和 coalescing；
- 双手 Lease＋整机 motion resource；
- alternative binding 和 state-bound frame token；
- canonical replay 与独立 Validator。

### 11.2 最优先的通用缺口

1. **完整业务 occurrence**：相同绑定的重复 Activity 已可用；继续补 RouteVisit/trigger-scoped Obligation 身份及 G11—G13，不再增加业务专用 Kernel 分支；
2. **typed Entity/Relation/Binding Domain**：把 reachability、process compatibility、LL side、handoff 和 directionality 从名字特判变成数据；
3. **Lease/holder 查询与跨资源不变量**：当前已支持提交前 `(resource, owner)` Lease 存在/不存在查询；继续明确活动期间持有及跨物理 holder 的合法交接/唯一性，不把 admission guard 当作持续不变量；
4. **Wait/Deadlock/Terminal**：定义未来事件推进、完整终态和不可行诊断；
5. **审计验收扩展**：CommitLog 与已选操作的精确对应已完成；后续新表达式、G11—G13 和候选索引均须加入独立审计/差分验收；
6. **动态时间义务**：真正支持 idle threshold、repeated obligation 和 next-Place deadline；
7. **Exchange golden case**：验证“提前持有 incoming wafer→Pick-out→Place-in”的一次性完整承诺。

### 11.3 需要业务确认、不能由代码猜测的事项

1. source `priority` 是硬 admission 顺序，还是只影响优化优先级；
2. 同 recipe 的 `wafer_index FIFO` 是硬约束，还是训练/搜索启发式；
3. 是否允许 preventive cleaning；
4. Swap/Exchange 是否必须作为原子 Intent，还是也允许两个独立 Intent；
5. LL 是否存在多 slot、slot directionality、batch Pump/Vent 和部分 slot 可访问等设备形态；
6. residency 的局部、PM/LL 类型默认与全局默认应采用何种覆盖顺序；
7. terminal 是否允许 wafer 暂留某些 Buffer/LL，还是必须全部返回原 IO/LP。

## 12. 下一轮扩充方法

下一轮不按“想到一个约束就加一行”的方式进行，而应为每个真实来源建立可追踪记录：

```text
真实设备/recipe/失败轨迹
    → 最小场景
    → 本矩阵中的 constraint ID
    → IR 表达或 gap
    → golden case
    → 数据生成覆盖
```

建议下一批优先收集：

1. 3—5 个真实双臂 exchange 时序；
2. 真实 cleaning priority 与重复触发案例；
3. PM/LL residency 和 completion-to-next-load 的具体边界；
4. 多 slot LL 的 pressure、cooling、方向和 batch 行为；
5. 曾经导致死锁、错误 mask 或 validator 漏检的轨迹。

只有当一个缺口至少服务两类真实约束，才优先增加新的 Kernel/AST 原语；否则应优先通过 Compiler lowering、关系数据或既有原语组合解决。
