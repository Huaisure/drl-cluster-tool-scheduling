# Cluster Tool Scheduling 语义基础 v1

状态：六项基础决策已确认  
范围：Constraint IR、Semantic Kernel、Schedule、Validator 和模型通用接口共享的语义合同  
暂不包含：神经网络结构、奖励权重、具体求解算法和现有代码迁移方案

## 1. 已确认的六项决策

1. **内部时间使用整数 Tick（离散时刻）**：输入和业务展示仍可使用秒；Compiler 负责按照实例声明的精度转换，无法精确表示时必须报错，不能静默取整。
2. **区间统一使用 `[start, end)`（左闭右开）**：开始时刻占用、结束时刻释放；两个区间可以在同一 Tick 无缝衔接。Deadline（截止时刻）采用包含边界的判定：`satisfy_time <= deadline`。
3. **Pump/Vent 在内部 Schedule 中显式存在**：模型可以选择较高层的 transport Intent（运输意图），但 Kernel 必须展开出可审计的 Pump/Vent 事件和区间。
4. **Clean 在内部 Schedule 中显式存在**：清洗不是隐藏的时间修正或 reward penalty，而是具有资源占用、前置条件、效果和可验证边界的真实事件。
5. **Process 是 Kernel 自动创建的显式区间**：它出现在状态和 Schedule 中，但不是策略直接选择的动作。满足 Place 等触发条件后，由 Kernel 按 IR 自动创建。
6. **Pick/Place 采用 Multi-boundary Operator（多边界算子）**：策略提交一个完整的 Operator boundary bundle，语义层一次性检查并承诺其声明的移动、资源预留和操作边界；模型不逐个控制内部边界。bundle 可以是 Direct transfer、Prefetch、Place continuation 或 Exchange；“完整”指全部声明边界被原子承诺，不要求每片 wafer 都在该 Intent 结束时到达 Module。

这六项决策共同定义了一个关键边界：模型决定“承诺哪个通用意图”，Kernel 决定“该意图依法产生哪些物理事件”。

## 2. 统一状态机的准确含义

这里的“统一状态机”正式命名为 **Unified Discrete-Event State Transition System（统一离散事件状态转移系统）**。它不是把所有情况压成一个巨大的枚举状态，而是把若干正交状态轴组成一个产品状态：

```text
S = EquipmentState
  × WaferState
  × ResourceState
  × ReservationState
  × ObligationState
  × Time
```

所有约束都通过同一套语义原语进入系统：

- `Guard`：某个边界发生时必须成立的条件；
- `ResourceClaim`：某个时间区间需要占用的容量或独占资源；
- `Effect`：事件边界对状态产生的确定性修改；
- `Trigger`：条件达成时由 Kernel 自动产生事件或义务；
- `Obligation`：已经产生、未来必须满足的要求，可带 Deadline；
- `Invariant`：所有可达状态或整个区间内始终必须成立的条件。

新增约束应优先编译成这些原语的组合，而不是增加模型特有的动作类型或神经网络分支。

## 3. 最少事实与通用活动（收敛后的合同）

持久动态事实只分为三类，另外保留时间与已承诺的事件/区间：

| 基础 | 含义 | 例子 |
|---|---|---|
| StateCell（状态值） | 无法从其他记录推导、且后续约束确实需要的事实 | 压力值、工艺进度、累计使用次数 |
| Lease（持有关系） | 谁持续占有哪个资源、多少容量 | 晶圆在槽位上、手臂持有晶圆 |
| Obligation（待办要求） | 已产生但尚未满足的要求，可有截止时间 | 后续必须取走晶圆、必须完成某项维护 |

“正在运行”从 `[start,end)` 区间推导；“已经结束”从对应实例的结束边界推导，不能仅凭当前没有活动就判定已完成，因为它也可能尚未开始。Reservation 从已承诺的资源使用推导，不再建立独立的业务占用标志。持片位置以 Lease 为事实来源，不再同时维护一份可写 `wafer.location`。

基本修改归为三组，沿用已有显式实现，不新增同义包装类：

- **更新状态**：设置值、自增、记录当前 Tick；保留三者的写入规则与冲突检查。
- **改变持有关系**：获取或释放 Lease；转移由同一声明 bundle 内的释放与获取表达。
- **改变待办**：创建或满足 Obligation。

Activity（活动）由条件、资源区间、边界、依赖和上述修改组成。加工、冷却、清洗、取放、Pump/Vent 都通过同一套结构表达；业务名字仅用于配置和审计，不构成新的执行原语。

### 3.1 LL 示例

LL（Load Lock，装载锁）只需保留压力值，以及晶圆所在槽位的 Lease。冷却与 Pump 是两个独立活动：

```text
t0: Place.end，触发活动 A（此例业务名称为冷却），区间 [t0,t3)
t1: Pump.start，压力转换区间 [t1,t5)
t3: 活动 A.end；Pump 仍然运行
t5: Pump.end，将 pressure_level 写为 vacuum
```

不额外维护 `thermal_phase`、`cooling_end_tick` 或 `pressure_transition` 状态；它们在此例中均可从 Schedule 推导。若真实约束依赖具体温度，才增加温度这一普通 StateCell，而不是预置一套冷却专用状态机。

LL 取片需要组合以下要求：本次访问所需活动已经结束、压力转换结束、压力侧匹配、槽位由目标晶圆持有、机械手与接口资源可用。前后活动的顺序使用边界依赖；跨 Intent 的重复访问识别和完整候选门控仍待实现，不能宣称 G06 已覆盖完整取片流程。

设备允许冷却与 Pump/Vent 重叠时分配独立资源；禁止重叠时声明共享互斥资源。二者是否并行不改变活动类型。

### 3.2 没有截止时间不等于没有要求

`deadline_tick=None`（模板中 `deadline_offset=None`）表示待办仍须满足，但没有超时边界：不会创建人为的大时间值，不参加 Deadline 检查。它可以在部分 Schedule 中保持未完成；要求终态时仍必须清偿。若业务本来没有未来要求，则不创建 Obligation。

同 Tick、同合并键仍先选最高优先级；最高优先级对应同一个待办时，取其中最早的有限截止时间，全部为 `None` 才保持 `None`。不同待办的最高优先级冲突仍报错。

### 3.3 当前实现边界

Reference slice 已落实 G06 的派生运行状态、可选 Deadline、同绑定重复实例、逐事件推进，以及已声明动态提交的独立审计，旧 Engine 和旧模型输入保持不变。审计逐次检查选择合法性、精确展开与最终状态，已接入恢复；它不保证候选集完整或排程必能完成。三组基本修改是现有 Effect 的语义归类。后续已独立实现[通用 IR 训练首版](./ir-training.md)，包括完整程序/状态/目标的匿名图和 PPO 接入；完整 RouteVisit/Obligation 发生身份、关系条件扩展及更大规模训练验证仍需后续完成。

现已提供[现有问题到 IR 的首版 Compiler](./problem-to-ir.md)：单机械手、IO/LP＋单槽 PM、普通有限路线和重复访问、候选站点、自动加工与返回。每片仅增加一个进度值，位置仍以 Lease 为准；声明式终态检查进度和真实返回占位。LL、清洗、JIT、运行中初态等输入暂时明确拒绝，不把“通用原语可表达”误称为“全部业务已接通”。参考协议因终态声明和绑定规范化修正升级到 `1.2-reference`。

候选提交前现可组合 StateCondition 与 LeaseCondition：直接读取指定资源是否持有指定对象，不增加 `wafer.location`。Lease 的“存在/不存在”不等于“唯一持有/资源空闲”，也不保证整个活动期间持续成立；持续持有和跨资源物理位置不变量仍待完善。

## 4. 时间与同 Tick 原子语义

内部时间为非负整数 Tick。持续时间为正整数 Tick；零时长只允许用于明确声明为瞬时的边界事件。

区间 `[s, e)` 的含义：

- `s` 时资源已经被占用；
- `e` 时资源已经释放；
- `[a, b)` 与 `[b, c)` 不冲突；
- 持续时间恒为 `e - s`。

同一 Tick 上可能同时出现多个结束、触发、开始和 Deadline 检查。Kernel 不在处理其中一个边界后立即向模型暴露中间态，而是反复应用确定性后果直至达到 fixed point（不动点/稳定态），再产生 Decision Epoch（决策时刻）。建议的逻辑阶段为：

1. 结束已到期的区间并释放资源；
2. 应用对应的结束 Effect；
3. 启动已经预先承诺且在该 Tick 生效的边界，并应用开始 Effect；
4. 触发自动 Process、cooling、obligation 等确定性后果；
5. 重复处理新产生的同 Tick 后果，直到状态不再变化；
6. 在该 Tick 的全部 Event 处理完后，检查到期义务和不变量；
7. 仅在合法稳定态上计算 enabled/committable Intent。

因此，恰好在 Deadline Tick 发生的满足事件是合法的，但它必须已经在之前的 commit 中得到承诺和预留；Kernel 不能先判定 Deadline 失败，再补执行一个同 Tick 的开始事件。

这个顺序必须形成 conformance tests（语义一致性测试），不能只依赖实现中的回调先后顺序。

### 4.1 当前时刻提交与下一事件推进

`commit()` 保留整个未来 bundle 的承诺，但只执行当前 Tick 的效果；它不再把 Session 直接跳到最晚完成时刻。`advance_next()` 推进到最近的未来 Event、区间边界或有限 Deadline，之后由 `frame()` 重新提供候选。没有未来边界返回 `None`，不是“调度已完成”的证明。`advance_to(tick)` 仅保留作显式回放/调试，不作为模型动作。

已预先承诺、在当前 Tick 到达的事件属于 `decision_round=0`。在其稳定态上作出的新提交，其当前 Tick 效果进入更高轮次；同一轮内部仍与事件列表排列无关。每轮应用状态/Lease/Obligation 后检查容量和 Deadline，避免后续一轮掩盖先前违规。轮次不增加物理时长，也不允许拆开已承诺 bundle。

预检查未来资源与效果时，尚未安排满足操作的待办可保留，因为中间决策可能完成它；当前 Tick 和真实时间推进不放松截止检查，已明确排到截止时间之后的满足操作也始终非法。例如短活动 5 秒完成、待办 8 秒截止、后台活动 20 秒完成时，可以在第 5 秒提交一个第 7 秒完成的满足操作。

## 5. Operator、Intent 与 Event

### 5.1 Operator Template

Operator Template（算子模板）是领域规则的声明式描述，至少包含：

```text
boundaries
duration expression
guards per boundary
resource claims / leases / reservations
effects per boundary
triggers
obligations
```

### 5.2 Intent

Intent（意图）是策略在 Decision Epoch 提交的参数化选择，例如：

```text
TransportIntent(
  wafer,
  source,
  destination,
  robot,
  hand,
  route_candidate,
)

ExchangeIntent(
  outgoing_wafer,
  incoming_wafer,
  module,
  robot,
  outgoing_hand,
  incoming_hand,
)
```

Intent 不等于一个瞬时动作，也不等于一个 Event。它是对完整 Operator bundle（算子边界包）的承诺请求。

完整 Operator bundle 不等于完整的单 wafer Module-to-Module 运输。Prefetch 可以结束于 hand，Place continuation 可以开始于 hand，Exchange 可以在同一个原子承诺内先 Pick-out 再 Place-in。具体生命周期和身份规则见 [动态 Intent 生命周期设计](./dynamic-intent-lifecycle.md)。

### 5.3 Event

Event（事件）是 Kernel 展开 Intent 或执行 Trigger 后产生的具体、带时刻边界，例如：

```text
MoveToSource.start
MoveToSource.end
Pick.start
Pick.end
MoveToDestination.start
MoveToDestination.end
Place.start
Place.end
Process.start        # automatic
Process.end          # automatic
Pump.start
Pump.end
Clean.start
Clean.end
```

Schedule 保存这些展开后的事件和对应区间，以便独立 Validator 完整复核。

## 6. 三层可行性词汇

为了避免把 action mask 误称为“保证可行”，v1 明确区分：

- **Enabled Intent（当前可启用）**：立即开始边界的 Guard 成立；
- **Committable Intent（当前可承诺）**：完整 bundle 所需的已知资源和时间窗口可以预留，且不会立即违反已知 obligation；
- **Completable State（可完成状态）**：存在至少一个合法后续能完成全部 wafer 和终态义务。

模型通常只从 Committable Intent 中选择。Completable 是更强的全局性质，需要搜索、专家或保守安全证明近似，不能由局部 action mask 自动保证。

## 7. v1 语义合同的验收项

第一阶段实现前，至少需要把以下内容固化为 schema 和 conformance tests：

1. Tick 单位声明、精确转换和非法精度输入；
2. `[start, end)` 的容量冲突与同 Tick 无缝衔接；
3. Deadline 边界等号成立；
4. 同 Tick 事件输入排列改变时，最终稳定态一致；
5. Pick/Place 完整 bundle 的 Guard、Claim、Reservation 和 Effect；
6. Process 自动创建且不出现在策略候选动作中；
7. Clean、Pump、Vent 在 Schedule 和 Validator 中可见；
8. LL cooling 与 pressure transition 可以重叠且分别结束；
9. 对不允许重叠的设备，仅通过 IR 资源/不变量配置即可禁止；
10. Snapshot 序列化再恢复后，enabled/committable 集合和后续状态一致。

## 8. 下一步

下一份可执行规格是 [Constraint IR v1](./constraint-ir-v1.md)：定义静态对象、状态变量、Operator Template、表达式、资源、Trigger、Obligation 和 Objective 的最小 schema；配套的 [golden cases](./golden-cases/README.md) 作为 Kernel 与独立 Validator 的共同语义基准。模型结构和训练数据 schema 应在这些 golden cases 通过后再定稿。
