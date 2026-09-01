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
6. **Pick/Place 采用 Multi-boundary Operator（多边界算子）**：策略提交一个完整运输 Intent，语义层一次性检查并承诺其声明的移动、资源预留和操作边界；模型不逐个控制内部边界。

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

## 3. Load Lock 的正交状态

LL（Load Lock，装载锁）的压力变化与 wafer（晶圆）的冷却/热稳定过程必须分开建模。一个建议的最小状态分解是：

```text
LoadLock:
  pressure_level      ∈ {atmosphere, vacuum}
  pressure_transition ∈ {idle, pumping, venting}
  slot_occupancy      : slot -> optional wafer_id

WaferInLoadLock:
  thermal_phase       ∈ {not_required, cooling, ready}
  cooling_end_tick    : optional Tick
```

合法并行示例：

```text
t0: wafer Place.end 到达 LL，触发 cooling=[t0, t3)
t1: Pump.start，pressure_transition=pumping
t3: cooling.end，thermal_phase=ready；Pump 仍可继续
t5: Pump.end，pressure_level=vacuum，pressure_transition=idle
t5: 若其他 Guard 和资源条件满足，wafer 可从 vacuum side Pick
```

因此不能定义 `pressure_state ∈ {atmosphere, vacuum, transitioning}` 后再把“正在冷却”塞进同一枚举，也不应产生 `vacuum_and_cooling`、`pumping_and_ready` 这类笛卡尔积状态。并发来自不同状态轴和不同资源区间的同时推进，而不是来自越来越多的特殊状态名。

LL 取片边界至少同时检查：

```text
pressure_transition == idle
pressure_level == required_interface_side
wafer.thermal_phase in {not_required, ready}
slot contains the requested wafer
robot and interface claims are available
```

如果具体设备允许在 Pump/Vent 期间继续冷却，二者可以重叠；如果某设备禁止重叠，应通过共享资源 Claim 或 Invariant 表达，而不是改动模型动作空间。

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
3. 触发自动 Process、cooling、obligation 等确定性后果；
4. 检查该 Tick 到期的义务和不变量；
5. 启动已经承诺且在该 Tick 生效的边界；
6. 重复处理新产生的同 Tick 后果，直到状态不再变化；
7. 仅在稳定态上计算 enabled/committable Intent。

这个顺序必须形成 conformance tests（语义一致性测试），不能只依赖实现中的回调先后顺序。

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
```

Intent 不等于一个瞬时动作，也不等于一个 Event。它是对完整 Operator bundle（算子边界包）的承诺请求。

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

下一份可执行规格应是 `Constraint IR v1`：定义静态对象、状态变量、Operator Template、表达式、资源、Trigger、Obligation 和 Objective 的最小 schema；同时给出 5—10 个小型 golden cases，作为 Kernel 与独立 Validator 的共同语义基准。模型结构和训练数据 schema 应在这些 golden cases 通过后再定稿。
