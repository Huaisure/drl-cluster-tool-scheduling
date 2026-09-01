# Cluster Tool 通用调度学习框架设计

状态：设计提案（经审查修订）  
目标：从调度问题本身出发，建立可覆盖 Pick、Place、Clean、Pump、Vent、LL、驻留、双臂几何和未来组合约束的通用学习与搜索框架。  
非目标：在第一阶段复用或增量修改现有 HGT + Transformer + PPO 架构。

## 1. 结论

本问题不适合让一个端到端策略网络同时学习物理合法性、时间推进、长程规划和目标优化。推荐的总体方案是：

```text
声明式 Constraint IR
        +
精确离散事件 Kernel
        +
独立语义 Validator
        +
通用残余约束图模型
        +
短 Beam Search
        +
不确定性驱动的 CP-SAT / LNS fallback
        +
专家、反事实、DAgger 数据飞轮
```

硬约束由 Kernel 保证，模型只学习：

1. 当前可行候选的相对优先级；
2. 状态的剩余代价与完成概率；
3. 深层死锁、deadline miss 和预测不确定性。

推荐的第一条主线不是纯 PPO，而是：

```text
专家搜索生成 decision-frame 数据
    -> listwise imitation / regret learning
    -> greedy baseline
    -> 同模型接入 Beam Search
    -> DAgger 补充模型诱发状态
    -> LNS 与 solver-in-the-loop 优化
```

## 2. 审查结论与修订

### 2.1 保留的核心判断

- 物理与约束合法性必须由确定性 Kernel 负责，不能依赖神经网络学会。
- 模型输入应是“尚未解决的约束与候选操作”，而不只是设备拓扑。
- 数据质量、反事实覆盖和模型诱发状态覆盖，比更换网络层类型更重要。
- 直接策略适合低延迟，学习型搜索适合主要质量路线，LNS 适合离线改进。
- Solver 应主要作为专家、fallback 和证明工具，而不是唯一线上路径。

### 2.2 原方案中过强的承诺

“Constraint IR 可以表达任意未来约束”是不成立的。任何模型若看不到影响未来可行性的状态或规则，都不可能推断其影响。

修订后的承诺是：

- IR 提供一组版本化的约束原语；
- LL、清洗、驻留、双臂等规则编译为这些原语的组合；
- 新规则若可由已有原语表达，只新增 Problem 配置或编译规则，模型结构不变；
- 新规则若不能由已有原语表达，必须扩展 IR Compiler、Kernel 和 Validator；模型接口仍保持不变，但需要针对新 IR schema 做兼容性验证和数据补充。

### 2.3 原方案缺失的语义

原方案只描述了单个 Operator 的区间资源占用，没有充分表达跨多个 Operator 持续存在的占用。例如：

- hand 从 `Pick.start` 占用到后续同一 wafer 的 `Place.end`；
- chamber 从 `Place.start` 占用到后续 `Pick.end`；
- 某些 valve、slot 或方向许可也可能跨事件存在。

修订方案增加具名 `Lease`：Operator 可以在 start/end 创建、转移或释放带 holder identity 的资源 lease。容量校验同时考虑活动 interval claim、持久 lease 和未来 reservation。

### 2.4 同一时间戳必须原子处理

仅规定 `[start, end)` 仍不足以消除实现歧义。Kernel 在时间 `t` 必须以固定顺序处理：

1. 收集并校验所有 `end == t` 的事件；
2. 原子应用 end effects 和资源释放；
3. 激活由 end boundary 触发的新 obligation；
4. 在 post-end 状态上联合校验所有 `start == t` 的事件；
5. 原子应用 start effects、resource claims 和 reservations；
6. 在允许 `t` 时刻满足的动作处理完成后，再判定 deadline violation；
7. 生成新的 decision frontier。

这样允许 `Pick.end == Clean.start`，也避免同一时间戳的动作因遍历顺序不同而产生不同结果。

### 2.5 学习预测不能承担证明责任

- learned value、deadlock risk 和 predicted bound 只能用于排序、扩展预算和 fallback 决策；
- 不能用 learned bound 做不可逆硬剪枝；
- 硬剪枝只能使用可证明 lower bound、dominance、状态等价和 Constraint Kernel 的冲突证明；
- 最终 Schedule 必须由独立 Validator 验收。

### 2.6 数据标签必须携带置信度

求解器 timeout 时得到的 incumbent 不是最优标签。每条 expert label 必须保存：

- expert 类型和版本；
- 求解预算；
- incumbent upper bound；
- proven lower bound；
- optimality gap；
- 是否最优、可行、超时或未知；
- 反事实分支的展开深度。

未知相对优劣不能被错误地当成负样本。

## 3. 原问题的计算性质

Cluster Tool 调度不是普通的 job-to-machine assignment，而是带状态、身份、持续时间和触发规则的离散事件规划问题。

### 3.1 主要困难

- 可变规模：wafer、route、module、robot、hand 和候选动作数量均变化；
- 长时域：一个局部选择可能在几十个事件后形成死锁；
- 硬约束：任何一次容量、身份、时序或设备状态违规都会使整个解失效；
- 强组合性：LL、清洗、驻留、候选 PM、双臂和 valve 约束会相互作用；
- 多解性：同一状态常有多个近似等价的好动作；
- 对称性：同构 module、robot、wafer 重命名不应改变策略；
- 数据分布偏移：专家轨迹状态与模型自己运行时访问的状态不同；
- 质量—时间权衡：线上可能要求毫秒响应，离线则允许秒到分钟搜索。

### 3.2 正确的决策过程

应把环境建模为事件驱动 Semi-MDP：

- Kernel 自动执行没有选择意义的强制事件和时间推进；
- 只有出现资源竞争、alternative binding、可选状态转换或 obligation 权衡时才建立 decision epoch；
- 一个模型决策可以提交单个 Intent，也可以提交同一 epoch 的兼容 Intent 集合；
- 所有未来影响都必须存在于 Markov snapshot 中，不依赖模型不可见的 Kernel 私有状态。

## 4. 总体架构

```text
Problem / Equipment / Rules
            |
            v
    Constraint Compiler
            |
            v
       Constraint IR  ------------------------------+
            |                                       |
            v                                       v
      Event Kernel                         Independent Validator
            |                                       ^
            v                                       |
      DecisionFrame                                 |
   (residual graph + intents)                       |
            |                                       |
            v                                       |
    Constraint Graph Model                          |
   policy/value/risk/uncertainty                     |
            |                                       |
      +-----+-----------+-------------+              |
      |                 |             |              |
      v                 v             v              |
   Greedy           Beam Search   Neural LNS         |
      |                 |             |              |
      +-----------------+-------------+              |
                        |                            |
                        v                            |
                    Schedule -----------------------+

Expert Adapters (CP-SAT / exact / heuristic / LNS)
                        |
                        v
              DecisionFrame Dataset
                        |
                        v
              Imitation -> DAgger -> RL
```

## 5. Deep Modules 与 Interface

### 5.1 ConstraintCompiler Module

Interface：

```python
compile(problem: ClusterProblem) -> ConstraintIR
```

它隐藏：

- 设备语义到 IR 原语的映射；
- route 展开；
- candidate module alternative；
- LL 状态机；
- 清洗 trigger 和 priority；
- residency obligation；
- robot/hand/geometry 兼容关系；
- schema 版本与静态一致性检查。

错误必须是结构化的 `InvalidProblem` 或 `UnsupportedConstraint`，不得静默降级。

### 5.2 EventKernel Module

Interface：

```python
state = kernel.reset(ir)
frame = kernel.next_decision(state)
state = kernel.commit(state, intent_set)
```

它隐藏：

- 事件边界排序；
- resource claim、lease 与 reservation；
- guard/effect 执行；
- obligation 激活、满足和过期；
- 增量合法候选生成；
- 强制事件闭包与时间推进；
- persistent/copy-on-write search state；
- canonical state hash 和 dominance key。

核心 invariant：每次公开返回的 `KernelState` 都处于稳定事件边界，不存在尚未应用的同时间戳 effect。

### 5.3 SemanticValidator Module

Interface：

```python
validate(problem: ClusterProblem, schedule: Schedule) -> ValidationReport
```

Validator 不读取 Kernel 内部变量，也不复用 Kernel 的动作合法性判断。它从原始 Problem 和 Schedule 独立 replay：

- wafer identity；
- resource occupancy；
- `[start, end)` 边界；
- route 顺序；
- robot/hand；
- LL 状态转换；
- cleaning trigger；
- residency 与其他 deadline。

这能降低 Compiler 或 Kernel 与 Validator 共享同一错误的风险。对关键语义还应加入第二 solver adapter 或小规模穷举做差分测试。

### 5.4 PolicyRanker Module

Interface：

```python
rank(frame: DecisionFrame) -> PolicyOutput
```

`PolicyOutput` 包含：

- 每个 Intent 的 score 或 probability；
- state cost-to-go 分位数；
- completion probability；
- deadlock/deadline risk；
- calibrated uncertainty；
- 可选的 pairwise compatibility refinement。

策略不能创建、修改或伪造 Intent，只能对 Kernel 生成的候选排序。

### 5.5 SearchScheduler Module

Interface：

```python
solve(ir: ConstraintIR, policy: PolicyRanker, limits: SearchLimits) -> SolveResult
```

返回：

```text
Solved(schedule, objective, bound, proof_status)
Timeout(incumbent, bound, gap)
Infeasible(core)
```

不同 adapter 可以实现 greedy、Beam、limited discrepancy、LNS 或 CP-SAT，但调用者使用相同 interface。

## 6. Constraint IR

### 6.1 基础类型

```python
ConstraintIR(
    entities,
    state_variables,
    resources,
    operator_templates,
    constraints,
    initial_state,
    objectives,
    schema_version,
)
```

#### Entity

具有稳定 identity 的对象，例如 wafer、module、robot、hand、slot、route step。

#### StateVariable

```python
StateVariable(
    owner,
    domain,
    initial_value,
)
```

可表示有限状态、计数器、timer anchor、位置、方向和 process type。

#### Resource

```python
Resource(
    capacity,
    compatible_holders,
    calendar,
)
```

Resource 不只保存 occupancy count，还保存 holder identity。

#### Lease

```python
Lease(
    resource,
    holder,
    amount,
    acquired_at,
    release_condition,
)
```

Lease 用于跨 Operator 持续占用，例如 chamber occupancy 和 robot hand holding。

#### OperatorTemplate

```python
OperatorTemplate(
    parameters,
    guards,
    duration,
    interval_claims,
    start_effects,
    end_effects,
)
```

Effect 可以修改 StateVariable、创建/释放 Lease、创建 obligation 或更新 objective accumulator。

#### Obligation

```python
Obligation(
    trigger,
    satisfaction,
    earliest_time,
    deadline,
    priority,
    violation_kind,
)
```

Obligation 必须保持受限、可检查，不能演化成任意脚本执行器。表达式由版本化、纯函数、无副作用的 AST 组成。

### 6.2 初始原语集合

第一版建议只支持：

- equality / inequality guard；
- finite-state transition；
- precedence 与 min/max lag；
- interval no-overlap；
- cumulative capacity；
- alternative binding；
- compatibility table；
- resource lease；
- trigger + obligation + deadline；
- counter/timer update；
- sequence-dependent duration；
- linear/lexicographic objective。

IR schema 必须版本化。新增原语时同时增加：

- Compiler 支持；
- Kernel semantics；
- Validator replay；
- feature encoder；
- conformance tests；
- checkpoint 和 dataset compatibility 规则。

## 7. 复杂约束如何编译

### 7.1 Pick / Place

Pick：

- `Pick.start` 检查 source occupant identity、process readiness、reachability、LL side 和 empty hand；
- 创建 hand lease；
- 占用 robot 与 source interface interval；
- `Pick.end` 释放 source chamber lease，更新 wafer location。

Place：

- `Place.start` 检查 hand holder identity、route target、reachability、状态和容量；
- 创建 target chamber lease；
- 占用 robot 与 target interface interval；
- `Place.end` 释放 hand lease，更新 wafer location，并触发 process/LL readiness。

### 7.2 LL

Conversion LL 编译为：

- slot capacity resource；
- atmosphere/vacuum/transitioning 状态变量；
- Pump 和 Vent Operator；
- Pick/Place 的 side guard；
- transition interval 对 LL/interface 的独占 claim；
- wafer occupancy lease；
- 根据 route candidate side 产生的 obligation。

IR 和 Schedule 中保留显式 Pump/Vent 事件。策略侧可以使用 transport macro Intent，但 Kernel 展开后仍生成完整事件，避免 Validator 看不到状态转换。

Vacuum transfer LL 不包含 atmosphere/vacuum transition，只保留普通容量和可选 hold/cool obligation。

### 7.3 清洗

每个 PM 保存：

- previous process type；
- completed wafer counter；
- last empty/clean time anchor；
- active cleaning obligations。

Idle、process-switch 和 wafer-count trigger 都产生 Clean obligation。Clean Operator：

- 要求 PM empty；
- 在 `[Clean.start, Clean.end)` 独占 PM；
- 重置对应 counter/timer；
- 更新 process type；
- 按问题定义处理多个 trigger 的优先级。

若 cleaning priority 未定义且多个 trigger 同时出现，Compiler 必须报 `UnderSpecifiedProblem`，不能自行猜测。

### 7.4 Residency

加工完成时创建 deadline obligation：

```text
Pick.start <= process_completion + max_residency
```

若约束是进入下一节点，则为：

```text
next Place.end <= process_completion + max_transfer_time
```

DecisionFrame 必须包含 absolute deadline anchor 和相对当前时间的 slack，不能只包含静态 residency limit。

### 7.5 双臂几何

- 每只 hand 是 capacity-1 resource；
- robot operation 是共享互斥 resource；
- orientation 是 StateVariable；
- module-hand-orientation 组合由 compatibility table 表达；
- Pick/Place effect 更新 orientation 和 holder lease。

如果设备允许真正的并行双臂操作，需要显式降低共享 robot claim，而不能仅把 hand capacity 设置为 2。

## 8. DecisionFrame 与候选 Intent

### 8.1 Markov snapshot

DecisionFrame 必须完整包含：

- 当前相对时间基准；
- 所有 entity 当前状态；
- 活动 interval、lease 和 reservation；
- 未完成 route/operator；
- 活动 obligation 和 deadline；
- 资源日历与 next-free time；
- objective accumulators；
- 当前合法 Intent；
- 每个 Intent 的 read/write/resource/obligation footprint。

模型不应依赖跨 episode 的隐藏 memory 来弥补状态缺失。若 snapshot 完整，memory 只用于压缩计算，不应改变决策语义。

### 8.2 Intent

```python
Intent(
    id,
    operator_template,
    bindings,
    earliest_start,
    latest_start,
    duration,
    reads,
    writes,
    claims,
    lease_effects,
    affected_obligations,
)
```

Intent ID 只在当前 frame 有效。Kernel commit 时必须重新验证版本和合法性，防止 stale frame。

### 8.3 单动作与批动作

建议同时实验两种方式：

1. Sequential select + STOP：策略逐个选择零时间提交的 Intent，直到选择 STOP；
2. Score + compatible subset：策略独立打分，由小型 exact selector 选择最大权兼容集合。

第二种更快且顺序无关，但独立分数可能忽略动作互补性；第一种表达力更强，但 horizon 和顺序噪声更大。首版推荐 compatible subset，并加入轻量 pairwise refinement。

Selector 只解决同一 decision epoch 的兼容集合，不负责未来排程，避免它演变成第二个大型求解器。

### 8.4 候选爆炸控制

- 增量维护受 changed resources/state variables 影响的 Intent；
- 使用合法的静态 dominance 消除等价候选；
- 对同构实体做 canonical ordering；
- 模型按 candidate-centric 子图编码，避免重复全图计算；
- 对极大 action set 使用分块打分和全局 top-k merge；
- 不得使用未经证明的启发式预过滤删除唯一可完成候选。

## 9. 通用模型设计

### 9.1 残余约束图

模型输入不是物理设备图，而是当前尚未解决的 residual constraint graph。

节点族：

- `entity`；
- `operator`；
- `resource`；
- `state_variable`；
- `constraint/obligation`；
- `intent`；
- `global/objective`。

关系角色：

- binds；
- reads / writes；
- claims / leases；
- precedes；
- conflicts；
- triggers；
- satisfies；
- alternative-of；
- contributes-to-objective。

关系使用 role embedding 和数值 edge features，不为 Pick、Clean、LL 建独立网络分支。

### 9.2 特征

时间特征统一相对 decision epoch 表达：

- earliest start；
- latest start；
- duration；
- remaining time；
- deadline slack；
- resource next-free time；
- critical-path lower bound；
- normalized objective lower/upper gap。

同时保留必要的 absolute anchor 差值关系，避免多个 deadline 之间的相对顺序丢失。时间可使用 `log1p`、分位数尺度或训练集统计归一化；schema 和 scaler 必须随 checkpoint 保存。

### 9.3 第一版网络

建议从中等规模的 edge-conditioned factor graph network 开始：

- hidden dimension：128 或 256；
- 4–8 次共享权重 message-passing iteration；
- typed/role-conditioned message MLP；
- gated node update；
- resource/constraint 局部 attention；
- candidate subgraph pooling；
- global objective token；
- permutation-equivariant aggregation。

共享权重迭代比堆叠大量不同层更接近约束传播计算，也更利于跨规模测试。Transformer 可以作为后续 ablation，而不是默认复杂化。

### 9.4 输出 heads

- listwise Intent policy；
- quantile cost-to-go；
- completion probability；
- short-horizon deadlock/deadline risk；
- lower-bound residual；
- uncertainty；
- 可选辅助 constraint-propagation head。

Risk head 不代替 Kernel legality。Value 与 risk 的目标应按 expert budget 和 label confidence 加权。

## 10. 四种推理方案

### 10.1 A：Direct Ranker

模型直接排序 Intent，Kernel 提交最高分兼容集合。

优点：

- 毫秒级；
- interface 小；
- 容易批量推理；
- 适合固定设备上的在线调度。

缺点：

- 只能保证局部合法；
- 不能保证最终完成；
- 深层死锁高度依赖数据覆盖。

定位：必要 baseline 和低延迟 adapter，不建议作为唯一最终方案。

### 10.2 B：Model-guided Beam Search

模型输出 policy prior、value、risk 和 uncertainty，Kernel 负责展开和状态推进。

搜索排序示意：

```text
priority = committed_cost
         + predicted_remaining_cost
         + risk_penalty
         - exploration_bonus
```

必须：

- 用 canonical state hash 合并等价状态；
- 用 dominance 保留更早/更低成本版本；
- learned value 只排序；
- 用 proven lower bound 做安全剪枝；
- uncertainty 高时扩大 beam 或 fallback；
- 报告延迟、incumbent 和 gap。

定位：推荐的主方案。首轮实验使用 Beam 8/16/32/64，构建延迟—质量 Pareto 曲线。

### 10.3 C：Neural LNS

从任意可行 Schedule 开始，模型选择 destroy region 和 repair priority：

- 关键 resource 时间窗；
- 造成等待的 wafer chain；
- cleaning/LL transition 附近；
- deadline slack 最小的 obligation；
- wait-for cycle 周边。

Kernel/CP adapter 修复并验收。LNS 适合离线高质量排程和专家数据生成，不是第一阶段线上路径。

### 10.4 D：Solver-native Heuristic

把 Constraint IR 编译为 CP-SAT/其他 exact solver，模型预测：

- branching variable；
- candidate ordering；
- restart；
- LNS neighborhood；
- incumbent improvement likelihood。

定位：

- 小中规模专家；
- 困难实例 fallback；
- lower bound/gap 提供者；
- 数据反事实展开器。

### 10.5 暂不推荐：纯 trajectory model / learned world model

- 事件动力学是已知且可精确执行的，没有必要用数据重新学习；
- world model 错误会在长时域累积并隐藏硬约束违规；
- trajectory model 很难保证组合约束和分布外合法性；
- 若后期尝试，应只用于 LNS/search controller 或候选重排序，所有 transition 仍由 Kernel 执行。

## 11. 数据设计

### 11.1 基本训练单元

数据集不只保存 trajectory，而保存完整 DecisionFrame：

```python
DecisionExample(
    problem_ir_hash,
    state_snapshot,
    residual_graph,
    candidates,
    expert_rankings,
    selected_intent_set,
    candidate_outcomes,
    value_bounds,
    proof_status,
    expert_metadata,
)
```

每个 candidate outcome 尽量包含：

- immediate cost；
- best known final objective；
- proven lower bound；
- regret interval；
- completion/timeout/deadlock；
- deadline miss；
- rollout/search budget；
- critical violated或tight obligation。

### 11.2 不能只存单条最优动作

同一状态可能有多个近似等价动作。推荐标签：

- top-k ranking；
- soft target `softmax(-regret / temperature)`；
- pairwise dominance；
- regret interval；
- compatible Intent set。

若两个候选的置信区间重叠，不应强制构造确定顺序。

### 11.3 Expert portfolio

数据来源：

- exact/CP-SAT；
- Beam；
- LNS；
- 安全启发式；
- 现有策略；
- 随机化次优策略；
- 人工或真实运行日志。

每条轨迹记录 expert、版本、随机种子、预算、gap 和 Validator 结果。使用独立 Validator 审计所有可行标签。

### 11.4 反事实数据

对专家状态至少展开：

- expert candidate；
- 第二梯队 candidate；
- 启发式看似合理但错误的 candidate；
- 近 deadline candidate；
- 近死锁 candidate；
- 高不确定 candidate。

反事实不一定展开到 episode 结束，可以使用固定搜索预算得到上下界和删失标签。

### 11.5 困难样本

重点生成：

- capacity 恰好达到上限；
- residency slack 接近 0；
- cleaning trigger 即将或同时发生；
- LL 处于错误侧或 transition 中；
- 多 LL、多 robot 相互等待；
- 双臂满载 wait-for cycle；
- PM 当前空闲但即将被 obligation 占用；
- 多个局部合法候选中只有少数最终可完成；
- 同拓扑不同时间参数导致策略反转；
- 零时长事件和同时间戳边界；
- candidate module 的对称与非对称组合。

### 11.6 数据增强

- entity/resource 重命名；
- 同构 module permutation；
- wafer permutation（保留 priority/FIFO 语义）；
- 一致时间缩放；
- 目标权重变化；
- 对称候选合并；
- 小幅时间扰动产生 near-boundary 样本。

增强后必须重新通过 Compiler 和 Validator，不能只修改张量。

### 11.7 DAgger 数据飞轮

```text
当前模型 rollout
    -> 收集失败/高不确定/专家分歧状态
    -> Expert Adapter 从该状态重求解
    -> 生成候选 regret 与 bounds
    -> Validator 审计
    -> 去重与难度分层
    -> 合并训练
```

这一步负责覆盖模型真正访问的状态分布，避免只在专家轨迹上表现良好。

### 11.8 数据切分

禁止仅随机按 seed 切分。至少保留：

- unseen topology；
- unseen scale；
- unseen duration distribution；
- unseen route length；
- unseen constraint composition；
- single-constraint-seen / composition-unseen；
- adversarial near-boundary；
- real-device holdout。

需要做 canonical hash 和同构检测，避免由模板或同源 Problem 造成泄漏。

## 12. 训练目标与阶段

### 12.1 多任务目标

```text
L = L_listwise_policy
  + lambda_value * L_quantile_value
  + lambda_complete * L_completion
  + lambda_risk * L_deadlock_deadline
  + lambda_bound * L_bound_residual
  + lambda_aux * L_constraint_propagation
```

- policy 目标按 regret 和 label confidence 加权；
- value 预测分位数而不是单点，表达搜索标签不确定性；
- objective 建议预测相对 proven lower bound 的 residual/gap；
- auxiliary head 可预测一次 Kernel propagation 后的 slack/domain 变化；
- 不要通过非法动作惩罚训练 legality，非法动作根本不进入可提交候选。

### 12.2 训练阶段

阶段 0：Constraint IR、Kernel、Validator 和 expert adapters，无神经模型。  
阶段 1：candidate MLP 与简单 graph ranker 的监督 baseline。  
阶段 2：factor graph model + listwise regret imitation。  
阶段 3：保持模型不变，接入 Beam Search。  
阶段 4：DAgger / advantage-weighted imitation。  
阶段 5：Neural LNS 和 solver-in-the-loop 优化。  
阶段 6：数据充分后再评估 conservative offline RL 或有限 on-policy fine-tuning。

不建议在阶段 0–3 之前投入大规模 PPO。

## 13. 实验矩阵

| 实验 | 数据 | 模型 | 推理 | 要验证的问题 |
|---|---|---|---|---|
| E0 | expert chosen action | candidate MLP | greedy | 非图特征下限 |
| E1 | 同 E0 | factor graph | greedy | 图结构是否有价值 |
| E2 | 同 E1 | 同 E1 | Beam-16 | 长程搜索的价值 |
| E3 | regret + counterfactual | 同 E1 | Beam-16 | 标签质量的价值 |
| E4 | DAgger | 同 E1 | Beam-16 | 状态分布覆盖的价值 |
| E5 | 同 E4 | Graph Transformer | Beam-16 | 更复杂架构的净价值 |
| E6 | repair 数据 | LNS policy | LNS | 长预算质量上限 |
| E7 | search traces | branch model | CP-SAT | 专家速度和 gap |

预期 E2、E3、E4 的收益大于 E5；实验必须允许这一假设被证伪。

### 13.1 Beam ablation

- width：1、4、8、16、32、64；
- model value on/off；
- risk penalty on/off；
- uncertainty fallback on/off；
- proven bound pruning on/off；
- state dedup/dominance on/off。

### 13.2 数据 ablation

- 单一 expert vs portfolio；
- chosen action vs regret；
- 无反事实 vs top-k 反事实；
- 无 DAgger vs DAgger；
- random split vs composition holdout；
- 普通样本 vs near-boundary oversampling。

## 14. 评估指标

优先级从高到低：

1. Validator validity，目标必须是 100%；
2. completion/success rate；
3. deadlock 和 deadline miss rate；
4. makespan、cycle time、throughput、tardiness；
5. 相对 expert/lower bound 的 gap；
6. P50/P95/P99 推理延迟；
7. 搜索节点数和内存；
8. unseen topology/scale/constraint composition 泛化；
9. uncertainty calibration 与 fallback coverage；
10. 质量—时间 Pareto front。

若 validity 不是 100%，应停止模型质量比较，先修复 Compiler/Kernel/Validator。

## 15. 阶段性验收门槛

### Gate 0：语义内核

- Pick/Place/Clean/Pump/Vent 边界测试齐全；
- 同时间戳 permutation test 结果一致；
- 小规模状态与穷举/第二 adapter 差分一致；
- 每类复杂约束有组合测试；
- snapshot 序列化恢复后行为一致。

### Gate 1：数据可信

- 所有 feasible label 经独立 Validator；
- 数据有 canonical 去重；
- label 记录 budget/bound/gap；
- train/validation/test 无模板泄漏；
- near-deadlock、deadline 和组合约束有足够覆盖。

### Gate 2：直接策略

- Greedy validity 100%；
- 在 in-distribution 上显著超过手工 dispatching baseline；
- OOD completion 不出现不可接受崩溃；
- uncertainty 能识别多数失败状态。

### Gate 3：学习型搜索

- Beam 相对 Greedy 形成稳定质量提升；
- width 增加时质量总体单调改善；
- 搜索状态 dedup 和 persistent state 控制住内存；
- fallback 能覆盖高风险状态。

### Gate 4：数据飞轮

- DAgger 明显降低模型诱发状态上的失败率；
- composition holdout 改善；
- 新数据增量的收益可追踪，避免只增加重复样本。

## 16. 主要风险与缓解

### 16.1 IR 太弱

症状：每加入一种设备规则都需要新 primitive。  
缓解：先收集真实规则 corpus，检查是否可归约到 state transition、resource、temporal、obligation 和 compatibility；无法归约时再版本化扩展。

### 16.2 IR 太强

症状：变成任意脚本语言，Kernel 无法增量传播，模型也无法泛化。  
缓解：限制为纯函数 AST、有限 domain、显式 effect 和可检查 obligation，不允许用户代码回调进入 Kernel。

### 16.3 Compiler 与 Kernel 同错

缓解：Validator 从原始 Problem 独立 replay；小规模实例与第二 solver/穷举差分；关键事件边界使用 property-based tests。

### 16.4 候选数量爆炸

缓解：增量生成、静态 dominance、同构合并、candidate-centric caching、分块打分；不能未经证明删除候选。

### 16.5 搜索状态复制成本

缓解：persistent/COW state、事件增量日志、共享静态 IR、局部 propagation cache。

### 16.6 Expert 数据昂贵

缓解：小规模 exact、中规模 Beam/LNS、大规模 bounded expert；主动选择高不确定和高影响状态补标，而不是均匀穷举。

### 16.7 模型学到 expert 偏差

缓解：expert portfolio、反事实 regret、不同预算标签、DAgger、solver-in-the-loop 优化。

### 16.8 目标函数变化

缓解：Objective 作为 IR 和 global token 的显式输入；训练保留 objective vector，不只保存一个固定加权总分；评估 Pareto 而非单一 makespan。

## 17. 推荐实施顺序

### 第一步：语义基础

1. 定义 IR v1 和 conformance tests；
2. 实现 event boundary、resource lease、reservation、obligation；
3. 显式建模 Pick/Place/Clean/Pump/Vent；
4. 建立独立 Validator 和差分测试；
5. 建立 canonical snapshot 和 persistent search state。

### 第二步：专家与数据

1. 实现安全启发式和 Beam Expert；
2. 接入 CP-SAT adapter；
3. 定义 DecisionExample schema；
4. 生成 chosen/top-k/counterfactual 数据；
5. 建立 topology/constraint composition holdout。

### 第三步：模型 baseline

1. candidate MLP；
2. factor graph ranker；
3. listwise regret + quantile value + risk；
4. greedy 评估与 uncertainty calibration。

### 第四步：主推理方案

1. Beam-8/16/32/64；
2. canonical dedup 和 proven bound；
3. risk/uncertainty adaptive budget；
4. solver fallback；
5. 绘制质量—延迟 Pareto。

### 第五步：数据飞轮与 LNS

1. DAgger；
2. hard-state active labeling；
3. Neural LNS；
4. solver branching model；
5. 最后再决定是否需要 offline/on-policy RL。

## 18. 建议的代码布局

以下仅表示 module seam，不要求一次性迁移现有代码：

```text
cluster_toolkit/
  constraint_ir/
    schema.py
    expressions.py
    conformance.py
  compiler/
    problem_compiler.py
    ll_rules.py
    cleaning_rules.py
    robot_rules.py
  event_kernel/
    kernel.py
    state.py
    frontier.py
    propagation.py
  validator/
    ... independent semantic replay ...

cluster_learning/
  decision_frame/
    schema.py
    graph_builder.py
  data/
    trajectory_store.py
    expert_adapters.py
    counterfactual.py
    splits.py
  models/
    candidate_mlp.py
    factor_graph.py
    heads.py
  search/
    greedy.py
    beam.py
    lns.py
    fallback.py
  training/
    imitation.py
    dagger.py
    offline_rl.py
```

外部主要 seam 应保持为：

```python
ir = compiler.compile(problem)
result = scheduler.solve(ir, policy, limits)
report = validator.validate(problem, result.schedule)
```

## 19. 当前推荐版本

如果只能选择一条近期最可能成功的路线：

```text
Constraint IR v1
    + exact event Kernel
    + independent Validator
    + factor graph Intent ranker
    + Beam-16/32
    + uncertainty adaptive budget
    + CP-SAT fallback
    + regret/counterfactual/DAgger 数据
```

直接策略、Beam、LNS 和 solver branching 不需要四套状态表示。它们共享同一 IR、DecisionFrame、模型表示和数据资产，只替换 SearchScheduler adapter。这是该设计最重要的 leverage 与 locality 来源。

## 20. 仍需明确的产品问题

这些问题不阻塞 IR/Kernel 设计，但会决定实验预算和默认 adapter：

- 线上单次决策延迟预算是 1 ms、10 ms、100 ms 还是秒级？
- 主要目标是 makespan、steady-state throughput、cycle time、tardiness 还是多目标？
- 是否存在随机故障、arrival、process-time uncertainty 等在线扰动？
- 清洗是否允许 preventive clean？多个 cleaning trigger 的 priority 是否总是定义？
- Pump/Vent 是可决策动作，还是设备控制层自动动作？
- 双臂是否只表示持片容量，还是存在真实并行与几何动作？
- 最大 wafer、route、module、candidate 数量是多少？
- 是否需要最优性 gap/proof，还是只需要稳定高质量可行解？

## 21. 参考依据

- [Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/11958dfee29b6709f48a9ba0387a2431-Abstract.html)：图策略和规模泛化的直接调度基线。
- [Exact Combinatorial Optimization with Graph Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2019/hash/d14c2267d848abeb81fd590f371d39bd-Abstract.html)：变量—约束图上的 learned branching 与 expert imitation。
- [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a)：DAgger 与模型诱发状态分布。
- [Neural Large Neighborhood Search for the Capacitated Vehicle Routing Problem](https://arxiv.org/abs/1911.09539)：学习启发式与 LNS 结合。
- [Graph Neural Networks are Dynamic Programmers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8248b1ded388fcdbbd121bcdfea3068c-Abstract-Conference.html)：图网络与动态规划式算法推理的结构关系。
- [Conservative Q-Learning for Offline Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html)：静态数据离线 RL 的分布外价值高估问题。
- [OR-Tools CP-SAT model interface](https://github.com/google/or-tools/blob/stable/ortools/sat/cp_model.h)：interval、NoOverlap、Cumulative 和 Automaton 等可组合约束原语。

