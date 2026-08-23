# Cluster Tool 调度数据生成 Pipeline

## 1. 文档目的

本文定义 Cluster Tool 调度数据从物理拓扑、Recipe 和 workload 生成，到多种求解器并行标注、质量归约及数据集切分的总体框架。

该框架面向三类下游用途：

- 强化学习训练与评测；
- 监督学习或模仿学习数据构造；
- CP-SAT、周期调度和启发式算法的 benchmark。

本文固定当前全大气版本的职责、稳定数据接口、求解预算和正确性口径；含 LOAD_LOCK 的扩展与经验难度阈值后续再补充。

### 1.1 当前实现范围

仓库已经提供可用于大规模生产的全大气纵切面：

- schema-v2 通用物理 Module：`IO | CHAMBER | LOAD_LOCK`，PM/AL/BUFFER 是 CHAMBER tag；
- 9 个 1–3 Cell 线性硬件 archetype、32 个有限 arm 变体和不可变拓扑快照；
- 每 Cell 一台单/双臂 Robot、每相邻 Cell 边界 1–2 个 BUFFER；
- topology-aware compiler 直接保存从 IO 出发并返回 IO 的完整可执行 Route；
- Engine、严格 Validator、direct CP-SAT 和 periodic CP-SAT 的多 Robot/BUFFER 换手语义；
- 独立的 cycle、startup、closedown CP-SAT，周期展开只作为可行 hint/fallback；
- genetic 四 seed、branch search 三个 horizon、short/long 晋级和子进程硬截止；
- `plan/run/resume/reduce/status` 生产 CLI、原子落盘和正交状态模型。

第一版生产数据不含 LOAD_LOCK、清洗、JIT/residency 和双臂几何。LOAD_LOCK 在 schema 中保留，但含 LL 的复杂 family 不阻塞全大气数据生产。

## 2. 核心原则

### 2.1 Problem 是不可变事实

一个 instance 表示有限批次调度问题：所有 wafer 在时间 `t=0` 位于虚拟 IO，调度从初始状态开始，直到所有 wafer 完成 Recipe 并返回 IO。

Problem 一旦生成便不可因求解结果而改变。求解器输出、验证结果、最优性证明和经验难度均作为独立的 labeling 产物保存。

新 pipeline 不复用旧 `ClusterProblem` schema。`SchedulingInstance` 是 generator、solver、Validator 和学习系统之间的新规范接口；旧 Engine 或已有算法需要通过显式 adapter 接入，不在 canonical problem 中保留兼容字段。

### 2.2 合法性优先，makespan 是唯一优化目标

所有候选调度必须先满足统一问题语义和 Validator。只有合法调度才参与比较，合法调度之间以完成全部 wafer 的 makespan 作为唯一优化目标。

### 2.3 生成与求解解耦

数据生成阶段负责产生结构合法、Route 可物理实现的 problem。Labeling 阶段针对同一个 problem 并行运行 CP-SAT、周期调度、遗传算法及未来其他算法。不同 solver 不修改 problem，也不并发修改同一个聚合文件。

### 2.4 区分结构属性、求解状态与难度

`wafer_count`、`recipe_count`、Route 长度、候选 PM 数等是实例的客观结构属性，不直接等同于难度。难度在固定 solver 配置和预算下通过 `status`、incumbent、best bound、gap 等结果校准，并且允许按 solver 分别记录。

## 3. 建模边界

第一版采用确定性、简化但一致的物理语义。

第一版包含：

- Module 和 Robot 的物理可达关系；
- PM、AL、BUFFER 和虚拟 IO；
- 所有物理 Module 的单片占用语义；
- Robot 操作互斥、移动、Pick/Place 时间和持片容量；
- 单臂 Robot 容量 1，双臂 Robot 抽象容量 2；
- Recipe 顺序、候选 PM 和确定性加工时间；
- 所有 wafer 在 `t=0` 可用，且初始 `priority=0`。

暂不包含：

- 清洗约束；
- JIT 和 residency；
- 随机或区间加工时间；
- 双臂几何、固定夹角和换臂姿态；
- Valve/interface；
- 真实 LP slot；
- 物理 chamber 容量大于 1；
- LOAD_LOCK Route 和 Pump/Vent 调度（仅预留 schema）。

生成器可以随机采样时间参数，但写入 instance 后必须成为确定的整数秒，调度执行期间不存在时间不确定性。

## 4. 分层模型

### 4.1 Topology Catalog

Topology 只描述物理结构：

- Module ID、类型、数量和所属 Cell；
- Robot ID、单双臂类型和可达 Module；
- Cell 之间的 LL、BUFFER 或其他物理连接；
- 从加工区域返回 IO/LP 的物理路径。

Topology 不描述 PM 工艺能力、Recipe、wafer 数量或时间参数。仓库中的 `topologies/**/*.json` 是物理拓扑的唯一 source of truth；`PipelineCatalog` 递归加载并校验这些文件。生产 `plan` 只选择 catalog 中已经存在的 topology，并将其内容写入 run 内的不可变 `topologies/` 快照，不在运行时通过 Python 常量重建物理结构。不同的 PM、BUFFER 或 Robot 组合具有不同的内容寻址 `topology_id`，同一个快照随后生成多个 instance。

Topology schema 不保存 PM capability 或可配置的 PM capacity。第一版所有物理 Module 均按单片占用解释；Robot 持片能力由 `arm_kind=single_arm|dual_arm` 决定。

当前 archetype-v1 固定以下硬件布局：

| `archetype_id` | 每 Cell PROCESS CHAMBER 数 | 每相邻边界 BUFFER 数 |
| --- | --- | --- |
| `single_compact` | `(2)` | `()` |
| `single_parallel` | `(6)` | `()` |
| `dual_balanced` | `(3,3)` | `(1)` |
| `dual_front_bottleneck` | `(2,6)` | `(1)` |
| `dual_rear_bottleneck` | `(6,2)` | `(1)` |
| `dual_parallel_handoff` | `(4,4)` | `(2)` |
| `triple_balanced` | `(3,3,3)` | `(1,1)` |
| `triple_middle_bottleneck` | `(5,2,5)` | `(1,1)` |
| `triple_asymmetric` | `(2,4,6)` | `(1,2)` |

每个布局都包含一个 `C0_AL`。是否访问 AL 属于 Recipe 采样；未访问时该 Module 对当前 instance 不占用资源。Robot arm 与布局正交，只从有限 profile 中选择：1 Cell 为 `single/dual`，2 Cell 为 `ss/dd/sd/ds`，3 Cell 为 `sss/ddd/sds/dsd`。因此完整 catalog 最多包含 32 个 `layout archetype × arm profile` topology；不再随机扰动 PM、BUFFER 数量或 Robot 可达关系。生产 spec 可通过 `topology_archetypes` 删除不需要的布局。

这 32 个完整物理 topology 位于 `topologies/atmospheric_archetypes/*.json`。每个文件直接保存 Module、Robot、arm kind、Cell 和可达关系；修改或移除原型只操作 catalog JSON。`topology_family.py` 不保存 archetype 表，只保留旧 schema-v1 随机 family 的兼容生成器。

固定 topology 后仍会采样 Recipe 数量、加工步骤、Route Cell 序列、候选 PM 子集、reentry、AL visit、周期比例、wafer 数量，以及 process/align/buffer/Robot 时间。只有 Cell、PM、BUFFER、Robot、arm 和可达关系被冻结；“只改变时间”的 `frozen_recipe` 专项模式不属于默认大规模生产。

### 4.2 Recipe Generation Profile

Recipe 生成采用“通用 topology-aware compiler + 可覆盖 profile”的方式：

- 通用 compiler 负责物理正确性，并根据 topology 判断是否需要插入 AL、LL、BUFFER 和返回路径；
- profile 控制 Recipe 数量、PM 步骤数、每步候选 PM 数、重入概率、加工时间分布及允许的加工 Cell 路径；
- 特殊 topology 可以覆盖默认 profile，但不把生成规则写入物理 topology 文件。

### 4.3 Recipe

Recipe 在 canonical problem 中直接保存完整 Route。每个 visit 声明候选 `module_ids` 和一个确定的 `process_time`；PM、AL、BUFFER 使用同一种 visit 结构，只有 tag 和加工时间不同。

第一版不维护 `PM.process_ids` capability matrix；Recipe 中的候选 PM 集合是该步骤可加工位置的唯一事实。相同 topology 的不同 instance 可以在同一物理 PM 位置上生成不同工艺。

canonical problem 不保存旧 schema 的 `process_id`、PM capability matrix 或 pool。候选 Module 不得跨 Cell；跨 Cell 必须显式经过连接相邻 Cell 的 BUFFER，Pick 与随后的 Place 必须由同一 Robot 完成。任意 Recipe 最多涉及 3 个 Cell，并允许重入。

### 4.4 Canonical Instance Schema

`SchedulingInstance` 自包含一次求解所需的确定性事实：

```text
SchedulingInstance
├── instance_id + schema_version
├── topology
│   ├── modules: kind + cell_id + LL side definition
│   └── robots: arm_kind + reachable module_ids
├── timing
│   ├── robot pick/place/travel
│   └── LL pump/vent
├── recipes[]
│   └── steps[]: candidate_module_ids + process_time
├── workload[]: recipe_id + wafer_count + release_time=0 + priority=0
├── source_module_id + sink_module_id
├── objective = makespan
└── provenance: topology/profile/generator version + seed + periodic ratio
```

所有时间参数在生成 instance 时一起采样并冻结为整数。同一个 topology 因此可以产生不同的设备时间、Recipe 候选 PM、加工时间和 workload，而 topology 文件本身始终只描述物理结构。

### 4.4 Workload

Workload 定义每种 Recipe 的 wafer 数量。所有 wafer 同时可用，不预先规定 `ABAB`、分批投片或产品优先级，产品交错顺序由调度器决定。

建议保留以下规模属性：

| `wafer_scale` | wafer 总数 |
| --- | ---: |
| `small` | 10–25 |
| `medium` | 26–50 |
| `large` | 51–100 |
| `xlarge` | 101–200 |

`wafer_scale` 只是规模，不是 easy/medium/hard。它通常显著影响直接非周期求解器的模型规模，但对重复固定稳态周期的求解影响不同。

## 5. 生成 Pipeline

```text
人工 Topology Catalog
        ↓
选择 topology 和 recipe generation profile
        ↓
生成 instance 级整数时间参数
        ↓
生成 Recipe PM 核心序列
        ↓
topology-aware compiler 编译完整 Route
        ↓
生成 workload
        ↓
Schema 与引用检查；复杂 topology 增加 Route witness 检查
        ↓
写入 Raw Corpus
```

生成器会在内存中构造逐片串行调度，并通过 Engine 和严格 Validator 完整回放，从而验证结构可行。该 witness 只用于生成审计，不保存动作；metadata 仅记录验证通过和 `serial_witness_saved=false`。

原始 instance 在通过 schema 和 Route witness 检查后即可写入 corpus，其 labeling 状态可以是 `pending`。是否进入训练或评测数据集由后续 Dataset View 筛选决定。

## 6. Recipe 和 Workload 的初始分布

以下数值作为 pilot 默认值，必须配置化并允许根据真实数据或 pilot 统计调整。

### 6.1 Recipe 结构

每个 problem 生成 1–3 种 Recipe。单条 Recipe 包含 1–5 个 PM 加工步骤，每一步包含 1–3 个候选 PM。

| 属性 | 默认分布 |
| --- | --- |
| PM 步骤数 | 1: 10%，2: 30%，3: 35%，4: 20%，5: 5% |
| 候选 PM 数 | 1: 40%，2: 45%，3: 15% |
| 重入 | 无重入 90%；一个步骤重复一次 10% |

候选 PM 数不能超过当前 Cell 的可用 PM 数。Recipe compiler 必须拒绝不能通过 topology materialize 的候选组合。

### 6.2 加工时间

pilot 使用整数时间锚点：

| 时间锚点 | 默认权重 |
| ---: | ---: |
| 30 秒 | 20% |
| 50 秒 | 30% |
| 300 秒 | 30% |
| 600 秒 | 20% |

可以在锚点上施加 `±10%` 扰动，最终四舍五入为正整数。同一步骤的所有候选 PM 共用一个加工时间。

### 6.3 多 Recipe 数量比例

一般 workload 可以从少量比例模板生成，并允许部分实例施加数量扰动。适合周期法的实例必须使用第 8 节定义的精确比例，不能包含 tail。

数据生成不应再预先写入 `easy/medium/hard`。应按 `topology_id × recipe_count × wafer_scale` 分层生成固定配额，其余结构属性在每个格子内随机并记录。

## 7. 并行 Labeling 与 Solution Reducer

```text
Raw Problem
    ↓
┌──────────────────┬──────────────────┬──────────────────┐
│ Direct noncyclic │ Fixed-ratio      │ Genetic / other  │
│ CP-SAT           │ periodic CP-SAT  │ heuristics       │
└──────────────────┴──────────────────┴──────────────────┘
    ↓ 每次运行写独立 solution record
统一 Validator
    ↓
Solution Reducer
    ↓
best validated solution + bounds + solver-specific statistics
```

每个 solver task 由以下四元组唯一标识：

```text
(instance_id, solver_name, solver_config_hash, seed)
```

每个 attempt 使用四个正交状态字段：

```text
solution_status:   UNKNOWN | FEASIBLE | OPTIMAL | INFEASIBLE
termination_reason: NORMAL | TIME_LIMIT | ERROR | INTERRUPTED | NOT_ELIGIBLE
validation_status: NOT_RUN | VALID | INVALID
workflow_status:   PENDING | RUNNING | TERMINAL
```

短、长预算以及所有失败 attempt 都保留。只要存在一个 `VALID` 完整解，instance 就可用；若结构可行性 witness 已通过但 solver 报告 `INFEASIBLE`，Reducer 将其隔离为 consistency failure。

Direct noncyclic CP-SAT 对所有实例先运行 10 分钟 short attempt。周期不适用的实例若未证明最优，再运行 30 分钟 long attempt；周期适用的实例不晋级 direct，但仍保留 short 结果。超时不是生成失败，所有 attempt 都保存原始状态、incumbent、best bound、gap、runtime 和预算。

### 7.1 当前 Direct CP-SAT 的实现边界

当前实现从 canonical instance 经 adapter 进入既有执行模型，采用完整有限批次的 action-level circuit 建模：

- 每个 Recipe 步骤直接选择一个候选 PM；
- PM 从 `Place.start` 占用至离开该 PM 的 `Pick.end`，加工从 `Place.end` 开始；
- 每次 transfer 动态选择一台能同时到达起点和终点的 Robot，Pick/Place 共用该选择；
- 每台物理 Robot 的全部 Pick/Place 动作由独立 circuit 排序，并按相邻动作所在 Module 决定 travel time；
- Robot 从 `Pick.start` 到对应 `Place.end` 持片，单臂容量为 1，双臂抽象容量为 2；
- Robot 初始位置未知，首动作不计初始移动时间；Pick/Place 输出使用配置中的精确整数时长；
- 所有完整解必须具有预期动作数量，并通过既有 `ValidatorSuite` 后才可保存。

为保证短预算也能留下可用标注，求解器先把串行可行调度作为完整 hint 固定一次，得到 CP-SAT 可行性见证；再使用剩余预算放开完整模型进行优化和证明。第一阶段固定 hint 得到的 `OPTIMAL` 只针对固定变量，外部统一记录为 `FEASIBLE`。只有第二阶段完整模型返回 `OPTIMAL`，才允许声明全问题 `PROVEN_OPTIMAL`。

该模型借鉴了四个参考项目中的 circuit 动作排序、候选腔室选择和“周期/进出稳态状态分离”思想，但没有直接复制其场景代码。带清洗版本绑定了 JIT、LL 状态和多类清洗规则；其中一版双臂实现还通过虚拟臂与固定 travel padding 近似 Robot 占用，和本仓库 Validator 的物理 Robot 互斥语义并不等价。旧 `baseline/cpsat/solver` 因裁剪后存在缺失约束函数和包内导入错误，仅保留作参考，不再作为当前 instance 的公共求解入口。

周期法只运行在显式生成的固定比例实例上，且所有候选 Module 集合必须两两完全相同或完全不相交；部分交叉集合返回 `NOT_ELIGIBLE`。Genetic 使用 4 个 seed，branch search 使用 horizon 1/3/5，二者每个 attempt 最多 10 分钟且不晋级。

Reducer 只比较通过统一 Validator 的完整解，并以最小 makespan 选择当前 best solution。后续新增 solver 结果时，problem 不变，只重新生成 solution index 和 Dataset View。

## 8. 固定比例周期调度

第一版周期比例目录如下，比例均使用 Recipe 的稳定顺序表示，并按最大公约数规范化：

| Recipe 数 | 允许的基础周期比例 |
| ---: | --- |
| 1 | `(1)` |
| 2 | `(1,1)`、`(1,2)`、`(2,1)` |
| 3 | `(1,1,1)`、`(1,2,1)`、`(2,1,1)`、`(1,1,2)`、`(1,2,2)`、`(2,2,1)`、`(2,1,2)` |

周期 workload 必须严格满足：

```text
counts = k × ratio_template
```

第一版不把任意 counts 分解为固定周期加 tail。例如 `(A=51, B=49)` 不按 `49 × (1,1) + (2,0)` 求解，而是归入一般非周期问题。

`cpsat_periodic` 支持 1–3 Cell、多 Robot 和显式 BUFFER Route。实现首先用 CP-SAT 求解一个基础比例批次的环形稳态周期，再依据每条 transfer 和 process dependency 是否跨越周期边界计算 wafer 的 period offset。有限 workload 通过以下方式物化：

1. 从空机台开始，在前 `k` 个周期各投入一个基础比例批次；
2. pipeline 填满后，完整重复稳态动作周期；
3. 第 `k` 个周期后停止投片，让已投 wafer 按同一周期 dependency 自然排空；
4. 将周期 token 映射成稳定的实际 `(recipe_id, wafer_index)`，分配双臂并通过统一 Validator。

这会产生可验证的 startup、steady 和 closedown 三段。实现先枚举全部动作边界，确定性展开、严格验证并按 makespan/depth/shift 排序，只把前两个 cut 送入边界 CP-SAT。每个 cut 的 startup 与 closedown 并行求解：startup 固定稳态边界并最小化到达时间，closedown 固定同一边界并最小化清空时间；周期展开是 guaranteed-feasible hint/fallback。short 预算为 cycle 5 分钟、startup/closedown 各 5 分钟；未全部证明最优则晋级为 cycle 10 分钟、startup/closedown 各 20 分钟。

稳态 CP-SAT 显式约束：

- 每台物理 Robot 的 Pick/Place 动作环及各自唯一跨周期弧；
- 双臂抽象持片容量和跨周期 holding interval；
- PM 从 `Place.start` 到离开 PM 的 `Pick.end` 的环形占用；
- Recipe 加工时间、候选 PM 一致性和 process wrap；
- 周期末到周期初的 Robot travel time。

周期 CP-SAT 使用可配置时间上限并保存原始状态；边界枚举与 composition 是有限、确定性的后处理，同时单独记录运行时间。`solve_cpsat_instance` 负责比例路由：支持的规范化比例进入 `cpsat_periodic`，其他比例进入 `cpsat_direct`。不支持的比例不会被强行拆分成周期加 tail。

需要严格区分子问题最优与完整问题最优：

- `cycle/startup/closedown.status = OPTIMAL` 只证明对应子模型最优；
- 拼接结果通过 Validator 后是完整问题的合法可行解；
- 即使所有子模型均为 `OPTIMAL`，也不能自动宣称拼接后的有限批次 makespan 全局最优；
- 只有完整问题的有效下界等于该解 makespan 时，才能标记 `global_optimality_status = PROVEN_OPTIMAL`。

建议的周期 solution 摘要如下：

```json
{
  "solver": "cpsat_periodic_composition",
  "ratio_template": {"A": 1, "B": 2, "C": 2},
  "components": {
    "cycle": {"status": "OPTIMAL", "objective": 120, "best_bound": 120},
    "startup": {"status": "FEASIBLE", "objective": 450, "best_bound": 410},
    "closedown": {"status": "OPTIMAL", "objective": 390, "best_bound": 390}
  },
  "composition": {
    "makespan": 6840,
    "validation_status": "VALID",
    "global_optimality_status": "UNPROVEN"
  }
}
```

solution record 顶层的 `best_bound` 仅允许表示完整有限批次问题的有效下界，并必须同时声明 `best_bound_scope = full_problem`。周期、startup 或 closedown 子模型的 bound 只能放在各自 `components` 中，Reducer 不得用这些局部 bound 计算全问题 gap。

## 9. 数据组织

每个问题使用独立目录：

```text
instances/
  <instance_id>/
    problem.json
    metadata.json
    solutions/
      cpsat_direct/
      cpsat_periodic/
      genetic/
      branch_search/
      other/
    solution_index.json
```

不会保存 serial witness。合法动作单独写入确定性的 `*.actions.json.gz`，solution JSON 保存 `actions_file`、`action_count` 和未压缩 JSON 的 SHA-256。

solver worker 只能写自己唯一的 solution 文件，不修改顶层 manifest。Reducer 负责验证和生成 `solution_index.json`，其中至少记录：

- 已验证 solution 数量；
- 当前 best solution 路径与 makespan；
- 完整问题 best bound 和 certified gap（若存在）；
- labeling 状态；
- reducer 和 Validator 版本。

Raw Corpus 保存所有通过生成检查的 instance。Dataset View 通过独立 manifest 引用符合用途和质量条件的 instance：

- RL View：至少存在一个合法参考 makespan；
- Supervised View：存在所选 solver 的完整合法动作序列；
- Benchmark View：保存需要比较的 solver 运行记录和预算。

## 10. 数据切分与难度

建议保留以下互斥评测视图：

- `validation_iid`：训练见过 topology，但 problem seed 未见过；
- `test_iid`：已见 topology 和生成分布下的独立实例；
- `test_recipe_ood`：已见 topology，但 Recipe Set 未参与训练；
- `test_topology_ood`：整个 topology template 未参与训练；
- `test_scale_ood`：后续可选，测试超出训练规模的 workload。

同一个 problem 只能属于一个 split。Solution 跟随 problem，不参与 split 分配。

实例 metadata 保存结构属性；labeling 后另外保存 solver-specific 结果，例如：

```text
difficulty.cpsat_direct
difficulty.cpsat_periodic
difficulty.genetic
```

具体 easy/medium/hard 阈值不在生成前定义。应先运行覆盖主要 topology、Recipe 数量和 wafer scale 的 pilot，收集固定配置下的状态、bound、gap 和求解时间分布，再确定阈值。

## 11. 正确性与可复现性

每个 instance 必须保存：

- 唯一 `instance_id`；
- `topology_id` 和版本；
- generator、compiler、problem schema 版本；
- master seed 和各阶段派生 seed；
- 完整生成参数及实际采样统计；
- Route materialization witness；
- 内容结构签名。

每个 solution 必须保存：

- solver 名称、代码版本和配置 hash；
- seed、预算和运行环境信息；
- 原始 solver status；
- objective、incumbent、完整问题 best bound、bound scope 和 gap；
- 动作序列或动作文件引用；
- Validator 版本和验证结果。

所有时间使用整数秒输入；事件和验证采用统一的 `[start, end)` 区间语义。

## 12. 后续专项设计

以下内容刻意留待后续，不阻塞全大气数据生产：

1. 含 LOAD_LOCK 的复杂 topology family 与 Pump/Vent 调度；
2. RL observation、训练样本转换接口和 Dataset View；
3. 真实双臂几何、清洗、JIT/residency；
4. 根据大规模运行统计校准经验难度阈值。

## 13. 当前代码入口

主要模块：

- `cluster_toolkit.cluster_generator.pipeline_models`：Topology、Profile、SchedulingInstance 和生成请求 schema；
- `cluster_toolkit.cluster_generator.pipeline_catalog`：版本化 catalog 加载；
- `cluster_toolkit.cluster_generator.pipeline`：确定性 instance 生成；
- `cluster_toolkit.cluster_generator.problem_adapter`：将当前 direct topology 转换为内存中的旧 `ClusterProblem schema v1`，供既有 Engine、Action 和 Validator 复用；
- `cluster_toolkit.cluster_generator.corpus`：不可变 problem 目录落盘；
- `cluster_toolkit.cluster_generator.solutions`：solution record、状态语义和 reducer；
- `cluster_toolkit.cluster_generator.production`：run spec、拓扑快照和严格生成审计；
- `cluster_toolkit.cluster_generator.labeling`：子进程 supervisor、晋级、resume 和 reducer；
- `cluster_toolkit.run_data_pipeline`：`plan/run/resume/reduce/status` 入口；
- `baseline.cpsat.solve_instance`：当前 direct instance 的完整非周期 CP-SAT 入口；
- `baseline.cpsat.solve_periodic_instance`：固定比例周期、startup、steady、closedown 组合入口；
- `baseline.cpsat.solve_cpsat_instance`：按规范化 workload 比例自动选择 periodic 或 direct；
- `baseline.genetic.solve`：经同一 adapter 执行的遗传算法入口。

示例：

```bash
python -m cluster_toolkit.run_data_pipeline plan datasets/raw/run-001 \
  --seed 100 --count 100 --topology-count 32
python -m cluster_toolkit.run_data_pipeline run datasets/raw/run-001
python -m cluster_toolkit.run_data_pipeline status datasets/raw/run-001
python -m cluster_toolkit.run_data_pipeline resume datasets/raw/run-001
```

对一个已物化 canonical problem 自动选择 CP-SAT 方法并输出完整动作：

```bash
python -m baseline cpsat \
  datasets/raw/instances/<instance_id>/problem.json \
  --mode auto \
  --time-limit 1800 \
  --output datasets/raw/instances/<instance_id>/solutions/cpsat/actions.json
```

同一生成请求得到稳定的 `instance_id` 和相同内容。已存在且内容一致的 instance 可幂等写入；同一 ID 对应不同内容时 writer 必须拒绝覆盖。

### 13.1 旧执行层 Adapter

`to_cluster_problem(instance)` 只负责执行表示转换，不改变或落盘 canonical problem。转换使用旧 schema v1，因此不会生成 PM capability 或 `process_id`：

```text
SchedulingInstance
        ↓ to_cluster_problem（仅内存）
ClusterProblem schema v1
        ↓
ClusterEngine / existing Actions / ValidatorSuite
```

adapter 支持 schema-v2 的 IO、普通 CHAMBER、AL tag 和 BUFFER tag，并保留多 Robot 可达关系。完整 Route 已在生成阶段物化；Robot 只能在显式 BUFFER visit 后换手。第一版仍拒绝 LOAD_LOCK 调度。

所有 Robot 在转换后的 `initial_state` 中使用 `position_module_id = null`。因此第一次 Robot 动作之前不计初始移动时间；之后相邻动作仍必须满足 travel time。IO 的临时执行容量设为 wafer 总数，其余物理 Module 容量设为 1。Workload 在 adapter 内稳定展开为 `(recipe_id, 0-based wafer_index)`，但 canonical problem 仍只保存每种 Recipe 的 wafer 数量。
