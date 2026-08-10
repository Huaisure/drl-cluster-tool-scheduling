# Cluster Tool 基准数据生成器

本包包含两种互补模式：

- `DatasetGenerator`：从外部problem JSON保留固定拓扑，随机化参数、Route和Wafer；
- `ProblemGenerator`：程序化生成具有AL/LL/ATM/VTM/Buffer/PM结构和显式工艺能力的
  半导体集束型设备实例，并生成经过完整回放验证的串行可行性调度。

## PPO课程实例

训练时可以直接逐episode采样，无需专家动作标签：

```python
from cluster_generator import ProblemGenerator

generator = ProblemGenerator()
problem = generator.sample(seed=123, difficulty="medium", split="train")

# 默认按30% easy、40% medium、20% hard、10% edge采样。
problem = generator.sample_curriculum(seed=124, split="train")
```

规模分为10～25、26～50和51～75片Wafer，Recipe为1～6种；`edge`与`hard`共享大规模
区间。Problem模型本身不限制Wafer数量，10～75只是当前生成分布。Recipe之间的Wafer
数量以均匀分配为中心做小幅扰动。结构签名经过不相交hash bucket切分，避免相同结构跨越
train/validation/test。

训练集拓扑按10%简化结构、60%单真空单元、30%双真空单元采样；验证集为5%/45%/50%，
测试集为0%/30%/70%。三类结构分别为：

```text
虚拟IO -> 统一TM -> PM×3-6

虚拟IO + AL×1 + ATM×1 + LL×1-2
                       -> VTM1 + PM×3-6

虚拟IO + AL×1 + ATM×1 + LL×1-2
                       -> VTM1 + PM×3-6
                       -> Buffer×1-2
                       -> VTM2 + PM×3-6
```

除虚拟IO外所有Module容量均为1。ATM以单臂为主，VTM以双臂为主。复杂Recipe显式写为
`AL -> LL -> 真空侧流程 -> LL`，虚拟IO不进入Route，出口LL后由ATM直接返回IO。
跨真空单元必须经过Buffer；只使用VTM2、只使用VTM1但到Buffer冷却，以及返回VTM1后
继续PM工艺都允许生成。

schema version 2显式保存PM的`process_ids`和PM RouteVisit的`process_id`。一个PM配置
2～3种工艺；同一种工艺通常由1～2个PM支持，少量由3～4个PM支持。两个真空单元的
工艺集合严格不重叠，Route候选PM必须等于配置该工艺的全部PM。每条Recipe包含1～8个
PM步骤，最多只有一种工艺重入：65%不重入、30%出现两次、5%出现三至四次。双单元
Recipe有25%概率包含Buffer冷却，路过Buffer时间为0，冷却访问为非零时间且不会连续。

每片Wafer写入必填静态`priority`。默认数据包含三类最终优先级结果：50%全部为0，30%
按Recipe分组，20%在Recipe内分成2～4个投片波次。JSON只保存逐片priority，不保存策略
名称。Wafer以`(recipe_id, wafer_index)`唯一标识；Engine在虚拟IO首次Pick时只开放全局
最小priority，Env可继续执行“同priority、同Recipe取最小wafer_index”的掩码。

需要指标与参考调度时使用：

```python
benchmark = generator.generate(seed=123, difficulty="hard", split="test")
problem = benchmark.problem
actions = benchmark.actions
metadata = benchmark.metadata
```

metadata包含拓扑类型、Module数量、候选比例、PM负载、平均合法动作数、reference
makespan、下界和近似gap。参考调度按照priority顺序串行完成Wafer，并在候选PM中选择
当前累计负载最小者；每份调度均由ClusterEngine完整回放并再次通过ValidatorSuite。
它只证明Problem至少存在一条合法完成路径，不声明调度质量或最优性。

### 时间与配置

移动、Pick和Place在8～15秒内采样；AL为10～20秒；Pump/Vent通常为10～20秒，少量
长尾不超过30秒。PM工艺先按20%/30%/30%/20%选择30、50、300、600秒档位，再做
±10%扰动。同一种工艺在一个Problem内加工时间固定，与候选PM无关。所有概率、范围和
档位都集中在可序列化的`ProblemGenerationConfig`中：

```python
from cluster_generator import ProblemGenerationConfig, ProblemGenerator

config = ProblemGenerationConfig(cooling_probability=0.20)
generator = ProblemGenerator(config=config)
```

参考调度不应用RL Env额外的“同priority、同Recipe只开放最小wafer_index”投片掩码，
因此用于证明Problem结构可行和回归Validator，不作为该mask下的专家动作标签。

固化数据集：

```bash
cluster-generate-rl rl-test/ \
  --split test \
  --difficulty hard \
  --count 1000 \
  --seed 42
```

每个实例输出problem JSON、`*.actions.json`参考调度，并在manifest记录全部难度指标。
训练集还可使用`--seed-only`只保存seed、难度和生成器版本；validation/test禁止此选项，
确保评估数据固化。`--without-reference-actions`可以只保存problem和manifest。

## 固定拓扑实例

`cluster_generator` 从一个合法problem JSON读取固定拓扑，随机生成可复现的设备参数、
Route、Wafer和初始source分配。该兼容模式仍可读取旧LP模板，只保证结构可行，不生成动作序列，并主动移除模板中的
JIT和Cleaning配置。

### 快速开始

```bash
cluster-generate validator/examples/all_actions_recipe.json dataset/ \
  --profile small \
  --count 10 \
  --seed 42
```

输出：

```text
dataset/
  manifest.json
  instance-00000.json
  instance-00001.json
  ...
```

输出目录必须不存在或为空。重新生成本工具已经创建的数据集时使用 `--overwrite`；
生成器只会删除旧manifest声明的文件，不会删除其他文件。

### Python接口

```python
from cluster_generator import DatasetGenerator, GenerationConfig

config = GenerationConfig(
    profile="medium",
    instance_count=100,
    seed=42,
)
generator = DatasetGenerator.from_template("template.json", config)

raw_instance = generator.generate_instance(index=0)
manifest = generator.generate("dataset/")
```

独立审计已有内存实例：

```python
from cluster_generator import validate_generated_instance

audit = validate_generated_instance(raw_instance)
```

审计会重新调用 `parse_problem()`，证明每条候选Route至少存在一条LP到LP的物理路径，
构造Cluster Engine初始状态，并检查每条Route至少有一个合法初始Pick。

### Profile

| Profile | Route数 | 总Wafer数 | Route步数 | PM加工时间 |
|---|---:|---:|---:|---:|
| small | 1–2 | 10–20 | 2–5 | 50–100 |
| medium | 2–4 | 21–50 | 4–10 | 50–300 |
| large | 4–6 | 51–75 | 8–20 | 100–600 |

上述表格和配置属于固定拓扑`DatasetGenerator`。PPO课程生成器使用前述现实时间与
Route分布，并通过`cluster-generate-rl`生成数据。

每个范围都可以通过配置JSON覆盖。区间格式为：

```json
{
  "minimum": 3,
  "maximum": 8
}
```

示例配置：

```json
{
  "profile": "small",
  "instance_count": 50,
  "seed": 7,
  "route_count": {"minimum": 3, "maximum": 5},
  "total_wafers": {"minimum": 15, "maximum": 30},
  "route_steps": {"minimum": 3, "maximum": 9},
  "process_time": {"minimum": 20, "maximum": 200},
  "candidate_probability": 0.5,
  "max_candidates": 3
}
```

```bash
cluster-generate template.json dataset/ --config generation.json --count 100
```

CLI参数覆盖配置文件，配置文件覆盖Profile默认值。

### 结构可行性的含义

两个Module在至少被一个相同TM访问时相邻。生成器用精确剩余步数动态规划构造：

```text
起始LP -> 非LP Route步骤 -> 最终LP
```

Route至少访问一个PM，内部不出现LP，也不连续重复同一Module。候选Module只在保留一条
确定可行的witness path时加入。Manifest记录每条Route的起始LP、最终LP和witness path。

该保证不等价于完整调度可行性或最优性。第一版不生成JIT、Cleaning、动作列表和makespan。
