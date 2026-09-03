# 通用 IR 模型：训练入口

这里是真正消费 Constraint IR 的新训练链路，不是旧 Pick/Place 网络的包装。

```text
原 Problem JSON / 编译后的 IR JSON
              ↓
IRSchedulingEnv（ReferenceSession）
              ↓
IRGraphEncoder（匿名通用关系图）
              ↓
IRActorCritic（关系消息传递＋候选统一评分）
              ↓
PPO → checkpoint → 独立审计的评估轨迹
```

## 快速开始

从仓库根目录执行。两个输出目录必须尚不存在，不覆盖已有数据或实验。

```bash
.venv/bin/python -m cluster_rl.ir.data datasets/ir_v1 --seed 17

.venv/bin/python -m cluster_rl.ir.train \
  --train datasets/ir_v1/train/manifest.json \
  --validation datasets/ir_v1/validation/manifest.json \
  --test datasets/ir_v1/test/manifest.json \
  --run-dir runs/ir_v1_first \
  --total-steps 256 --rollout-steps 64 --minibatch-size 16 \
  --epochs 2 --width 32 --layers 3 --evaluation-interval 2
```

生成器默认固定 8/4/4 个 train/validation/test 小实例，覆盖 1–2 片晶圆、2–3 个 PM、1–3 次访问、候选 PM、不同移动/加工时长及单/双臂。它生成原始 JSON、已编译 IR 和带内容 hash 的 manifest，跨 split 去重；不在训练过程中重新生成数据。所有路线都在可达普通 PM 上，无 JIT/清洗等额外约束，逐片串行执行可构造可行解；生成器本身不生成最优参考排程。

`--train`、`--validation`、`--test` 都可接收多个原 Problem JSON、IR JSON 或 manifest。旧 manifest 的 `problem_file` 也可读取，但每个源问题必须在[首版 Compiler 支持范围](../../dev/problem-to-ir.md)内，不能静默跳过不支持实例。训练、验证、测试间有重复 IR hash 会拒绝；这不是图同构去重，也不代表已构造分布外测试。

默认 CPU、单计算线程。可显式传 `--device cuda` 或 `--device mps`；本轮验收只覆盖 CPU，不承诺其他设备的速度和数值复现性。先看日志中的 `rollout_seconds` 与 `ppo_seconds`，再扩大规模。

## 环境合同

`IRSchedulingEnv.reset()` 返回 `(IRGraph, info)`；`step(index)` 返回标准五元组。环境在一个实例内有固定的动作空间上界，实际合法索引是观测中紧凑的 `action_nodes`，`info.action_mask` 对齐固定上界。非法索引明确报错，不改动会话。

- 每次选择一个 Intent，提交完整 bundle。仍有可做操作时停留当前 Tick，允许同 Tick 连续承诺不同并发工作；必须一起原子提交的操作应表达成 Composite Intent。
- 下一事件存在时，最后一个候选是通用 Wait（等待），即使还有其他 Intent 也可选择它。
- 没有候选时，自动推进至下一个已承诺事件、区间边界、有限义务截止或 `elapsed_at_least` 时间条件触发点。
- 只有声明的目标值、精确终点 Lease、无未满足义务、无运行/未来已承诺工作全部成立，才算成功。没有候选不等于成功。
- 不存在下一推进点又未完成时为 `deadlock`；错过义务期限为 `deadline_missed`。不把其他 Kernel 异常吞掉伪装成失败样本。
- `max_decisions` 与可选 `max_time_seconds` 是本次训练任务的硬预算。耗尽返回 `truncated=True`、失败奖励；剩余预算会进入观测。这里的预算截断按吸收失败训练，不 bootstrap；仅 PPO rollout 长度切分使用价值 bootstrap。达到操作次数预算后，最后一次操作仍可自动执行至完成。

输入要求显式 `TerminalStateSpec`，时间单位为 `second`。当前有限、无循环自动展开程序可编码；循环 automatic emission graph 明确拒绝。Reference IR 不支持的条件不会通过另一个业务判断补进环境。

奖励没有晶圆、加工或清洗特定项。令 `C(t)=t/(S+t)`，默认 `S=100 秒`：

```text
step reward = -0.5 × (C(t_after) - C(t_before))
完成时 +1；死锁、期限失败或预算耗尽时 -1
```

时间项在 episode 内望远镜相消，初始被迫等待也计入第一次奖励；在默认 `gamma=1` 下，成功回报大于 0.5，失败回报不大于 -1。增加同 Tick 操作不凭空获得正奖励。评估优先比较完成率，再比较成功实例的平均 makespan（总完工时间）；不能只比较不同成功子集的平均耗时。

## 模型实际看到什么

`IRGraph` 仅含五个数值数组：节点原语类型、数值特征、边连接、边关系类型、当前可选节点索引。

| 内容 | 表达 |
|---|---|
| 系统事实 | 容量资源、类型化状态值、对象持有关系、活动义务、有效互斥选择 |
| 全部后续工作 | 有限绑定行具体化的所有计划，不限于当前合法候选；包括绑定后的条件、区间、边界效果、自动后果和步骤依赖 |
| 已承诺未来 | 未结束区间的资源使用、未来事件效果、剩余期限、有效 scope 的释放时刻 |
| 候选操作 | 对应完整计划、起止窗口、资源预约和预测状态/持有关系变化；未满足义务门槛和创建/满足义务效果也在计划图中 |
| 剩余目标 | 显式目标状态、精确目标 Lease 集合，以及当前状态；不是只给局部 delta |

`audit_kind` 被丢弃。资源、实体、枚举值、模板、规则、scope 和参数的 ID 只用于构图连接，不做字符串分词、hash 数值输入或 ID embedding（身份向量）。枚举值是匿名符号节点；数值做 `asinh` 压缩，时间字段先换算为秒。普通整数 StateCell 保留整数语义，不猜测业务意义。

网络只使用有限通用原语词表，例如容量资源、相等条件、占位获取、状态更新、义务创建、边界和等待。它没有 Pick/Place/Clean 输出头，也没有 PM/LL 类型特征。消息传递保留邻接数量，避免平均聚合丢掉持有/容量相关的数量信息；Actor 统一评分候选，Critic 读取整图。没有节点编号或序列位置 embedding，节点重排应只导致对应重排。

节点原语与边角色列表是版本化的模型输入协议 `ir-graph-1`。增加原语必须更新编码器、测试和协议，不允许未知字段静默丢失。这个编码器消费参考协议 `1.2-reference`；其 ID 查找是参考候选接口的 Adapter，不是模型语义。

## PPO 与实验产物

`cluster_rl/ppo.py` 提供共用 GAE（广义优势估计）、只在有选择的状态上归一化 Actor 优势、裁剪 Actor/价值损失。旧训练器和 IR 训练器都调用它。新训练器支持变长候选 batch、rollout 末端 bootstrap、epoch/minibatch 更新、熵项、梯度裁剪、KL 提前停止、非有限损失/梯度拒绝。

```text
run-dir/
  config.json             输入 hash、模型及训练配置
  baselines.json          随机、最短时长、初始策略的验证结果
  metrics.jsonl           PPO 指标、吞吐和验证指标
  episodes.jsonl          episode 完成/失败原因和回报
  best.pt                 按验证集选择；可能是第 0 步的初始模型
  last.pt                 最后实际训练更新后的模型与优化器
  result.json             最佳步数、参数变化量、最终验证/测试
  validation_traces/      可恢复且经过独立审计的 SessionSnapshot
  test_traces/            最终测试轨迹，不参与选模型
```

完整独立审计只用于评估，不放在每个训练 step 内；每条评估轨迹都审计，成功轨迹额外要求完整终态。失败轨迹审计通过只表示其合法前缀可信，不表示完成了任务。当前 `best.pt` 只按完成率和成功耗时选择，`best_step=0` 会如实显示没有超过初始策略。

单独重新加载并评估：

```bash
.venv/bin/python -m cluster_rl.ir.train \
  --evaluate-checkpoint runs/ir_v1_first/best.pt \
  --validation datasets/ir_v1/validation/manifest.json \
  --run-dir runs/ir_v1_reload
```

旧业务网络 checkpoint 不兼容，加载时校验图特征协议。保存优化器不等于已实现精确断点续训：当前提供重新加载推理/评估，尚未保存 RNG、进行中的环境和数据游标以供无缝恢复训练。

## 当前边界与验收

这是可以开始实验的第一版，不是生产训练平台。当前单环境采样、有限绑定行具体化和参考 Session 的多次回放仍会随问题规模增长变慢；未做向量环境、索引候选、静态图压缩、GPU 吞吐优化或超参数搜索。

源输入 Compiler 仍不支持 LL、清洗、JIT 等配置；但直接用通用 IR 声明的 Lease 条件、期限/无期限义务及条件效果可以进入环境与编码器，测试包含非 PM 语义案例。新增业务应扩充翻译与数据，不增加模型业务分支。

运行测试：

```bash
.venv/bin/python -m pytest tests/test_ir_training.py -q
.venv/bin/python -m pytest -q
```

本轮实施与实测结果见 [dev/ir-training.md](../../dev/ir-training.md)。小样本训练成功只证明基础设施可用，后续仍需多 seed、更多规模、真实约束组合与分布外评估。
