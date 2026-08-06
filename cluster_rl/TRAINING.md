# HGT + Transformer Decoder PPO 训练

模型以 `ClusterHeteroGraphBuilder` 生成的异构图作为唯一状态输入：

1. HGT encoder 在 `global`、`wafer`、`route_step`、`module` 和 `robot`
   节点及其异构关系上进行消息传递；
2. Transformer decoder 为每个 `(wafer, robot)` Pick、`(module, robot)`
   Place 和 `ADVANCE` 动作构造 query，并对 HGT 节点 memory 解码；
3. Actor 输出与环境完全一致的
   `((W + M) * R + 1)` 个 masked logits，Critic 从 global 节点输出状态价值。

环境按物理时间增量返回奖励，因此成功 episode 的原始回报是 `-makespan`。
PPO 使用 first-legal 调度的 makespan 进行归一化：

```text
成功 episode 归一化回报 = 1 - makespan / reference_makespan
```

## 环境与启动

本地开发先安装相邻的统一工具包：

```bash
conda activate rl
python -m pip install -e ../cluster-tool-validator
```

在线生成训练需要提供固定的验证集和测试集 manifest：

```bash
python -m cluster_rl.train \
  --train-mode generator \
  --num-envs 8 \
  --cpu-workers 0 \
  --generator-seed 42 \
  --validation-manifest datasets/validation/manifest.json \
  --test-manifest datasets/test/manifest.json \
  --total-steps 100000
```

每个环境槽位在 episode 结束后都会用新的确定性种子生成完整实例，并同步替换
`ClusterEnv`、观察编码器和初始观察。训练参考值由 RL 环境的 FIFO-aware
逐片串行 `first_legal` rollout 计算；工具包 manifest 中的启发式 actions 不作为RL基线。
训练结束后的 greedy evaluation 只读取固定 validation/test manifest。

首次验证链路可以运行：

```bash
python -m cluster_rl.train \
  --train-mode generator \
  --num-envs 2 \
  --validation-manifest datasets/validation/manifest.json \
  --total-steps 384 \
  --rollout-steps 128 \
  --epochs 1 \
  --hgt-layers 1 \
  --num-layers 1 \
  --model-dim 32 \
  --feedforward-dim 64
```

`--hgt-layers` 控制 HGT encoder 深度，`--num-layers` 控制 Transformer
decoder 深度。Apple Silicon 可添加 `--device mps`，NVIDIA GPU 可添加
`--device cuda`；默认使用 CPU。

生成模式可通过 `--cpu-workers N` 启用持久化环境进程，但小图逐步跨进程传输的IPC
开销可能高于并行收益，默认和推荐值均为0。PPO使用针对固定异构图schema的专用
minibatch拼接路径，避免通用PyG Batch的额外拆装。
每个update日志分别输出`rollout`和`PPO`耗时，便于判断瓶颈位于环境侧还是模型更新侧。

需要进一步定位时可添加`--profile-timing`。该模式会打印图编码、主进程拼批与传输、
策略推理、环境/IPC等待、worker内部环境推进/参考调度/图编码，以及PPO minibatch
拼接、前向和反向的耗时。为获得准确GPU计时，它会在各阶段同步CUDA，因此只应用于
一到两个诊断update，不要用于正式长时间训练。

## 输出与日志

每次新训练默认创建独立时间戳目录：

```text
runs/ppo_cluster_YYYYMMDD_HHMMSS/
├── checkpoint.pt
├── config.json
├── train.log
├── updates.csv
├── episodes.csv
├── evaluation.csv
└── training_curves.png
```

- `train.log` 保留完整控制台训练日志；
- `updates.csv` 保存 PPO loss、entropy、KL、choice fraction 和梯度范数；
- `episodes.csv` 保存分场景 makespan 与归一化回报；
- `evaluation.csv` 保存训练结束后的 greedy rollout 结果；
- `training_curves.png` 可视化 makespan、归一化回报、loss、entropy 和 KL；
- greedy rollout 会通过独立 `ValidatorSuite` 检查动作序列。

A800正式大规模训练可在已分配的Slurm节点中启动：

```bash
chmod +x scripts/train_a800_large.sh
./scripts/train_a800_large.sh
```

脚本默认训练500万步，可通过环境变量覆盖训练量和输出目录，例如：

```bash
TOTAL_STEPS=2000000 RUN_DIR=runs/my_large_run \
  ./scripts/train_a800_large.sh
```

从 checkpoint 继续训练：

```bash
python -m cluster_rl.train \
  --resume runs/ppo_cluster_YYYYMMDD_HHMMSS/checkpoint.pt \
  --total-steps 200000
```
