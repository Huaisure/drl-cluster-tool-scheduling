# HGT + Transformer Decoder PPO 训练

本文描述旧业务图模型。新的通用 Constraint IR 环境、匿名图模型和 PPO 入口已可运行，见 [ir/README.md](ir/README.md)；命令为 `python -m cluster_rl.ir.train`。两者共用 `cluster_rl/ppo.py` 的优势估计和裁剪损失，但输入协议与 checkpoint 不兼容。

模型以 `ClusterHeteroGraphBuilder` 生成的异构图作为唯一状态输入：

1. HGT encoder 在 `global`、`wafer`、`route_step`、`module` 和 `robot`
   节点及其异构关系上进行消息传递；
2. Transformer decoder只为当前合法的Pick、Place和`ADVANCE`动作构造query，
   并对HGT节点memory解码；
3. Actor 输出紧凑的合法动作logits，并通过batch内保存的环境动作索引完成双向映射；
   Critic从global节点输出状态价值。

Module类型特征只保留`is_io`、`is_pm`和`is_ll`。AL与BUFFER作为特殊的加工/暂存
Module共享`is_pm`，不增加额外类型位；旧的`is_lp`已经移除。

环境按物理时间增量返回奖励，因此成功 episode 的原始回报是 `-makespan`。
PPO 使用数据集生成时已验证的参考调度 makespan 构造有界时间代价，并额外区分成功与死锁。

```text
成功 episode 归一化回报
= 1 - 0.5 * makespan / (reference_makespan + makespan)
```

## 环境与启动

本地开发先安装相邻的统一工具包：

```bash
conda activate rl
python -m pip install -e ../cluster-tool-validator
```

先一次性固化训练、验证和测试数据：

```bash
chmod +x scripts/generate_datasets.sh
./scripts/generate_datasets.sh
```

默认生成1000个训练实例、100个验证实例和100个测试实例，全部保存Problem JSON和
reference metadata，不保存训练不需要的参考动作列表。数量和seed可通过环境变量覆盖：

```bash
TRAIN_COUNT=2000 VALIDATION_COUNT=200 TEST_COUNT=200 DATASET_SEED=42 \
  ./scripts/generate_datasets.sh
```

训练只读取固化后的manifest，不在episode切换时调用generator：

```bash
python -m cluster_rl.train \
  --train-mode dataset \
  --num-envs 8 \
  --cpu-workers 0 \
  --train-manifest datasets/train/manifest.json \
  --validation-manifest datasets/validation/manifest.json \
  --test-manifest datasets/test/manifest.json \
  --total-steps 100000
```

每个环境槽位在episode结束后按`slot_index + episode_index * num_envs`确定性轮换训练集，
并同步替换`ClusterEnv`、观察编码器和初始观察。训练和评估参考值直接读取manifest中的
已验证reference makespan。训练期间每隔`--evaluation-interval`个update在固定的
`--validation-cases`个validation实例上评估；子集会优先覆盖topology/difficulty组合。
模型首先按success rate、再按成功实例的
mean normalized cost保存`best_checkpoint.pt`。最终validation/test评估使用该最佳模型。
`--train-mode generator`仍保留用于在线生成实验。

首次验证链路可以运行：

```bash
python -m cluster_rl.train \
  --train-mode dataset \
  --num-envs 2 \
  --train-manifest datasets/train/manifest.json \
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
├── best_checkpoint.pt
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
- `evaluation.csv` 同时保存周期validation和最终validation/test结果，包含instance ID、
  difficulty、topology family、seed、success和termination reason；仅成功episode计算
  normalized cost与relative gain；
- `training_curves.png` 可视化 makespan、归一化回报、loss、entropy 和 KL；
- greedy rollout 会通过独立 `ValidatorSuite` 检查动作序列。

A800正式大规模训练可在已分配的Slurm节点中启动：

```bash
chmod +x scripts/train_a800_large.sh
./scripts/train_a800_large.sh
```

脚本默认训练100万步，可通过环境变量覆盖训练量和输出目录，例如：

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
