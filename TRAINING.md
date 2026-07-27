# PPO 训练

默认使用 `examples/scenarios/` 下的三个场景并行采样，训练实体 Token
Transformer Actor-Critic。每个场景会先运行一次确定性的 first-legal
参考调度，并使用以下归一化回报：

```text
成功 episode 回报 = 1 - makespan / reference_makespan
```

参考策略的回报为 0，不同规模问题的回报处于相近范围。

```bash
python -m train --total-steps 100000
```

每次运行默认创建独立的时间戳目录：

```text
runs/ppo_cluster_YYYYMMDD_HHMMSS/
├── checkpoint.pt
├── updates.csv
├── episodes.csv
├── evaluation.csv
└── training_curves.png
```

控制台会输出分场景 makespan、归一化回报、PPO 指标和最终对比表。训练曲线包括：

- 各场景 makespan；
- 各场景归一化回报；
- Policy/Value loss；
- Entropy 和 approximate KL。

训练结束后会对三个场景执行 greedy rollout，并使用独立 Validator 检查动作序列。

从 checkpoint 继续训练：

```bash
python -m train \
  --resume runs/ppo_cluster_YYYYMMDD_HHMMSS/checkpoint.pt \
  --total-steps 200000
```

由于网络输入和奖励定义已更新，旧版未归一化训练生成的 checkpoint
不能继续使用，需要开始一次新的训练。

Apple Silicon 可以添加 `--device mps`，NVIDIA GPU 可以添加
`--device cuda`。默认使用 CPU，以获得最稳定的首次运行体验。
