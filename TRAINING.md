# PPO 训练

默认使用 `examples/scenarios/` 下的三个场景并行采样，训练实体 Token
Transformer Actor-Critic：

```bash
python -m train --total-steps 100000
```

训练过程输出 PPO 指标、完成 episode 的 makespan 和成功率。最终模型默认保存到：

```text
checkpoints/ppo_cluster.pt
```

训练结束后会对三个场景执行 greedy rollout，并使用独立 Validator 检查动作序列。

从 checkpoint 继续训练：

```bash
python -m train \
  --resume checkpoints/ppo_cluster.pt \
  --checkpoint checkpoints/ppo_cluster.pt \
  --total-steps 200000
```

Apple Silicon 可以添加 `--device mps`，NVIDIA GPU 可以添加
`--device cuda`。默认使用 CPU，以获得最稳定的首次运行体验。
