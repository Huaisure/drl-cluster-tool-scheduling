# 异构图小框架

这个目录把异构图拆成三层：

1. `schema.py`：只定义图的通用数据结构，不了解调度业务。
2. `builder.py`：把静态 `ClusterProblem` 和动态 Env observation 组合成图。
3. `env_adapter.py`：在 `reset/step` 后调用构图器，并维护图节点与 Env 动作的索引约定。

当前实现是一个可运行示例，不是最终图设计。节点特征、边特征和关系都应根据实验结果调整。

## 一张异构图通常如何构建

先确定节点类型，再确定关系，最后才选择特征。

当前示例包含三种节点：

- `wafer`：每片 wafer 一个节点；
- `module`：每个 LP/PM/LL 一个节点；
- `robot`：每个 TM 一个节点。

示例关系分为两类：

- 静态关系：`robot -can_access-> module`，来自 problem 拓扑；
- 动态关系：`wafer -located_in-> module`、`wafer -held_by-> robot`，来自当前 Env 状态；
- 任务关系：`wafer -can_move_to-> module`，来自 wafer 当前 route step。

图中的索引必须稳定。这里严格复用：

- `wafer` 节点顺序 = `env.wafer_keys`；
- `module` 节点顺序 = `env.module_ids`；
- Env 动作顺序 = `[所有 pick wafer, 所有 place module]`。

因此模型可以为 wafer 节点输出 pick logits，为 module 节点输出 place logits，然后拼接并应用 `graph.action_mask`。

## 图如何与 Env 联系

推荐让 Env 继续拥有状态转移语义，图只做 observation 的另一种表示：

```python
import torch

from cluster_env import ClusterEnv
from hetero_graph import GraphEnvAdapter

graph_env = GraphEnvAdapter(ClusterEnv(problem))
graph, info = graph_env.reset()

# GNN 输出与 [wafer nodes, module nodes] 同序的 logits。
logits = model(graph)
action_mask = torch.as_tensor(graph.action_mask, device=logits.device)
masked_logits = logits.masked_fill(~action_mask, float("-inf"))
action = int(masked_logits.argmax().item())

graph, reward, terminated, truncated, info = graph_env.step(action)
```

数据流是：

```text
ClusterProblem ───────────────┐
                              v
ClusterEnv -> raw observation -> GraphBuilder -> HeteroGraph -> GNN
     ^                                                        |
     └──────────── integer action <- mask + policy logits <────┘
```

`GraphEnvAdapter` 不重新判断动作是否合法，也不计算时间和 reward。动作合法性仍以 `ClusterEnv` 产生的 `action_mask` 为准。调度事件的占用和状态变化仍由 Env 按事件边界处理；图是每个决策时刻的快照。

## 你主要需要填写的位置

在 `ClusterHeteroGraphBuilder` 中修改：

- `_build_nodes`：决定各节点特征；
- `_build_edges`：决定关系和边特征；
- `build`：决定全局特征；
- 如需清洗、驻留时间、LL Pump/Vent 等状态，先让 Env observation 暴露这些动态信息，再在构图器中读取。

常见的下一步是：

- 加入反向边，使消息可以双向传播；
- 对时间特征做统一归一化；
- 将静态拓扑缓存，只在每步更新动态特征和动态边；
- 将本目录的 `HeteroGraph` 转为 PyTorch Geometric `HeteroData` 或 DGL graph；
- 为每种节点类型分别编码，再用 relation-specific message passing 聚合；
- 保留 Env 的 mask，禁止模型采样非法动作。

这个包目前不引入 PyTorch Geometric，避免在图设计尚未稳定时增加重依赖。确定模型库后，只需新增一个转换器，不需要改 Env 或图 schema。
