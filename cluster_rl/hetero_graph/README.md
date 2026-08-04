# 异构图 observation

这个目录负责把 `ClusterEnv` 的决策状态转换为与具体 GNN 库无关的异构图：

1. `schema.py` 定义节点、关系和完整图快照的数据结构；
2. `feature_schema.py` 集中定义所有节点的特征顺序和语义；
3. `builder.py` 将 `ClusterProblem` 和 Env observation 转换为图；
4. `env_adapter.py` 在 `reset/step` 后构图，并转换图节点索引与 Env 动作。

## Env 接口

构图器使用以下 observation 字段：

- `wafer_loc: [W]`：`[0, M)` 表示 module，`[M, M + R)` 表示 robot；
- `wafer_step: [W]`：wafer 已完成的 route step；
- `process_remaining: [W]`：当前加工剩余时间；
- `robot_loc: [R]`：robot 所在 module，`M` 表示初始位置未知；
- `robot_holding: [R, K]`：有序持片列表，`W` 表示空位；
- `robot_phase: [R]`：idle、前往 Pick、Picking、前往 Place 或 Placing；
- `robot_operation_wafer: [R]`：pending operation 的 wafer，`W` 表示无；
- `robot_operation_module: [R]`：Pick 源或 Place 目标，`M` 表示无；
- `time_to_operation_start/end: [R]`：pending operation 到 start/end 的相对时间；
- 环境的 `action_mask: [(W + M) * R + 1]`：按实体优先扁平编码，最后一位是 ADVANCE。
- 图中的 `action_mask: [W + M, R]`：builder 拆出的运输动作；`can_advance` 单独表示 ADVANCE。

其中 `W`、`M`、`R` 分别是 wafer、module、robot 数量。运输动作编码为：

```python
action = entity_index * R + robot_index
```

`entity_index < W` 表示 pick wafer，否则表示 place 到
`module_index = entity_index - W`。`action == (W + M) * R` 表示 ADVANCE。

Env observation 和事件计算中的时间单位保持为秒。构图时，所有时间类节点特征统一除以
`TIME_SCALE_SECONDS = 100.0`，包括加工、驻留、Pick、Place、移动和 pending
operation 的相对剩余时间。不同类型的时间共用同一个尺度，以保留它们之间的物理比例。

## 图结构

节点类型为 `global`、`wafer`、`route_step`、`module` 和 `robot`。

每条 route 的实际 visit 对应一个 `route_step` 节点，另外增加一个加工时间为 0 的最终返回 LP 步骤。这样初始 wafer 连接第一个 next step，完成加工路线后仍能通过最终步骤连接 LP。

每种关系都显式包含正向边和反向边，使 HGT 中各节点类型都能接收相关实体的消息：

- `wafer -located_in-> module` / `module -contains-> wafer`；
- `wafer -held_by-> robot` / `robot -holds-> wafer`；
- `wafer -at_step-> route_step` / `route_step -current_for-> wafer`；
- `wafer -next_step-> route_step` / `route_step -next_for-> wafer`；
- `route_step -can_run_on-> module` / `module -supports_step-> route_step`；
- `route_step -precedes-> route_step` / `route_step -follows-> route_step`；
- `robot -can_access-> module` / `module -accessible_by-> robot`；
- `robot -located_at-> module` / `module -has_robot-> robot`。
- `robot -operates_on-> wafer` / `wafer -operation_of-> robot`；
- `robot -operation_at-> module` / `module -has_operation-> robot`。

`held_by/holds` 根据 `robot_holding` 构建，因此在 Pick 执行期间，wafer
可以同时连接源 module 和持有它的 robot，准确表达 `[Pick.start, Pick.end)`。

`global` 与其他所有节点类型双向连接。它保存 wafer 完成比例、步骤完成比例和剩余加工量比例，同时接收实体节点信息供 value head 聚合。全局特征不包含绝对时间，保证相同调度状态的图表示不随时间原点变化。

当前按标准 HGT 设计，边只表达关系类型和连接关系，不保存逐边特征。加工时间和驻留时间保存在 `route_step` 节点中。

节点顺序始终复用 `env.wafer_keys`、`env.module_ids` 和按 ID 排序的 robot，保证模型输出可以稳定映射回 Env 动作。

节点特征的增删、排序和含义统一维护在 `feature_schema.py`。特征顺序是模型输入协议的一部分，修改后需要同步更新 `builder.py` 中对应节点的特征值构造。
