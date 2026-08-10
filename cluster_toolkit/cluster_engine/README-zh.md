# Cluster Engine

`cluster_engine` 是供RL环境使用的最小有状态离散事件内核。它读取
`problem.ClusterProblem`，维护当前动态状态，并返回经过物理约束和IO投片优先级过滤的
语义动作。

## 核心接口

```python
from cluster_engine import ADVANCE, ClusterEngine, PickAction, PlaceAction
from problem import load_problem

engine = ClusterEngine(load_problem("problem.json"))
state = engine.reset()

actions = engine.available_actions()
record = engine.step(actions[0])

if ADVANCE in engine.available_actions():
    engine.step(ADVANCE)
```

Engine自己持有`state`并原地推进。多个并行RL环境应各自创建一个Engine实例。

`load_lock_observation(module_id)`返回Conversion LL的只读模型观测，包括Pump/Vent
时间、空LL从上次`Pick.end`开始的另一侧转换进度，以及占用LL从`Place.end`开始的
出口侧和转换进度。所有进度都限制在`[0, 1]`，不暴露绝对时间戳。

## 动作

Engine只接受三种RL动作：

- `PickAction(robot_id, wafer_key)`；
- `PlaceAction(wafer_key, target_module_id)`；
- `AdvanceAction()`，共享常量为`ADVANCE`。

双臂只表示Robot持片容量为2，不暴露`arm_id`。Place显式选择wafer，持有该wafer的
Robot由状态唯一推导。

Pick/Place自动包含Robot移动。派发时立即锁定Robot；Pick预约wafer，Place预约目标容量。
只有Advance会把系统时间推进到下一个Travel、Pick/Place、工艺完成或LL可达性事件边界。

Pick/Place返回`DispatchRecord`，包含动作的开始和结束时间。Advance返回`None`。Engine不
保存完整动作历史，由Env决定是否收集这些记录。

## 动作可行性

`available_actions()`返回语义动作对象，不生成NumPy扁平mask。Env负责维护动作编号并将
动作集合投影为RL action mask。

第一版检查：

- Wafer位置、Route顺序和工艺完成时间；
- Module容量及Place预约；
- Robot可达性、移动、忙碌和持片容量；
- Conversion LL侧别和自动转换完成时间；
- 仅作用于IO首次Pick的全局最小`priority`组；优先级在Pick派发时释放；
- 不对设备内部Wafer执行FIFO或priority筛选。

Pick不前瞻检查下一目标容量。RL可以先持有暂时无处可放的wafer，并承担潜在死锁。

## Conversion LL

Pump/Vent不是RL动作。Wafer进入LL后，Engine根据下一Route节点唯一推导出口侧，并自动
加入相应转换时间。LL变空后，允许上次取出侧立即Place；另一侧需等待Pump/Vent时间。

所有LL之后的同一Route候选模块必须位于同一侧，否则Engine拒绝该转换。

## 明确不负责

第一版不包含：

- Pump/Vent/Clean决策动作；
- PM清洗、JIT和residency；
- 调度序列replay与详细验证报告；
- 状态JSON、回滚、copy-on-write；
- Gym observation、reward和扁平action mask；
- 双臂几何、真实LP/slot、Valve和PM工艺能力。

Validator目前验证Pick/Place的工艺顺序、时间与资源约束，但尚未验证IO首次Pick的priority
派发顺序；这是已知的首版校验缺口。

完整序列验证继续由`validator`负责；Gym和模型接口由外层`ClusterEnv`负责。对于包含自动
LL转换的序列，旧Validator仍要求显式Pump/Vent记录，需要由外围适配器补齐后再验证。
