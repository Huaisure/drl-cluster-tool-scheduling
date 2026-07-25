# Problem Reader

`problem` 是与 `validator` 并列的独立 Python 包，只负责读取、标准化和检查机台结构及 Wafer 工艺流程。

## 公开接口

从JSON文件读取：

```python
from problem import load_problem

problem = load_problem("validator/examples/naura_task1.json")
```

从已经解析的字典读取：

```python
from problem import parse_problem

problem = parse_problem(raw_problem)
```

成功后返回 `ClusterProblem`。文件不存在时保留 `FileNotFoundError`；JSON格式或字段定义不合法时由Pydantic抛出 `ValidationError`。

## 统一初始状态

初始状态直接写在问题JSON中，并只以每片Wafer为事实来源：

```json
{
  "initial_state": {
    "wafers": [
      {
        "route_id": "A",
        "wafer_index": "0-24",
        "step_index": 0,
        "location": {
          "kind": "module",
          "module_id": "LP1"
        },
        "process_end_time": null
      }
    ]
  }
}
```

Wafer也可以位于Robot手臂：

```json
"location": {
  "kind": "robot",
  "robot_id": "TM1",
  "arm_id": "arm0"
}
```

字段含义：

- `(route_id, wafer_index)` 唯一标识一片Wafer；
- 初始状态JSON中的 `wafer_index` 必须是字符串表达式：
  - `"3"` 表示单片；
  - `"1,4,7"` 表示多个离散编号；
  - `"1-5"` 表示首尾均包含的连续区间；
  - `"1,3-5,8"` 可以组合单值与区间；
- 表达式允许逗号和连字符两侧出现空格，但不允许负数、倒序区间、空项或重复编号；
- `step_index=0` 表示初始LP，`1..N`表示Route中的步骤，`N+1`表示最终LP；
- `process_end_time` 表示时间轴从0开始时，当前加工预计结束的时刻；`null`或`0`表示当前没有未完成的加工。

读取时会先把 `wafer_index` 表达式展开为逐片、整数编号的 `WaferInitialState`。
Module occupants和Robot arms不在JSON中重复填写。它们由所有Wafer的 `location`
自动推导。读取阶段会检查Wafer唯一性、引用、工艺步骤、Module容量以及Robot手臂容量。
只有位于Module中的Wafer才能设置大于0的 `process_end_time`；Wafer位于Robot时不能仍处于加工中。

调用 `problem.initial_state.to_snapshot()` 会生成只读的 `InitialSnapshot`：

```python
snapshot.wafers_by_key
snapshot.module_occupants
snapshot.tm_arms
snapshot.tm_positions
```

快照只是初始事实的派生索引，不是第二份输入。ValidatorSuite每次验证只生成一次快照，
再把对应主体的初始切片传给Module、Robot和Wafer Validator。

Robot自身的初始位置是可选的：

```json
"initial_state": {
  "robots": {
    "TM1": {
      "position_module_id": "LP1"
    },
    "TM2": {
      "position_module_id": null
    }
  }
}
```

- 提供Module ID时，Robot从该Module位置开始；
- 值为 `null` 或没有该Robot条目时，初始位置为anywhere；
- anywhere表示第一条动作前不计算移动时间，第一条动作之后必须正常维护位置；
- 初始位置必须是问题中存在且该Robot能够访问的Module。

Robot位置与Robot arms是两类信息：位置来自可选的 `robots` 配置，arms仍只从Wafer位置推导。

## 包边界

`problem` 负责：

- Module、Robot和Route的字段类型；
- Module容量默认由类型派生：LP为25，其余Module为1；问题JSON显式提供
  `capacity` 时会发出覆盖默认值的警告，并按传入的正整数容量处理；
- Robot动作时间在模型内部统一为 `place_time` / `pick_time`，输入兼容旧字段 `load_time` / `unload_time`；
- Wafer初始位置和加工完成时间；
- 从初始事实生成只读 `InitialSnapshot`；
- 单Module与候选Module写法的标准化；
- 时间参数必须非负；
- Robot、Route、LL和Cleaning的引用必须存在；
- 拒绝未定义字段。

`problem` 不负责：

- 读取或回放动作序列；
- 动作执行过程中发生的Module容量、Robot移动、Wafer工艺时间等约束；
- 创建 `ModuleValidator`、`RobotValidator` 或 `WaferValidator`。

这些职责仍属于并列的 `validator` 包。

## 依赖与测试

当前实现依赖 Pydantic 2。运行测试：

```bash
pytest -q problem/tests
```

`ValidatorSuite(problem)` 目前通过公开模型读取配置，但问题解析仍完全留在本包中。后续接口稳定后，可以整体迁出当前仓库单独发布和维护。
