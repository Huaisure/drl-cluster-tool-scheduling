# 一个主体一个实例的 Action Sequence Validator

## 目标

Validator 只判断给定的动作序列是否违反规则，不负责调度搜索，也不判断死锁。

动作序列分别从三个主体观察：

```text
Module：观察引用该 module_id 的动作
Robot：观察引用该 tm_id 的动作
Wafer：观察引用该 (route_id, wafer_index) 的动作
```

一个具体主体对应一个 Validator 实例。例如：

```text
ModuleValidator("PM1") 只持有 PM1 的动作和状态
RobotValidator("TM1")  只持有 TM1 的动作和状态
WaferValidator(("A", 0)) 只持有这一片 Wafer 的动作和状态
```

同一个动作可以同时被 Module、Robot 和 Wafer 三个实例观察，但三个实例检查的是不同方面。

## 当前目录

```text
项目根目录/
├── problem/                  # 读取和检查机台、工艺JSON
│   ├── models.py
│   └── loader.py
└── validator/                # 验证给定的动作序列
    ├── models.py
    ├── module_validator.py
    ├── robot_validator.py
    ├── wafer_validator.py
    ├── pipeline.py
    ├── common/
    │   ├── actions.py
    │   └── intervals.py
    └── docs/
        └── README-zh.md
```

`problem` 与 `validator` 是并列包。前者判断“问题定义能否被正确理解”，后者判断“动作序列在给定问题下是否合法”。`ValidatorSuite` 接收解析完成的 `ClusterProblem`，但不会参与JSON解析。

初始状态也属于 `problem`。Wafer状态的唯一输入是 `problem.initial_state.wafers`：
每片Wafer记录位置、工艺步骤和加工完成时间。Module占用以及Robot手臂占用
均由Wafer位置推导，不作为第二份输入。

初始状态JSON中的 `wafer_index` 使用字符串表达式。例如 `"3"`、`"1-5"`、
`"1,3-5,8"`。问题读取阶段先把表达式展开成逐片整数编号；动作中的Wafer引用
仍然只对应一片Wafer，不支持范围表达式。

`InitialState.to_snapshot()` 将这些事实一次性投影为只读的 `InitialSnapshot`：

```text
wafers_by_key
module_occupants
tm_arms
tm_positions
```

ValidatorSuite每次验证只创建一个快照，再把各主体需要的初始切片传入对应Validator。

Robot还可以通过 `problem.initial_state.robots[robot_id].position_module_id` 提供可选机械位置。缺失或 `null` 表示anywhere：第一条Robot动作前不检查移动时间，第一条动作后正常维护位置并检查后续移动。

## 数据流

```text
问题JSON ─→ problem.load_problem() ─→ ClusterProblem
                                      ↓
                               InitialSnapshot
                                      ↓
全部原始 actions ────────────────→ ValidatorSuite(problem)
                                      ↓
                  按 Module、Robot、Wafer 分发动作和初始切片
       ↓
全部动作只转换一次为 ActionRecord，再按 (start, end, input index) 排序
       ↓
为每个具体主体创建一个 Validator
       ↓
各实例以初始切片为起点，在方法局部变量中回放并验证动作
       ↓
ValidatorSuite 合并所有 ValidationReport
```

例如一个动作同时引用 `PM1`、`TM1` 和 Wafer `("A", 0)`，它会分别进入：

```text
ModuleValidator("PM1").actions
RobotValidator("TM1").actions
WaferValidator(("A", 0)).actions
```

这不是重复验证同一条约束：Module关注占用，Robot关注手臂和移动，Wafer关注工艺路线和时间。

## ValidatorSuite

`ValidatorSuite` 是总控和分发层。创建时必须传入解析完成的问题：

```python
from problem import load_problem
from validator import ValidatorSuite

problem = load_problem("validator/examples/naura_task1.json")
suite = ValidatorSuite(problem)
report = suite.validate(actions)
```

Pipeline随后维护：

```python
suite.module_validators
suite.robot_validators
suite.wafer_validators
```

这三个列表中的每个元素都只对应一个具体主体。重复调用 `suite.validate()` 时会根据新输入重建这些列表，不会累积上一次的实例。

## 项目入口

项目根目录的 `run_validation.py` 提供了完整调用入口：

```bash
python run_validation.py \
  validator/examples/all_actions_recipe.json \
  validator/examples/all_actions_actions.json
```

第一个参数是问题JSON，第二个参数是动作JSON。动作JSON的根必须是数组，每个元素必须是一个动作对象。

命令退出码：

- `0`：验证通过；
- `1`：动作序列违反了一条或多条约束；
- `2`：输入文件无法读取或解析。

也可以在Python代码中复用入口函数：

```python
from run_validation import print_report, validate_action_sequence

report = validate_action_sequence("problem.json", "actions.json")
print_report(report)

if not report.ok:
    for issue in report.issues:
        print(issue.constraint_id, issue.message)
```

`run_validation.py` 只负责读取、调用和展示结果。问题定义仍由 `problem` 负责，约束仍由各个Validator负责。

创建规则是：

- 问题中定义的每个Module都会创建一个 `ModuleValidator`，即使它没有动作；
- 问题中定义的每个Robot都会创建一个 `RobotValidator`，即使它没有动作；
- 每个 `problem.initial_state.wafers` 中的具体Wafer都会创建一个 `WaferValidator`，动作不能凭空创建新Wafer。

分组规则目前是：

- Module 使用 `module_id`，兼容 `pm_id`；
- Robot 使用 `tm_id`；
- Wafer 使用 `(route_id, wafer_index)`，兼容用 `count` 表示 Wafer 序号。
- 动作类型在入口统一为 `pick` / `place`，同时兼容输入中的 `unload` / `load`。

动作引用问题中不存在的Module、Robot或Wafer时，Pipeline会立即拒绝；初始状态自身的引用则在 `problem` 读取阶段检查。

## ModuleValidator

创建方式：

```python
ModuleValidator(
    module_id="PM1",
    config=problem.Modules["PM1"],
    actions=pm1_actions,
    initial_occupants={("A", 0)},
)
```

实例直接持有：

```python
validator.module_id
validator.config
validator.actions
validator.initial_occupants
```

`validator.initial_occupants` 是快照中该Module初始占用的不可变切片。

所有Module统一使用 `ModuleValidator`。`Module.capacity` 默认由Module类型派生：
LP为25，其他Module为1。问题JSON显式提供正整数 `capacity` 时会发出警告，
并按该容量覆盖类型默认值。

当前 `ModuleValidator.validate()` 只实现容量约束：

- Place（JSON中的 `load`）在 `Place.start` 占用Module容量；
- Pick（JSON中的 `unload`）在 `Pick.end` 释放Module容量；
- 任意事件时刻的Wafer数量不能超过 `validator.capacity`；
- 同一时刻发生 `Pick.end` 和 `Place.start` 时，先释放再占用。

容量失败使用 `module.capacity`，报告触发Place的动作序号、事件时间、容量和占用快照。

基础 ModuleValidator 暂不检查动作中的Wafer是否确实存在、重复Place、Cleaning、Pump或Vent；这些属于后续的占用一致性或派生Module规则。

## RobotValidator

一个 `RobotValidator` 只负责一台具体Robot，并通过 `validator.config` 持有该Robot的
`ClusterCell` 配置。`validator.initial_arms` 来自快照中的TM手臂占用，
`validator.initial_position_module_id` 来自可选Robot初始位置；其中 `None`
表示anywhere。

当前实现四类规则：

- `robot.action_overlap`：同一Robot的动作区间 `[start, end)` 不能重叠；首尾相接允许；
- `robot.capacity`：single-arm容量为1，dual-arm容量为2，Robot持有的Wafer总数不能超过容量；
- `robot.movement_time`：相邻动作位于不同Module时，两个动作之间必须留出足够移动时间；
- `robot.action_duration`：load和unload动作必须满足各自的最小时长。

Robot容量按以下事件回放：

- Pick（JSON中的 `unload`）在 `Pick.start` 占用Robot容量；
- Place（JSON中的 `load`）在 `Place.end` 释放Robot容量；
- 同一时刻发生 `Place.end` 和 `Pick.start` 时，先释放再占用。

初始容量来自 `validator.initial_arms`。校验过程使用局部集合回放，不会修改初始切片。

移动时间使用：

```text
next.start - previous.end >= robot.travel_time(previous_module, next_module)
```

如果两个动作位于同一Module，则不需要移动时间。等号边界合法。

如果提供了 `initial_position_module_id`，Validator把它看作Robot在时间0的位置，并检查到第一条动作的移动：

```text
first.start >= robot.travel_time(initial_module, first_module)
```

初始位置为 `None`（anywhere）时不检查第一段移动；第一条动作发生后，Robot位置确定，后续动作正常检查。

动作最小时长在内部统一使用Place/Pick名称：

```text
place.end - place.start >= robot.place_time
pick.end - pick.start >= robot.pick_time
```

问题JSON同时兼容旧字段 `load_time` / `unload_time`，但读取后统一为
`place_time` / `pick_time`。等于配置时长或超过配置时长都合法。
其他动作类型暂不参与这条规则。

当前暂不检查具体 `arm_id` 是否空闲或持有正确Wafer，也不检查Module reachability和双臂几何关系。

## WaferValidator

一个 `WaferValidator` 只负责一片具体Wafer，并持有对应的 `route`、全局
`just_in_time` 配置以及快照中的 `initial_wafer`。Validator不会再创建语义重复的
`WaferState`。

当前已经实现两类规则：

- `wafer.process_order`：从初始位置开始，Pick与Place必须交替；Pick必须来自Wafer当前所在Module；Place必须进入Route的下一步骤及其候选Module；完成全部步骤后只能Place回LP；
- `wafer.interval_overlap`：同一Wafer的Pick、Place和PM加工区间不得重叠。

这里使用半开区间 `[start, end)`。PM加工区间从Place结束开始：

```text
Place:   [place.start, place.end)
Process: [place.end, place.end + process_time)
Pick:    [pick.start, pick.end)
```

因此 `pick.start == place.end + process_time` 合法，早于这个时刻则会和加工区间重叠。

如果Wafer初始位于PM并且 `process_end_time > 0`，Validator会把
`[0, process_end_time)` 作为初始加工区间。位于Robot上的Wafer不能携带未完成的加工时间。

当前有意不检查：

- JIT / residency time；
- LL的加工时间、Pump和Vent；
- 动作序列是否覆盖整条Route。前缀序列可以合法，只有已经出现的动作会被检查。

虽然LL的时间规则尚未实现，但LL上的Pick/Place仍参与位置推进和动作区间互斥。

## 公共代码边界

`common/actions.py` 负责在入口把原始动作一次性规范化为 `ActionRecord`，
再按指定主体分组和排序。后续Validator共享同一批ActionRecord，不再读取原始动作字典。

`common/intervals.py` 只提供不属于任何单一主体的机械时间函数：

- `intervals_overlap()`：判断两个 `[start, end)` 区间是否重叠；
- `within_closed_window()`：判断时间是否位于闭区间 `[lower, upper]`。

业务规则不要放入 `common/` 或 `pipeline.py`。PM capacity 属于 ModuleValidator，Robot 移动时间属于 RobotValidator，工艺时间属于 WaferValidator。

## 当前状态

当前已完成实例边界、动作分发、状态所有权和报告合并，并实现了Module容量、Robot动作互斥与容量、Wafer工艺顺序以及Wafer时间区间互斥。后续仍按“一次实现一个主体的一条规则，并同步增加回归测试”的方式推进。
