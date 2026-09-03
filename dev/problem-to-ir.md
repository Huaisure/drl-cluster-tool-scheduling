# 现有问题到 Constraint IR：首版合同

## 1. 这次交付什么

入口是 `compile_problem(ClusterProblem, TimeDomain) -> ConstraintIRV1`，实现位于 `cluster_toolkit/constraint_ir/problem_compiler.py`。它不调用旧 Engine，只读取已校验的业务输入并生成通用声明。转换结果可以序列化、重新读取、生成动态候选、提交、逐事件推进、中途恢复和独立审计。

这一步打通的是“真实问题 → 通用语义 → 可验证调度”，不是训练迁移，也不是新求解算法。Kernel 没有增加 PM、Pick、清洗或 LL 的执行分支。后续新增的[通用 IR 训练首版](./ir-training.md)已独立接通环境、模型输入和 PPO；本文继续描述 Compiler 本身。

## 2. 首版范围

| 支持 | 暂不支持，留待迭代 |
|---|---|
| 一个单臂或双臂机械手 | 多机械手、跨机械手交接 |
| IO/LP、物理容量为 1 的 PM | LL、AL、BUFFER、多槽 PM |
| 有限路线、候选 PM、重复访问同一 PM | 运行中初态、动态新增晶圆 |
| 晶圆全部从 IO/LP 的第 0 步开始、优先级相同 | 混合优先级 |
| 显式正数取片、放片、加工时长 | 缺失加工时长、零时长取放/加工 |
| 零或正数移动时间、已知/未知初始位置 | JIT、局部 residency、清洗配置 |
| 指定返回站点、schema v2 工艺能力校验 | 自动生成 Exchange 复合候选 |

已列出的不支持输入配置返回 `UNSUPPORTED_FEATURE` 和字段路径，不静默删掉约束。即使 `cleaning`/`just_in_time` 配置没有实际启用规则，首版也保守拒绝非 `None` 的配置。Exchange 不是源输入字段：当前可以选择普通 Pick/Place 完成 swap，但不会自动生成复合 Exchange 候选。模块能力由原 Problem 的 schema 校验，机械手不可达的候选不进入绑定表；某次访问没有任何可达候选则拒绝。

源时间单位为秒。`TimeDomain(unit="second", ticks_per_unit=1000)` 表示每秒 1000 个 Tick；不能精确表示的时长报 `TIME_PRECISION_LOSS`，不取整。非法源对象会重新校验并拒绝；最终站点容量不足等静态矛盾也不能生成有效 IR。

同优先级同 recipe 的 FIFO 暂不作为硬约束：这里对齐基础 Engine，而不是旧 RL 环境额外的 ActionSafetyFilter。没有最晚开始时间时继续保持 `None`。

## 3. 业务如何变成通用事实

| 原问题 | 编译后的通用表达 |
|---|---|
| PM/IO/LP 容量 | Resource（容量资源）＋ Lease（对象持有/占位关系） |
| 每个机械臂可持一片 | 每臂一个容量为 1 的资源 |
| 两臂不能同时动作 | 两臂的移动、取放区间共用一个容量为 1 的 motion 资源 |
| 机械手当前位置 | 一个枚举 StateCell（状态值）；决定本次移动时长 |
| 某片晶圆进行到哪一步 | 每片一个整数进度 StateCell |
| 候选站点、手臂、晶圆 | Typed Binding Domain（有类型的有限参数绑定表） |
| 现在能不能做 | Guard（提交前条件）：进度、位置、指定持有关系 |
| 同一工序选一个方案 | Choice Scope（互斥选择标识）：晶圆＋阶段 |
| 执行动作 | Interval（持续区间）＋资源使用＋起止边界的 Effect（状态修改） |
| 放入后自动加工 | Place.end 触发普通 Automatic Activity（自动活动） |
| 全部完成并返回 | TerminalStateSpec：最终进度＋精确返回占位集合 |

只有 Adapter 理解业务名称。IR 中的 `Pick`、`Process` 等 `audit_kind` 用于检查和解释，不是 Kernel 分支；它们和可读业务 ID 不应作为模型语义标签。后续新增的 `cluster_rl/ir/graph.py` 已独立实现匿名通用模型投影，丢弃审计标签，ID 只用于连接。

## 4. 一片晶圆的例子

路线为 `LP → PM1 → PM1 → LP`，同一个 PM 上有两次不同的访问：

```text
进度 0：源站待取
   Pick.end → 1
   Place.end → 自动活动开始；进度仍为 1
   活动结束 → 2
   Pick.end → 3
   Place.end → 第二次自动活动开始；进度仍为 3
   活动结束 → 4
   Pick.end → 5
   返回 Place.end → 6
```

进度只有在加工结束后才进入可取片阶段，因此不能提前 Pick。重复访问使用不同阶段 scope，不会因“以前已经在 PM1 做过”而被错误屏蔽。另一片晶圆复用模板，用不同绑定行表达。

持片位置以 Lease 为唯一事实来源，不另存 `wafer.location`。对齐旧 Engine 的交接边界：Pick.start 获取手臂占位、Pick.end 释放源站占位；Place.start 获取目标占位、Place.end 释放手臂占位。交接区间有意同时保留两端记录，不等于晶圆被复制。通用跨资源物理位置不变量仍未完成。

加工区间占用独立 activity 资源，晶圆仍持有 module Lease，避免把同一片晶圆在同一模块重复计算容量。加工和其他腔室加工、机械手动作可并发；共享机械手动作仍互斥。

## 5. 真正的完成条件

`require_terminal=True` 不再简单要求所有 Lease 消失。

- 声明的终态状态值必须满足；未声明的状态值（如最后机械手位置）不受限制。
- 最终 Lease 必须与声明集合精确相等，包括资源、对象及占用量。
- 不得仍有运行区间或未满足的 Obligation（义务/待办）。
- Session 审计还要求没有尚未执行的已承诺边界。

如果未声明 `terminal_state`，保留原来的“所有 Lease 必须关闭”规则。首版 Compiler 会声明每片晶圆的最终进度和指定返回站点占位。仅仅 `frame()` 没有候选或 `advance_next()` 返回 `None`，都不能当作完成。

## 6. 使用方法

```bash
.venv/bin/python -m cluster_toolkit.run_ir_compilation \
  examples/scenarios/long_route_1w.json \
  --output /tmp/long-route.ir.json --ticks-per-second 1000
```

不指定 `--output` 时输出到标准输出。输出路径若已存在会拒绝覆盖；编译失败不创建产物。

```python
from cluster_toolkit.problem import load_problem
from cluster_toolkit.constraint_ir import (
    ReferenceKernel, ReferenceValidator, TimeDomain, compile_problem,
)

ir = compile_problem(load_problem("examples/scenarios/long_route_1w.json"),
                     TimeDomain(unit="second", ticks_per_unit=1000))
session = ReferenceKernel.start(ir)
frame = session.frame()
# 策略选择一个候选；示例只提交第一步，不承诺该策略最终可完成。
session.commit(frame.frame_token, (frame.intents[0].candidate_key,))
session.advance_next()
report = ReferenceValidator.validate_session(ir, session.snapshot())
```

## 7. 验收与后续边界

测试文件为 `cluster_toolkit/constraint_ir/tests/test_problem_compiler.py`，覆盖真实 JSON 转换、规范化往返、完整路线与旧 Engine 的取放边界对齐、独立 Session 终态审计，以及重复访问、候选互斥、双臂互斥与预取、加工并发、schema v2、不可达候选、不同返回站点和非法配置。

完整调度使用逐片串行见证策略，只证明存在合法完整轨迹，不是最优策略。整数秒案例另经旧 Validator 验证；小数时间由精确 Tick IR 审计及逐次 Engine 时间对比验收，旧 Validator 的浮点严格相等问题未在本次修改。

本轮验证：全仓库 `.venv/bin/python -m pytest -q --tb=short` 为 **506 passed**（约 8 分 26 秒）；IR 专项排除三个耗时真实场景后为 **170 passed**。真实场景为 `long_route_1w`、`mixed_3pm_20w`、`mixed_5pm_24w`，均执行到完整终态。全量有 71 条警告，来自 torch 弃用提示及原 Problem 的显式 capacity 提示；编译时重新校验输入也会触发后者。命令行导出/重新加载、compileall 和空白检查通过。

参考执行器与独立审计器会反复回放前缀；大一点的多晶圆案例运行明显较慢。后续新增的 IR 训练环境已接通小规模训练、等待/失败分类和耗时奖励，独立审计仅用于评估；高吞吐执行、索引化候选及声明式 IR Objective 仍未实现。不能把小规模可训练性当作大规模训练性能保证。

参考协议升级为 `1.2-reference`，语义版本为 `1.2`。除终态字段外，还修正了绑定表序列化：列与对应值必须一起重排，不能各自排序，否则可能把晶圆绑定到错误资源。旧参考 IR/快照须从源数据重新生成；旧 Problem、Engine、RL 输入及 checkpoint 不变。

下一轮优先用更多真实输入确认范围，然后补最小的 residency/JIT 编译及逐次义务身份，再增加 LL/清洗。每次增加编译规则都需要真实轨迹验收，不应只增加业务名称或 Kernel 特判。
