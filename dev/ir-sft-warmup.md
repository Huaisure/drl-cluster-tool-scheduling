# IR 模型 GA / Branch Search 监督 warm-up 实验

## 数据来源与整理

- 原始运行：`datasets/raw/run-ga-branch-1000-20260831-113524`。
- 运行计划含 1000 个实例、7000 个预期尝试；现场已有 3201 个终态尝试。
- 官方 reducer 生成 1000 个 `solution_index.json`，其中 266 个实例 usable，0 个
  quarantined。
- `datasets/ir_sft_ga_branch_v1` 是首轮兼容性实验：233 个接收、33 个因 0 秒
  BUFFER/过渡处理被旧编译器拒绝。该结果促成了编译器的非负 route duration
  适配，作为失败实验保留。
- `datasets/ir_sft_ga_branch_v2` 是修正后的正式数据：266/266 个 usable 实例
  全部接收，无拒绝；train/validation/test 为 210/28/28，seed=1701，按 wafer
  scale、Cell 数、胜出 expert 分层且实例不交叉。
- 最佳合法标签中 GA 212 个、Branch Search 54 个。每个 manifest 条目保留最佳
  expert、两类 solver 各自最佳 makespan、源动作 SHA-256、规模与拓扑元数据。

编译适配覆盖任意数量的 single/dual-arm Robot、PM/AL/BUFFER 中间站、跨 Cell
handoff 和 0 秒中间站处理；Load Lock pressure transition、JIT、cleaning 仍需其
各自明确 IR 语义，未被静默丢弃。

## 数据标签合同

源 solver 轨迹已经 legacy Validator 标记为 VALID。整理阶段还会重新计算压缩
动作内容 SHA-256，并验证每个 `(action type, phase, module, robot, wafer)` 都有
对应的 IR operator 与 binding row。训练时按 solver 决策顺序在 Reference
Session 重放；动作暂不可用时只允许推进到已声明的下一个事件，并把 Wait 作为
监督标签。完整回放必须达到 IR terminal state 且通过独立 Session audit。

## 已完成实验

### `runs/ir_sft_smoke_v1`

- 数据：1 train / 1 validation / 1 test，均为独立 split 中的最小实例。
- 模型：width=16、layers=2、batch=4、AdamW lr=1e-3，CPU 8 threads。
- 训练：109 个多候选监督状态、31 个 Wait、28 次 optimizer step；验证 expert
  top-1 accuracy 0.2317。
- 闭环验证：1/1 success，0 deadlock，makespan 7323，GA/Branch Search 比值
  1.9175。
- 闭环测试：1/1 success，0 deadlock，makespan 15420，GA 比值 2.5176。
- 结论：端到端数据、反向传播、兼容 checkpoint 和独立闭环审计均成立；单实例
  训练远未达到接近 GA 的质量，因此只作为冒烟证据，不作为最终模型结论。

### `runs/ir_sft_warmup_v1`

计划 20 train / 5 validation / 5 test，最大 25 wafers；width=32、layers=3、
batch=8、AdamW lr=5e-4、CPU 8 threads。首例耗时 253 秒，预计总耗时过高，
主动中止；目录保留配置、首例 `cases.jsonl` 与可加载 `last.pt`。该实验没有完整
结果，不参与模型选择。

### `runs/ir_sft_warmup_v2`

改为 width=16、layers=2 并从 smoke checkpoint 继续，仍计划 20 个训练实例。
首例耗时 228 秒后中止，保存 `last.pt`；随后以该权重开始有界的 v3 实验。

### `runs/ir_sft_warmup_v3`

- 数据：5 train / 3 validation / 3 test，最大 25 wafers，三个 split 完全隔离。
- 初始化：v2 `last.pt`；width=16、layers=2、batch=8、AdamW lr=3e-4。
- 训练：696 个多候选监督状态、89 次 optimizer step；held-out expert top-1
  accuracy 0.4436。
- 原始 greedy 闭环：validation 0/3、test 0/3，全部在 3--4 个决策后 deadlock。
- 结论：更高的 teacher-forced accuracy 没有转化为闭环能力。轨迹诊断显示模型在
  一个 wafer 进入单候选 PM 后继续把同 route wafer 预取到最后一个空 hand；PM
  完成时无空 hand 可卸片，形成确定死锁。因此 v3 不能作为最终 warm-up 模型。

后续实验加入一阶 IR safety shield：按网络 logit 排序，对候选在克隆的 Session
Snapshot 上试提交；会立即落入 deadlock/deadline/budget failure 的动作被拒绝。
报告必须同时保留 raw greedy 与 shielded 两组指标，避免把安全层效果冒充纯模型
效果。

### safety shield 与 `runs/ir_sft_warmup_v4_graph2`

- 一阶 shield 在 validation 首例把 raw 的 4 步 deadlock 修正为完整成功：83 个
  决策、126 个危险候选拒绝、makespan=3819，与该例 GA/Branch Search 最佳值
  相同。
- 同一 shield 在 test 首例运行 89 步后仍进入深层 trap：当前所有候选都会立即
  deadlock，最终失败；说明一阶安全不能保证全局可完成。
- 可回溯版本在该 test 上运行 15 分钟仍未终止，因存在指数搜索风险而中止。它只
  保留为审计原型，不是最终部署策略。
- 针对可见性，graph protocol 升级为 `ir-graph-2`：候选的显式 state delta 直接
  连接到由该新状态启用的 successor plans，使两层网络能读取下一操作的资源与
  目标。这是通用 IR dataflow，未读取业务 ID。
- graph2 v4 从头训练 10 个实例：1468 个多候选监督状态、187 次 optimizer step；
  held-out top-1=0.3638。raw validation 0/3、test 0/3，均 deadlock。由此可知单轮
  单标签行为克隆仍不足；v5 从 v4 best checkpoint 继续第二轮训练。

### `runs/ir_sft_warmup_v5_graph2_epoch2`

- 从 v4 best 继续在同 10 个样本训练第二轮：1468 个监督状态、187 次新增
  optimizer step，在线 top-1=0.5817，held-out top-1=0.3677。
- raw validation 0/3、test 0/3，仍全部 deadlock。重复单标签 BC 没有转化为
  闭环改进，因此停止继续堆叠 epoch。
- 复盘发现旧的 `limit` 逻辑先按 wafer 数截断，10 个训练样本全部为 GA 胜出；
  有界实验没有保留完整 manifest 的 expert/cell 分层。随后将 loader 改为按
  `(expert_solver, topology_cell_count)` 轮询取样。

### `runs/ir_sft_warmup_v6_balanced`

- 平衡后的 12 个样本覆盖 GA 10、Branch Search 2，cell 1/2/3 为 5/5/2；在
  `wafer_count <= 25` 下没有 Branch Search 胜出的 3-cell 样本。
- 从 v5 best 初始化；完成 7/12 个 case、1287 个监督选择、163 次 optimizer
  step。已完成部分真实包含一个 15-wafer、2-cell Branch Search 胜出轨迹。
- 第 8 个 18-wafer、3-cell case 在旧实现中运行超过 30 分钟仍未结束，实验在
  case 边界前中止；`last.pt` 保留前 7 个完整 case。随后缓存静态 binding 展开与
  problem hash，并让候选预览、commit、advance 从可信当前 snapshot 增量执行。
  Constraint IR 全套测试 174/174 通过，套件耗时从 141 秒降至 82 秒；该极端
  case 的无图严格回放最终仍需 320.7 秒。
- `last.pt` 的单例 teacher-forced top-1=0.3537；raw validation/test 各首例均在
  4 个决策后 deadlock，完成率为 0，不能作为可部署模型。
- 进程内可信 `ReferenceSession.fork()` 取代 shield 每个分支的序列化、全量 audit
  restore；validation 首例 shield 耗时由约 2 分钟降至 19.9 秒。首个 validation
  与 test 均成功，test makespan 仅比 GA 慢 2.55%、比 Branch Search 慢 2.61%。
- 扩大到平衡的 3 validation + 3 test 后，两边均为 2/3 成功、1/3 deadlock。
  成功 test 的 paired ratio 为 GA 1.050、Branch Search 1.021；validation 为
  GA 1.104、Branch Search 1.182。失败样本把回退上限从 25 提高到 100 仍无法
  完成。
- 结论：平衡 SFT + shield 已在成功样本上接近 GA，但没有优于 Branch Search，
  且 33% held-out deadlock 明显不符合“几乎不死锁”。下一阶段应只用 train split
  做安全负例/DAgger 或学习型可完成性 critic；不得把当前 checkpoint 标为达标。

### `runs/ir_sft_warmup_v7_safety_distill`

- 从 v6 `last.pt` 出发，只在 4 个 train 实例收集一阶 shield rollout；2 条成功、
  2 条 deadlock 后整条丢弃。成功轨迹共 494 个状态，训练 62 个 batch。
- raw validation 为 2/3 成功、1/3 deadlock；raw test 为 3/3 成功、0 deadlock，
  证明 shield 轨迹确实能把安全行为蒸馏进网络。
- 代价是质量明显退化：raw test paired ratio 为 GA 1.836、Branch Search 1.448；
  原因是最长成功 train 轨迹有 372 个决策、makespan 27612，本身安全但很慢。
  因而 v7 不满足 makespan 验收。

### `runs/ir_sft_warmup_v8_corrective_distill`

- 为避免覆盖专家排序，只保留“模型最高分动作被 shield 拒绝”的纠正状态；限制
  收集样本不超过 12 wafers，并把学习率降为 `3e-5`。
- 4 条 train rollout 中 2 条成功，只产生 86 个纠正状态、11 次 optimizer step。
- raw validation/test 均为 0/3；shield 结果回到 v6 的 2/3 成功、1/3 deadlock，
  test 成功轨迹相对 GA 1.050、Branch Search 1.021。说明这一剂量不足以改变关键
  动作排序。
- v7/v8 给出当前 Pareto 边界：强安全蒸馏可消除这 3 个 test 样本的 deadlock，
  但显著损害 makespan；弱纠正保留 makespan，却没有消除 deadlock。下一步需要
  联合 expert imitation、安全 margin/critic 与调度质量目标，而不是继续单一 CE。

### `runs/ir_sft_warmup_v9_anchored_distill`

- 使用与 v8 完全相同的 86 个 train-only 纠正状态，把学习率恢复到 `3e-4`，并对
  原 v6 策略加入权重 5 的 KL anchor；共 11 次更新。
- raw validation/test 均仅 1/3 成功；test 成功样本相对 GA 2.065、Branch Search
  1.954。shield 仍为 2/3 成功，但 test paired ratio 退化到 GA 1.553、Branch
  Search 1.498。
- 结论：简单 KL anchor 没有解决安全纠正与调度质量的冲突，v9 同样不达标。
  后续不应在这 3 个留出样本上继续调参，以免形成验证/测试过拟合；应扩大
  train-only DAgger 数据并单独训练可完成性 value/critic，再一次性评估完整 test。

### 完成 v6、`runs/ir_sft_warmup_v10_more_expert` 与 v11

- 增加 case-boundary 断点续训后，从 v6 已保存的第 7 个 case 恢复模型、AdamW、
  监督计数与确定性样本顺序，完成剩余 5 个样本。完整 v6 为 12/12 case、3143
  个监督选择、396 次优化，held-out top-1=0.4212；raw test 3/4 成功、25%
  deadlock，成功轨迹相对 GA 1.836、Branch Search 1.448。
- v10 从完整 v6 继续训练所有可用的 `<=12` wafer train 实例（实际 14 个），
  2275 个监督状态、290 次优化；小规模 validation/test 均 0% 成功。更多同类
  单标签 BC 再次破坏闭环安全，没有改善质量。
- 复盘源动作的 `start` 字段后，将同一最早 start 组中当前合法的动作视为等价
  多标签，使用 `-log(sum p(action))` 损失；未来 start 的动作不会提前进入标签。
  缺少 start 的手工/兼容输入仍按原序号形成单标签。
- 两个 3-cell train 轨迹分别含 29/14 个并发 start 组，说明该修正确实覆盖真实
  多 robot 数据。v11 从 v5 重跑同一 12-case 平衡集合，3144 个监督状态、397
  次优化，held-out top-1=0.4211；raw test 仍为 3/4、25% deadlock，paired
  ratio 与 v6 相同。多标签修正消除了伪冲突，但不足以解决全局可完成性与质量。

### `runs/ir_sft_warmup_v12_expert_margin` 与 v13

- 在 train-only 专家状态上检查冻结模型 top-1；仅当它不属于专家集合且一步试探
  会直接进入 deadlock/deadline/budget failure 时，生成 `专家集合 > 危险动作`
  的 pairwise margin。安全的非专家动作不作为负例。
- 4 个 10-wafer train case 共检查 612 个专家状态；440 个 top-1 不是专家动作，
  但只有 6 个是一步危险动作，说明标签很保守。
- v12 对这 6 个样本做 3 次带 KL anchor 的更新，raw/shield test 均与 v11 基本
  相同：3/4 成功、25% deadlock，成功 makespan 仍为 GA 1.836、Branch 1.448。
- v13 将相同样本强化到 50 次更新并减弱 anchor，raw test 退化为 1/4 成功、
  75% deadlock；shield 也只有 2/4。直接改变共享 actor 来拟合少量局部负例会
  扰动全局排序，不能解决覆盖问题。
- 下一步需要冻结专家 actor，新增独立 action-safety/value head；用 train rollout
  的成功状态和失败前缀训练可完成性，再在推理时作为 reranker。该 head 必须与
  actor makespan 排序分离，避免复现 v7/v9/v13 的质量破坏。

### v14--v17 独立 safety head

- `cluster_rl.ir.safety_head` 冻结 v11 actor，复制一套独立图网络作为动作安全头；
  专家动作只来自 train split 的已验证回放，失败负例只来自冻结 actor 的 train
  rollout 尾部。推理时安全头仅过滤候选，保留候选仍按冻结 actor logit 排序。
- v14/v15 暴露了工程问题：保留数百个完整 IRGraph 会占用大量内存，且完整专家
  回放没有必要重复数据构建阶段的审计。两个运行在首例完成前主动停止，只保留
  配置，不参与模型比较。采集器随后改为定长失败尾部，并在每例收满专家正例配额
  后停止回放；actor 采集另设 200 决策上限。
- v16 在 4 个分层 train case 上得到 16 个专家正例、23 个失败尾部负例，平衡后
  46 个样本、48 次更新，BCE 从 0.6625 降至 0.6391。绝对阈值 -1/0/1 在 3 个
  validation case 上均未改变结果；严格验证选择回退到冻结 actor。test 为 3/3
  成功、0 deadlock，但 paired ratio 仍为 GA 1.836、Branch Search 1.448。
- v17 改用状态内相对安全 margin 0/0.5；尽管 margin 0 在验证轨迹中大量过滤候选，
  最终动作与冻结 actor 相同，validation 仍为 2/3 成功、1/3 deadlock。验证再次
  选择无过滤基线，test 指标与 v16 相同。粗粒度“失败尾部即负例”没有学到能改变
  关键分叉的因果排序，继续扩大同类标签缺乏依据。

### `runs/ir_sft_warmup_v18_shield_budget` 与 v19

- `cluster_rl.ir.evaluate_shield` 只在 validation 比较安全执行参数，选定后对 test
  评估一次。v18 比较回溯预算 25/100：两者 validation 均为 2/3 成功、1/3
  deadlock，成功轨迹相对 GA 1.104、Branch Search 1.182；预算 100 在失败例确实
  用满 100 次回溯仍未完成，因此选择预算 25。
- v19 固定预算 25，比较一次回退 1/4/8 层。三个步长的 validation 成功率与成功
  makespan 完全相同，失败例分别在 37/35/27 个决策后 deadlock；固定跳层只能更早
  失败，不能找到可完成分支，验证选择 stride 1。
- 所选策略的 test 为 2/3 成功、1/3 deadlock；两个成功轨迹相对 GA 1.050、
  Branch Search 1.021。该结果接近 GA 但未优于 Branch Search，33.3% deadlock
  也不满足验收。后续安全数据应通过实际替代分支到达成功来产生因果动作对，安全
  搜索只适合作为离线教师，不能把高成本回溯计为轻量 actor 的模型能力。

### v20--v22 因果反事实动作对

- `cluster_rl.ir.causal_safety` 在 train-only 失败轨迹尾部保存精确 Session 分叉，
  实际执行原动作和若干替代动作；只有相同续跑教师下原分支失败、替代分支完整
  success 时才生成 `替代 > 原动作` 的独立 safety-head 排序标签。
- v20 首先暴露了语义错误：50 步采集 `decision_limit` 只是预算截断，不是任务
  deadlock，不应触发反事实负例。该运行主动停止；实现随后只对真实 deadlock 或
  deadline failure 挖掘分支。
- v21 对两个短 deadlock train case 使用原 v11 actor 续跑，各检查 13 个替代分支，
  共 26 次尝试，没有完整成功分支；因此没有训练、没有访问 validation/test。
- v22 改用 raw test 曾 3/3 成功的 v7 作为续跑教师，并对每个状态先验证“原动作+
  同教师”确实失败。两个短 deadlock case 分别做 30/107 次反事实尝试，仍为 0 个
  因果动作对；没有训练，也没有访问 validation/test。结果以
  `status=no_causal_pairs` 保存。
- 这表明这些陷阱不能由失败前 25 层中的单一动作替换加固定续跑教师修复。下一步
  需要从 train split 做多步成功搜索并提取整条分歧前缀，或训练真正的状态可达性
  value；继续扩大粗尾部标签、单点替代数或 held-out 调参均缺乏证据。

### graph-3 数值通道与 v23--v26

- 图协议升级为 `ir-graph-3`：候选节点除完整关系子图外，直接暴露通用的
  earliest/latest slack、duration、资源数量/总量、状态变化数和直接后继数。旧图的
  候选数值特征全为 0，两层网络难以从子节点稳定读取这些决策信息。
- v23 从头训练 4 个 `<=12` wafer 实例，validation/test 均为 0/2；v24 扩大到
  14 个小实例仍为 0/2；v25 把消息层数提到 4 仍为 0/2。因此新通道是必要的
  可见性修正，但不会自动解决闭环暴露偏差。
- `migrate_checkpoint` 把 graph-2 数值投影原样复制到 graph-3 第 0 通道，其余列置
  0。v7 迁移后的 validation 2/3、test 3/3 及 makespan 逐例与迁移前完全一致，
  证明迁移不改变函数。v26 冻结其余网络，只学新数值列并用 KL anchor；
  validation 质量变差，严格选模回退到迁移基线，test 仍为 3/3、0 deadlock、
  GA 1.836、Branch Search 1.448。

### 源动作时间顺序修正与 v27--v29

- 轨迹诊断发现源 `actions` 文件并不保证按 `start` 单调排序，而旧回放只对
  相邻的相同时刻分组。全数据扫描中，train 63/210、validation 9/28、test 6/28
  个文件至少有一次相邻逆序，分别累计 4149/666/332 次。这会把未来动作错当为
  当前标签，是监督数据合同缺陷，不是调参问题。
- `replay_expert` 现先按 `(start, original_index)` 稳定排序，再对同一最早时刻构建
  多标签。在关键 validation 实例 `instance-db3a9fca25419656` 上，修正后完整
  IR replay 成功且通过 audit，makespan 3207，仅比源 expert 3188 慢 0.60%；这验证了
  修正后标签时间线的可执行性。
- v27 用修正后的 4 个小实例从头训练，held-out top-1=0.4109，validation
  从 v23 的 0/2 提高到 1/2，但 test 仍为 0/2。v28 从 v27 继续训练 14 个小
  实例，validation 仍 1/2、test 0/2。
- v29 从 v27 继续训练分层平衡的 12 个 `<=25` wafer 实例，共 3213 个监督
  选择、407 次更新，held-out top-1=0.4468。validation 为 2/4，test 为 3/4；
  test deadlock 25%，成功轨迹平均为 GA 1.836、Branch Search 1.448。尽管
  teacher-forced accuracy 提高，闭环结果与旧 v11 相同，仍未达标。

### v30 有界随机 rollout 搜索

- `sample_search` 在每个实例上用固定 seed 生成有界随机轨迹，只在 validation
  选择 temperature/budget，然后对 test 执行一次所选策略；每条候选轨迹都通过
  Session audit。
- v30 对 v29 比较贪心与 temperature=0.5、budget=2/4。贪心 validation 为
  2/4，两个随机预算都为 0/4，因此选模正确回退到贪心。一次 test 结果为
  3/4、25% deadlock、GA 1.836、Branch Search 1.448。高温采样不是当前的解法。
- v31 继续比较 temperature=0.05/0.1、budget=2/4；四组随机候选同样都是
  validation 0/4，因此仍选贪心，test 指标与 v30 一致。这排除了“只是 0.5
  温度过高”的解释。复盘后将搜索实现改为每个 portfolio 始终以贪心轨迹保底，
  其余预算才用随机轨迹；这保证增加预算不会降低完成率。v31 旧报告的随机
  候选未包含该保底，但因所有额外轨迹都失败，修正不会改变选模结论。

### v32--v33 安全权重的 expert rehearsal

- 为了在不破坏 v7 已学安全行为的前提下恢复 makespan，从无损迁移的 v7
  graph-3 checkpoint 出发，用修正后的 expert 时间线做 `3e-5` 低学习率 rehearsal。
- v32 只训练 4 个 `<=12` wafer GA 实例，632 个监督选择、80 次更新；过滤后
  test 仅剩 2 例，虽然 2/2 成功、0 deadlock，但质量为 GA 2.686，子集太小且
  质量无改善，不用于最终结论。
- v33 改为分层的 4 个 `<=25` wafer 实例，包含两条 Branch Search 和两条 GA
  轨迹，共 1057 个监督选择、134 次更新，held-out top-1=0.4457。完整
  validation 为 2/4，test 为 3/4、25% deadlock、GA 1.836、Branch Search 1.448，
  与 v29/v11 的闭环轨迹一致。低学习率 rehearsal 没有破坏策略，也没有修复
  关键分支；不应继续对同一局部极值做学习率微调。
- v34 首次把 v7 安全蒸馏权重扩展到完整 4 个 test 实例。第 4 例在第 13 个
  决策后 deadlock，因此旧的 3/3、0 deadlock 不能外推；完整口径也是 3/4、
  25% deadlock、GA 1.836、Branch Search 1.448。v7、v29 和 v33 在这 4 个 test
  上实际生成相同的三条成功 makespan 和同一条失败。

### v35 shield 完整 4-case 复核

- 将 v8 corrective-distill checkpoint 无损迁移到 graph-3，按 validation 固定的
  budget=25、stride=1 对完整 4 validation + 4 test 复核。validation 为 2/4，
  test 也只有 2/4，deadlock 50%；成功 test 轨迹为 GA 1.050、Branch Search
  1.021。新增的第 4 例耗尽 25 次回溯仍在第 59 个决策后 deadlock。
- 离线取 raw v7 与 v8 shield 两条轨迹中较短的成功者，完成集仍只是 3/4，
  因为二者都无法完成第 4 例。成功轨迹的 GA paired ratio 可从 1.836 改善到
  1.530，有 Branch Search 参考的两例为 1.015，但 25% deadlock 没有改善。
- 因此当前最好证据仍是两个不达标端点：raw actor 完成率 75% 但成功调度慢，
  shield 成功调度接近 GA 但只完成 50%。已有数据不支持声称模型超过 Branch
  Search 或“几乎不死锁”。

### v36 消息传播容量消融

- 用与 v33 完全相同的 4 个分层实例，将模型从 width=16/layers=2 改为
  width=32/layers=4 并从头训练。共 1057 个监督选择、134 次更新，held-out
  top-1=0.3699。
- validation/test 均为 0/4，多数在 3--6 个决策后 deadlock。四层理论上能读取
  更远的资源关系，但在当前数据量和单 epoch 下更难拟合；不继续盲目扩容。

### 共享 handoff 的无后继 pick 修正与 v3 数据集

- v29/v38 失败轨迹显示，共享 buffer 的上游 Robot 能把 wafer pick 回自己的手，
  却无法到达路线下一站；Compiler 旧 binding 只检查 Robot 能否到达 pick 源站，
  因而向模型暴露了物理上无后继的“局部合法”动作。
- pick binding 现要求同一 Robot 同时覆盖当前源站与至少一个下一目标站。
  跨 Cell 移动仍通过路线中显式共享 buffer 分段；测试同时验证两个方向只生成
  对应下游 Robot 的 pick。
- 由于 binding 改变 IR hash，从原 reducer 运行重新物化
  `datasets/ir_sft_ga_branch_v3`：266/266 接受、0 拒绝，train/validation/test 仍为
  210/28/28，GA/Branch Search 最佳 expert 仍为 212/54。所有源 expert 动作均仍在
  修正后的 IR 动作域内可表达。

### v37--v43 精简动作域训练与选模

- v37 直接把 v29 checkpoint 用在 v3：validation/test 完成率仍为 2/4 和 3/4，
  但关键 validation 失败从 14 步推迟到 175 步，共同 test 失败从 12 步推迟到
  39 步。这证明 Compiler 修正删除了真实结构性死路，但不是全部死锁来源。
- v38 从 v29 开始在 v3 的 4 个分层样本上微调，共 1038 个监督选择、131 次
  更新；一个 handoff 样本的标签状态从 417 降为 398，Wait 从 211 降为 194。
  闭环结果与 v37 相同，说明这一轮未改变关键排序。
- v39 比较 Wait penalty 0/1/禁用，v40 又针对失败轨迹的 0.8401 logit 间隙
  比较 0/0.85。1、禁用和 0.85 都使 validation 从 2/4 降为 0/4，因此严格回退
  penalty=0。Wait 是状态依赖的必要决策，不能用全局置信度修正。
- v41 的宽反事实搜索因单例分支过多主动停止。v42 限制为两个小规模 train
  实例、失败前 20 层、每层两个替代；失败例实际续跑 55 个分支，仍为 0 个
  “单动作替代→完整成功”因果对，因此没有训练或访问 validation/test。
- SFT 选模从“held-out imitation accuracy”改为闭环 validation 的完成率、死锁率、
  Branch Search/GA paired ratio 字典序；`initialize_from` 权重作为 epoch 0 候选，
  test 只评最终所选 checkpoint 一次。v43 的新 epoch 与 epoch 0 闭环完全平局，
  因此正确保留 `best_epoch=0`；test 仍为 3/4、25% deadlock、GA 1.836、Branch
  Search 1.448。

## 验收口径

最终模型必须在完全隔离 test split 上同时报告：完成率、deadlock rate、成功
makespan、相对 Branch Search 与 GA 的 paired ratio。只有完成率高于 Branch
Search、makespan 接近 GA 且 deadlock 很少时，才能宣称达到目标；训练 loss 或
teacher-forced accuracy 本身不足以支持该结论。
