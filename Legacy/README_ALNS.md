# 单方向中断与待令列车恢复 ALNS（第一版）

本实现读取项目原有 `DataBase/npy` 的列车计划和双向 OD。原数据文件保持只读。

## 运行

在 `D:\推理数据集\MetroReasoning` 目录执行：

```powershell
python run_alns.py --decisions 3 --iterations 200 --time-limit 30
python run_alns.py --until-recovered --max-decisions 50 --weight-index 0
python run_alns.py --benchmark --decisions 3 --iterations 20 --time-limit 5
python -m unittest Env.test_alns -v
python -m Env.calibration
```

`Env/test_alns.py` 保留回归测试，检查方向编码、封锁、限流、折返清客、信息隔离、乘客守恒、搜索与结果记录。每个函数开头均有中文说明，可先读对应 `def` 的文档字符串了解作用。

默认算例为 `20 蒲黄榆 → 21 刘家窑` 下行（方向 0）在第 2100 秒中断，实际持续 1200 秒，最初预测持续 900 秒。方向 1 为上行，沿站序递减。`run_alns.py --help` 列出可改的场景和预算参数。每次运行单独写入 `Result/alns/run_时间戳/weight_*/{alns,lns,rule}/`；每个决策时刻一个 JSON，并有汇总和跨方法比较文件。

`--sample-seed N` 调用原有 `Env/scenario.py` 抽样单方向中断；如果开始时刻已有列车占用该区间，会按后续种子重抽并记录实际采用的种子。第一版不虚构区间内列车的紧急制动状态。

## 程序流程

1. `Env/configure.py` 保存中断真值和公开搜索设置。求解器只收到 `DecisionConfig`、当前 `Snapshot` 和只使用快照预测的评价函数。
2. `Env/ALNSEnv.py` 从原始数据重建中断时刻的列车、车上乘客、站台及站外队列；用已观察到的最近 5 分钟 OD 均值预测后续每分钟客流。
3. `Env/ALNS.py` 从安全待令方案开始，先按当前快照枚举本窗口到站的折返动作，逐个组合成完整方案并试算，再破坏/修复具体列车待令时间、清客折返与车站进站人数控制；保存可行候选和否决原因计数，并更新算子选择权重。迭代数和墙钟时限先到即停。
4. 选中方案仅执行下一个 300 秒窗口。环境使用实际客流和实际封锁状态推进；若预测恢复落空，下一轮观察到仍封锁并重新求解。
5. 若区间在窗口中途实际开放，环境在开放时刻提前返回新快照。恢复阶段按已有车序重新安排尚未执行的待令车发车时刻；站台进站配额只对本轮窗口有效，下一轮重新判断并下达。已执行的时刻不回写。
6. `Env/ResultRecorder.py` 分别保存决策时可见状态、预测候选、所选方案和实际反馈。`--benchmark` 从同一初始状态分别运行规则、固定算子 LNS、自适应 ALNS。

## 恢复阶段的范围

`snapshot.phase` 标明当前观察为中断或恢复；求解器仍看不到 `actual_duration`。开放后的首轮立即切换恢复算子；若通过 `Config.recovery_weight_sets` 提供经过校准的恢复期权重，同轮也切换权重。默认仍沿用 W0/W1 的试验权重，不声称已标定阶段偏好。`held_departures` 保存因本次中断而延后的车次及原计划发车时刻，`release` 动作只可把未执行的发车提前至当前时刻之后，且不能早于原计划。候选必须满足故障边界既有车序及最小间隔。折返产生的原车与反向车次关系记录在 `turnback_links`，目前**不优化车底后续接续**，不能把折返车直接视为恢复原交路。

整场汇总中的 `reopened_at` 是观察到开放的时刻，`release_completed_at` 是待令车全部放行的时刻；两者都不是完整交路恢复时刻。`recovery_status` 仅报告待令列车是否已放行，`pending_held_trains` 是尚未消化的待令车数。`--decisions` 限定共同观察轮数；`--until-recovered` 则在待令车放完后停止，最多运行 `--max-decisions` 轮。到达轮数或 OD 数据末尾不等于完整交路恢复。

## 目标与约束

可行方案的分数为加权归一化成本：预测窗口内的站外及站台等待人秒、窗口末等待人数、当前未发车事件的剩余晚点秒、站台超预警人秒、调整动作代价。`end_waiting_pax` 是窗口末仍在等待的人，并非最终无法完成出行。`remaining_delay_sec` 每轮按当前尚未发车事件重算，不能跨轮求和；`realized_delay_sec` 在实际发车的执行窗口记账，整场可汇总。普通列车动作和限流动作各计 1，清客折返暂计 3；这些业务权重尚未做运营标定。`cancelled_original_stops` 与候选的 `audit.cancelled_stops` 只用于报告，当前不计入分数。

评分尺度保存于 `Env/metric_scales.json`，由 `python -m Env.calibration` 从固定的 3 个区间 × 3 个时长 × 3 种客流压力组合生成。各指标取正值第 75 百分位；实测高峰负荷始终没有产生超预警指标，因此仅拥挤尺度借用明确标注的合成压力试验。0.5 倍和 4 倍客流均不是实测平峰或超高峰。这组尺度是可复现的试验尺度，尚需更多独立场景检查稳定性；运行时不会再用本轮基础方案的零值作分母。不同权重组的分数不可直接互比，须看原始分项指标。

候选校验：方向与相邻区间匹配、封锁时段不新驶入故障区间、同方向出发间隔至少 70 秒、模型站台上限、列车容量和乘客守恒。站台预警值只产生软成本。折返还需列车在本窗口到达假定可用站、原方向后续行程能够取消、同站折返资源不重复占用；新增反向车次按追踪间隔插入，受影响的既有反向车次会顺延。

**折返站假设：**将 `Env/Constant.py` 中 7 个历史候选站（编号 0、3、6、10、16、19、22）暂作可折返；两端站只开放能够反向驶出的来车方向，中间站暂开放双向。最短作业时间默认 300 秒，站 10 暂取 120 秒。折返到站后清客，未到目的地的乘客进入同方向站台等待队列；站台模型上限不足时转到站外等待队列。`turnback_routes=()` 可关闭全部折返。

## 结果文件怎么读

- `comparison.json`：同一次运行中各权重、各方法的总等待、同客流无中断对照、额外等待、末轮等待人数、折返列数和搜索用时。无中断对照只在事后计算，不送给求解器。
- `weight_*/alns/summary.json`：一组权重的逐轮索引与整场汇总，`total_turnbacks` 是实际下发的折返动作总数。
- `decision_时间.json`：`snapshot` 是决策前可见状态；`candidates` 是并列候选；`chosen_plan` 是最终方案；`feedback.actual_metrics` 是下一窗口执行后的真实指标；`next_snapshot` 用于下一轮。`search.turnback_trials` 逐车写出折返试算是否可行、分项结果和分数；`turnback_options`、`turnback_feasible`、`rejection_reasons` 分别汇总折返尝试数、可行数和候选否决原因。`station_control_window` 标明本轮进站配额、起止时刻。候选中的 `turnback` 表示“该列车在指定站清客并反向开行”；`cleared_pax` 是本窗口实际清客人数，`induced_train_count` 是因插入反向车次而顺延的既有车次数。
- `decision_时间.json` 的 `snapshot.phase` 可区分中断与恢复；恢复候选的 `release.target_departure` 是重新安排的未来发车时刻。

## 模型边界

- 站台面积、上限、列车容量与行车间隔沿用 `Env/Constant.py` 的建模假设，不代表当前运营许可。
- 第一版不包含逐站联锁进路、道岔冲突、站台股道占用及区间内多车停车位置。因此“可行”只表示**通过本仿真器列出的约束**，不能解释成现场可直接下令。
- 中断开始时已有列车占用故障区间的场景会被拒绝；该类紧急制动状态需另建环境。
- 历史候选站的双向折返属于**明确的仿真假设**；缺少现行联锁进路表，输出不能当作真实运营许可。现有数据未来客流仅用于环境执行，候选评价使用决策时根据过去 5 分钟生成的预测。
- `--benchmark` 是程序对照入口。要证明统计收益，仍需事先固定多场景测试集并报告逐场原始指标、失败率与运行时分布。

## 函数作用一览

| 文件 | 函数 | 作用 |
| --- | --- | --- |
| `run_alns.py` | `run_case` | 从同一初始状态完成一组权重和一种方法的多轮决策，保存逐轮记录。 |
| `run_alns.py` | `main` | 读取命令行参数，构建中断场景并逐组运行、打印中文摘要。 |
| `Env/ALNS.py` | `_fingerprint` | 按列车动作和车站控制生成方案去重键。 |
| `Env/ALNS.py` | `Candidate.to_dict` | 把候选方案及其评价转成结果记录字典。 |
| `Env/ALNS.py` | `SearchResult.to_dict` | 把本轮搜索过程和候选结果转成可写入 JSON 的字典。 |
| `Env/ALNS.py` | `ALNSSolver.__init__` | 设定是否启用算子权重自适应；关闭时作为固定 LNS 对照。 |
| `Env/ALNS.py` | `ALNSSolver.search` | 在时间或迭代预算内搜索并返回最好方案及折返比较候选。 |
| `Env/ALNS.py` | `ALNSSolver.search.choose` | 按当前算子权重随机挑选一个破坏或修复算子。 |
| `Env/ALNS.py` | `ALNSSolver.search.normalize` | 补齐未折返列车的基础待令并维持故障边界发车间隔。 |
| `Env/ALNSEnv.py` | `safe_initial_plan` | 按当前恢复预测为受影响列车生成最早可行的基础待令方案。 |
| `Env/ALNSEnv.py` | `TrainPlan.to_dict` | 把方案转成可写入 JSON 的字典。 |
| `Env/ALNSEnv.py` | `Snapshot.digest` | 计算快照指纹，用于检查候选试算是否改动原状态。 |
| `Env/ALNSEnv.py` | `Snapshot.to_dict` | 把可见状态展开为结果记录所需的字典。 |
| `Env/ALNSEnv.py` | `EvalResult.score` | 用业务权重和同场景归一化尺度计算候选综合成本。 |
| `Env/ALNSEnv.py` | `ALNSEnv.__init__` | 读取时刻表与 OD，核对数据形状并建立仿真初始状态。 |
| `Env/ALNSEnv.py` | `ALNSEnv.public_config` | 返回求解器可见的参数，不含真实恢复时长。 |
| `Env/ALNSEnv.py` | `ALNSEnv.reset` | 从数据起点推演到中断开始时刻并返回第一份决策快照。 |
| `Env/ALNSEnv.py` | `ALNSEnv.observe` | 复制当前状态，并用过去五分钟客流生成后续客流预测。 |
| `Env/ALNSEnv.py` | `ALNSEnv.initial_plan` | 返回本轮 ALNS 搜索使用的安全基础待令方案。 |
| `Env/ALNSEnv.py` | `ALNSEnv._apply_plan` | 把列车动作写入时刻表副本，并构造折返车次与进站控制检查结果。 |
| `Env/ALNSEnv.py` | `ALNSEnv._append_reverse_train` | 按假定折返时间生成反向车次，并为插入的车次修复追踪间隔。 |
| `Env/ALNSEnv.py` | `ALNSEnv._validate_schedule` | 检查新时刻表的追踪间隔和封锁区间通行条件。 |
| `Env/ALNSEnv.py` | `ALNSEnv.evaluate` | 在快照副本上用预测客流试算候选，不修改实际环境。 |
| `Env/ALNSEnv.py` | `ALNSEnv._guard_actual_window` | 若实际仍封锁，把本窗口可能越过故障区间的列车继续扣停。 |
| `Env/ALNSEnv.py` | `ALNSEnv.step` | 执行选中方案一个决策窗口，按真实封锁与客流推进并反馈。 |
| `Env/ALNSEnv.py` | `ALNSEnv._advance` | 在初始化阶段按原计划推进到中断开始时刻。 |
| `Env/ALNSEnv.py` | `ALNSEnv._simulate` | 按需求、到站、清客、上车和发车事件推演指定时间窗。 |
| `Env/ALNSEnv.py` | `ALNSEnv._metrics` | 汇总等待、剩余乘客、延误、拥挤、折返清客和守恒指标。 |
| `Env/ALNSEnv.py` | `ALNSEnv._simulate.crowd` | 计算各站站台峰值和超过预警线的人数。 |
| `Env/ALNSEnv.py` | `ALNSEnv._simulate.admit` | 按进站配额和站台模型上限把站外乘客放入站台。 |
| `Env/configure.py` | `Weights.__post_init__` | 要求五项权重非负且总和为一，避免目标分数失去含义。 |
| `Env/configure.py` | `Config.__post_init__` | 检查有向区间、时长、搜索预算和折返假设是否自洽。 |
| `Env/configure.py` | `Config.public` | 向 ALNS 提供公开配置，不传入实际恢复时长。 |
| `Env/configure.py` | `load_metric_scales` | 读取并核验离线校准的固定评分尺度。 |
| `Env/configure.py` | `Config.weights_for` | 按当前中断或恢复阶段取对应权重。 |
| `Env/calibration.py` | `derive_scales` | 从正值指标样本计算第 75 百分位评分尺度。 |
| `Env/calibration.py` | `_free_start` | 寻找故障区间无车占用的中断开始时刻。 |
| `Env/calibration.py` | `calibrate` | 试算 27 组校准场景并记录候选指标。 |
| `Env/calibration.py` | `main` | 生成并保存固定评分尺度文件。 |
| `Env/ResultRecorder.py` | `ResultRecorder.__init__` | 创建结果目录并保存其位置。 |
| `Env/ResultRecorder.py` | `ResultRecorder.record_decision` | 将本轮快照、权重、候选、已选方案和反馈原子写入 JSON。 |
| `Env/scenario.py` | `sample_scenario` | 按种子抽取一个有向区间、开始时刻和时长，并确保结束不超出数据范围。 |
| `Env/scenario.py` | `DisruptionScenario.end_seconds` | 返回这次中断的实际恢复时刻，供仿真环境使用。 |
| `Env/test_alns.py` | `config` | 构造固定的单方向中断测试配置，允许覆盖个别参数。 |
| `Env/test_alns.py` | `ALNSTest.test_direction_must_match_directed_edge` | 方向编号和有向相邻站不一致时必须报错。 |
| `Env/test_alns.py` | `ALNSTest.test_observation_hides_actual_duration_and_evaluate_is_pure` | 快照不含实际恢复时长，多次试算也不改真实环境。 |
| `Env/test_alns.py` | `ALNSTest.test_blocked_traversal_rejected_and_initial_plan_held` | 无保护方案不得穿越封锁区间，基础待令方案应通过校验。 |
| `Env/test_alns.py` | `ALNSTest.test_step_uses_truth_and_conserves_passengers` | 实际执行推进一个窗口且乘客总数守恒。 |
| `Env/test_alns.py` | `ALNSTest.test_forecast_evaluation_ignores_unobserved_actual_demand` | 修改未来实际需求不能改变已经生成的候选预测。 |
| `Env/test_alns.py` | `ALNSTest.test_admission_control_moves_waiting_outside` | 限流会把等待乘客保留在站外而不凭空丢失。 |
| `Env/test_alns.py` | `ALNSTest.test_search_and_json_record` | 搜索返回可行候选并能写出完整决策记录。 |
| `Env/test_alns.py` | `ALNSTest.test_empty_routes_forbid_turnback` | 显式关闭折返进路时，折返动作必须不可行。 |
| `Env/test_alns.py` | `ALNSTest.test_assumed_route_turns_train_and_conserves_passengers` | 历史候选站按仿真假设开放后，逐车折返并清客。 |
