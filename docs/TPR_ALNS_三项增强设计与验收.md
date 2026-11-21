# MetroReasoning V1：三项 ALNS 增强设计与验收说明

## 1. 版本定位与不可变边界

在用户确认的五目标版 `MetroReasoning_5_FiveObjectives_MinimalChange.zip` 上增量实现算法层功能；**没有修改** `Constant.py`、`configure.py`、`Env.py`、`TrainOperator.py`、`PassengerOperator.py` 的原有物理规则，也没有增加 V1 动作或中断类型。

- V1 中断：**单方向、单相邻区间、完全封锁**。
- V1 列车指令：`Holding / Delay / Turnback / Release`；乘客指令：`EntryControl`。
- 决策间隔：300 秒；单次预测与评价：原设定 1200 秒。
- 五个原始指标：站外与站台等待人秒、车内超原计划停留人秒（延误**代理**）、拥挤超预警人秒、运行图偏移秒数、窗口末未完成出行人数。原版默认等权0.2，未经过权重校准。
- 硬约束、物理仿真逻辑和信息隔离均沿用旧项目。评分不用于抵消违反硬约束的方案。

## 2. 增强一：拓扑感知搜索（`Env/ALNS.py`）

新增 `TopologyContext.from_snapshot(snapshot, config.public())`：只从决策时可观察的 `Snapshot` 与公开折返设施中提取：封锁边界、方向、受影响列车、可用的候选折返站、与故障边界的距离、站台饱和度，以及**预测**剩余封锁秒数。

新增算子：

- `related_destroy`：移除某列车动作，并同步移除对应站点/方向的限流动作（如有）。
- `topology_insert`：在按故障位置、折返时间、站台饱和度排序的靠前列车动作中抽样。
- `greedy_repair`：从靠前的候选动作中最多抽取3个完整方案调用仿真，选择当前局部预测成本最低的可行方案。

这些排序是**启发式先验**，不宣称已验证为最优策略；实际是否接受仍以统一的五目标/鲁棒 `Evaluator` 结果为准。基础算子和原模拟退火/自适应权重更新保留。

## 3. 增强二：车客协同修复（`Env/ALNS.py`）

新增 `coupled_turnback`。给某个真实可观测的折返候选 `TurnbackAction(train_id, station)`，基于公开快照估计：

1. 待清客人数：车内仍需前往折返站以远目的地的乘客；目的地正好为折返站的乘客正常下车，不重复计入清客。
2. 现有站台人数：分方向站台按该方向计，岛式共用站台按上下行合计。
3. 决策窗口内该站同方向新增预测OD客流。
4. 站台预警人数上限及相对该预警人数3%的启发式缓冲。

生成一个**联合方案**：`TurnbackAction + EntryControlAction`，进站配额在0～300之间整型调整。

这只是候选生成器，不会强制让物理环境执行。**即使配额为0仍无法容纳清客人群，整个折返候选仍会被Env拒绝。** 无法靠限流绕开站台硬容量、折返资源和其他约束。

增强版候选站点在原边界站及上游一站基础上，额外包含本窗口可能折返的站点。仍然只有**一种**乘客控制动作，没有增加站台疏散等新机制。

## 4. 增强三：预测多场景鲁棒评价（`Env/Evaluator.py`）

新增 `ForecastScenario` 和 `RobustEvaluator`。同一时刻针对同一候选完整方案，使用**共同的可观测快照、预测OD、归一化尺度及固定场景**评估：

| 场景 | 基于预测的恢复时刻扰动 | 未来预测客流倍率 | 概率 |
|---|---:|---:|---:|
| central | 0秒 | 1.00 | 0.50 |
| early_low | -300秒 | 0.85 | 0.25 |
| late_high | +300秒 | 1.15 | 0.25 |

在已知尚未解除封锁的情况下，预测解封事件至少安排在下一个300秒决策边界，防止出现与当前已知封锁状态冲突的预测场景。`actual_end`和真实未来OD**不参与场景构建**。所有随机搜索候选共用上述确定性场景，不因动作选择而更改外生预测。

若任何预测场景被物理校验判为不可行，则该联合候选被标记为鲁棒不可行（保守型全场景可行规则）。对所有场景均可行的候选：

\[
J_{\rm robust}=\mathbb{E}[J]+\lambda\bigl(\operatorname{CVaR}_{\alpha}(J)-\mathbb{E}[J]\bigr)
\]

其中 `J` 是五目标**归一化后的综合分数**，默认 `λ=0.35, α=0.8`。CVaR是离散场景概率下最差 `(1-α)` 尾部的平均分；其余分项指标在结果中记录概率加权期望。**鲁棒总分不等于简单的五项期望贡献之和**，多出的项为 `robustness_premium`。

这三组场景、扰动大小及风险参数属于**待校准建模假设**，不是已验证的真实中断时长概率分布。不能据此宣称现场级风险控制。需要在后续不同区间/时长/客流、不同权重和场景扰动下进行敏感性验证。

## 5. 可核验候选日志（`Env/ResultRecorder.py`）

逐轮 `round_XXX.json` 原有 `snapshot`、`selected_prediction`、`actual_feedback` 保持分离；新增：

- `search_evidence.search_profile`：`standard` 或 `enhanced`；
- `search_evidence.topology_summary`：已知拓扑状态，含候选折返站与其站台饱和度；
- `search_evidence.operator_calls` 与 `operator_weights`：搜索算子记录；
- `search_evidence.candidate_audit`：所有**去重后被仿真评价**的候选动作组合、来源算子、可行性、成本及每个预测场景评分/失败原因；
- `selected_prediction.robust_details`：场景设置、场景结果、期望评分、CVaR、风险溢价。

日志中“可行方案”不等于“全局最优”；因模拟退火未接受的可行候选不算物理负样本；`candidate_audit` 是**数据生成依据**，不是现成的忠实CoT标签。

## 6. CLI运行方法

在 `MetroReasoning/` 根目录执行：

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_alns.py --mode benchmark --rounds 1 --iterations 3 --profile enhanced
```

算法消融：

```bash
# 只禁用拓扑感知/车客协同，保留同样的鲁棒评价，比较算法结构
python run_alns.py --mode alns --rounds 1 --iterations 3 --profile standard
python run_alns.py --mode alns --rounds 1 --iterations 3 --profile enhanced

# 只关闭多场景，保留增强算子，对比评价方式
python run_alns.py --mode alns --rounds 1 --iterations 3 --profile enhanced --single-scenario

# 修改风险偏好
python run_alns.py --mode alns --rounds 1 --iterations 3 --profile enhanced --risk-aversion 0.5 --cvar-alpha 0.8
```

同一场景、同一预算、同一随机种子、相同的五目标权重与评价模式才能公平对比算法搜索能力。三个预测场景使每次新候选的评估计算量明显增大，建议先从1～3次迭代检查功能，再进行正式预算标定；**不能把提高搜索时间本身当成方法性能提高**。

## 7. 现有工程边界与后续验收

本版只修改算法搜索层、评价层、CLI、日志及文档测试；以下旧问题不在本次擅自调整范围内：

- Delay后后续时刻表与区间实际运行时间的一致性，需继续专项核验；
- 热启动后实际区间占用与中断场景静态抽样的匹配问题；
- 真实车底交路、终点正常折返与道岔联锁进路许可仍缺支撑数据；
- `passenger_delay_proxy`不是实际OD全程晚点；
- 预测OD由过去5分钟均值外推，尚未经过预测精度标定；
- 新增拓扑与风险评分是否带来真实改进，仍需多实例、多种子及消融实验验证。

**结论：此交付是三项增强功能可运行的研究原型，不是保证现场运营安全的最终调度系统，也不能仅凭单轮测试生成Gold标签。**
