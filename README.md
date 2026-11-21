# MetroReasoning v2：北京地铁5号线中断环境与 ALNS 接入

本包以用户上传的 `MetroReasoning(4).zip` 为原始项目基础，保留 `DataBase` 数据目录和旧版代码归档；当前 V1 在物理环境上接入了新的 ActionPlan 版 ALNS。

v2 在 v1 的车内延误、折返服务变更和五目标评价基础上，修正了搜索预试算占满预算的问题，并使时间上限覆盖预试算。当前口径见 `docs/v2_修整说明.md`；`docs/v1_修正说明.md` 留作上版变更记录。五目标等权仅为实验初值。

## 1. V1范围

- 单方向、单区间、完全封锁。
- 决策间隔：300秒。
- 仿真推进：事件驱动；窗口内按列车到站、折返完成、故障解除、OD到达、停站作业完成和发车事件推进。
- 列车动作：Holding、Delay、Short-turning、Release。
- 客流动作：Entry Control（进站配额）。
- 不包含：多区间/双方向中断、降级通行、跳停、取消车次、新增车底、换乘限流、车站关闭。

## 2. 目录

```text
MetroReasoning/
├── DataBase/
│   └── beijing_5/           # CSV 与 NPY 文件同层，原始内容未修改
├── Env/                      # V1物理环境、预测试算和 ALNS
│   ├── Constant.py
│   ├── configure.py
│   ├── TrainOperator.py
│   ├── PassengerOperator.py
│   ├── Env.py
│   ├── Evaluator.py
│   ├── ALNS.py
│   ├── ResultRecorder.py
│   └── __init__.py
├── Legacy/                   # 原版Env、ALNS和入口程序归档，不参与V1导入
├── Result/                   # 原项目结果及本版 alns_v1 结果
├── tests/                    # V1自动化测试
├── examples/run_demo.py
├── run_alns.py
├── docs/
└── requirements.txt
```

## 3. 安装与运行

建议 Python 3.10 或更高版本。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/run_demo.py
python run_alns.py --mode benchmark --rounds 3 --iterations 10
```

Windows PowerShell同样在项目根目录运行以上命令。

默认数据位置由 `Env/configure.py` 相对于项目目录定位为 `DataBase/beijing_5`，不依赖 `D:` 盘符。传入相对的 `data_root` 时也以项目目录为基准；需要读取项目外数据时可传绝对路径。

## 4. 最小调用方式

```python
from Env import Env, ActionPlan, EntryControlAction

env = Env()                 # 默认：20站→21站下行，2100秒开始中断
observation = env.reset()   # 推演至中断开始时刻
result = env.step(ActionPlan(
    passenger_actions=(EntryControlAction(station=20, direction=0, max_admit=100),)
))
print(result.snapshot.time)
print(result.metrics)
```

列车动作示例（动作必须满足列车当前状态和车站条件）：

```python
from Env import HoldingAction, DelayAction, TurnbackAction, ReleaseAction

# ActionPlan(train_actions=(HoldingAction(train_id=..., station=...),))
# ActionPlan(train_actions=(DelayAction(train_id=..., station=..., delay_seconds=120),))
# ActionPlan(train_actions=(TurnbackAction(train_id=..., station=...),))
# ActionPlan(train_actions=(ReleaseAction(train_id=..., station=...),))
```

动作规则：
- Holding只允许对当前站、停站作业已完成且尚未发车的列车执行；持续有效，直到显式Release。
- Delay只允许作用于当前站的停站/待发列车，延长其计划发车时刻。
- Turnback方向由折返设施和来车方向推导，不接受决策器指定目标方向；可对当前停站或本窗口将到站的列车预先下令。尚未完成停站的指令状态先记为`pending`，实际清客时再核验站台容量及折返资源；若条件不足，记录为`rejected`，列车按原行程继续运行。
- Release只解除扣车，不直接保证发车；重排时刻按当前时刻、原计划、最小追踪间隔和停站/折返完成时刻确定，发车事件再检查封锁。
- 同一列车同一决策时刻不能接收多个动作；下令时校验失败会抛出`InfeasibleAction`，不会部分执行。预下达的折返指令以`StepResult.executed_actions`中的最终状态为准。
- 车站同方向只允许一列车停站或待令；后车不能占用前车仍在使用的站台。运行中的列车可在站外等候进站，本版未建立完整的联锁进路和区间闭塞模型。

## 5. 信息隔离

`configure.DisruptionTruth`由环境配置持有，含`actual_duration`；决策器拿到的是`Snapshot.disruption`（`DisruptionObservation`），仅含起止区间、开始时刻、预测结束时刻和当前是否封锁，不含真实持续时间/真实结束时刻。不要把`Env._truth`或`Env._config`传给优化器。Python对象的私有字段不是安全沙箱，正式实验应只向求解器传递`Snapshot`和明确筛选后的公开配置。

候选评价的时域必须满足：

`H_decision <= H_eval <= H_forecast`

ALNS 只接收 `Snapshot`、`Config.public()` 返回的公开运行参数和 `Evaluator`；`Evaluator` 在预测副本中试算，未来 OD 来自快照中的预测，封锁结束按预测时刻处理。环境在实际恢复事件发生前不会把真实恢复时刻写入列车重试发车时间。

**五目标更新（本包）：** 原有单目标等待人秒已改为五项目标的归一化加权评分：①站外+站台等待；②车内超原定发车时刻停留人秒（仅为乘客延误代理，不等于完整OD到达延误）；③站台超过预警线的拥挤人秒；④本轮评价新增运行图偏移；⑤时域末未完成出行人数（含车上、站台、站外，且不等于永久出行失败）。默认权重均为0.2，属于未标定的对照初值。硬约束保持一票否决，不混入评分。完整公式、归一化尺度与边界见 `docs/V1_五目标评价设计与修改清单.md`。ALNS算子权重只控制搜索抽样概率，与目标权重无关。

`run_alns.py` 默认跑 5 个决策窗口、每窗口 30 次搜索；`--mode rule/lns/alns/benchmark` 分别运行无新动作规则、固定算子权重 LNS、自适应 ALNS 或三者对照。每轮 JSON 分开保存决策时快照、候选评分、选中指令和真实执行反馈；`summary.json` 保存全场汇总及仅供审计的中断真值。结果默认在项目内 `Result/alns_v1/run_.../`。这些记录尚不是经专家核验的大模型微调样本。

## 6. 乘客与守恒

- `outside`、`platform`、`onboard`是三个物理位置。
- `stranded_platform`只是站台乘客的状态标记，不是第四个位置；其人数已包含在`platform`中。
- 清客前检查完整站台容量；不足时整个折返动作不可行，不将乘客塞入站外队列。
- 故障解除事件清除滞留标记，乘客仍在原站台队列中等待服务。
- 每个事件后检查：`arrivals_total == completed_total + outside + platform + onboard`。
- 上行按目的站递减顺序分配，下行按目的站递增顺序分配。

## 7. 参数边界

`Constant.py`沿用原项目车站、站台、列车容量、追踪间隔、停站/区间运行时间和折返候选站设置。若原始项目未提供可追溯来源，本包在`docs/参数来源清单.xlsx`中标为“原代码值/待核实”或“建模假设”，不将其包装成已核实运营参数。尤其折返候选站和300秒/120秒折返作业时间，不能解释为现行联锁许可。

原始149条记录是车次计划，数据未提供可靠车底接续关系。`train_id`保留原始计划索引，`vehicle_id`暂为`None`；折返后用新的`service_id`标记反向服务，不宣称149条计划对应149组实体车底。终点站的历史折返设施仍在配置中，但本版折返动作仅处理尚有原方向后续行程的车次。

## 8. 旧代码

原版`ALNSEnv.py`、`ALNS.py`、`configure.py`、`scenario.py`等已放入`Legacy/Env`。旧版`run_alns.py`放入`Legacy/`。它们用于函数迁移审计，不会被V1物理环境自动导入；本版根目录的 `run_alns.py` 使用新接口。对应关系见`docs/旧代码迁移映射.md`。

本次没有删除原项目的`Result`结果目录；它们是历史实验输出，不是V1代码验收结果。

## 9. 当前验收状态

当前测试覆盖物理环境、真实恢复时间与未来客流的信息隔离、完整联合动作的试算、折返候选和结果记录。原`docs/测试报告.md`是接入前的历史报告，以当前实际运行结果为准。

这些测试证明代码通过了本包所列的单元/集成测试，不等于线路参数已获运营核验，也不等于已经完成全场景物理有效性认证。尤其原始折返设施、站台容量和行车参数仍需数据来源复核。

---

## 2026-11-20：TPR-ALNS 三项增强（保留原 V1 物理环境）

新增拓扑感知搜索、折返-客流联合修复、三场景预测鲁棒评价。
默认 `--profile enhanced` 且启用鲁棒场景；使用 `--profile standard` 对照标准算子基线，`--single-scenario` 切回原单场景五目标。

```bash
python run_alns.py --mode benchmark --rounds 1 --iterations 3 --profile enhanced
python -m unittest discover -s tests -v
```

详细公式、预测场景及不变约束见 `docs/TPR_ALNS_三项增强设计与验收.md`。增加鲁棒场景会提高CPU开销，请先小迭代验证后再开展正式多种子对照。**此版本没有修改原有物理规则或现有4+1调度动作。**
