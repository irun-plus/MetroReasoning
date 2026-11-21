"""五目标预测评价：只修改评分/观测，不修改V1物理状态转移和约束。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import heapq
from math import inf, isfinite
from typing import Mapping

from . import Constant as C
from .Env import ActionPlan, Env
from .TrainOperator import ReleaseAction, TrainStatus

# 顺序与用户确定的五个目标一一对应。第二项是“车内超计划停留”的延误代理，
# 由于缺少逐乘客入站/到达时间戳，不可冒充完整OD实际到达延误。
OBJECTIVE_KEYS = (
    "passenger_waiting", "passenger_delay_proxy", "platform_crowding",
    "train_schedule_adjustment", "unfinished_trips",
)
OBJECTIVE_LABELS = {
    "passenger_waiting": "乘客站外及站台等待时间",
    "passenger_delay_proxy": "车内超计划到发时刻停留（乘客延误代理）",
    "platform_crowding": "站台超过预警线的拥挤人秒",
    "train_schedule_adjustment": "新增晚点及折返服务变更量（列车秒）",
    "unfinished_trips": "评价窗口结束时未完成出行人数",
}


@dataclass(frozen=True)
class ObjectiveWeights:
    """初始等权占位方案，不是经论文/现场标定的最优权重；支持外部配置。"""
    passenger_waiting: float = 0.20
    passenger_delay_proxy: float = 0.20
    platform_crowding: float = 0.20
    train_schedule_adjustment: float = 0.20
    unfinished_trips: float = 0.20

    def __post_init__(self) -> None:
        vals = self.as_dict()
        if any(not isfinite(v) or v < 0 for v in vals.values()):
            raise ValueError("目标权重必须是有限且非负的数")
        if abs(sum(vals.values()) - 1.0) > 1e-9:
            raise ValueError("五个目标权重的总和必须为1")

    def as_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in OBJECTIVE_KEYS}

    @classmethod
    def from_sequence(cls, values) -> ObjectiveWeights:
        values = tuple(float(v) for v in values)
        if len(values) != len(OBJECTIVE_KEYS):
            raise ValueError("必须依次输入五个目标权重")
        return cls(**dict(zip(OBJECTIVE_KEYS, values)))


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    score: float
    metrics: dict[str, float]
    executed_actions: dict
    end_time: int
    reason: str | None = None
    objective_values: dict[str, float] = field(default_factory=dict)
    objective_scales: dict[str, float] = field(default_factory=dict)
    objective_weights: dict[str, float] = field(default_factory=dict)
    objective_contributions: dict[str, float] = field(default_factory=dict)
    robust_details: dict = field(default_factory=dict)


class Evaluator:
    """五目标试算。每个候选的同一轮状态、预测、权重和归一化刻度完全相同。"""

    def __init__(self, env: Env, weights: ObjectiveWeights | None = None):
        config = env._config
        if config.evaluation_horizon % config.decision_interval:
            raise ValueError("评价时域必须是决策窗口的整数倍")
        self.weights = weights if weights is not None else ObjectiveWeights()
        self._base = env.fork_for_evaluation(env.observe())
        self._start_metrics = self._base.metrics()
        self._steps = min(
            config.evaluation_horizon // config.decision_interval,
            (len(self._base.demand) - 1 - self._base.current_time) // config.decision_interval,
        )
        if self._steps < 1:
            raise StopIteration("剩余仿真时间不足一个决策窗口")
        self._horizon = self._steps * config.decision_interval
        snapshot = env.observe()
        remaining = max(0.0, float(snapshot.arrivals_total - snapshot.completed_total))
        forecast = sum(float(od.sum()) for minute, od in zip(snapshot.forecast_times,
                         snapshot.forecast_od)
                       if snapshot.time < minute <= snapshot.time + self._horizon)
        # 以上只使用决策时可见状态/预测，不能看真实未来客流或actual_end。
        population = max(1.0, remaining + forecast)
        passenger_time_scale = max(1.0, population * self._horizon)
        # 预警到硬容量之间的缓冲，是一个明确的站台拥挤物理刻度。
        crowd_headroom = sum(
            max(0, int(C.PLATFORM_CAPACITY_PEOPLE[i])
                    - int(C.PLATFORM_WARNING_PEOPLE[i]))
            * (2 if i in C.SEPARATE_DIRECTION_CAPACITY_STATIONS else 1)
            for i in range(C.STATION_NUM)
        )
        self.scales = {
            "passenger_waiting": passenger_time_scale,
            "passenger_delay_proxy": passenger_time_scale,
            "platform_crowding": max(1.0, crowd_headroom * self._horizon),
            "train_schedule_adjustment": max(1.0, len(snapshot.trains) * self._horizon),
            "unfinished_trips": population,
        }
        # 保留与原ALNS模拟退火温度同数量级的评分基准，不改变其算法逻辑。
        # 注意合成score不是实际人秒，只有分项指标具有原始物理单位。
        self._score_unit = passenger_time_scale

    @staticmethod
    def _continuation_plan(env: Env) -> ActionPlan:
        """完全沿用原版恢复期策略，不干预ALNS动作空间。"""
        if env.observe().phase != "recovery":
            return ActionPlan()
        held = sorted(
            (tr for tr in env._trains.values() if tr.status == TrainStatus.HELD),
            key=lambda tr: (tr.planned_departure if tr.planned_departure is not None else inf,
                            tr.train_id),
        )
        seen = set()
        releases = []
        for train in held:
            key = (train.current_station, train.direction)
            if key not in seen and train.current_station is not None:
                releases.append(ReleaseAction(train.train_id, train.current_station))
                seen.add(key)
        return ActionPlan(tuple(releases))

    def evaluate(self, plan: ActionPlan) -> Evaluation:
        # 基础运行图/外生OD在试算期间只读；共享字节而隔离可变车客状态。
        # 这使鲁棒多场景不会为每次候选重复复制整份7200秒需求矩阵。
        immutable = (self._base.base_schedule, self._base.demand)
        trial = copy.deepcopy(self._base, {id(array): array for array in immutable})
        try:
            first = trial.step(plan)
            for action in first.executed_actions["train_actions"]:
                if action.get("status") in ("rejected", "not_executed", "pending"):
                    return Evaluation(False, inf, {}, first.executed_actions,
                                      trial.current_time,
                                      action.get("reason", "本窗口列车动作未执行"))
            for _ in range(1, self._steps):
                trial.step(self._continuation_plan(trial))
        except ValueError as exc:
            return Evaluation(False, inf, {}, {}, trial.current_time, str(exc))
        end = trial.metrics()
        schedule_parts = {
            "window_schedule_shift_sec": max(0.0, end["accumulated_schedule_shift_sec"]
                                             - self._start_metrics["accumulated_schedule_shift_sec"]),
            "window_pending_schedule_shift_delta_sec": (end["pending_schedule_shift_sec"]
                                                         - self._start_metrics["pending_schedule_shift_sec"]),
            "window_cancelled_service_sec": (end["cancelled_service_sec"]
                                             - self._start_metrics["cancelled_service_sec"]),
            "window_added_reverse_service_sec": (end["added_reverse_service_sec"]
                                                 - self._start_metrics["added_reverse_service_sec"]),
            "window_cancelled_stop_count": (end["cancelled_stop_count"]
                                            - self._start_metrics["cancelled_stop_count"]),
            "window_added_reverse_stop_count": (end["added_reverse_stop_count"]
                                                - self._start_metrics["added_reverse_stop_count"]),
        }
        schedule_parts["window_schedule_adjustment_sec"] = max(0.0, sum(
            schedule_parts[key] for key in (
                "window_schedule_shift_sec", "window_pending_schedule_shift_delta_sec",
                "window_cancelled_service_sec", "window_added_reverse_service_sec")))
        metrics = {
            "wait_pax_sec": end["wait_pax_sec"] - self._start_metrics["wait_pax_sec"],
            "onboard_late_pax_sec": end["onboard_late_pax_sec"] - self._start_metrics["onboard_late_pax_sec"],
            "crowd_excess_pax_sec": end["crowd_excess_pax_sec"] - self._start_metrics["crowd_excess_pax_sec"],
            "end_waiting_pax": end["end_waiting_pax"],
            "end_onboard_pax": end["onboard_pax"],
            "end_unfinished_pax": end["arrivals_pax"] - end["completed_pax"],
            "completed_pax": end["completed_pax"] - self._start_metrics["completed_pax"],
            "accumulated_schedule_shift_sec": end["accumulated_schedule_shift_sec"],
            **schedule_parts,
            "platform_peak": end["platform_peak"],
            "conservation_gap": end["conservation_gap"],
        }
        values = {
            "passenger_waiting": metrics["wait_pax_sec"],
            "passenger_delay_proxy": metrics["onboard_late_pax_sec"],
            "platform_crowding": metrics["crowd_excess_pax_sec"],
            "train_schedule_adjustment": metrics["window_schedule_adjustment_sec"],
            "unfinished_trips": metrics["end_unfinished_pax"],
        }
        contributions = {
            key: (self.weights.as_dict()[key] * max(0.0, values[key]) / self.scales[key])
            for key in OBJECTIVE_KEYS
        }
        score = self._score_unit * sum(contributions.values())
        return Evaluation(True, score, metrics, first.executed_actions, trial.current_time,
                          objective_values=values,
                          objective_scales=dict(self.scales),
                          objective_weights=self.weights.as_dict(),
                          objective_contributions=contributions)


@dataclass(frozen=True)
class ForecastScenario:
    """仅修改决策者可用的预测恢复时间与OD预测，不读取任何未来真值。"""
    name: str
    duration_offset_seconds: int = 0
    demand_multiplier: float = 1.0
    probability: float = 1.0

    def __post_init__(self):
        if not self.name or not isfinite(self.demand_multiplier) or self.demand_multiplier < 0:
            raise ValueError("鲁棒场景名称和客流倍率必须有效")
        if not isfinite(self.probability) or self.probability <= 0:
            raise ValueError("鲁棒场景概率必须为正数")


DEFAULT_FORECAST_SCENARIOS = (
    ForecastScenario("central", 0, 1.00, 0.50),
    ForecastScenario("early_low", -300, 0.85, 0.25),
    ForecastScenario("late_high", 300, 1.15, 0.25),
)


class RobustEvaluator:
    """同一观察快照下的三场景预测仿真；所有候选共用固定场景与归一化尺度。

    使用期望 + λ*(CVaR_α-期望)；默认严格要求每个预测场景均满足硬约束。
    场景是对预测的扰动，不是环境真实actual_end/未来OD。
    """

    def __init__(self, env: Env, weights: ObjectiveWeights | None = None,
                 scenarios: tuple[ForecastScenario, ...] = DEFAULT_FORECAST_SCENARIOS,
                 *, risk_aversion: float = 0.35, alpha: float = 0.80):
        if not scenarios:
            raise ValueError("至少需要一个预测场景")
        if not isfinite(risk_aversion) or not 0 <= risk_aversion <= 1:
            raise ValueError("risk_aversion必须在[0,1]")
        if not 0 <= alpha < 1:
            raise ValueError("CVaR alpha必须在[0,1)")
        if abs(sum(s.probability for s in scenarios) - 1.0) > 1e-9:
            raise ValueError("鲁棒场景概率合计必须为1")
        self.scenarios = tuple(scenarios)
        self.risk_aversion = float(risk_aversion)
        self.alpha = float(alpha)
        self._snapshot = env.observe()
        self._template = Evaluator(env, weights)
        self.weights = self._template.weights
        self.scales = dict(self._template.scales)
        self._evaluators = tuple(self._for_scenario(s) for s in self.scenarios)

    def _for_scenario(self, scenario: ForecastScenario) -> Evaluator:
        ev = copy.copy(self._template)
        # 模板本来就是用forecast构造的预测副本；从不访问真实env._truth。
        base = self._template._base
        ev._base = copy.deepcopy(base, {id(base.base_schedule): base.base_schedule})
        fork = ev._base
        snap = self._snapshot
        now = snap.time
        # 对所有场景使用相同外生预测；乘客随机数流不会因为动作顺序改变。
        for minute, od in zip(snap.forecast_times, snap.forecast_od):
            if minute < len(fork.demand):
                fork.demand[minute] = np_rint_int64(od, scenario.demand_multiplier)
        if snap.disruption.blocked_now:
            # 截止时间只由公开predicted_end与假设扰动组成；不得取actual_end。
            estimated_end = max(now + fork._config.decision_interval,
                                snap.disruption.predicted_end + scenario.duration_offset_seconds)
            fork._truth = replace(fork._truth,
                                  actual_duration=int(estimated_end - fork._truth.start_time))
            fork._config = replace(fork._config, disruption=fork._truth)
            fork._event_queue = [e for e in fork._event_queue if e[4] != "DISRUPTION_END"]
            heapq.heapify(fork._event_queue)
            fork._push_event(estimated_end, "DISRUPTION_END", -1, -1)
        return ev

    @staticmethod
    def _cvar(values: list[tuple[float, float]], alpha: float) -> float:
        """对离散损失概率质量的最坏(1-alpha)尾部求条件均值。"""
        tail = 1.0 - alpha
        remaining = tail
        total = 0.0
        for loss, probability in sorted(values, key=lambda item: item[0], reverse=True):
            mass = min(probability, remaining)
            total += loss * mass
            remaining -= mass
            if remaining <= 1e-12:
                break
        return total / tail

    def evaluate(self, plan: ActionPlan) -> Evaluation:
        results = [(s, e.evaluate(plan)) for s, e in zip(self.scenarios, self._evaluators)]
        details = {
            "mode": "forecast_scenario_robust",
            "alpha": self.alpha,
            "risk_aversion": self.risk_aversion,
            "feasibility_policy": "all_scenarios",
            "scenario_results": [{
                "name": s.name, "probability": s.probability,
                "duration_offset_seconds": s.duration_offset_seconds,
                "demand_multiplier": s.demand_multiplier,
                "feasible": r.feasible,
                "score": float(r.score) if r.feasible else None,
                "reason": r.reason,
                "objective_values": r.objective_values,
            } for s, r in results],
        }
        rejected = [(s, r) for s, r in results if not r.feasible]
        if rejected:
            s, r = rejected[0]
            return Evaluation(False, inf, {}, r.executed_actions, r.end_time,
                              f"鲁棒场景[{s.name}]不可行：{r.reason}",
                              objective_weights=self.weights.as_dict(),
                              objective_scales=dict(self.scales), robust_details=details)
        expected = sum(s.probability * r.score for s, r in results)
        cvar = self._cvar([(r.score, s.probability) for s, r in results], self.alpha)
        score = expected + self.risk_aversion * (cvar - expected)
        details.update(expected_score=expected, cvar_score=cvar,
                       robustness_premium=score - expected)
        all_metric_keys = results[0][1].metrics.keys()
        metrics = {k: sum(s.probability * r.metrics[k] for s, r in results)
                   for k in all_metric_keys}
        values = {k: sum(s.probability * r.objective_values[k] for s, r in results)
                  for k in OBJECTIVE_KEYS}
        contributions = {k: sum(s.probability * r.objective_contributions[k]
                                for s, r in results) for k in OBJECTIVE_KEYS}
        central = results[0][1]
        return Evaluation(True, score, metrics, central.executed_actions, central.end_time,
                          objective_values=values, objective_scales=dict(self.scales),
                          objective_weights=self.weights.as_dict(),
                          objective_contributions=contributions, robust_details=details)


def np_rint_int64(array, factor):
    """在预计算OD预测上施加非负倍率；不改变观测原值或真实未来客流。"""
    import numpy as np
    return np.rint(np.asarray(array, dtype=np.float64) * factor).astype(np.int64)
