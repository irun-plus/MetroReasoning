"""在只含已知状态和预测未来的环境副本上试算一份动作方案。"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from math import inf

from .Env import ActionPlan, Env
from .TrainOperator import ReleaseAction, TrainStatus


@dataclass(frozen=True)
class Evaluation:
    feasible: bool
    score: float
    metrics: dict[str, float]
    executed_actions: dict
    end_time: int
    reason: str | None = None


class Evaluator:
    """每个候选都从同一预测副本出发，评分只看本次评价时域的增量。"""

    def __init__(self, env: Env):
        """冻结当前时刻的预测副本和本轮共同的评价时域。"""
        config = env._config
        if config.evaluation_horizon % config.decision_interval:
            raise ValueError("评价时域必须是决策窗口的整数倍")
        self._base = env.fork_for_evaluation(env.observe())
        self._start_metrics = self._base.metrics()
        self._steps = min(
            config.evaluation_horizon // config.decision_interval,
            (len(self._base.demand) - 1 - self._base.current_time) // config.decision_interval,
        )
        if self._steps < 1:
            raise StopIteration("剩余仿真时间不足一个决策窗口")

    @staticmethod
    def _continuation_plan(env: Env) -> ActionPlan:
        """预测恢复后，按原计划次序逐站向解除扣车；其余窗口无新动作。"""
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
        """返回可行性、实际执行状态和预测指标；不改变真实环境。"""
        trial = copy.deepcopy(self._base)
        try:
            first = trial.step(plan)
            for action in first.executed_actions["train_actions"]:
                if action.get("status") in ("rejected", "not_executed", "pending"):
                    return Evaluation(False, inf, {}, first.executed_actions, trial.current_time,
                                      action.get("reason", "本窗口列车动作未执行"))
            for _ in range(1, self._steps):
                trial.step(self._continuation_plan(trial))
        except ValueError as exc:
            return Evaluation(False, inf, {}, {}, trial.current_time, str(exc))
        end = trial.metrics()
        metrics = {
            "wait_pax_sec": end["wait_pax_sec"] - self._start_metrics["wait_pax_sec"],
            "crowd_excess_pax_sec": end["crowd_excess_pax_sec"] - self._start_metrics["crowd_excess_pax_sec"],
            "end_waiting_pax": end["end_waiting_pax"],
            "completed_pax": end["completed_pax"] - self._start_metrics["completed_pax"],
            "accumulated_schedule_shift_sec": end["accumulated_schedule_shift_sec"],
            "platform_peak": end["platform_peak"],
            "conservation_gap": end["conservation_gap"],
        }
        return Evaluation(True, metrics["wait_pax_sec"], metrics, first.executed_actions,
                          trial.current_time)
