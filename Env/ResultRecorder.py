"""把每轮预测决策与真实执行反馈分别保存为可核验 JSON。"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from .Env import ActionPlan, Snapshot, StepResult
from .Evaluator import Evaluation
from .ALNS import SearchResult
from .PassengerOperator import EntryControlAction
from .TrainOperator import DelayAction, HoldingAction, ReleaseAction, TurnbackAction


def plan_to_dict(plan: ActionPlan) -> dict:
    """将逐车动作和进站配额转换为带动作类型的 JSON 数据。"""
    names = {DelayAction: "delay", HoldingAction: "hold",
             ReleaseAction: "release", TurnbackAction: "turnback",
             EntryControlAction: "entry_control"}
    def encode(action):
        """为动作参数补上明确的类型名。"""
        return {"type": names[type(action)], **asdict(action)}
    return {"train_actions": [encode(x) for x in plan.train_actions],
            "passenger_actions": [encode(x) for x in plan.passenger_actions]}


def evaluation_to_dict(result: Evaluation) -> dict:
    """保存原始分项指标以及当前单一等待指标评分。"""
    return {"feasible": result.feasible, "score_wait_pax_sec": result.score,
            "metrics": result.metrics, "reason": result.reason,
            "predicted_end_time": result.end_time,
            "predicted_first_step_execution": result.executed_actions}


class ResultRecorder:
    """一场运行对应一个结果目录；每轮写一份记录和一份全场汇总。"""

    def __init__(self, output_root: Path, mode: str):
        """在项目相对结果根目录下建立当前求解模式的目录。"""
        self.directory = output_root / mode
        self.directory.mkdir(parents=True, exist_ok=True)

    def record_round(self, index: int, snapshot: Snapshot, search: SearchResult | None,
                     plan: ActionPlan, predicted: Evaluation, actual: StepResult) -> None:
        """同时保留决策时可见输入、候选证据、预测选择和真实反馈。"""
        evidence = None
        if search is not None:
            evidence = {
                "baseline": evaluation_to_dict(search.baseline_evaluation),
                "candidates_generated": search.candidates_generated,
                "candidates_feasible": search.candidates_feasible,
                "infeasible_by_reason": search.infeasible_by_reason,
                "iterations_completed": search.iterations_completed,
                "search_seconds": search.search_seconds,
                "operator_weights": search.operator_weights,
                "top_candidates": [
                    {"plan": plan_to_dict(p), "evaluation": evaluation_to_dict(e)}
                    for p, e in search.candidates
                ],
                "best_turnback": (
                    {"plan": plan_to_dict(search.best_turnback[0]),
                     "evaluation": evaluation_to_dict(search.best_turnback[1])}
                    if search.best_turnback else None
                ),
            }
        payload = {
            "decision_time": snapshot.time,
            "phase": snapshot.phase,
            "snapshot": snapshot.to_dict(),
            "search_evidence": evidence,
            "selected_plan": plan_to_dict(plan),
            "selected_prediction": evaluation_to_dict(predicted),
            "actual_feedback": {
                "executed_actions": actual.executed_actions,
                "metrics": actual.metrics,
                "next_time": actual.snapshot.time,
            },
        }
        self._write(f"round_{index:03d}.json", payload)

    def record_summary(self, summary: dict) -> None:
        """记录场景真值和全场结果；真值不写进逐轮模型输入。"""
        self._write("summary.json", summary)

    def _write(self, name: str, data: dict) -> None:
        """以 UTF-8 编码保存中文字段和动作说明。"""
        (self.directory / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
