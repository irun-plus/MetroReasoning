"""把每轮调度输入、候选、选中动作和真实反馈写成可核验 JSON。"""

import json
from dataclasses import asdict
from pathlib import Path

from .ALNS import SearchResult
from .configure import Weights
from .ALNSEnv import Feedback, Snapshot


class ResultRecorder:
    """负责在指定结果目录持久化逐决策记录。"""
    def __init__(self, directory: Path):
        """创建结果目录并保存其位置。"""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def record_decision(self, snapshot: Snapshot, weights: Weights,
                        search: SearchResult, feedback: Feedback) -> Path:
        """将本轮快照、权重、候选、已选方案和反馈原子写入 JSON。"""
        document = {
            "snapshot": snapshot.to_dict(),
            "weight_set": asdict(weights),
            "search": search.to_dict(),
            "candidates": [candidate.to_dict() for candidate in search.candidates],
            "chosen_plan": search.best_plan.to_dict() if search.best_plan else None,
            "station_control_window": {"start": snapshot.time,
                                       "end": feedback.next_snapshot.time,
                                       "controls": [asdict(control) for control in
                                                    feedback.executed_plan.station_controls]},
            "feedback": {
                "executed_plan": feedback.executed_plan.to_dict(),
                "actual_metrics": feedback.actual_metrics,
                "next_snapshot": feedback.next_snapshot.to_dict(),
            },
        }
        path = self.directory / f"decision_{snapshot.time}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path
