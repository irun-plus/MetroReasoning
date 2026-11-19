"""运行 V1 环境中的规则、固定 LNS 或自适应 ALNS。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from Env import ActionPlan, Config, Env
from Env.ALNS import ALNS
from Env.Evaluator import Evaluator
from Env.ResultRecorder import ResultRecorder


PROJECT_ROOT = Path(__file__).resolve().parent


def run_one(mode: str, config: Config, *, rounds: int, iterations: int, seed: int,
            output_root: Path) -> dict:
    """从同一中断初态逐轮决策，只执行每轮选中方案的下一个 300 秒。"""
    if mode not in ("rule", "lns", "alns") or rounds <= 0:
        raise ValueError("mode 必须是 rule、lns 或 alns，rounds 必须为正数")
    env = Env(config)
    env.reset()
    recorder = ResultRecorder(output_root, mode)
    first_metrics = env.metrics()
    total_search_seconds = 0.0
    turnback_count = 0
    feasible_turnback_rounds = 0
    completed = 0
    for index in range(1, rounds + 1):
        snapshot = env.observe()
        try:
            evaluator = Evaluator(env)
        except StopIteration:
            break
        if mode == "rule":
            plan = ActionPlan()
            predicted = evaluator.evaluate(plan)
            search = None
        else:
            search = ALNS(iterations=iterations, seed=seed + index - 1,
                          adaptive=(mode == "alns")).solve(snapshot, config.public(), evaluator)
            plan = search.best_plan
            predicted = search.best_evaluation
            total_search_seconds += search.search_seconds
            feasible_turnback_rounds += int(search.best_turnback is not None)
        actual = env.step(plan)
        turnback_count += sum(
            a.get("type") == "turnback" and a.get("status") == "executed"
            for a in actual.executed_actions["train_actions"])
        recorder.record_round(index, snapshot, search, plan, predicted, actual)
        completed += 1
    final = env.metrics()
    summary = {
        "mode": mode,
        "rounds": completed,
        "scenario_truth_for_audit_only": asdict(config.disruption),
        "objective": "预测评价时域内站外和站台乘客等待人秒最小",
        "evaluation_horizon_seconds": config.evaluation_horizon,
        "decision_interval_seconds": config.decision_interval,
        "iterations_per_decision": iterations if mode != "rule" else 0,
        "seed": seed,
        "actual_wait_pax_sec_since_disruption": final["wait_pax_sec"] - first_metrics["wait_pax_sec"],
        "actual_end_waiting_pax": final["end_waiting_pax"],
        "actual_turnbacks_started": turnback_count,
        "rounds_with_feasible_turnback_candidate": (
            None if mode == "rule" else feasible_turnback_rounds),
        "search_seconds": total_search_seconds,
        "final_time": env.current_time,
        "conservation_gap": final["conservation_gap"],
        "result_directory": str(recorder.directory),
    }
    recorder.record_summary(summary)
    return summary


def main() -> None:
    """解析运行参数，并在项目内 Result/alns_v1 保存逐轮与汇总结果。"""
    parser = argparse.ArgumentParser(description="北京地铁5号线 V1 ALNS 决策实验")
    parser.add_argument("--mode", choices=("rule", "lns", "alns", "benchmark"), default="alns")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_root = args.output or (
        PROJECT_ROOT / "Result" / "alns_v1" / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    modes = ("rule", "lns", "alns") if args.mode == "benchmark" else (args.mode,)
    summaries = []
    for mode in modes:
        summary = run_one(mode, Config(), rounds=args.rounds, iterations=args.iterations,
                          seed=args.seed, output_root=output_root)
        summaries.append(summary)
        turnback_note = ("未搜索折返候选" if summary["rounds_with_feasible_turnback_candidate"] is None
                         else f"可行折返候选出现 {summary['rounds_with_feasible_turnback_candidate']} 轮")
        print(f"{mode}: 决策 {summary['rounds']} 轮，实际等待 "
              f"{summary['actual_wait_pax_sec_since_disruption']:.0f} 人秒，"
              f"折返启动 {summary['actual_turnbacks_started']} 列，"
              f"{turnback_note}，"
              f"搜索 {summary['search_seconds']:.2f} 秒")
    if len(summaries) > 1:
        (output_root / "comparison.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果目录：{output_root}")


if __name__ == "__main__":
    main()
