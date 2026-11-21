"""运行 V1 环境中的规则、固定 LNS 或自适应 ALNS。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

from Env import ActionPlan, Config, Env
from Env.ALNS import ALNS
from Env.Evaluator import Evaluator, RobustEvaluator, ObjectiveWeights
from Env.ResultRecorder import ResultRecorder


PROJECT_ROOT = Path(__file__).resolve().parent


def run_one(mode: str, config: Config, *, rounds: int, iterations: int, seed: int,
            output_root: Path, objective_weights: ObjectiveWeights | None = None,
            search_profile: str = "enhanced", robust: bool = True,
            risk_aversion: float = 0.35, cvar_alpha: float = 0.80,
            max_evaluations: int | None = None) -> dict:
    """从同一中断初态逐轮决策，只执行每轮选中方案的下一个 300 秒。"""
    if mode not in ("rule", "lns", "alns") or rounds <= 0:
        raise ValueError("mode 必须是 rule、lns 或 alns，rounds 必须为正数")
    env = Env(config)
    env.reset()
    recorder = ResultRecorder(output_root, mode)
    first_metrics = env.metrics()
    total_search_seconds = 0.0
    total_unique_evaluations = 0
    turnback_count = 0
    feasible_turnback_rounds = 0
    completed = 0
    for index in range(1, rounds + 1):
        snapshot = env.observe()
        try:
            evaluator = (RobustEvaluator(env, weights=objective_weights,
                                         risk_aversion=risk_aversion, alpha=cvar_alpha)
                         if robust else Evaluator(env, weights=objective_weights))
        except StopIteration:
            break
        if mode == "rule":
            plan = ActionPlan()
            predicted = evaluator.evaluate(plan)
            search = None
        else:
            search = ALNS(iterations=iterations, seed=seed + index - 1,
                          adaptive=(mode == "alns"), profile=search_profile,
                          max_evaluations=max_evaluations).solve(
                              snapshot, config.public(), evaluator)
            plan = search.best_plan
            predicted = search.best_evaluation
            total_search_seconds += search.search_seconds
            total_unique_evaluations += search.unique_candidates_evaluated
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
        "objective": "预测评价时域内五目标归一化加权最小化（合成分数，非人秒）",
        "objective_weights": (objective_weights or ObjectiveWeights()).as_dict(),
        "search_profile": search_profile,
        "evaluation_mode": "robust_forecast_scenarios" if robust else "single_forecast",
        "risk_aversion": risk_aversion if robust else None,
        "cvar_alpha": cvar_alpha if robust else None,
        "passenger_delay_metric_note": "车内停站超计划发车及区间排队超计划到站人秒，仅为OD到达延误代理",
        "schedule_adjustment_metric_note": "新增到发晚点、待发晚点变化、折返取消及新增服务时长的列车秒合计",
        "evaluation_horizon_seconds": config.evaluation_horizon,
        "decision_interval_seconds": config.decision_interval,
        "iterations_per_decision": iterations if mode != "rule" else 0,
        "max_evaluations_per_decision": max_evaluations if mode != "rule" else None,
        "total_unique_evaluations": total_unique_evaluations if mode != "rule" else 1 * completed,
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
    parser.add_argument("--max-evaluations", type=int,
                        help="每轮最多调用多少次物理试算（含初始方案与搜索探针）；对比策略时设为相同值")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objective-weights", type=float, nargs=5,
                        metavar=("WAIT", "DELAY", "CROWD", "SHIFT", "UNFINISHED"),
                        default=(0.2, 0.2, 0.2, 0.2, 0.2),
                        help="五目标权重，按等待/车内延误代理/拥挤/运行图调整/未完成出行输入，合计必须为1")
    parser.add_argument("--profile", choices=("standard", "enhanced"), default="enhanced",
                        help="standard=原随机邻域基线；enhanced=拓扑感知+车客耦合")
    parser.add_argument("--single-scenario", action="store_true",
                        help="关闭鲁棒多场景评价，使用原单场景五目标Evaluator")
    parser.add_argument("--risk-aversion", type=float, default=0.35,
                        help="CVaR尾部风险混合比例[0,1]")
    parser.add_argument("--cvar-alpha", type=float, default=0.80,
                        help="CVaR置信水平[0,1)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_root = args.output or (
        PROJECT_ROOT / "Result" / "alns_v1" / f"run_{datetime.now():%Y%m%d_%H%M%S_%f}")
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    modes = ("rule", "lns", "alns") if args.mode == "benchmark" else (args.mode,)
    weights = ObjectiveWeights.from_sequence(args.objective_weights)
    summaries = []
    for mode in modes:
        summary = run_one(mode, Config(), rounds=args.rounds, iterations=args.iterations,
                          seed=args.seed, output_root=output_root, objective_weights=weights,
                          search_profile=args.profile, robust=not args.single_scenario,
                          risk_aversion=args.risk_aversion, cvar_alpha=args.cvar_alpha,
                          max_evaluations=args.max_evaluations)
        summaries.append(summary)
        turnback_note = ("未搜索折返候选" if summary["rounds_with_feasible_turnback_candidate"] is None
                         else f"可行折返候选出现 {summary['rounds_with_feasible_turnback_candidate']} 轮")
        print(f"{mode}: 决策 {summary['rounds']} 轮，实际等待 "
              f"{summary['actual_wait_pax_sec_since_disruption']:.0f} 人秒，"
              f"折返启动 {summary['actual_turnbacks_started']} 列，"
              f"{turnback_note}，"
              f"试算 {summary['total_unique_evaluations']} 次，"
              f"搜索 {summary['search_seconds']:.2f} 秒")
    if len(summaries) > 1:
        (output_root / "comparison.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果目录：{output_root}")


if __name__ == "__main__":
    main()
