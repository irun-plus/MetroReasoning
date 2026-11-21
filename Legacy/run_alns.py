"""运行一场单方向中断，并可选做规则、固定 LNS 与 ALNS 对照。"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from Env.ALNS import ALNSSolver, Candidate, SearchResult
from Env.configure import Config
from Env.ALNSEnv import ALNSEnv
from Env.scenario import sample_scenario
from Env.ResultRecorder import ResultRecorder


def run_case(config: Config, output: Path, decisions: int, method: str,
             weight_index: int, until_recovered: bool = False):
    """从同一初始状态完成一组权重和一种方法的多轮决策，保存逐轮记录。"""
    env = ALNSEnv(config)
    snapshot = env.reset()
    initial_snapshot = snapshot
    recorder = ResultRecorder(output / f"weight_{weight_index}" / method)
    steps = []
    reopened_at = None
    release_completed_at = None
    for _ in range(decisions):
        if snapshot.time >= len(env.demand) - 1:
            break
        weights = config.weights_for(weight_index, snapshot.phase)
        if method == "rule":
            plan = env.initial_plan(snapshot)
            evaluation = env.evaluate(plan)
            if not evaluation.feasible:
                raise RuntimeError(f"rule baseline infeasible: {evaluation.violations[:3]}")
            scales = env.public_config(weights).metric_scales
            candidate = Candidate(plan, evaluation, evaluation.score(weights, scales), "rule", "rule")
            search = SearchResult(plan, evaluation, (candidate,), 0, 0.0, scales,
                                  {}, 1, 1)
        else:
            search = ALNSSolver(adaptive=(method == "alns")).search(
                snapshot, env.public_config(weights), env.evaluate)
            if search.best_plan is None:
                raise RuntimeError(f"{method} found no feasible baseline at t={snapshot.time}")
        feedback = env.step(search.best_plan)
        if snapshot.blocked and not feedback.next_snapshot.blocked:
            reopened_at = feedback.next_snapshot.time
        if not feedback.next_snapshot.blocked and not env.held_train_ids and release_completed_at is None:
            release_completed_at = feedback.next_snapshot.time
        record_path = recorder.record_decision(snapshot, weights, search, feedback)
        steps.append({"time": snapshot.time, "record": str(record_path),
                      "search_sec": search.elapsed_sec, "iterations": search.iterations,
                      "forecast_metrics": search.best_eval.metrics,
                      "actual_metrics": feedback.actual_metrics})
        snapshot = feedback.next_snapshot
        if until_recovered and release_completed_at is not None:
            break
    normal_wait = (env._simulate(initial_snapshot, initial_snapshot.schedule.copy(),
                                 snapshot.time, (), forecast=False)["wait_pax_sec"]
                   if steps else 0)
    actual_wait = sum(step["actual_metrics"]["wait_pax_sec"] for step in steps)
    summary = {
        "method": method, "weight_index": weight_index,
        "interruption_weights": asdict(config.weight_sets[weight_index]),
        "recovery_weights": (asdict(config.recovery_weight_sets[weight_index])
                             if config.recovery_weight_sets else None),
        "scenario": {"from": config.break_from, "to": config.break_to,
                     "direction": config.break_direction, "start": config.break_start_time,
                     "actual_duration": config.actual_duration,
                     "predicted_duration": config.predicted_duration,
                     "assumed_turnback_stations": sorted({route.station for route in config.turnback_routes}),
                     "turnback_permission": "仿真假设，未核验现行运营进路"},
        "decisions": steps,
        "total_search_sec": sum(step["search_sec"] for step in steps),
        "total_wait_pax_sec": actual_wait,
        "normal_wait_pax_sec": normal_wait,
        "extra_wait_pax_sec": actual_wait - normal_wait,
        "total_crowd_excess_pax_sec": sum(step["actual_metrics"]["crowd_excess_pax_sec"] for step in steps),
        "final_end_waiting_pax": steps[-1]["actual_metrics"]["end_waiting_pax"] if steps else None,
        "realized_departure_delay_sec": sum(step["actual_metrics"]["realized_delay_sec"] for step in steps),
        "full_timetable_delay_sec": int((env.schedule[:len(env.base_schedule), :, 1] - env.base_schedule[:, :, 1]).clip(min=0).sum()),
        "total_turnbacks": sum(step["actual_metrics"]["turnback_count"] for step in steps),
        "total_cleared_pax": sum(step["actual_metrics"]["cleared_pax"] for step in steps),
        "conservation_gap": steps[-1]["actual_metrics"]["conservation_gap"] if steps else None,
        "reopened_at": reopened_at,
        "release_completed_at": release_completed_at,
        "recovery_scope": "held_train_release",
        "recovery_status": ("not_opened" if reopened_at is None else
                            "release_in_progress" if env.held_train_ids else "held_trains_released"),
        "pending_held_trains": len(env.held_train_ids),
        "turnback_links_unassigned": len(env.turnback_links),
    }
    path = recorder.directory / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    """读取命令行参数，构建中断场景并逐组运行、打印中文摘要。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "Result" / "alns")
    parser.add_argument("--from-station", type=int, default=20)
    parser.add_argument("--to-station", type=int, default=21)
    parser.add_argument("--direction", type=int, default=0)
    parser.add_argument("--start", type=int, default=2100)
    parser.add_argument("--actual-duration", type=int, default=1200)
    parser.add_argument("--predicted-duration", type=int, default=900)
    parser.add_argument("--sample-seed", type=int,
                        help="调用 Env/scenario.py 抽取单方向中断位置、开始时间和实际时长")
    parser.add_argument("--decisions", type=int, default=5)
    parser.add_argument("--until-recovered", action="store_true",
                        help="运行到待令车放完；受 --max-decisions 上限保护")
    parser.add_argument("--max-decisions", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--weight-index", type=int, choices=(0, 1))
    parser.add_argument("--benchmark", action="store_true",
                        help="从同一初始状态对比规则、固定算子 LNS 和自适应 ALNS")
    args = parser.parse_args()
    sampled = None
    if args.sample_seed is not None:
        data_root = args.data_root or Path(__file__).resolve().parent / "DataBase" / "npy"
        schedule = np.load(data_root / "train_schedule.npy")
        for offset in range(1000):
            candidate = sample_scenario(args.sample_seed + offset,
                start_range_seconds=(1800, 2400), duration_range_seconds=(900, 1500))
            departure = schedule[:, candidate.from_station, 1]
            arrival = schedule[:, candidate.to_station, 0]
            occupied = np.any((departure > 0) & (departure <= candidate.start_seconds)
                              & (arrival > candidate.start_seconds))
            if not occupied:
                sampled = candidate
                break
        if sampled is None:
            parser.error("连续 1000 次抽样均未找到中断开始时无列车占用的有向区间")
        print(f"实际采用的中断场景种子：{sampled.seed}")
    config = Config(data_root=args.data_root,
                    break_from=sampled.from_station if sampled else args.from_station,
                    break_to=sampled.to_station if sampled else args.to_station,
                    break_direction=sampled.direction if sampled else args.direction,
                    break_start_time=sampled.start_seconds if sampled else args.start,
                    actual_duration=sampled.duration_seconds if sampled else args.actual_duration,
                    predicted_duration=args.predicted_duration,
                    seed=args.sample_seed if sampled else 42,
                    alns_iterations=args.iterations, alns_time_limit=args.time_limit)
    if args.decisions <= 0:
        parser.error("--decisions 必须为正整数")
    if args.max_decisions <= 0:
        parser.error("--max-decisions 必须为正整数")
    indices = range(len(config.weight_sets)) if args.weight_index is None else (args.weight_index,)
    methods = ("rule", "lns", "alns") if args.benchmark else ("alns",)
    run_output = args.output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    summaries = []
    for index in indices:
        for method in methods:
            budget = args.max_decisions if args.until_recovered else args.decisions
            summary = run_case(config, run_output, budget, method, index,
                               until_recovered=args.until_recovered)
            summaries.append(summary)
            label = {"rule": "规则", "lns": "固定算子 LNS", "alns": "自适应 ALNS"}[method]
            recovery_label = {"not_opened": "仍封锁", "release_in_progress": "正在放行",
                              "held_trains_released": "待令车已放行"}[summary["recovery_status"]]
            print(f"W{index} {label}：决策 {len(summary['decisions'])} 轮，"
                  f"累计等待 {summary['total_wait_pax_sec']:.0f} 人秒，"
                  f"比无中断多 {summary['extra_wait_pax_sec']:.0f} 人秒，"
                  f"末轮站外及站台等待 {summary['final_end_waiting_pax']:.0f} 人，"
                  f"折返 {summary['total_turnbacks']:.0f} 列，"
                  f"待令余车 {summary['pending_held_trains']} 列，"
                  f"恢复状态 {recovery_label}，"
                  f"搜索 {summary['total_search_sec']:.2f} 秒")
    (run_output / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果目录：{run_output}")


if __name__ == "__main__":
    main()
