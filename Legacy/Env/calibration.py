"""用固定的中断、时长和客流压力组合生成可复现的评分尺度。"""

import argparse
import json
from pathlib import Path

import numpy as np

from .ALNSEnv import ALNSEnv, StationControl, TrainAction, TrainPlan
from .configure import Config, METRIC_NAMES


def derive_scales(rows: list[dict[str, float]],
                  fallback_rows: list[dict[str, float]] | None = None) -> dict[str, float]:
    """取实测负荷正值的第 75 百分位；仅缺失指标才借助压力场景。"""
    scales = {}
    for name in METRIC_NAMES:
        values = [float(row[name]) for row in rows if float(row[name]) > 0]
        if not values and fallback_rows:
            values = [float(row[name]) for row in fallback_rows if float(row[name]) > 0]
        if not values:
            raise ValueError(f"校准场景中没有正值：{name}；请补覆盖该指标的场景")
        scales[name] = float(np.percentile(values, 75))
    return scales


def _free_start(schedule: np.ndarray, station: int) -> int:
    """选取接近 2100 秒且故障区间内无车的中断开始时刻。"""
    departure = schedule[:, station, 1]
    arrival = schedule[:, station + 1, 0]
    for offset in range(901):
        for second in (2100 + offset, 2100 - offset):
            if second < 1200 or second > 3000:
                continue
            occupied = np.any((departure > 0) & (departure <= second)
                              & (arrival > second))
            if not occupied:
                return second
    raise ValueError(f"站{station}→{station + 1}找不到无车占用的中断起点")


def calibrate(data_root: Path) -> dict:
    """试算 27 个组合中的待令、限流和逐车折返方案，保留可行指标。"""
    schedule = np.load(data_root / "train_schedule.npy")
    positions = (6, 12, 20)
    durations = (300, 900, 1800)
    demand_factors = (0.5, 1.0, 4.0)
    rows = []
    measured_rows = []
    stress_rows = []
    cases = []
    for station in positions:
        start = _free_start(schedule, station)
        for duration in durations:
            for factor in demand_factors:
                env = ALNSEnv(Config(data_root=data_root, break_from=station,
                                     break_to=station + 1, break_start_time=start,
                                     actual_duration=duration,
                                     predicted_duration=duration))
                if factor != 1.0:
                    env.demand = np.rint(env.demand * factor).astype(np.int64)
                    env.demand_times = np.flatnonzero(
                        env.demand.sum(axis=(1, 2, 3))).tolist()
                snapshot = env.reset()
                base = env.initial_plan(snapshot)
                plans = [base]
                for limit in (0, 300, 600):
                    plans.append(TrainPlan(base.train_actions,
                                           (StationControl(station, 0, limit),)))
                for route in snapshot.turnback_routes:
                    turn_station = route.station
                    if route.approach_direction != 0 or not 0 < turn_station <= station:
                        continue
                    for train_id in range(len(snapshot.schedule)):
                        arrival = int(snapshot.schedule[train_id, turn_station, 0])
                        if (int(snapshot.schedule[train_id, turn_station, 2]) != 0
                                or not snapshot.time < arrival <= snapshot.time + 300
                                or snapshot.schedule[train_id, turn_station + 1, 0] <= arrival):
                            continue
                        actions = tuple(a for a in base.train_actions if a.train_id != train_id)
                        plans.append(TrainPlan(actions + (
                            TrainAction(train_id, turn_station, "turnback", 0),)))
                feasible = 0
                for plan in plans:
                    result = env.evaluate(plan, snapshot)
                    if result.feasible:
                        rows.append(result.metrics)
                        if factor == 1.0:
                            measured_rows.append(result.metrics)
                        elif factor == 4.0:
                            stress_rows.append(result.metrics)
                        feasible += 1
                cases.append({"from_station": station, "start": start,
                              "duration": duration, "demand_factor": factor,
                              "tried": len(plans), "feasible": feasible})
    sources = {name: ("measured_peak" if any(row[name] > 0 for row in measured_rows)
                      else "synthetic_stress") for name in METRIC_NAMES}
    return {"method": "measured_positive_p75_with_stress_fallback", "scenarios": len(cases),
            "note": "0.5 和 4.0 倍客流是合成压力试验，不是实测平峰或超高峰",
            "cases": cases, "feasible_candidates": len(rows),
            "scale_sources": sources,
            "scales": derive_scales(measured_rows, stress_rows)}


def main():
    """运行离线校准并保存固定评分尺度；不修改原始 OD 或时刻表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "DataBase" / "npy")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("metric_scales.json"))
    args = parser.parse_args()
    result = calibrate(args.data_root)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"校准 {result['scenarios']} 场，评分尺度保存到 {args.output}")


if __name__ == "__main__":
    main()
