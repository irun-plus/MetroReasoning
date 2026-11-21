"""V1中断配置与场景抽样。实际恢复真值仅由Env内部持有；快照只暴露预测信息。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Iterable
import numpy as np
from . import Constant as C


@dataclass(frozen=True)
class DisruptionTruth:
    """环境内部真值；不得传给决策器。"""
    from_station: int
    to_station: int
    direction: int
    start_time: int
    actual_duration: int
    predicted_duration: int

    @property
    def actual_end(self) -> int:
        return self.start_time + self.actual_duration

    @property
    def predicted_end(self) -> int:
        return self.start_time + self.predicted_duration


@dataclass(frozen=True)
class TurnbackFacility:
    station: int
    approach_direction: int
    turnaround_seconds: int
    resource_id: str
    source: str = "待核实假设"


@dataclass(frozen=True)
class DecisionConfig:
    """仅含求解器需要的公开运行参数，不暴露中断真值。"""
    decision_interval: int
    turnback_facilities: tuple[TurnbackFacility, ...]


@dataclass(frozen=True)
class Config:
    data_root: Path | str | None = None
    disruption: DisruptionTruth = field(default_factory=lambda: DisruptionTruth(20, 21, 0, 2100, 1200, 900))
    decision_interval: int = C.DECISION_INTERVAL_SECONDS
    evaluation_horizon: int = 1200
    forecast_horizon: int = 1200
    seed: int = 42
    turnback_facilities: tuple[TurnbackFacility, ...] = field(default_factory=lambda: tuple(TurnbackFacility(**x) for x in C.TURNBACK_FACILITIES))
    demand_forecast_minutes: int = 5

    def __post_init__(self):
        d = self.disruption
        if d.direction not in (C.DOWN_DIRECTION, C.UP_DIRECTION):
            raise ValueError("方向只能为0=下行或1=上行")
        if not (0 <= d.from_station < C.STATION_NUM and 0 <= d.to_station < C.STATION_NUM):
            raise ValueError("中断车站编号越界")
        if d.to_station - d.from_station != C.DIRECTION_STEP[d.direction]:
            raise ValueError("中断区间站序与方向不一致")
        if d.start_time < 0 or d.actual_duration <= 0 or d.predicted_duration <= 0:
            raise ValueError("中断时刻或持续时间无效")
        if self.decision_interval <= 0 or self.evaluation_horizon < self.decision_interval:
            raise ValueError("必须满足H_decision <= H_eval")
        if self.forecast_horizon < self.evaluation_horizon:
            raise ValueError("必须满足H_eval <= H_forecast")
        if self.demand_forecast_minutes <= 0:
            raise ValueError("客流预测回看分钟数必须为正")
        keys = [(f.station, f.approach_direction) for f in self.turnback_facilities]
        if len(keys) != len(set(keys)):
            raise ValueError("折返设施配置存在重复的车站-来车方向")
        for f in self.turnback_facilities:
            if not (0 <= f.station < C.STATION_NUM) or f.approach_direction not in (0, 1) or f.turnaround_seconds <= 0:
                raise ValueError("折返设施参数无效")

    @property
    def resolved_data_root(self) -> Path:
        """以项目目录为基准定位数据，避免依赖盘符或启动时的工作目录。"""
        project_root = Path(__file__).resolve().parents[1]
        if self.data_root is not None:
            path = Path(self.data_root)
            return path if path.is_absolute() else project_root / path
        return project_root / "DataBase" / "beijing_5"

    def public(self) -> DecisionConfig:
        """提取当前决策所需的折返设施和窗口长度。"""
        return DecisionConfig(self.decision_interval, self.turnback_facilities)


def edge_occupancy(schedule: np.ndarray, station: int, direction: int) -> list[tuple[int, int, int]]:
    """由无中断基础运行图计算有向区间占用：(进入时刻,离开时刻,train_id)。"""
    if direction not in (0, 1) or not 0 <= station < C.STATION_NUM:
        raise ValueError("区间或方向无效")
    nxt = station + C.DIRECTION_STEP[direction]
    if not 0 <= nxt < C.STATION_NUM:
        raise ValueError("指定方向不存在相邻区间")
    occupied = []
    for tid in range(schedule.shape[0]):
        dep = int(schedule[tid, station, 1])
        arr = int(schedule[tid, nxt, 0])
        d = int(schedule[tid, station, 2])
        if d == direction and dep >= 0 and arr > dep and (dep > 0 or arr > 0):
            occupied.append((dep, arr, tid))
    return sorted(occupied)


def sample_disruption(
    schedule: np.ndarray,
    *, seed: int,
    start_range: tuple[int, int],
    duration_range: tuple[int, int],
    predicted_duration: int,
    time_step: int = 60,
    horizon_seconds: int = 7200,
) -> DisruptionTruth:
    """先均匀抽有向区间，再在该区间可行时刻中抽样；无可行时刻直接报告，不重抽区间。"""
    lo, hi = start_range
    dmin, dmax = duration_range
    if lo < 0 or hi < lo or dmin <= 0 or dmax < dmin or time_step <= 0:
        raise ValueError("中断抽样范围无效")
    rng = Random(seed)
    edges = [(segment if d == C.DOWN_DIRECTION else segment + 1, d)
             for segment in range(C.STATION_NUM - 1) for d in (0, 1)]
    station, direction = rng.choice(edges)
    duration = rng.randint(dmin, dmax)
    if predicted_duration <= 0:
        raise ValueError("预测持续时间必须为正数，且应独立于实际持续时间设置")
    if hi + duration > horizon_seconds:
        raise ValueError("抽样时段与持续时间可能超出仿真数据范围")
    occupied = edge_occupancy(schedule, station, direction)
    candidates = []
    for start in range(((lo + time_step - 1) // time_step) * time_step, hi + 1, time_step):
        if not any(a <= start < b for a, b, _ in occupied):
            candidates.append(start)
    if not candidates:
        raise RuntimeError(f"抽中的区间({station},{direction})在指定时段无可行中断开始时刻；未重抽区间以避免分布偏差")
    start = rng.choice(candidates)
    to_station = station + C.DIRECTION_STEP[direction]
    return DisruptionTruth(station, to_station, direction, start, duration, predicted_duration)
