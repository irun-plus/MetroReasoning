"""保存中断真值、可见决策参数及仿真假设折返进路。"""

from dataclasses import dataclass, field
import json
from math import isfinite
from pathlib import Path

from .Constant import HISTORICAL_TURNBACK_CANDIDATE_STATIONS, TURNBACK_ROUTES


METRIC_NAMES = ("wait_pax_sec", "end_waiting_pax", "remaining_delay_sec",
                "crowd_excess_pax_sec", "change_cost")


def load_metric_scales(override: dict[str, float] | None = None) -> dict[str, float]:
    """读取离线校准后冻结的五项尺度，并拒绝缺项或无效数值。"""
    if override is None:
        path = Path(__file__).with_name("metric_scales.json")
        if not path.exists():
            raise FileNotFoundError(f"缺少评分尺度文件：{path}；请先运行 python -m Env.calibration")
        override = json.loads(path.read_text(encoding="utf-8"))["scales"]
    if set(override) != set(METRIC_NAMES) or any(
            not isfinite(float(value)) or float(value) <= 0 for value in override.values()):
        raise ValueError("评分尺度必须包含五项指标，且均为正的有限数")
    return {name: float(override[name]) for name in METRIC_NAMES}


@dataclass(frozen=True)
class TurnbackRoute:
    """仿真中允许某方向列车在指定车站清客后反向开行。"""

    station: int
    approach_direction: int
    turnaround_seconds: int
    assumption: str = "历史候选站按可折返处理；未核验现行进路"


@dataclass(frozen=True)
class Weights:
    """定义等待、窗口末等待、延误、拥挤和调整动作五项业务权重。"""
    wait: float = 0.40
    end_waiting: float = 0.25
    delay: float = 0.15
    crowd: float = 0.15
    change: float = 0.05

    def __post_init__(self):
        """要求五项权重非负且总和为一，避免目标分数失去含义。"""
        values = (self.wait, self.end_waiting, self.delay, self.crowd, self.change)
        if any(value < 0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("目标权重必须非负且总和为 1")


@dataclass(frozen=True)
class DecisionConfig:
    """只包含求解器可以看到的搜索预算、权重和折返假设。"""
    decision_interval: int
    evaluation_horizon: int
    alns_iterations: int
    alns_time_limit: float
    seed: int
    weights: Weights
    turnback_routes: tuple[TurnbackRoute, ...]
    metric_scales: dict[str, float]


@dataclass(frozen=True)
class Config:
    """保存整场中断的真值和运行参数，其中实际时长只供环境执行。"""
    data_root: Path | None = None
    break_from: int = 20
    break_to: int = 21
    break_direction: int = 0
    break_start_time: int = 2100
    actual_duration: int = 1200
    predicted_duration: int = 900
    decision_interval: int = 300
    evaluation_horizon: int = 1200
    alns_iterations: int = 200
    alns_time_limit: float = 30.0
    seed: int = 42
    weight_sets: tuple[Weights, ...] = field(default_factory=lambda: (
        Weights(),
        Weights(wait=0.35, end_waiting=0.25, delay=0.10, crowd=0.25, change=0.05),
    ))
    turnback_routes: tuple[TurnbackRoute, ...] = field(default_factory=lambda: tuple(
        TurnbackRoute(*route) for route in TURNBACK_ROUTES
    ))
    metric_scales: dict[str, float] | None = None
    recovery_weight_sets: tuple[Weights, ...] | None = None

    def __post_init__(self):
        """检查有向区间、时长、搜索预算和折返假设是否自洽。"""
        if not 0 <= self.break_from < 23 or not 0 <= self.break_to < 23:
            raise ValueError("车站编号超出 23 站范围")
        if self.break_direction not in (0, 1):
            raise ValueError("方向只能取 0=下行或 1=上行")
        step = 1 if self.break_direction == 0 else -1
        if self.break_to - self.break_from != step:
            raise ValueError("中断区间的站序与方向不一致")
        if self.break_start_time < 0 or min(self.actual_duration, self.predicted_duration) <= 0:
            raise ValueError("中断开始时间或持续时间无效")
        if self.decision_interval <= 0 or self.evaluation_horizon < self.decision_interval:
            raise ValueError("决策间隔或评价窗口无效")
        if self.alns_iterations <= 0 or self.alns_time_limit <= 0:
            raise ValueError("搜索迭代数或时间上限无效")
        if any(route.station not in HISTORICAL_TURNBACK_CANDIDATE_STATIONS
               or route.approach_direction not in (0, 1)
               or (route.station == 0 and route.approach_direction != 1)
               or (route.station == 22 and route.approach_direction != 0)
               or route.turnaround_seconds <= 0 for route in self.turnback_routes):
            raise ValueError("折返假设的车站、方向或作业时间无效")
        if not self.weight_sets:
            raise ValueError("至少需要一组目标权重")
        if self.recovery_weight_sets is not None and len(self.recovery_weight_sets) != len(self.weight_sets):
            raise ValueError("恢复期权重组数必须与中断期一致")
        if self.metric_scales is not None:
            load_metric_scales(self.metric_scales)

    def weights_for(self, index: int, phase: str) -> Weights:
        """按已观察到的封锁状态取权重；未提供恢复权重时沿用试验权重。"""
        if phase == "recovery" and self.recovery_weight_sets is not None:
            return self.recovery_weight_sets[index]
        return self.weight_sets[index]

    def public(self, weights: Weights | None = None) -> DecisionConfig:
        """向 ALNS 提供公开配置，不传入实际恢复时长。"""
        return DecisionConfig(
            self.decision_interval,
            self.evaluation_horizon,
            self.alns_iterations,
            self.alns_time_limit,
            self.seed,
            weights or self.weight_sets[0],
            self.turnback_routes,
            load_metric_scales(self.metric_scales),
        )
