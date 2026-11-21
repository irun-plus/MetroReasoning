"""按随机种子复现单方向区间中断场景。

列车计划和客流仍从 DataBase 读取；抽样范围由运行方指定，属于实验假设。
"""

from dataclasses import dataclass
from random import Random

try:
    from .Constant import DOWN_DIRECTION, STATION_NUM, UP_DIRECTION
except ImportError:  # Allow direct execution while exploring a single case.
    from Constant import DOWN_DIRECTION, STATION_NUM, UP_DIRECTION


@dataclass(frozen=True)
class DisruptionScenario:
    """记录一次中断的有向区间、开始时刻和实际持续时间。"""
    seed: int
    from_station: int
    to_station: int
    direction: int
    start_seconds: int
    duration_seconds: int

    @property
    def end_seconds(self) -> int:
        """返回这次中断的实际恢复时刻，供仿真环境使用。"""
        return self.start_seconds + self.duration_seconds


def sample_scenario(
    seed: int,
    *,
    start_range_seconds: tuple[int, int],
    duration_range_seconds: tuple[int, int],
    horizon_seconds: int = 7200,
) -> DisruptionScenario:
    """按种子抽取一个有向区间、开始时刻和时长，并确保结束不超出数据范围。"""
    start_min, start_max = start_range_seconds
    duration_min, duration_max = duration_range_seconds
    if start_min < 0 or start_min > start_max:
        raise ValueError("中断开始时间抽样范围无效")
    if duration_min <= 0 or duration_min > duration_max:
        raise ValueError("中断持续时间抽样范围无效")
    if start_max + duration_max > horizon_seconds:
        raise ValueError("抽样中断可能延伸到数据范围之外")

    rng = Random(seed)
    segment = rng.randrange(STATION_NUM - 1)
    direction = rng.choice((DOWN_DIRECTION, UP_DIRECTION))
    if direction == DOWN_DIRECTION:
        from_station, to_station = segment, segment + 1
    else:
        from_station, to_station = segment + 1, segment

    return DisruptionScenario(
        seed=seed,
        from_station=from_station,
        to_station=to_station,
        direction=direction,
        start_seconds=rng.randint(start_min, start_max),
        duration_seconds=rng.randint(duration_min, duration_max),
    )
