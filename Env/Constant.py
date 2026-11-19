"""北京地铁5号线V1物理环境常量。

注意：数据来源或运营许可尚未核实的数值均显式标注为建模假设，
不能将本文件参数解释为现场行车命令或安全认证参数。
"""
from __future__ import annotations

STATION_NAMES = (
    "天通苑北", "天通苑", "天通苑南", "立水桥", "立水桥南", "北苑路北",
    "大屯路东", "惠新西街北口", "惠新西街南口", "和平西桥", "和平里北街",
    "雍和宫", "北新桥", "张自忠路", "东四", "灯市口", "东单", "崇文门",
    "磁器口", "天坛东门", "蒲黄榆", "刘家窑", "宋家庄",
)
STATION_NUM = len(STATION_NAMES)
DOWN_DIRECTION = 0  # 下行：站序递增
UP_DIRECTION = 1    # 上行：站序递减
DIRECTION_STEP = {DOWN_DIRECTION: 1, UP_DIRECTION: -1}

PLATFORM_FORMS = (
    "side", "side", "side", "side", "side", "side", "side", "side",
    "island", "island", "island", "staggered_island", "island", "island",
    "island", "island", "island", "island", "island", "island", "island",
    "island", "separated_island",
)
SEPARATE_DIRECTION_CAPACITY_STATIONS = frozenset((*range(8), 22))
PLATFORM_AREA_M2 = (
    2000, 2000, 2000, 1800, 2000, 1700, 2000, 1900, 1900, 1300,
    1300, 2100, 1300, 1300, 2200, 1300, 1900, 2400, 1300, 1300,
    1300, 1300, 3000,
)
# 原型沿用的密度假设：0.85人/m²预警，1.0人/m²模型上限。
PLATFORM_WARNING_DENSITY_ASSUMPTION = 0.85
PLATFORM_UPPER_DENSITY_ASSUMPTION = 1.0
PLATFORM_WARNING_PEOPLE = tuple(int(x * PLATFORM_WARNING_DENSITY_ASSUMPTION) for x in PLATFORM_AREA_M2)
PLATFORM_CAPACITY_PEOPLE = tuple(int(x * PLATFORM_UPPER_DENSITY_ASSUMPTION) for x in PLATFORM_AREA_M2)
OUTSIDE_QUEUE_LIMIT_PEOPLE = (None,) * STATION_NUM  # 未取得可靠数据，不虚构硬上限

TRAIN_CAPACITY_PEOPLE = 1440  # 待核实的模型参数
MIN_TRAIN_HEADWAY_SECONDS = 70  # 待核实的模型参数
DECISION_INTERVAL_SECONDS = 300
DWELL_SECONDS = (
    35, 40, 30, 50, 30, 30, 50, 30, 50, 30, 30, 50, 30, 30, 45,
    35, 50, 50, 45, 35, 45, 44, 38,
)
RUN_SECONDS_DOWN = (
    86, 86, 119, 117, 111, 181, 129, 94, 91, 93, 104, 85, 74,
    87, 81, 82, 79, 79, 97, 133, 82, 150,
)
RUN_SECONDS_UP = (
    85, 86, 120, 122, 110, 180, 133, 99, 89, 93, 98, 81, 78,
    89, 81, 83, 79, 80, 98, 133, 81, 136,
)

# V1暂用历史候选站构造仿真折返设施；现行联锁进路许可仍需核实。
HISTORICAL_TURNBACK_CANDIDATE_STATIONS = frozenset((0, 3, 6, 10, 16, 19, 22))
TURNBACK_TIME_ASSUMPTION_SECONDS = 300
STATION_10_TURNBACK_TIME_ASSUMPTION_SECONDS = 120
TURNBACK_FACILITIES = tuple(
    {
        "station": station,
        "approach_direction": direction,
        "turnaround_seconds": STATION_10_TURNBACK_TIME_ASSUMPTION_SECONDS
            if station == 10 else TURNBACK_TIME_ASSUMPTION_SECONDS,
        "resource_id": f"turnback_station_{station}",
        "source": "历史候选站与作业时间假设，未核验现行联锁进路",
    }
    for station in sorted(HISTORICAL_TURNBACK_CANDIDATE_STATIONS)
    for direction in ((1,) if station == 0 else ((0,) if station == STATION_NUM - 1 else (0, 1)))
)

assert len(PLATFORM_FORMS) == len(PLATFORM_AREA_M2) == len(DWELL_SECONDS) == STATION_NUM
assert len(RUN_SECONDS_DOWN) == len(RUN_SECONDS_UP) == STATION_NUM - 1
assert len(PLATFORM_CAPACITY_PEOPLE) == STATION_NUM
