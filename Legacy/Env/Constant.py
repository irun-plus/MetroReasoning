"""MetroReasoning 原型使用的北京地铁 5 号线基础参数。

车站顺序和站台形式依据本地 5 号线核验资料。
"""

STATION_NAMES = (
    "天通苑北", "天通苑", "天通苑南", "立水桥", "立水桥南", "北苑路北",
    "大屯路东", "惠新西街北口", "惠新西街南口", "和平西桥", "和平里北街",
    "雍和宫", "北新桥", "张自忠路", "东四", "灯市口", "东单", "崇文门",
    "磁器口", "天坛东门", "蒲黄榆", "刘家窑", "宋家庄",
)
STATION_NUM = len(STATION_NAMES)
DOWN_DIRECTION = 0  # 下行：从站 0 到站 22
UP_DIRECTION = 1    # 上行：从站 22 到站 0

# 站台形式：side 为侧式，island 为岛式，staggered_island 为错层岛式，
# separated_island 为分离岛式。宋家庄为分离岛式；旧仿真按方向分别计算容量。
PLATFORM_FORMS = (
    "side", "side", "side", "side", "side", "side", "side", "side",
    "island", "island", "island", "staggered_island", "island", "island",
    "island", "island", "island", "island", "island", "island", "island",
    "island", "separated_island",
)
# 这些车站的上下行候车区域在模型中分别计算容量。
SEPARATE_DIRECTION_CAPACITY_STATIONS = frozenset((*range(8), 22))

# 官方安全限值。对上下行分开计算的车站，旧模型给两个方向复用同一面积数值；
# 这只是简化假设，不代表两个方向的站台面积均已分别测量。
PLATFORM_AREA_M2 = (
    2000, 2000, 2000, 1800, 2000, 1700, 2000, 1900, 1900, 1300,
    1300, 2100, 1300, 1300, 2200, 1300, 1900, 2400, 1300, 1300,
    1300, 1300, 3000,
)
PLATFORM_WARNING_DENSITY_ASSUMPTION = 0.85
PLATFORM_UPPER_DENSITY_ASSUMPTION = 1.0
PLATFORM_WARNING_PEOPLE = tuple(
    int(area * PLATFORM_WARNING_DENSITY_ASSUMPTION) for area in PLATFORM_AREA_M2
)
PLATFORM_MODEL_UPPER_PEOPLE = tuple(
    int(area * PLATFORM_UPPER_DENSITY_ASSUMPTION) for area in PLATFORM_AREA_M2
)
# 保留旧字段名，含义仍是仿真模型的上限，并非真实站台物理极限。
PLATFORM_ADMISSION_LIMIT_PEOPLE = PLATFORM_MODEL_UPPER_PEOPLE

# 本阶段不建模闸机通行能力和进闸后的步行时间，沿用旧项目的简化方式。
# 逐站站外排队上限尚无依据；None 表示未知，不能当作已满足的安全约束。
OUTSIDE_QUEUE_LIMIT_PEOPLE = (None,) * STATION_NUM

TRAIN_CAPACITY_PEOPLE = 1440
MIN_TRAIN_HEADWAY_SECONDS = 70
TURNBACK_TIME_ASSUMPTION_SECONDS = 300
STATION_10_TURNBACK_TIME_ASSUMPTION_SECONDS = 120

DWELL_SECONDS = (
    35, 40, 30, 50, 30, 30, 50, 30, 50, 30, 30, 50, 30, 30, 45,
    35, 50, 50, 45, 35, 45, 44, 38,
)
# 对区间 s：下行为 s→s+1，上行为 s+1→s；两方向运行时间分别存储。
RUN_SECONDS_DOWN = (
    86, 86, 119, 117, 111, 181, 129, 94, 91, 93, 104, 85, 74,
    87, 81, 82, 79, 79, 97, 133, 82, 150,
)
RUN_SECONDS_UP = (
    85, 86, 120, 122, 110, 180, 133, 99, 89, 93, 98, 81, 78,
    89, 81, 83, 79, 80, 98, 133, 81, 136,
)

# 以下仅为有历史折返证据的候选车站；列在这里不代表任一方向已获准折返。
HISTORICAL_TURNBACK_CANDIDATE_STATIONS = frozenset((0, 3, 6, 10, 16, 19, 22))

# 暂将历史候选站作为可用折返站，仅供本版仿真；尚未核验现行进路许可。
# 两端站只设置能反向驶出的来车方向，中间站暂按双向可折返处理。
# 每项依次为车站编号、来车方向、折返作业秒数。
TURNBACK_ROUTES = tuple(
    (station, direction,
     STATION_10_TURNBACK_TIME_ASSUMPTION_SECONDS
     if station == 10 else TURNBACK_TIME_ASSUMPTION_SECONDS)
    for station in sorted(HISTORICAL_TURNBACK_CANDIDATE_STATIONS)
    for direction in ((1,) if station == 0 else
                      ((0,) if station == STATION_NUM - 1 else (0, 1)))
)

assert len(PLATFORM_FORMS) == len(PLATFORM_AREA_M2) == len(DWELL_SECONDS) == STATION_NUM
assert len(RUN_SECONDS_DOWN) == len(RUN_SECONDS_UP) == STATION_NUM - 1
assert len(OUTSIDE_QUEUE_LIMIT_PEOPLE) == STATION_NUM
assert all(warning < upper for warning, upper in zip(PLATFORM_WARNING_PEOPLE, PLATFORM_MODEL_UPPER_PEOPLE))
