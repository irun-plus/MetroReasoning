"""利用现有 5 号线数据仿真单方向中断、逐车调整与乘客流动。

站台上限、列车容量及折返进路均为模型假设，不代表运营许可。
"""

from bisect import bisect_right
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from .configure import Config, DecisionConfig, TurnbackRoute, Weights
from . import Constant as C


@dataclass(frozen=True, order=True)
class TrainAction:
    """描述一列车的待令、晚发、折返或恢复放行动作。"""
    train_id: int
    station: int
    action: str  # hold、delay、turnback 或 release
    delay_sec: int
    target_departure: int | None = None


@dataclass(frozen=True, order=True)
class StationControl:
    """规定下一决策窗口内某站某方向最多允许多少人进站。"""
    station: int
    direction: int
    max_admit: int  # persons admitted from outside during the next decision window


@dataclass(frozen=True)
class TrainPlan:
    """汇总一轮决策中的全部列车动作和车站进站控制。"""
    train_actions: tuple[TrainAction, ...] = ()
    station_controls: tuple[StationControl, ...] = ()

    def to_dict(self):
        """把方案转成可写入 JSON 的字典。"""
        return asdict(self)


@dataclass(frozen=True)
class Snapshot:
    """保存决策时刻可见的列车、乘客、封锁和客流预测状态。"""
    time: int
    break_from: int
    break_to: int
    break_direction: int
    break_start_time: int
    predicted_end: int
    blocked: bool
    schedule: np.ndarray
    outside: np.ndarray
    platform: np.ndarray
    onboard: np.ndarray
    arrivals_total: int
    completed_total: int
    forecast_times: tuple[int, ...]
    forecast_od: tuple[np.ndarray, ...]
    turnback_routes: tuple[TurnbackRoute, ...] = ()
    held_departures: tuple[tuple[int, int], ...] = ()
    turnback_links: tuple[tuple[int, int], ...] = ()

    @property
    def phase(self):
        """只按当前已观察到的封锁状态区分中断与恢复阶段。"""
        return "interruption" if self.blocked else "recovery"

    def digest(self):
        """计算快照指纹，用于检查候选试算是否改动原状态。"""
        h = sha256()
        h.update(str((self.time, self.predicted_end, self.blocked,
                      self.arrivals_total, self.completed_total)).encode())
        for array in (self.schedule, self.outside, self.platform, self.onboard):
            h.update(array.tobytes())
        for t, array in zip(self.forecast_times, self.forecast_od):
            h.update(str(t).encode())
            h.update(array.tobytes())
        h.update(str((self.turnback_routes, self.held_departures, self.turnback_links)).encode())
        return h.hexdigest()

    def to_dict(self):
        """把可见状态展开为结果记录所需的字典。"""
        return {
            "time": self.time,
            "phase": self.phase,
            "break_info": {
                "from_station": self.break_from,
                "to_station": self.break_to,
                "direction": self.break_direction,
                "start_time": self.break_start_time,
                "predicted_end": self.predicted_end,
                "currently_blocked": self.blocked,
            },
            "schedule": self.schedule.tolist(),
            "outside_by_od": self.outside.tolist(),
            "platform_by_od": self.platform.tolist(),
            "onboard_by_train_destination": self.onboard.tolist(),
            "arrivals_total": self.arrivals_total,
            "completed_total": self.completed_total,
            "demand_forecast": [
                {"time": t, "od": array.tolist()}
                for t, array in zip(self.forecast_times, self.forecast_od)
            ],
            "available_turnback_routes": [asdict(route) for route in self.turnback_routes],
            "held_departures": [{"train_id": i, "planned_departure": t}
                                for i, t in self.held_departures],
            "turnback_links": [{"original_train_id": i, "reverse_train_id": j}
                               for i, j in self.turnback_links],
        }


@dataclass(frozen=True)
class EvalResult:
    """保存候选方案可行性、违规原因及预测指标。"""
    feasible: bool
    violations: tuple[str, ...]
    metrics: dict[str, float]
    audit: dict | None = None

    def score(self, weights: Weights, scales: dict[str, float]):
        """用业务权重和同场景归一化尺度计算候选综合成本。"""
        if not self.feasible:
            return float("inf")
        return sum(getattr(weights, key) * self.metrics[metric] / scales[metric]
                   for key, metric in (("wait", "wait_pax_sec"),
                                       ("end_waiting", "end_waiting_pax"),
                                       ("delay", "remaining_delay_sec"),
                                       ("crowd", "crowd_excess_pax_sec"),
                                       ("change", "change_cost")))


@dataclass(frozen=True)
class Feedback:
    """保存已执行方案、实际窗口指标和下一决策快照。"""
    executed_plan: TrainPlan
    actual_metrics: dict[str, float]
    next_snapshot: Snapshot


def safe_initial_plan(snapshot: Snapshot) -> TrainPlan:
    """封锁时生成基础待令，开放后按既有车序生成最早放行方案。"""
    if not snapshot.blocked:
        station, next_station = snapshot.break_from, snapshot.break_to
        direction = snapshot.break_direction
        held = dict(snapshot.held_departures)
        rows = sorted((int(snapshot.schedule[i, station, 0]), i)
                      for i in range(len(snapshot.schedule))
                      if int(snapshot.schedule[i, station, 2]) == direction
                      and snapshot.schedule[i, station, 1] > snapshot.time
                      and snapshot.schedule[i, next_station, 0] > snapshot.schedule[i, station, 1])
        previous = max((int(snapshot.schedule[i, station, 1])
                        for i in range(len(snapshot.schedule))
                        if int(snapshot.schedule[i, station, 2]) == direction
                        and 0 < snapshot.schedule[i, station, 1] <= snapshot.time), default=-10**9)
        actions = []
        for _, train_id in rows:
            departure = int(snapshot.schedule[train_id, station, 1])
            earliest = max(snapshot.time + 1, held.get(train_id, departure),
                           previous + C.MIN_TRAIN_HEADWAY_SECONDS)
            target = earliest if train_id in held else max(departure, earliest)
            if target < departure:
                actions.append(TrainAction(train_id, station, "release", 0, target))
            elif target > departure:
                actions.append(TrainAction(train_id, station, "delay", target - departure))
            previous = target
        return TrainPlan(tuple(actions))
    station, next_station = snapshot.break_from, snapshot.break_to
    direction = snapshot.break_direction
    rows = sorted((int(snapshot.schedule[i, station, 1]), i)
                  for i in range(len(snapshot.schedule))
                  if int(snapshot.schedule[i, station, 2]) == direction
                  and snapshot.schedule[i, station, 1] >= snapshot.time
                  and snapshot.schedule[i, next_station, 0] > snapshot.schedule[i, station, 1])
    previous = max((int(snapshot.schedule[i, station, 1])
                    for i in range(len(snapshot.schedule))
                    if int(snapshot.schedule[i, station, 2]) == direction
                    and 0 < snapshot.schedule[i, station, 1] < snapshot.time), default=-10**9)
    actions = []
    for departure, train_id in rows:
        earliest = snapshot.predicted_end if departure < snapshot.predicted_end else departure
        target = max(departure, earliest, previous + C.MIN_TRAIN_HEADWAY_SECONDS)
        if target > departure:
            actions.append(TrainAction(train_id, station, "hold", target - departure))
        previous = target
    return TrainPlan(tuple(actions))


class ALNSEnv:
    """按事件推进列车与乘客，试算时只使用当前可见的预测。"""

    def __init__(self, config: Config):
        """读取时刻表与 OD，核对数据形状并建立仿真初始状态。"""
        self.config = config
        self.data_root = Path(config.data_root or Path(__file__).resolve().parents[1] / "DataBase" / "npy")
        self.base_schedule = np.load(self.data_root / "train_schedule.npy").astype(np.int64)
        down = np.load(self.data_root / "od_matrices_down.npy", allow_pickle=True).astype(np.int32)
        up = np.load(self.data_root / "od_matrices_up.npy", allow_pickle=True).astype(np.int32)
        self.demand = np.stack((down, up), axis=1)
        if self.base_schedule.shape[1:] != (C.STATION_NUM, 5):
            raise ValueError("列车计划数组形状与 23 站模型不匹配")
        if self.demand.shape[2:] != (C.STATION_NUM, C.STATION_NUM):
            raise ValueError("OD 数组形状与 23 站模型不匹配")
        if config.break_start_time + max(config.actual_duration, config.predicted_duration) >= len(self.demand):
            raise ValueError("中断或恢复预测超过 OD 数据范围")
        edge_departures = self.base_schedule[:, config.break_from, 1]
        edge_arrivals = self.base_schedule[:, config.break_to, 0]
        if np.any((edge_departures > 0) & (edge_departures <= config.break_start_time)
                  & (edge_arrivals > config.break_start_time)):
            raise ValueError("中断开始时已有列车占用故障区间，当前模型不处理区间内停车")
        for direction in (0, 1):
            for station in range(C.STATION_NUM):
                bad = range(station + 1) if direction == 0 else range(station, C.STATION_NUM)
                if np.any(self.demand[:, direction, station, list(bad)]):
                    raise ValueError("OD 包含逆向或同站出行记录")
        self.demand_times = np.flatnonzero(self.demand.sum(axis=(1, 2, 3))).tolist()
        self.time = -1
        self.schedule = self.base_schedule.copy()
        self.outside = np.zeros((2, C.STATION_NUM, C.STATION_NUM), dtype=np.int64)
        self.platform = self.outside.copy()
        self.onboard = np.zeros((len(self.schedule), C.STATION_NUM), dtype=np.int64)
        self.arrivals_total = 0
        self.completed_total = 0
        self.held_train_ids = set()
        self.turnback_links = {}

    def public_config(self, weights: Weights | None = None) -> DecisionConfig:
        """返回求解器可见的参数，不含真实恢复时长。"""
        return self.config.public(weights)

    def reset(self, config: Config | None = None) -> Snapshot:
        """从数据起点推演到中断开始时刻并返回第一份决策快照。"""
        if config is not None:
            self.__init__(config)
        self._advance(self.schedule, min(self.config.break_start_time, len(self.demand) - 1), None)
        return self.observe()

    def observe(self) -> Snapshot:
        """复制当前状态，并用过去五分钟客流生成后续客流预测。"""
        t = self.time
        blocked = self.config.break_start_time <= t < self.config.break_start_time + self.config.actual_duration
        predicted_end = self.config.break_start_time + self.config.predicted_duration
        if blocked:
            predicted_end = max(predicted_end, t + self.config.decision_interval)
        arrays = [array.copy() for array in (self.schedule, self.outside, self.platform, self.onboard)]
        for array in arrays:
            array.setflags(write=False)
        history = [self.demand[minute] for minute in self.demand_times
                   if max(0, t - 300) <= minute <= t]
        forecast = np.rint(np.mean(history, axis=0)).astype(np.int32) if history else np.zeros_like(self.demand[0])
        forecast.setflags(write=False)
        forecast_times = tuple(minute for minute in range(((t // 60) + 1) * 60,
                              min(t + self.config.evaluation_horizon, len(self.demand) - 1) + 1, 60))
        return Snapshot(t, self.config.break_from, self.config.break_to,
                        self.config.break_direction, self.config.break_start_time,
                        predicted_end, blocked, *arrays,
                        self.arrivals_total, self.completed_total,
                        forecast_times, (forecast,) * len(forecast_times),
                        self.config.turnback_routes,
                        tuple(sorted((i, int(self.base_schedule[i, self.config.break_from, 1]))
                                     for i in self.held_train_ids)),
                        tuple(sorted(self.turnback_links.items())))

    def initial_plan(self, snapshot: Snapshot | None = None) -> TrainPlan:
        """返回本轮 ALNS 搜索使用的安全基础待令方案。"""
        return safe_initial_plan(snapshot or self.observe())

    def _apply_plan(self, snapshot: Snapshot, plan: TrainPlan):
        """把列车动作写入时刻表副本，并构造折返车次与进站控制检查结果。"""
        schedule = snapshot.schedule.copy()
        violations = []
        seen_trains = set()
        turnbacks = {}
        turned_stations = set()
        induced_ids = set()
        for action in sorted(plan.train_actions, key=lambda item: item.action == "turnback"):
            if action.action not in ("hold", "delay", "turnback", "release"):
                violations.append(f"不支持的列车动作：{action.action}")
                continue
            if action.train_id in seen_trains:
                violations.append(f"T{action.train_id} 出现重复动作")
                continue
            seen_trains.add(action.train_id)
            i, station = action.train_id, action.station
            if not (0 <= i < len(schedule) and 0 <= station < C.STATION_NUM):
                violations.append("列车或车站编号越界")
                continue
            if action.action == "turnback":
                direction = int(schedule[i, station, 2])
                route = next((route for route in snapshot.turnback_routes
                              if route.station == station and route.approach_direction == direction), None)
                arrival = int(schedule[i, station, 0])
                if route is None or action.delay_sec != 0 or station in turned_stations:
                    violations.append(f"T{i}在车站{station}无可用折返假设或进路冲突")
                    continue
                if (direction == 0 and station == 0) or (direction == 1 and station == C.STATION_NUM - 1):
                    violations.append(f"T{i}在车站{station}反向后没有可运行区段")
                    continue
                if not snapshot.time < arrival <= snapshot.time + self.config.decision_interval:
                    violations.append(f"T{i}不在当前决策窗口到达折返站{station}")
                    continue
                next_station = station + (1 if direction == 0 else -1)
                if not 0 <= next_station < C.STATION_NUM or schedule[i, next_station, 0] <= arrival:
                    violations.append(f"T{i}在车站{station}没有可取消的后续行程")
                    continue
                for old_id in range(len(self.base_schedule), len(schedule)):
                    if schedule[old_id, station, 0] > 0 and abs(int(schedule[old_id, station, 1]) - (arrival + route.turnaround_seconds)) < route.turnaround_seconds:
                        violations.append(f"车站{station}折返资源仍被占用")
                        break
                if violations:
                    continue
                schedule[i, station, 1] = 0
                onward = range(station + 1, C.STATION_NUM) if direction == 0 else range(station - 1, -1, -1)
                for s in onward:
                    schedule[i, s] = 0
                schedule, shifted = self._append_reverse_train(schedule, station, arrival, route)
                induced_ids.update(shifted)
                turnbacks[i] = (station, direction, arrival, len(schedule) - 1)
                turned_stations.add(station)
                continue
            if action.action == "release":
                planned = dict(snapshot.held_departures).get(i)
                target = action.target_departure
                current = int(schedule[i, station, 1])
                if (snapshot.blocked or station != snapshot.break_from or planned is None
                        or action.delay_sec != 0
                        or target is None or not max(snapshot.time + 1, planned,
                            int(schedule[i, station, 0])) <= target < current):
                    violations.append(f"T{i} 的恢复放行时刻无效")
                    continue
                shift = target - current
                schedule[i, station, 1] = target
                stops = range(station, C.STATION_NUM) if int(schedule[i, station, 2]) == 0 else range(station, -1, -1)
                for following in list(stops)[1:]:
                    if schedule[i, following, 1] > snapshot.time:
                        schedule[i, following, 0:2] += shift
                continue
            if (action.target_departure is not None or action.delay_sec < 0
                    or schedule[i, station, 1] < snapshot.time or schedule[i, station, 1] == 0):
                violations.append(f"T{i} 的待令时长无效或试图修改已发生的发车")
                continue
            direction = int(schedule[i, station, 2])
            stops = range(station, C.STATION_NUM) if direction == 0 else range(station, -1, -1)
            schedule[i, station, 1] += action.delay_sec
            for following in list(stops)[1:]:
                if schedule[i, following, 1] <= 0:
                    continue
                schedule[i, following, 0:2] += action.delay_sec
        seen_controls = set()
        for control in plan.station_controls:
            key = (control.station, control.direction)
            if key in seen_controls or control.direction not in (0, 1) or not 0 <= control.station < C.STATION_NUM or control.max_admit < 0:
                violations.append(f"车站进站控制无效或重复：{key}")
            seen_controls.add(key)
        return schedule, violations, turnbacks, induced_ids

    def _append_reverse_train(self, schedule: np.ndarray, station: int, arrival: int,
                              route: TurnbackRoute):
        """按假定折返时间生成反向车次，并为插入的车次修复追踪间隔。"""
        direction = 1 - route.approach_direction
        train_id = len(schedule)
        row = np.zeros((C.STATION_NUM, 5), dtype=np.int64)
        departure = arrival + route.turnaround_seconds
        row[station] = (arrival, departure, direction, train_id, station)
        stops = list(range(station - 1, -1, -1)) if direction == 1 else list(range(station + 1, C.STATION_NUM))
        previous = station
        for s in stops:
            running = C.RUN_SECONDS_UP[s] if direction == 1 else C.RUN_SECONDS_DOWN[previous]
            at = int(row[previous, 1]) + running
            row[s] = (at, at + C.DWELL_SECONDS[s], direction, train_id, s)
            previous = s
        schedule = np.concatenate((schedule, row[None, :, :]), axis=0)
        shifted_ids = set()
        for s in [station] + stops:
            next_station = s + (1 if direction == 0 else -1)
            if not 0 <= next_station < C.STATION_NUM:
                continue
            ordered = sorted((int(schedule[i, s, 1]), i)
                             for i in range(len(schedule))
                             if schedule[i, s, 2] == direction
                             and schedule[i, s, 1] > 0
                             and schedule[i, next_station, 0] > schedule[i, s, 1])
            previous_departure = -10**9
            for depart, i in ordered:
                delay = max(0, previous_departure + C.MIN_TRAIN_HEADWAY_SECONDS - depart)
                if delay:
                    shifted_ids.add(i)
                    schedule[i, s, 1] += delay
                    following = range(s + 1, C.STATION_NUM) if direction == 0 else range(s - 1, -1, -1)
                    for later in following:
                        if schedule[i, later, 1] > 0:
                            schedule[i, later, 0:2] += delay
                previous_departure = depart + delay
        return schedule, shifted_ids

    def _validate_schedule(self, snapshot: Snapshot, schedule: np.ndarray):
        """检查新时刻表的追踪间隔和封锁区间通行条件。"""
        violations = []
        for direction in (0, 1):
            step = 1 if direction == 0 else -1
            for station in range(C.STATION_NUM):
                next_station = station + step
                if not 0 <= next_station < C.STATION_NUM:
                    continue
                departures = sorted((int(schedule[i, station, 1]), i)
                                    for i in range(len(schedule))
                                    if int(schedule[i, station, 2]) == direction
                                    and schedule[i, station, 1] >= snapshot.time
                                    and schedule[i, next_station, 0] > schedule[i, station, 1])
                for (earlier, _), (later, _) in zip(departures, departures[1:]):
                    if later - earlier < C.MIN_TRAIN_HEADWAY_SECONDS:
                        violations.append(f"区间{station}->{next_station}追踪间隔仅{later-earlier}秒")
                if station == snapshot.break_from and next_station == snapshot.break_to and snapshot.blocked:
                    for departure, train_id in departures:
                        if snapshot.break_start_time <= departure < snapshot.predicted_end:
                            violations.append(f"T{train_id}在{departure}秒驶入封锁区间{station}->{next_station}")
                if station == snapshot.break_from and next_station == snapshot.break_to and not snapshot.blocked:
                    arrival_order = sorted((int(snapshot.schedule[i, station, 0]), i)
                                           for _, i in departures)
                    for (_, front), (_, following) in zip(arrival_order, arrival_order[1:]):
                        front_departure = int(schedule[front, station, 1])
                        following_departure = int(schedule[following, station, 1])
                        if following_departure - front_departure < C.MIN_TRAIN_HEADWAY_SECONDS:
                            violations.append(f"T{following}不能在故障边界越过先到的T{front}")
        return violations

    def evaluate(self, plan: TrainPlan, snapshot: Snapshot | None = None) -> EvalResult:
        """在快照副本上用预测客流试算候选，不修改实际环境。"""
        snapshot = snapshot or self.observe()
        schedule, violations, turnbacks, induced_ids = self._apply_plan(snapshot, plan)
        violations.extend(self._validate_schedule(snapshot, schedule))
        if violations:
            return EvalResult(False, tuple(violations), {})
        end = min(snapshot.time + self.config.evaluation_horizon, len(self.demand) - 1)
        state = self._simulate(snapshot, schedule, end, plan.station_controls,
                               forecast=True, turnbacks=turnbacks)
        metrics = self._metrics(snapshot, schedule, plan, state)
        metrics["induced_train_count"] = float(len(induced_ids))
        cancelled = np.argwhere((self.base_schedule[:, :, 1] > snapshot.time)
                                & (schedule[:len(self.base_schedule), :, 1] == 0))
        audit = {"cancelled_stops": [
            {"train_id": int(i), "station": int(station),
             "planned_departure": int(self.base_schedule[i, station, 1])}
            for i, station in cancelled]}
        if metrics["conservation_gap"]:
            violations.append("乘客守恒检查未通过")
        return EvalResult(not violations, tuple(violations), metrics, audit)

    def _guard_actual_window(self, snapshot: Snapshot, schedule: np.ndarray):
        """若实际仍封锁，把本窗口可能越过故障区间的列车继续扣停。"""
        if not snapshot.blocked:
            return schedule, 0
        station, next_station = snapshot.break_from, snapshot.break_to
        end = snapshot.time + self.config.decision_interval
        direction = snapshot.break_direction
        departures = sorted((int(schedule[i, station, 1]), i)
                            for i in range(len(schedule))
                            if int(schedule[i, station, 2]) == direction
                            and schedule[i, station, 1] >= snapshot.time
                            and schedule[i, next_station, 0] > schedule[i, station, 1])
        previous = max((int(schedule[i, station, 1])
                        for i in range(len(schedule))
                        if int(schedule[i, station, 2]) == direction
                        and 0 < schedule[i, station, 1] < snapshot.time), default=-10**9)
        interventions = 0
        for departure, train_id in departures:
            target = max(departure, previous + C.MIN_TRAIN_HEADWAY_SECONDS)
            if departure < end:
                target = max(target, end)
            delay = target - departure
            if delay > 0:
                interventions += 1
                schedule[train_id, station, 1] += delay
                following = range(station + 1, C.STATION_NUM) if direction == 0 else range(station - 1, -1, -1)
                for s in following:
                    if schedule[train_id, s, 1] > 0:
                        schedule[train_id, s, 0:2] += delay
            previous = target
        return schedule, interventions

    def step(self, plan: TrainPlan) -> Feedback:
        """执行选中方案一个决策窗口，按真实封锁与客流推进并反馈。"""
        snapshot = self.observe()
        forecast = self.evaluate(plan, snapshot)
        if not forecast.feasible:
            raise ValueError(f"不能执行不可行方案：{forecast.violations[:3]}")
        schedule, _, turnbacks, induced_ids = self._apply_plan(snapshot, plan)
        schedule, safety_holds = self._guard_actual_window(snapshot, schedule)
        end = min(snapshot.time + self.config.decision_interval, len(self.demand) - 1)
        actual_end = self.config.break_start_time + self.config.actual_duration
        if snapshot.blocked and snapshot.time < actual_end < end:
            end = actual_end
        state = self._simulate(snapshot, schedule, end, plan.station_controls,
                               turnbacks=turnbacks)
        metrics = self._metrics(snapshot, schedule, plan, state)
        metrics["induced_train_count"] = float(len(induced_ids))
        metrics["safety_holds"] = safety_holds
        if metrics["conservation_gap"]:
            raise RuntimeError("实际执行后的乘客守恒检查未通过")
        self.schedule = schedule
        self.held_train_ids.update(action.train_id for action in plan.train_actions
                                   if action.action == "hold")
        self.held_train_ids.difference_update(action.train_id for action in plan.train_actions
                                              if action.action == "turnback")
        self.held_train_ids = {i for i in self.held_train_ids
                               if (self.schedule[i, self.config.break_from, 1] > end
                                   and self.schedule[i, self.config.break_from, 1]
                                   > self.base_schedule[i, self.config.break_from, 1])}
        self.turnback_links.update({i: reverse_id for i, (_, _, _, reverse_id) in turnbacks.items()})
        self.outside, self.platform, self.onboard = state["outside"], state["platform"], state["onboard"]
        self.arrivals_total, self.completed_total, self.time = state["arrivals_total"], state["completed_total"], end
        return Feedback(plan, metrics, self.observe())

    def _advance(self, schedule: np.ndarray, end: int, controls):
        """在初始化阶段按原计划推进到中断开始时刻。"""
        snapshot = self.observe() if self.time >= 0 else Snapshot(-1, self.config.break_from,
            self.config.break_to, self.config.break_direction, self.config.break_start_time,
            self.config.break_start_time + self.config.predicted_duration, False,
            self.schedule, self.outside, self.platform, self.onboard, 0, 0, (), ())
        state = self._simulate(snapshot, schedule, end, controls or ())
        self.outside, self.platform, self.onboard = state["outside"], state["platform"], state["onboard"]
        self.arrivals_total, self.completed_total, self.time = state["arrivals_total"], state["completed_total"], end

    def _simulate(self, snapshot: Snapshot, schedule: np.ndarray, end: int, controls,
                  forecast: bool = False, turnbacks: dict | None = None):
        """按需求、到站、清客、上车和发车事件推演指定时间窗。"""
        outside, platform, onboard = snapshot.outside.copy(), snapshot.platform.copy(), snapshot.onboard.copy()
        if len(onboard) < len(schedule):
            onboard = np.pad(onboard, ((0, len(schedule) - len(onboard)), (0, 0)))
        turnbacks = turnbacks or {}
        cleared_pax = 0
        cleared_to_outside_pax = 0
        arrivals_total, completed_total = snapshot.arrivals_total, snapshot.completed_total
        events = {}
        for i in range(len(schedule)):
            for station in range(C.STATION_NUM):
                arrival, departure = map(int, schedule[i, station, :2])
                if snapshot.time < arrival <= end:
                    events.setdefault(arrival, [[], []])[0].append((i, station))
                if snapshot.time < departure <= end:
                    events.setdefault(departure, [[], []])[1].append((i, station))
        if forecast:
            demand_events = {t: od for t, od in zip(snapshot.forecast_times, snapshot.forecast_od)
                             if snapshot.time < t <= end}
        else:
            left = bisect_right(self.demand_times, snapshot.time)
            right = bisect_right(self.demand_times, end)
            demand_events = {t: self.demand[t] for t in self.demand_times[left:right]}
        for t in demand_events:
            events.setdefault(t, [[], []])
        events.setdefault(end, [[], []])
        budgets = np.full((2, C.STATION_NUM), 2**31 - 1, dtype=np.int64)
        for control in controls:
            budgets[control.direction, control.station] = control.max_admit
        first_window_end = snapshot.time + self.config.decision_interval
        wait_pax_sec = 0
        crowd_excess_pax_sec = 0
        platform_peak = 0
        last = max(snapshot.time, 0)

        def crowd():
            """计算各站站台峰值和超过预警线的人数。"""
            peak = 0
            excess = 0
            for station in range(C.STATION_NUM):
                if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS:
                    counts = (int(platform[d, station].sum()) for d in (0, 1))
                else:
                    counts = (int(platform[:, station].sum()),)
                for count in counts:
                    peak = max(peak, count)
                    excess += max(0, count - C.PLATFORM_WARNING_PEOPLE[station])
            return peak, excess

        def admit():
            """按进站配额和站台模型上限把站外乘客放入站台。"""
            for station in range(C.STATION_NUM):
                for direction in (0, 1):
                    waiting = outside[direction, station]
                    if not np.any(waiting) or budgets[direction, station] <= 0:
                        continue
                    cap_used = int(platform[direction, station].sum()) if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS else int(platform[:, station].sum())
                    room = min(C.PLATFORM_MODEL_UPPER_PEOPLE[station] - cap_used, int(budgets[direction, station]))
                    if room <= 0:
                        continue
                    for dest in np.flatnonzero(waiting):
                        moved = min(room, int(waiting[dest]))
                        waiting[dest] -= moved
                        platform[direction, station, dest] += moved
                        budgets[direction, station] -= moved
                        room -= moved
                        if room <= 0:
                            break

        for t in sorted(events):
            dt = max(0, t - last)
            wait_pax_sec += dt * int(outside.sum() + platform.sum())
            peak, excess = crowd()
            platform_peak = max(platform_peak, peak)
            crowd_excess_pax_sec += dt * excess
            last = t
            if t > first_window_end:
                budgets[:] = 2**31 - 1
            if t in demand_events:
                arrivals = demand_events[t]
                outside += arrivals
                arrivals_total += int(arrivals.sum())
            arrivals, departures = events[t]
            for train_id, station in arrivals:
                completed_total += int(onboard[train_id, station])
                onboard[train_id, station] = 0
                if train_id in turnbacks:
                    _, direction, _, _ = turnbacks[train_id]
                    for destination in np.flatnonzero(onboard[train_id]):
                        count = int(onboard[train_id, destination])
                        occupied = int(platform[direction, station].sum()) if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS else int(platform[:, station].sum())
                        inside = min(count, max(0, C.PLATFORM_MODEL_UPPER_PEOPLE[station] - occupied))
                        platform[direction, station, destination] += inside
                        outside[direction, station, destination] += count - inside
                        cleared_pax += count
                        cleared_to_outside_pax += count - inside
                        onboard[train_id, destination] = 0
            if t in demand_events:
                admit()
            for train_id, station in departures:
                direction = int(schedule[train_id, station, 2])
                next_station = station + (1 if direction == 0 else -1)
                if not 0 <= next_station < C.STATION_NUM or schedule[train_id, next_station, 0] <= t:
                    continue
                room = C.TRAIN_CAPACITY_PEOPLE - int(onboard[train_id].sum())
                destinations = range(station + 1, C.STATION_NUM) if direction == 0 else range(station - 1, -1, -1)
                for dest in destinations:
                    boarded = min(room, int(platform[direction, station, dest]))
                    platform[direction, station, dest] -= boarded
                    onboard[train_id, dest] += boarded
                    room -= boarded
                    if room <= 0:
                        break
            if departures:
                admit()
        peak, _ = crowd()
        platform_peak = max(platform_peak, peak)
        return dict(outside=outside, platform=platform, onboard=onboard,
                    arrivals_total=arrivals_total, completed_total=completed_total,
                    wait_pax_sec=wait_pax_sec,
                    crowd_excess_pax_sec=crowd_excess_pax_sec, platform_peak=platform_peak,
                    cleared_pax=cleared_pax, cleared_to_outside_pax=cleared_to_outside_pax,
                    end=end)

    def _metrics(self, snapshot: Snapshot, schedule: np.ndarray, plan: TrainPlan, state: dict):
        """汇总等待、剩余乘客、延误、拥挤、折返清客和守恒指标。"""
        horizon = state["end"]
        original = schedule[:len(self.base_schedule)]
        planned = self.base_schedule[:, :, 1]
        adjusted = original[:, :, 1]
        changed = (adjusted - planned).clip(min=0)
        remaining_events = ((planned > 0) & (planned <= horizon)
                            & (adjusted > snapshot.time))
        remaining = int((adjusted - np.maximum(planned, snapshot.time)).clip(min=0)[remaining_events].sum())
        departed = (adjusted > snapshot.time) & (adjusted <= horizon)
        realized = int(changed[departed].sum())
        cancelled = (planned > snapshot.time) & (adjusted == 0)
        future = adjusted > snapshot.time
        residual = int(state["outside"].sum() + state["platform"].sum())
        onboard_count = int(state["onboard"].sum())
        gap = state["arrivals_total"] - state["completed_total"] - residual - onboard_count
        return {
            "wait_pax_sec": float(state["wait_pax_sec"]),
            "end_waiting_pax": float(residual),
            "onboard_pax": float(onboard_count),
            "future_timetable_delay_sec": float(changed[future].sum()),
            "remaining_delay_sec": float(remaining),
            "realized_delay_sec": float(realized),
            "cancelled_original_stops": float(cancelled.sum()),
            "crowd_excess_pax_sec": float(state["crowd_excess_pax_sec"]),
            "change_cost": float(sum(3 if action.action == "turnback" else 1
                                     for action in plan.train_actions) + len(plan.station_controls)),
            "turnback_count": float(sum(action.action == "turnback" for action in plan.train_actions)),
            "cleared_pax": float(state["cleared_pax"]),
            "cleared_to_outside_pax": float(state["cleared_to_outside_pax"]),
            "platform_peak": float(state["platform_peak"]),
            "completed_pax": float(state["completed_total"] - snapshot.completed_total),
            "conservation_gap": float(gap),
        }
