"""V1事件驱动物理环境：单方向、单区间、完全封锁。

本模块只负责物理状态与状态转移，不包含ALNS或其他优化算法。
决策间隔固定为300秒；窗口内按事件时刻推进。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Iterable
import heapq
import itertools
import copy
import numpy as np

from . import Constant as C
from .configure import Config, DisruptionTruth
from .TrainOperator import (
    TrainAction, TrainStatus, TrainState, StopPlan, TrainOperator,
    HoldingAction, DelayAction, TurnbackAction, ReleaseAction,
)
from .PassengerOperator import (
    EntryControlAction, PassengerAction, PassengerOperator,
)


class InfeasibleAction(ValueError):
    """动作违反V1物理条件。"""


@dataclass(frozen=True)
class ActionPlan:
    train_actions: tuple[TrainAction, ...] = ()
    passenger_actions: tuple[PassengerAction, ...] = ()

@dataclass(frozen=True)
class DisruptionObservation:
    from_station: int
    to_station: int
    direction: int
    start_time: int
    predicted_end: int
    blocked_now: bool

@dataclass(frozen=True)
class Snapshot:
    """决策时可见状态；结构上不包含actual_duration或actual_end。"""
    time: int
    disruption: DisruptionObservation
    trains: tuple[dict, ...]
    outside: np.ndarray
    platform: np.ndarray
    onboard: np.ndarray
    stranded_platform: np.ndarray
    arrivals_total: int
    completed_total: int
    forecast_times: tuple[int, ...]
    forecast_od: tuple[np.ndarray, ...]
    phase: str

    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "phase": self.phase,
            "disruption": asdict(self.disruption),
            "trains": list(self.trains),
            "outside": self.outside.tolist(),
            "platform": self.platform.tolist(),
            "onboard": self.onboard.tolist(),
            "stranded_platform": self.stranded_platform.tolist(),
            "arrivals_total": self.arrivals_total,
            "completed_total": self.completed_total,
            "forecast": [{"time": t, "od": x.tolist()} for t, x in zip(self.forecast_times, self.forecast_od)],
        }

@dataclass(frozen=True)
class StepResult:
    snapshot: Snapshot
    metrics: dict[str, float]
    executed_actions: dict


class Env:
    """北京地铁5号线V1物理仿真环境。"""
    # 同一时刻事件优先级：到站下车、折返完成、故障解除、OD到达、停站作业完成、发车。
    EVENT_PRIORITY = {
        "ARRIVAL": 10,
        "TURNBACK_COMPLETE": 20,
        "DISRUPTION_END": 25,
        "DEMAND": 30,
        "DWELL_COMPLETE": 40,
        "DEPARTURE": 70,
    }

    def __init__(self, config: Config | None = None):
        self._config = config or Config()
        self._truth: DisruptionTruth = self._config.disruption
        self.data_root = self._config.resolved_data_root
        self.base_schedule = np.load(self.data_root / "train_schedule.npy").astype(np.int64)
        down = np.asarray(np.load(self.data_root / "od_matrices_down.npy", allow_pickle=True), dtype=np.int64)
        up = np.asarray(np.load(self.data_root / "od_matrices_up.npy", allow_pickle=True), dtype=np.int64)
        self.demand = np.stack((down, up), axis=1)
        self._validate_data()
        self._reset_state()

    def _validate_data(self) -> None:
        if self.base_schedule.ndim != 3 or self.base_schedule.shape[1:] != (C.STATION_NUM, 5):
            raise ValueError(f"train_schedule形状应为(N,{C.STATION_NUM},5)，实际为{self.base_schedule.shape}")
        if self.demand.ndim != 4 or self.demand.shape[1:] != (2, C.STATION_NUM, C.STATION_NUM):
            raise ValueError(f"OD形状应为(T,2,{C.STATION_NUM},{C.STATION_NUM})，实际为{self.demand.shape}")
        if np.any(self.demand < 0):
            raise ValueError("OD需求不能为负数")
        # 检查OD方向，拒绝反向或同站需求，避免静默错分。
        for t in range(self.demand.shape[0]):
            down_bad = np.tril(self.demand[t, 0], k=0)
            up_bad = np.triu(self.demand[t, 1], k=0)
            if np.any(down_bad) or np.any(up_bad):
                raise ValueError(f"OD存在反向或同站需求，时间索引={t}")
        if self._truth.actual_end >= self.demand.shape[0]:
            raise ValueError("中断实际恢复时刻超出OD数据范围")

    def _reset_state(self) -> None:
        self.time = 0
        self.outside = np.zeros((2, C.STATION_NUM, C.STATION_NUM), dtype=np.int64)
        self.platform = np.zeros_like(self.outside)
        self.stranded_platform = np.zeros_like(self.outside)  # 逻辑标记；人数仍计入platform
        self.onboard = np.zeros((self.base_schedule.shape[0], C.STATION_NUM), dtype=np.int64)
        self.arrivals_total = 0
        self.completed_total = 0
        self.wait_pax_sec = 0.0
        self.onboard_pax_sec = 0.0
        self.crowd_excess_pax_sec = 0.0
        self.event_log: list[tuple[int, str, int]] = []
        self.platform_peak = 0
        self.blocked_departure_count = 0
        self._event_queue: list[tuple[int, int, int, int, str, int]] = []
        self._seq = itertools.count()
        self._trains: dict[int, TrainState] = {}
        self._resource_reservations: dict[str, tuple[int, int]] = {}
        self._pending_turnbacks: dict[int, tuple[int, dict]] = {}
        self._last_departure: dict[tuple[int, int], int] = {}
        self._quota_remaining: dict[tuple[int, int], int] = {}
        self._last_integral_time = 0
        self._defer_admission_time: int | None = None
        self._last_step_metrics = {"wait_pax_sec": 0.0, "crowd_excess_pax_sec": 0.0}
        self._initialize_trains()
        for t in range(0, self.demand.shape[0], 60):
            self._push_event(t, "DEMAND", -1, -1)
        self._push_event(self._truth.actual_end, "DISRUPTION_END", -1, -1)
        # 初始化阶段使用无中断基础运行图，推演至故障开始时刻。
        self._advance_until(self._truth.start_time, defer_departures_at_end=True)
        self._check_conservation()

    def reset(self) -> Snapshot:
        """重置环境并返回中断开始时刻的第一份观测。"""
        self._reset_state()
        return self.observe()

    def fork_for_evaluation(self, snapshot: Snapshot | None = None) -> Env:
        """以当前观测构造预测副本；真实恢复时刻和未来 OD 不进入试算。"""
        snapshot = snapshot or self.observe()
        if snapshot.time != self.time:
            raise ValueError("试算快照与当前环境时刻不一致")
        fork = copy.deepcopy(self)
        estimated_end = max(snapshot.disruption.predicted_end, self.time + self._config.decision_interval) \
            if snapshot.disruption.blocked_now else self.time
        predicted_truth = replace(self._truth, actual_duration=estimated_end - self._truth.start_time)
        fork._truth = predicted_truth
        fork._config = replace(self._config, disruption=predicted_truth)
        fork._event_queue = [e for e in fork._event_queue if e[4] != "DISRUPTION_END"]
        heapq.heapify(fork._event_queue)
        if snapshot.disruption.blocked_now:
            fork._push_event(estimated_end, "DISRUPTION_END", -1, -1)
        fork.demand[self.time + 1:] = 0
        for minute, od in zip(snapshot.forecast_times, snapshot.forecast_od):
            fork.demand[minute] = np.rint(od).astype(np.int64)
        return fork

    def _initialize_trains(self) -> None:
        n = self.base_schedule.shape[0]
        for tid in range(n):
            stops: list[StopPlan] = []
            for station in range(C.STATION_NUM):
                row = self.base_schedule[tid, station]
                arr, dep, direction, row_tid, row_station = map(int, row)
                # 空行默认全0；对train 0需用row_station或时间字段区分真实记录。
                valid = (row_tid == tid) and (arr > 0 or dep > 0 or row_station == station and (tid != 0 or station != 0))
                if valid:
                    stops.append(StopPlan(station, arr, dep))
            if not stops:
                continue
            direction = int(self.base_schedule[tid, stops[0].station, 2])
            stops.sort(key=lambda s: (s.arrival if s.arrival > 0 else 0, s.station if direction == 0 else -s.station))
            train = TrainState(tid, None, direction, stops, service_id=f"T{tid}")
            first = stops[0]
            if first.arrival > 0:
                train.status = TrainStatus.NOT_STARTED
                self._schedule_train_event(train, "ARRIVAL", first.arrival)
            else:
                train.current_station = first.station
                train.stop_index = 0
                train.planned_departure = first.departure
                train.dwell_complete_time = first.arrival + C.DWELL_SECONDS[first.station]
                if first.departure == 0:
                    train.status = TrainStatus.READY_TO_DEPART
                    self._schedule_train_event(train, "DEPARTURE", 0)
                else:
                    train.status = TrainStatus.DWELLING
                    self._schedule_train_event(train, "DWELL_COMPLETE", train.dwell_complete_time)
            self._trains[tid] = train

    def _push_event(self, time: int, kind: str, train_id: int, token: int) -> None:
        heapq.heappush(self._event_queue, (int(time), self.EVENT_PRIORITY[kind], next(self._seq), train_id, kind, token))

    def _schedule_train_event(self, train: TrainState, kind: str, time: int) -> None:
        train.event_token += 1
        train.next_event_kind = kind
        train.next_event_time = int(time)
        self._push_event(time, kind, train.train_id, train.event_token)

    def _active_blocked(self, time: int) -> bool:
        return self._truth.start_time <= time < self._truth.actual_end

    def _station_occupant(self, station: int, direction: int, exclude_id: int) -> TrainState | None:
        """同方向一座车站只允许一列正在停站或待令的车占用。"""
        return next((tr for tr in self._trains.values()
                     if tr.train_id != exclude_id and tr.current_station == station
                     and tr.direction == direction
                     and tr.status not in (TrainStatus.NOT_STARTED, TrainStatus.RUNNING,
                                           TrainStatus.OUT_OF_SERVICE)), None)

    def _integrate_waiting(self, time: int) -> None:
        dt = max(0, int(time) - self._last_integral_time)
        if dt:
            waiting = int(self.outside.sum() + self.platform.sum())
            self.wait_pax_sec += dt * waiting
            self.onboard_pax_sec += dt * int(self.onboard.sum())
            excess = 0
            for station in range(C.STATION_NUM):
                if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS:
                    counts = [int(self.platform[d, station].sum()) for d in (0, 1)]
                else:
                    counts = [int(self.platform[:, station].sum())]
                excess += sum(max(0, x - C.PLATFORM_WARNING_PEOPLE[station]) for x in counts)
            self.crowd_excess_pax_sec += dt * excess
        self._last_integral_time = int(time)

    def _admit_all(self) -> None:
        for station in range(C.STATION_NUM):
            capacity = PassengerOperator.platform_capacity(station)
            if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS:
                for direction in (0, 1):
                    key = (station, direction)
                    quota = self._quota_remaining.get(key, 2**31 - 1)
                    moved = PassengerOperator.admit(self.outside, self.platform, station,
                                                    direction, quota, capacity)
                    if key in self._quota_remaining:
                        self._quota_remaining[key] -= moved
                continue
            # 共用站台容量按两方向当前可进站需求比例分配，避免固定方向优先。
            used = int(self.platform[:, station].sum())
            room = max(0, capacity - used)
            requests = []
            for direction in (0, 1):
                key = (station, direction)
                quota = self._quota_remaining.get(key, 2**31 - 1)
                requests.append(min(int(self.outside[direction, station].sum()), quota))
            total_request = sum(requests)
            if room <= 0 or total_request <= 0:
                continue
            alloc0 = min(requests[0], (room * requests[0]) // total_request)
            alloc1 = min(requests[1], room - alloc0)
            # 整数取整后剩余容量再分配给尚有需求的一侧。
            leftover = room - alloc0 - alloc1
            if leftover:
                add0 = min(leftover, requests[0] - alloc0)
                alloc0 += add0
                leftover -= add0
                alloc1 += min(leftover, requests[1] - alloc1)
            for direction, allocation in ((0, alloc0), (1, alloc1)):
                key = (station, direction)
                moved = PassengerOperator.admit(self.outside, self.platform, station,
                                                direction, allocation, capacity)
                if key in self._quota_remaining:
                    self._quota_remaining[key] -= moved

    def _process_arrival(self, train: TrainState, time: int) -> None:
        if train.status == TrainStatus.OUT_OF_SERVICE:
            return
        destination = train.stops[0].station if train.status == TrainStatus.NOT_STARTED else train.next_station
        occupant = self._station_occupant(destination, train.direction, train.train_id)
        if occupant is not None:
            retry = max(time + 1, (occupant.next_event_time or time + 30) + 1)
            self._schedule_train_event(train, "ARRIVAL", retry)
            return False
        if train.status == TrainStatus.NOT_STARTED:
            train.current_station = train.stops[0].station
            train.stop_index = 0
        else:
            train.stop_index += 1
            train.current_station = train.stops[train.stop_index].station
        station = int(train.current_station)
        train.status = TrainStatus.DWELLING
        alighted = PassengerOperator.alight(self.onboard[train.train_id], station)
        self.completed_total += int(alighted)
        if train.stop_index >= len(train.stops) - 1:
            train.status = TrainStatus.OUT_OF_SERVICE
            train.next_event_kind = None
            train.next_event_time = None
            train.event_token += 1
            return True
        stop = train.stops[train.stop_index]
        train.planned_departure = int(stop.departure)
        train.dwell_complete_time = max(int(time), int(time) + C.DWELL_SECONDS[station])
        self._schedule_train_event(train, "DWELL_COMPLETE", train.dwell_complete_time)
        return True

    def _process_dwell_complete(self, train: TrainState, time: int) -> None:
        if train.status != TrainStatus.DWELLING:
            return
        train.status = TrainStatus.READY_TO_DEPART
        pending = self._pending_turnbacks.pop(train.train_id, None)
        if pending is not None:
            station, entry = pending
            try:
                self._perform_turnback(train, station, time, entry)
            except InfeasibleAction as exc:
                entry.update(status="rejected", reason=str(exc))
            else:
                return
        target = max(int(train.planned_departure or time), int(time))
        self._schedule_train_event(train, "DEPARTURE", target)

    def _process_departure(self, train: TrainState, time: int) -> None:
        if train.status not in (TrainStatus.READY_TO_DEPART, TrainStatus.HELD):
            return False
        if train.status == TrainStatus.HELD or train.held:
            return False
        station = train.current_station
        nxt = train.next_station
        if station is None or nxt is None:
            train.status = TrainStatus.OUT_OF_SERVICE
            return False
        direction = train.direction
        occupant = self._station_occupant(nxt, direction, train.train_id)
        if occupant is not None:
            run_time = max(1, train.stops[train.stop_index + 1].arrival
                           - train.stops[train.stop_index].departure)
            expected_arrival = time + run_time
            if occupant.next_event_kind != "DEPARTURE" or occupant.next_event_time is None or occupant.next_event_time >= expected_arrival:
                retry = max(time + 1, (occupant.next_event_time or time + 29) + 1 - run_time)
                self._schedule_train_event(train, "DEPARTURE", retry)
                return False
        self._admit_all()
        PassengerOperator.board(
            self.platform[direction, station], self.onboard[train.train_id], station,
            direction, C.TRAIN_CAPACITY_PEOPLE,
            self.stranded_platform[direction, station],
            allow_stranded=not self._active_blocked(time),
        )
        if station == self._truth.from_station and nxt == self._truth.to_station and direction == self._truth.direction and self._active_blocked(time):
            train.blocked_departures += 1
            self.blocked_departure_count += 1
            self._schedule_train_event(train, "DEPARTURE", time + self._config.decision_interval)
            return False
        key = (station, direction)
        previous = self._last_departure.get(key)
        if previous is not None and time < previous + C.MIN_TRAIN_HEADWAY_SECONDS:
            self._schedule_train_event(train, "DEPARTURE", previous + C.MIN_TRAIN_HEADWAY_SECONDS)
            return False
        self._last_departure[key] = int(time)
        scheduled = int(train.stops[train.stop_index].departure)
        shift = max(0, int(time) - scheduled)
        if shift:
            train.schedule_shift_seconds += shift
            for j in range(train.stop_index + 1, len(train.stops)):
                train.stops[j].arrival += shift
                train.stops[j].departure += shift
        train.status = TrainStatus.RUNNING
        next_stop = train.stops[train.stop_index + 1]
        self._schedule_train_event(train, "ARRIVAL", next_stop.arrival)
        return True

    def _process_turnback_complete(self, train: TrainState, time: int, resource_id: str) -> None:
        reservation = self._resource_reservations.get(resource_id)
        if reservation and reservation[1] == train.train_id:
            del self._resource_reservations[resource_id]
        if train.status != TrainStatus.TURNING_BACK:
            return
        old_direction = train.direction
        new_direction = 1 - old_direction
        station = int(train.current_station)
        step = C.DIRECTION_STEP[new_direction]
        stations = list(range(station, C.STATION_NUM if new_direction == 0 else -1, step))
        new_stops = [StopPlan(station, int(time), int(time))]
        depart = int(time)
        for nxt in stations[1:]:
            edge = min(station, nxt)
            run = C.RUN_SECONDS_DOWN[edge] if new_direction == 0 else C.RUN_SECONDS_UP[edge]
            arrival = depart + int(run)
            terminal = nxt == (C.STATION_NUM - 1 if new_direction == 0 else 0)
            departure = arrival if terminal else arrival + C.DWELL_SECONDS[nxt]
            new_stops.append(StopPlan(nxt, arrival, departure))
            depart = departure
        train.direction = new_direction
        train.service_id = f"T{train.train_id}-R"
        train.stops = new_stops
        train.stop_index = 0
        train.status = TrainStatus.READY_TO_DEPART
        train.held = False
        train.planned_departure = int(time)
        train.dwell_complete_time = int(time)
        train.turnback_resource = None
        self._schedule_train_event(train, "DEPARTURE", int(time))

    def _process_event(self, event: tuple) -> None:
        time, _, _, train_id, kind, token = event
        if train_id >= 0:
            train = self._trains.get(train_id)
            if train is None or token != train.event_token:
                return
        occurred = True
        if train_id >= 0:
            if kind == "ARRIVAL":
                occurred = self._process_arrival(train, time)
            elif kind == "DWELL_COMPLETE":
                self._process_dwell_complete(train, time)
            elif kind == "DEPARTURE":
                occurred = self._process_departure(train, time)
            elif kind == "TURNBACK_COMPLETE":
                self._process_turnback_complete(train, time, train.turnback_resource or "")
        elif kind == "DEMAND":
            if time < self.demand.shape[0]:
                self.outside += self.demand[time]
                self.arrivals_total += int(self.demand[time].sum())
                if self._defer_admission_time != time:
                    self._admit_all()
        elif kind == "DISRUPTION_END":
            PassengerOperator.release_stranded(self.stranded_platform)
            for train in self._trains.values():
                if (train.current_station == self._truth.from_station
                        and train.next_station == self._truth.to_station
                        and train.direction == self._truth.direction
                        and train.status == TrainStatus.READY_TO_DEPART
                        and train.blocked_departures):
                    self._schedule_train_event(train, "DEPARTURE", time)
        if occurred:
            self.event_log.append((int(time), kind, int(train_id)))

    def _advance_until(self, end_time: int, defer_departures_at_end: bool = True) -> None:
        end_time = int(end_time)
        self._defer_admission_time = end_time if defer_departures_at_end else None
        while self._event_queue and self._event_queue[0][0] <= end_time:
            if (defer_departures_at_end and self._event_queue[0][0] == end_time
                    and self._event_queue[0][4] == "DEPARTURE"):
                break  # 决策边界先观察/下达动作，再处理同刻发车事件
            event = heapq.heappop(self._event_queue)
            t = event[0]
            self._integrate_waiting(t)
            self._process_event(event)
            self.platform_peak = max(self.platform_peak, self._current_platform_peak())
            self._check_conservation()
            if event[4] == "DEPARTURE":
                # 确保同刻到达/需求已先处理；事件堆的priority保证这一点。
                pass
        self._integrate_waiting(end_time)
        self._defer_admission_time = None
        self.time = end_time
        self.platform_peak = max(self.platform_peak, self._current_platform_peak())

    def _current_platform_peak(self) -> int:
        peak = 0
        for station in range(C.STATION_NUM):
            if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS:
                peak = max(peak, *(int(self.platform[d, station].sum()) for d in (0, 1)))
            else:
                peak = max(peak, int(self.platform[:, station].sum()))
        return peak

    def _check_conservation(self) -> None:
        residual = int(self.outside.sum() + self.platform.sum() + self.onboard.sum())
        if self.arrivals_total != self.completed_total + residual:
            raise AssertionError(
                f"乘客守恒失败：arrivals={self.arrivals_total}, completed={self.completed_total}, residual={residual}"
            )
        if np.any(self.stranded_platform > self.platform):
            raise AssertionError("stranded标记人数超过其所在站台实际人数")

    def _perform_turnback(self, train: TrainState, station: int, time: int, entry: dict) -> None:
        facility = self._find_turnback_facility(station, train.direction)
        if train.current_station != station or train.status != TrainStatus.READY_TO_DEPART or facility is None:
            raise InfeasibleAction("列车未在折返站完成停站作业")
        reservation = self._resource_reservations.get(facility.resource_id)
        if reservation and reservation[0] > time:
            raise InfeasibleAction("折返资源仍被其他列车占用")
        count = int(self.onboard[train.train_id].sum())
        if not PassengerOperator.can_clear_to_platform(self.platform, station, train.direction, count):
            raise InfeasibleAction("清客后站台容量不足，折返动作不可行")
        cleared = PassengerOperator.clear_onboard(
            self.onboard[train.train_id], self.platform, self.stranded_platform,
            station, train.direction,
        )
        end = time + facility.turnaround_seconds
        self._resource_reservations[facility.resource_id] = (end, train.train_id)
        train.status = TrainStatus.TURNING_BACK
        train.held = False
        train.turnback_resource = facility.resource_id
        self._schedule_train_event(train, "TURNBACK_COMPLETE", end)
        entry.update(status="executed", resource_id=facility.resource_id,
                     turnaround_seconds=facility.turnaround_seconds,
                     cleared_pax=cleared, new_service_id=f"T{train.train_id}-R")

    def _validate_and_apply(self, plan: ActionPlan) -> dict:
        train_ids = [a.train_id for a in plan.train_actions]
        if len(train_ids) != len(set(train_ids)):
            raise InfeasibleAction("同一列车在同一决策时刻不能收到多个动作")
        control_keys = [(a.station, a.direction) for a in plan.passenger_actions]
        if len(control_keys) != len(set(control_keys)):
            raise InfeasibleAction("同一车站同一方向不能重复设置进站配额")
        for a in plan.passenger_actions:
            PassengerOperator.validate_entry_control(a)
        turnback_resources = set()
        # 第一遍只校验，不改变状态，确保方案原子性。
        for a in plan.train_actions:
            train = self._trains.get(a.train_id)
            if train is None:
                raise InfeasibleAction(f"不存在列车{a.train_id}")
            if isinstance(a, HoldingAction):
                TrainOperator.validate_holding(train, a.station)
            elif isinstance(a, DelayAction):
                TrainOperator.validate_delay(train, a.station, a.delay_seconds)
            elif isinstance(a, ReleaseAction):
                TrainOperator.validate_release(train, a.station)
            elif isinstance(a, TurnbackAction):
                at_station = (train.current_station == a.station and
                              train.status in (TrainStatus.DWELLING, TrainStatus.READY_TO_DEPART))
                arriving = (train.status == TrainStatus.RUNNING and train.next_station == a.station
                            and train.next_event_kind == "ARRIVAL" and train.next_event_time is not None
                            and train.next_event_time <= self.time + self._config.decision_interval
                            and train.stop_index + 2 < len(train.stops))
                if not (at_station or arriving) or train.next_station is None:
                    raise InfeasibleAction("Turnback仅允许在本站停站或本窗口将到达折返站的列车")
                facility = self._find_turnback_facility(a.station, train.direction)
                if facility is None:
                    raise InfeasibleAction("该车站及来车方向没有配置可用折返设施")
                if facility.resource_id in turnback_resources:
                    raise InfeasibleAction("同一决策时刻重复占用折返资源")
                turnback_resources.add(facility.resource_id)
                reservation = self._resource_reservations.get(facility.resource_id)
                eligible_time = self.time if train.status == TrainStatus.READY_TO_DEPART else (
                    train.dwell_complete_time if at_station else train.next_event_time + C.DWELL_SECONDS[a.station])
                if reservation and reservation[0] > eligible_time:
                    raise InfeasibleAction("折返资源仍被其他列车占用")
                count = int(self.onboard[a.train_id].sum())
                if train.status == TrainStatus.READY_TO_DEPART and not PassengerOperator.can_clear_to_platform(self.platform, a.station, train.direction, count):
                    raise InfeasibleAction("清客后站台容量不足，折返动作不可行")
            else:
                raise InfeasibleAction(f"不支持的列车动作类型：{type(a).__name__}")
        executed = {"train_actions": [], "passenger_actions": []}
        self._quota_remaining = {(a.station, a.direction): int(a.max_admit) for a in plan.passenger_actions}
        for a in plan.train_actions:
            train = self._trains[a.train_id]
            if isinstance(a, HoldingAction):
                TrainOperator.apply_holding(train)
                executed["train_actions"].append({"type": "hold", "train_id": a.train_id, "station": a.station})
            elif isinstance(a, DelayAction):
                old = int(train.planned_departure or self.time)
                train.planned_departure = old + int(a.delay_seconds)
                if train.status == TrainStatus.READY_TO_DEPART:
                    self._schedule_train_event(train, "DEPARTURE", max(self.time, train.planned_departure))
                executed["train_actions"].append({"type": "delay", "train_id": a.train_id, "station": a.station, "delay_seconds": a.delay_seconds})
            elif isinstance(a, ReleaseAction):
                key = (a.station, train.direction)
                previous = self._last_departure.get(key, -10**9)
                target = TrainOperator.apply_release(train, self.time, previous + C.MIN_TRAIN_HEADWAY_SECONDS)
                self._schedule_train_event(train, "DEPARTURE", target)
                executed["train_actions"].append({"type": "release", "train_id": a.train_id, "station": a.station, "target_departure": target})
            elif isinstance(a, TurnbackAction):
                entry = {"type": "turnback", "train_id": a.train_id, "station": a.station,
                         "status": "pending"}
                executed["train_actions"].append(entry)
                if train.status == TrainStatus.READY_TO_DEPART:
                    self._perform_turnback(train, a.station, self.time, entry)
                else:
                    self._pending_turnbacks[a.train_id] = (a.station, entry)
        for a in plan.passenger_actions:
            executed["passenger_actions"].append({"type": "entry_control", "station": a.station, "direction": a.direction, "max_admit": a.max_admit})
        self.platform_peak = max(self.platform_peak, self._current_platform_peak())
        self._admit_all()  # 决策边界下达配额后，先处理当前已在站外等待的乘客
        self.platform_peak = max(self.platform_peak, self._current_platform_peak())
        self._check_conservation()
        return executed

    def _find_turnback_facility(self, station: int, approach_direction: int):
        return next((f for f in self._config.turnback_facilities
                     if f.station == station and f.approach_direction == approach_direction), None)

    def step(self, plan: ActionPlan | None = None) -> StepResult:
        """执行一轮动作，并推进一个300秒决策窗口。"""
        plan = plan or ActionPlan()
        if not isinstance(plan, ActionPlan):
            raise TypeError("plan必须是ActionPlan")
        before_wait = self.wait_pax_sec
        before_crowd = self.crowd_excess_pax_sec
        before_completed = self.completed_total
        executed = self._validate_and_apply(plan)
        end = min(self.time + self._config.decision_interval, self.demand.shape[0] - 1)
        if end <= self.time:
            raise StopIteration("OD数据时间范围已结束")
        self._advance_until(end, defer_departures_at_end=True)
        for train_id, (_, entry) in tuple(self._pending_turnbacks.items()):
            entry.update(status="not_executed", reason="列车在本决策窗口未完成到站和停站作业")
            del self._pending_turnbacks[train_id]
        metrics = self.metrics()
        metrics["window_wait_pax_sec"] = self.wait_pax_sec - before_wait
        metrics["window_crowd_excess_pax_sec"] = self.crowd_excess_pax_sec - before_crowd
        metrics["window_completed_pax"] = float(self.completed_total - before_completed)
        metrics["blocked_departure_count"] = float(self.blocked_departure_count)
        return StepResult(self.observe(), metrics, executed)

    def observe(self) -> Snapshot:
        """构造不含实际恢复时刻的观测，并用历史客流生成未来预测。"""
        d = self._truth
        blocked = self._active_blocked(self.time)
        end = min(self.time + self._config.forecast_horizon, self.demand.shape[0] - 1)
        step_times = tuple(range(((self.time // 60) + 1) * 60, end + 1, 60))
        lookback = self._config.demand_forecast_minutes
        current_minute = self.time // 60
        first_minute = max(0, current_minute - lookback + 1)
        history_indices = np.arange(first_minute, current_minute + 1, dtype=np.int64) * 60
        history = self.demand[history_indices]
        if len(history):
            # OD数组按秒索引，但只有每60秒一个需求批次；均值必须按需求批次计算，不能把59个零秒计入分母。
            mean_od = history.mean(axis=0)
        else:
            mean_od = np.zeros((2, C.STATION_NUM, C.STATION_NUM), dtype=float)
        forecast = tuple(mean_od.copy() for _ in step_times)
        train_rows = []
        for tid, tr in sorted(self._trains.items()):
            remaining = tr.stops[tr.stop_index:]
            crosses_blocked_edge = tr.direction == d.direction and any(
                left.station == d.from_station and right.station == d.to_station
                for left, right in zip(remaining, remaining[1:])
            )
            train_rows.append({
                "train_id": tr.train_id,
                "vehicle_id": tr.vehicle_id,
                "service_id": tr.service_id,
                "direction": tr.direction,
                "status": tr.status.value,
                "current_station": tr.current_station,
                "next_station": tr.next_station,
                "planned_departure": tr.planned_departure,
                "next_event_kind": tr.next_event_kind,
                "next_event_time": tr.next_event_time,
                "has_stop_after_next": tr.stop_index + 2 < len(tr.stops),
                "crosses_blocked_edge": crosses_blocked_edge,
                "schedule_shift_seconds": tr.schedule_shift_seconds,
                "blocked_departures": tr.blocked_departures,
            })
        return Snapshot(
            time=self.time,
            disruption=DisruptionObservation(d.from_station, d.to_station, d.direction,
                                             d.start_time, d.predicted_end, blocked),
            trains=tuple(train_rows),
            outside=self._readonly(self.outside),
            platform=self._readonly(self.platform),
            onboard=self._readonly(self.onboard),
            stranded_platform=self._readonly(self.stranded_platform),
            arrivals_total=self.arrivals_total,
            completed_total=self.completed_total,
            forecast_times=step_times,
            forecast_od=forecast,
            phase="interruption" if blocked else "recovery",
        )

    @staticmethod
    def _readonly(array: np.ndarray) -> np.ndarray:
        result = array.copy()
        result.setflags(write=False)
        return result

    def metrics(self) -> dict[str, float]:
        residual = int(self.outside.sum() + self.platform.sum())
        onboard = int(self.onboard.sum())
        return {
            "wait_pax_sec": float(self.wait_pax_sec),
            "onboard_pax_sec": float(self.onboard_pax_sec),
            "end_waiting_pax": float(residual),
            "onboard_pax": float(onboard),
            "stranded_pax": float(self.stranded_platform.sum()),
            "crowd_excess_pax_sec": float(self.crowd_excess_pax_sec),
            "platform_peak": float(self.platform_peak),
            "completed_pax": float(self.completed_total),
            "arrivals_pax": float(self.arrivals_total),
            "conservation_gap": float(self.arrivals_total - self.completed_total - residual - onboard),
            "accumulated_schedule_shift_sec": float(sum(
                max(0, tr.schedule_shift_seconds) for tr in self._trains.values()
            )),
            "blocked_departure_count": float(self.blocked_departure_count),
        }

    def train_state(self, train_id: int) -> TrainState:
        """返回列车状态的副本，避免外部直接修改环境内部对象。"""
        import copy
        if train_id not in self._trains:
            raise KeyError(train_id)
        return copy.deepcopy(self._trains[train_id])

    @property
    def current_time(self) -> int:
        return self.time

    @property
    def train_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._trains))
