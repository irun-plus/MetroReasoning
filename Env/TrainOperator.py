"""列车决策动作与状态转换。物理事件和跨模块状态协调由Env统一执行。"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class TrainStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    DWELLING = "DWELLING"
    READY_TO_DEPART = "READY_TO_DEPART"
    HELD = "HELD"
    TURNING_BACK = "TURNING_BACK"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


@dataclass(frozen=True)
class HoldingAction:
    train_id: int
    station: int

@dataclass(frozen=True)
class DelayAction:
    train_id: int
    station: int
    delay_seconds: int

@dataclass(frozen=True)
class TurnbackAction:
    train_id: int
    station: int

@dataclass(frozen=True)
class ReleaseAction:
    train_id: int
    station: int

TrainAction = HoldingAction | DelayAction | TurnbackAction | ReleaseAction


@dataclass
class StopPlan:
    station: int
    arrival: int
    departure: int

@dataclass
class TrainState:
    train_id: int
    vehicle_id: int | None
    direction: int
    stops: list[StopPlan]
    service_id: str = ""
    stop_index: int = 0
    current_station: int | None = None
    status: TrainStatus = TrainStatus.NOT_STARTED
    next_event_kind: str | None = None
    next_event_time: int | None = None
    planned_departure: int | None = None
    dwell_complete_time: int | None = None
    held: bool = False
    schedule_shift_seconds: int = 0
    blocked_departures: int = 0
    event_token: int = 0
    turnback_resource: str | None = None

    @property
    def next_station(self) -> int | None:
        if self.stop_index + 1 < len(self.stops):
            return self.stops[self.stop_index + 1].station
        return None


class TrainOperator:
    """只负责列车动作的合法性和列车局部状态转换。"""
    @staticmethod
    def validate_holding(train: TrainState, station: int) -> None:
        if train.current_station != station:
            raise ValueError("Holding只能作用于列车当前所在车站")
        if train.status != TrainStatus.READY_TO_DEPART:
            raise ValueError("Holding仅允许作用于停站作业完成、尚未发车的列车")

    @staticmethod
    def apply_holding(train: TrainState) -> None:
        train.held = True
        train.status = TrainStatus.HELD
        train.next_event_kind = None
        train.next_event_time = None
        train.event_token += 1

    @staticmethod
    def validate_delay(train: TrainState, station: int, delay_seconds: int) -> None:
        if train.current_station != station:
            raise ValueError("Delay只能作用于列车当前所在车站")
        if train.status not in (TrainStatus.DWELLING, TrainStatus.READY_TO_DEPART):
            raise ValueError("Delay仅允许作用于车站停站或待发状态")
        if delay_seconds <= 0:
            raise ValueError("delay_seconds必须为正数")

    @staticmethod
    def validate_release(train: TrainState, station: int) -> None:
        if train.current_station != station:
            raise ValueError("Release只能作用于列车当前所在车站")
        if train.status != TrainStatus.HELD or not train.held:
            raise ValueError("Release只能解除已扣车列车")

    @staticmethod
    def apply_release(train: TrainState, now: int, min_departure: int) -> int:
        """解除扣车并返回重排后的最早发车时刻；不代表发车成功。"""
        train.held = False
        train.status = TrainStatus.READY_TO_DEPART
        old = train.planned_departure if train.planned_departure is not None else now
        target = max(now, old, min_departure, train.dwell_complete_time or now)
        train.planned_departure = target
        return target
