"""乘客决策动作与乘客分配规则；容量约束由Env调用本模块统一核验。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from . import Constant as C

@dataclass(frozen=True)
class EntryControlAction:
    station: int
    direction: int
    max_admit: int

PassengerAction = EntryControlAction


class PassengerOperator:
    """负责进站配额、乘客上下车和折返清客的乘客侧计算。"""
    @staticmethod
    def validate_entry_control(action: EntryControlAction) -> None:
        if not 0 <= action.station < C.STATION_NUM or action.direction not in (0, 1):
            raise ValueError("进站限流车站或方向无效")
        if action.max_admit < 0:
            raise ValueError("max_admit不能为负数")

    @staticmethod
    def platform_count(platform: np.ndarray, station: int, direction: int) -> int:
        if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS:
            return int(platform[direction, station].sum())
        return int(platform[:, station].sum())

    @staticmethod
    def platform_capacity(station: int) -> int:
        return int(C.PLATFORM_CAPACITY_PEOPLE[station])

    @classmethod
    def can_clear_to_platform(cls, platform: np.ndarray, station: int, direction: int, count: int) -> bool:
        used = cls.platform_count(platform, station, direction)
        return count >= 0 and used + count <= cls.platform_capacity(station)

    @staticmethod
    def clear_onboard(onboard_row: np.ndarray, platform: np.ndarray, stranded: np.ndarray,
                      station: int, original_direction: int) -> int:
        """清客前调用can_clear；乘客留在原方向站台并打stranded标记，不创建虚拟空间。"""
        count = int(onboard_row.sum())
        platform[original_direction, station] += onboard_row
        stranded[original_direction, station] += onboard_row
        onboard_row[:] = 0
        return count

    @staticmethod
    def admit(outside: np.ndarray, platform: np.ndarray, station: int, direction: int,
              quota: int, capacity: int) -> int:
        """按出行方向排序分配进站，返回实际放行人数。"""
        if quota <= 0:
            return 0
        if station in C.SEPARATE_DIRECTION_CAPACITY_STATIONS:
            used = int(platform[direction, station].sum())
        else:
            used = int(platform[:, station].sum())
        room = max(0, capacity - used)
        moved_total = min(room, quota, int(outside[direction, station].sum()))
        if moved_total <= 0:
            return 0
        dests = range(station + 1, C.STATION_NUM) if direction == 0 else range(station - 1, -1, -1)
        remaining = moved_total
        for dest in dests:
            moved = min(remaining, int(outside[direction, station, dest]))
            if moved:
                outside[direction, station, dest] -= moved
                platform[direction, station, dest] += moved
                remaining -= moved
            if remaining == 0:
                break
        return moved_total

    @staticmethod
    def alight(onboard_row: np.ndarray, station: int) -> int:
        count = int(onboard_row[station])
        onboard_row[station] = 0
        return count

    @staticmethod
    def board(platform_row: np.ndarray, onboard_row: np.ndarray, station: int,
              direction: int, train_capacity: int, stranded_row: np.ndarray | None = None,
              allow_stranded: bool = True) -> int:
        room = max(0, train_capacity - int(onboard_row.sum()))
        dests = range(station + 1, C.STATION_NUM) if direction == 0 else range(station - 1, -1, -1)
        boarded_total = 0
        for dest in dests:
            available = int(platform_row[dest])
            if stranded_row is not None and not allow_stranded:
                available -= int(stranded_row[dest])
            take = min(room, max(0, available))
            if take:
                platform_row[dest] -= take
                onboard_row[dest] += take
                if stranded_row is not None and allow_stranded:
                    stranded_row[dest] = max(0, int(stranded_row[dest]) - take)
                room -= take
                boarded_total += take
            if room == 0:
                break
        return boarded_total

    @staticmethod
    def release_stranded(stranded: np.ndarray) -> int:
        count = int(stranded.sum())
        stranded[:] = 0
        return count
