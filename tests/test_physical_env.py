import unittest
import numpy as np
from pathlib import Path

from Env import (
    Env, Config, DisruptionTruth, ActionPlan, InfeasibleAction,
    HoldingAction, DelayAction, TurnbackAction, ReleaseAction,
    EntryControlAction, TrainStatus,
)
from Env import Constant as C
from Env.TrainOperator import StopPlan, TrainState
from Env.PassengerOperator import PassengerOperator
from Env.configure import edge_occupancy, sample_disruption

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DataBase" / "beijing_5"


def make_env(**kwargs):
    cfg = Config(data_root=DATA, **kwargs)
    return Env(cfg)


class PhysicalEnvTests(unittest.TestCase):
    def test_data_path_is_relative_to_project(self):
        self.assertEqual(Config().resolved_data_root, DATA)
        self.assertEqual(Config(data_root="DataBase/beijing_5").resolved_data_root, DATA)

    def test_default_data_load_and_snapshot_hides_truth(self):
        env = make_env()
        snap = env.observe()
        self.assertEqual(env.current_time, 2100)
        self.assertEqual(len(env.train_ids), 149)
        self.assertFalse(hasattr(snap.disruption, "actual_duration"))
        self.assertFalse(hasattr(snap.disruption, "actual_end"))
        self.assertEqual(snap.disruption.predicted_end, 3000)
        self.assertEqual(snap.outside.flags.writeable, False)

    def test_decision_interval_is_300_seconds_and_conservation_holds(self):
        env = make_env()
        result = env.step()
        self.assertEqual(result.snapshot.time, 2400)
        self.assertEqual(result.metrics["conservation_gap"], 0)
        self.assertEqual(result.metrics["window_wait_pax_sec"] >= 0, True)

    def test_horizon_constraint(self):
        with self.assertRaises(ValueError):
            Config(disruption=DisruptionTruth(20, 21, 0, 2100, 1200, 900),
                   decision_interval=300, evaluation_horizon=200, forecast_horizon=1200)
        with self.assertRaises(ValueError):
            Config(disruption=DisruptionTruth(20, 21, 0, 2100, 1200, 900),
                   decision_interval=300, evaluation_horizon=1200, forecast_horizon=600)

    def test_passenger_up_direction_priority(self):
        outside = np.zeros((2, C.STATION_NUM, C.STATION_NUM), dtype=np.int64)
        platform = np.zeros_like(outside)
        # 上行站10，目的地按递减方向分配：9先于8。
        outside[1, 10, 9] = 5
        outside[1, 10, 8] = 5
        moved = PassengerOperator.admit(outside, platform, 10, 1, 6, 1000)
        self.assertEqual(moved, 6)
        self.assertEqual(platform[1, 10, 9], 5)
        self.assertEqual(platform[1, 10, 8], 1)

    def test_entry_control_is_window_action_and_capacity_bounded(self):
        env = make_env()
        env.outside[:] = 0
        env.platform[:] = 0
        env.outside[0, 10, 11] = 20
        env._quota_remaining = {(10, 0): 4}
        env._admit_all()
        self.assertEqual(int(env.platform[0, 10, 11]), 4)
        self.assertEqual(int(env.outside[0, 10, 11]), 16)

    def test_turnback_clear_requires_full_platform_room(self):
        env = make_env()
        station, direction = 19, 0
        env.platform[direction, station, 20] = C.PLATFORM_CAPACITY_PEOPLE[station]
        onboard = np.zeros(C.STATION_NUM, dtype=np.int64)
        onboard[20] = 1
        with self.assertRaises(InfeasibleAction):
            # 通过构造ready列车检验前置容量条件。
            tr = env._trains[0]
            tr.current_station = station
            tr.direction = direction
            tr.status = TrainStatus.READY_TO_DEPART
            tr.stops = [StopPlan(station, env.time - 10, env.time), StopPlan(20, env.time + 100, env.time + 130)]
            tr.stop_index = 0
            tr.planned_departure = env.time
            env.onboard[tr.train_id] = onboard
            env._validate_and_apply(ActionPlan((TurnbackAction(tr.train_id, station),), ()))

    def test_turnback_keeps_vehicle_and_train_identity_and_uses_resource_time(self):
        env = make_env()
        tr = env._trains[0]
        tr.current_station = 19
        tr.direction = 0
        tr.status = TrainStatus.READY_TO_DEPART
        tr.stops = [StopPlan(19, env.time - 10, env.time), StopPlan(20, env.time + 100, env.time + 130)]
        tr.stop_index = 0
        tr.planned_departure = env.time
        env.onboard[tr.train_id, 20] = 10
        env.arrivals_total += 10  # 测试注入的车内乘客同步计入系统总到达量
        vehicle_id, train_id = tr.vehicle_id, tr.train_id
        env._validate_and_apply(ActionPlan((TurnbackAction(train_id, 19),), ()))
        self.assertEqual(tr.status, TrainStatus.TURNING_BACK)
        self.assertEqual(int(env.stranded_platform[0, 19, 20]), 10)
        self.assertGreaterEqual(int(env.platform[0, 19, 20]), 10)
        self.assertEqual(tr.vehicle_id, vehicle_id)
        self.assertEqual(tr.train_id, train_id)
        result = env.step()
        self.assertEqual(result.metrics["conservation_gap"], 0)
        self.assertEqual(result.snapshot.time, 2400)
        self.assertEqual(tr.direction, 1)
        self.assertNotEqual(tr.status, TrainStatus.OUT_OF_SERVICE)

    def test_holding_only_ready_and_release_only_held(self):
        env = make_env()
        tr = env._trains[0]
        tr.current_station = 19
        tr.status = TrainStatus.DWELLING
        with self.assertRaises(ValueError):
            env._validate_and_apply(ActionPlan((HoldingAction(tr.train_id, 19),), ()))
        tr.status = TrainStatus.READY_TO_DEPART
        tr.planned_departure = env.time + 30
        env._validate_and_apply(ActionPlan((HoldingAction(tr.train_id, 19),), ()))
        self.assertEqual(tr.status, TrainStatus.HELD)
        # Release解除held并安排确定性重排。
        env._validate_and_apply(ActionPlan((ReleaseAction(tr.train_id, 19),), ()))
        self.assertEqual(tr.status, TrainStatus.READY_TO_DEPART)
        self.assertGreaterEqual(tr.planned_departure, env.time)
        with self.assertRaises(ValueError):
            env._validate_and_apply(ActionPlan((ReleaseAction(tr.train_id, 19),), ()))

    def test_delay_rejects_nonpositive_and_wrong_station(self):
        env = make_env()
        tr = env._trains[0]
        tr.current_station = 19
        tr.status = TrainStatus.READY_TO_DEPART
        with self.assertRaises(ValueError):
            env._validate_and_apply(ActionPlan((DelayAction(tr.train_id, 19, 0),), ()))
        with self.assertRaises(ValueError):
            env._validate_and_apply(ActionPlan((DelayAction(tr.train_id, 18, 10),), ()))

    def test_sampling_uses_selected_edge_without_resampling(self):
        schedule = np.load(DATA / "train_schedule.npy")
        occ = edge_occupancy(schedule, 20, 0)
        self.assertTrue(isinstance(occ, list))
        # 构造全部有向区间在t=0均被占用的基图，必须明确报错而不是重抽区间。
        synthetic = np.zeros((44, C.STATION_NUM, 5), dtype=np.int64)
        tid = 0
        for station in range(C.STATION_NUM - 1):
            for direction in (0, 1):
                nxt = station + C.DIRECTION_STEP[direction]
                synthetic[tid, station] = [0, 0, direction, tid, station]
                synthetic[tid, nxt] = [100, 130, direction, tid, nxt]
                tid += 1
        with self.assertRaises(RuntimeError):
            sample_disruption(synthetic, seed=2, start_range=(0, 0), duration_range=(100, 100), predicted_duration=90, time_step=60)

    def test_no_action_keeps_service_count_without_inventing_vehicle_ids(self):
        env = make_env()
        before = len(env._trains)
        env.step()
        self.assertEqual(len(env._trains), before)
        self.assertTrue(all(t.vehicle_id is None for t in env._trains.values()))

    def test_delay_keeps_events_in_time_order(self):
        env = make_env()
        tr = env._trains[25]
        env.step(ActionPlan((DelayAction(tr.train_id, tr.current_station, 120),)))
        events = [(time, kind) for time, kind, tid in env.event_log
                  if tid == tr.train_id and time >= 2100 and kind in ("DEPARTURE", "ARRIVAL")]
        self.assertEqual([time for time, _ in events], sorted(time for time, _ in events))

    def test_blocked_boundary_does_not_stack_trains_at_one_platform(self):
        env = make_env()
        for _ in range(4):
            env.step()
        at_boundary = [tr for tr in env._trains.values()
                       if tr.current_station == 20 and tr.direction == 0
                       and tr.status not in (TrainStatus.RUNNING, TrainStatus.OUT_OF_SERVICE)]
        self.assertLessEqual(len(at_boundary), 1)

    def test_turnback_can_be_ordered_for_arrival_in_current_window(self):
        env = make_env()
        tr = env._trains[16]
        result = env.step(ActionPlan((TurnbackAction(tr.train_id, 19),)))
        self.assertEqual(tr.status, TrainStatus.TURNING_BACK)
        self.assertTrue(any(a["type"] == "turnback" and a["train_id"] == tr.train_id
                            for a in result.executed_actions["train_actions"]))
        env.step()
        self.assertEqual(tr.direction, 1)
        self.assertEqual(tr.service_id, f"T{tr.train_id}-R")

    def test_sampled_up_edge_is_always_within_line(self):
        schedule = np.load(DATA / "train_schedule.npy")
        truth = sample_disruption(schedule, seed=86, start_range=(1200, 1800),
                                  duration_range=(300, 300), predicted_duration=300)
        self.assertTrue(0 <= truth.to_station < C.STATION_NUM)


    def test_event_priority_is_deterministic(self):
        order = ["ARRIVAL", "TURNBACK_COMPLETE", "DISRUPTION_END", "DEMAND", "DWELL_COMPLETE", "DEPARTURE"]
        values = [Env.EVENT_PRIORITY[x] for x in order]
        self.assertEqual(values, sorted(values))

    def test_release_reorders_against_previous_train_headway(self):
        env = make_env()
        tr = env._trains[0]
        tr.current_station = 19
        tr.direction = 0
        tr.status = TrainStatus.HELD
        tr.held = True
        tr.planned_departure = env.time - 100
        env._last_departure[(19, 0)] = env.time + 100
        env._validate_and_apply(ActionPlan((ReleaseAction(tr.train_id, 19),), ()))
        self.assertGreaterEqual(tr.planned_departure, env.time + 170)
        self.assertEqual(tr.status, TrainStatus.READY_TO_DEPART)

    def test_turnback_resource_is_exclusive(self):
        env = make_env()
        first = env._trains[0]
        first.current_station, first.direction = 19, 0
        first.status = TrainStatus.READY_TO_DEPART
        first.stops = [StopPlan(19, env.time - 1, env.time), StopPlan(20, env.time + 100, env.time + 130)]
        first.stop_index = 0
        first.planned_departure = env.time
        env._validate_and_apply(ActionPlan((TurnbackAction(first.train_id, 19),), ()))
        second = env._trains[1]
        second.current_station, second.direction = 19, 0
        second.status = TrainStatus.READY_TO_DEPART
        second.stops = [StopPlan(19, env.time - 1, env.time), StopPlan(20, env.time + 100, env.time + 130)]
        second.stop_index = 0
        second.planned_departure = env.time
        with self.assertRaises(InfeasibleAction):
            env._validate_and_apply(ActionPlan((TurnbackAction(second.train_id, 19),), ()))

    def test_station_capacity_can_reopen_entry_after_boarding(self):
        outside = np.zeros((2, C.STATION_NUM, C.STATION_NUM), dtype=np.int64)
        platform = np.zeros_like(outside)
        onboard = np.zeros(C.STATION_NUM, dtype=np.int64)
        stranded = np.zeros(C.STATION_NUM, dtype=np.int64)
        station, direction = 10, 0
        cap = C.PLATFORM_CAPACITY_PEOPLE[station]
        platform[direction, station, 11] = cap
        outside[direction, station, 11] = 10
        moved0 = PassengerOperator.admit(outside, platform, station, direction, 10, cap)
        self.assertEqual(moved0, 0)
        PassengerOperator.board(platform[direction, station], onboard, station, direction, 1000, stranded)
        moved1 = PassengerOperator.admit(outside, platform, station, direction, 10, cap)
        self.assertEqual(moved1, 10)


    def test_shared_platform_capacity_is_not_direction_order_biased(self):
        outside = np.zeros((2, C.STATION_NUM, C.STATION_NUM), dtype=np.int64)
        platform = np.zeros_like(outside)
        station = 10  # 岛式站：按模型共用站台容量
        outside[0, station, 11] = 10
        outside[1, station, 9] = 10
        env = make_env()
        env.outside[:] = 0
        env.platform[:] = 0
        env.outside[:, station] = outside[:, station]
        env.platform[0, station, 11] = C.PLATFORM_CAPACITY_PEOPLE[station] - 10
        env._quota_remaining = {}
        # 只剩10个空位、两方向各有10人，按需求比例各分5个空位。
        env._admit_all()
        self.assertEqual(int(env.platform[0, station, 11]), C.PLATFORM_CAPACITY_PEOPLE[station] - 5)
        self.assertEqual(int(env.platform[1, station, 9]), 5)


    def test_forecast_averages_minute_demand_batches_not_zero_seconds(self):
        env = make_env()
        snap = env.observe()
        indices = np.arange(env.time // 60 - 4, env.time // 60 + 1) * 60
        expected = env.demand[indices].mean(axis=0)
        np.testing.assert_allclose(snap.forecast_od[0], expected)
        self.assertGreater(float(snap.forecast_od[0].sum()), 0.0)


    def test_decision_boundary_precedes_same_time_departure(self):
        env = make_env()
        tr = env._trains[0]
        tr.current_station = 19
        tr.direction = 0
        tr.status = TrainStatus.READY_TO_DEPART
        tr.stops = [StopPlan(19, env.time - 20, env.time), StopPlan(20, env.time + 100, env.time + 130)]
        tr.stop_index = 0
        tr.planned_departure = env.time
        env._schedule_train_event(tr, "DEPARTURE", env.time)
        # 决策边界处理先停在发车事件之前，允许调度动作介入。
        env._advance_until(env.time, defer_departures_at_end=True)
        self.assertEqual(tr.status, TrainStatus.READY_TO_DEPART)
        env._validate_and_apply(ActionPlan((HoldingAction(tr.train_id, 19),), ()))
        env.step()
        self.assertEqual(tr.status, TrainStatus.HELD)


    def test_entry_control_is_applied_at_decision_boundary(self):
        env = make_env()
        env.outside[:] = 0
        env.platform[:] = 0
        env.outside[0, 10, 11] = 20
        env.arrivals_total = env.completed_total + int(env.onboard.sum()) + 20
        env._validate_and_apply(ActionPlan((), (EntryControlAction(10, 0, 4),)))
        self.assertEqual(int(env.platform[0, 10, 11]), 4)
        self.assertEqual(int(env.outside[0, 10, 11]), 16)


    def test_sampled_prediction_is_independent_of_actual_duration(self):
        schedule = np.zeros((1, C.STATION_NUM, 5), dtype=np.int64)
        truth = sample_disruption(schedule, seed=42, start_range=(0, 0),
                                  duration_range=(600, 600), predicted_duration=900,
                                  time_step=60, horizon_seconds=7200)
        self.assertEqual(truth.actual_duration, 600)
        self.assertEqual(truth.predicted_duration, 900)


if __name__ == "__main__":
    unittest.main(verbosity=2)
