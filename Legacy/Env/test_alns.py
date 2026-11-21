import json
import os
import unittest
from dataclasses import asdict
from pathlib import Path

from Env.configure import Config, Weights
from Env.ALNSEnv import ALNSEnv, EvalResult, StationControl, TrainAction, TrainPlan
from Env.ALNS import ALNSSolver
from Env.ResultRecorder import ResultRecorder
from Env.calibration import derive_scales
from Env.configure import METRIC_NAMES


DATA_ROOT = Path(os.environ.get("METRO_DATA_ROOT", Path(__file__).resolve().parents[1] / "DataBase" / "npy"))


def config(**changes):
    """构造固定的单方向中断测试配置，允许覆盖个别参数。"""
    values = dict(
        data_root=DATA_ROOT,
        break_from=20,
        break_to=21,
        break_direction=0,
        break_start_time=2100,
        actual_duration=1200,
        predicted_duration=900,
        decision_interval=300,
        evaluation_horizon=1200,
        alns_iterations=20,
        alns_time_limit=5.0,
        seed=42,
    )
    values.update(changes)
    return Config(**values)


class ALNSTest(unittest.TestCase):
    """覆盖方向、信息隔离、逐车动作、折返、守恒、搜索与记录。"""

    def test_direction_must_match_directed_edge(self):
        """方向编号和有向相邻站不一致时必须报错。"""
        with self.assertRaises(ValueError):
            config(break_direction=1)

    def test_observation_hides_actual_duration_and_evaluate_is_pure(self):
        """快照不含实际恢复时长，多次试算也不改真实环境。"""
        env = ALNSEnv(config())
        snapshot = env.reset()
        self.assertNotIn("actual_duration", str(asdict(snapshot)))
        self.assertNotIn("actual_duration", str(asdict(env.public_config())))
        before = env.observe().digest()
        plan = env.initial_plan(snapshot)
        first = env.evaluate(plan)
        second = env.evaluate(plan)
        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(before, env.observe().digest())
        self.assertTrue(first.feasible, first.violations)

        longer = ALNSEnv(config(actual_duration=1500))
        other_snapshot = longer.reset()
        self.assertEqual(snapshot.digest(), other_snapshot.digest())
        self.assertEqual(first.metrics, longer.evaluate(longer.initial_plan()).metrics)

    def test_blocked_traversal_rejected_and_initial_plan_held(self):
        """无保护方案不得穿越封锁区间，基础待令方案应通过校验。"""
        env = ALNSEnv(config())
        env.reset()
        self.assertFalse(env.evaluate(TrainPlan()).feasible)
        plan = env.initial_plan()
        self.assertTrue(any(a.station == 20 and a.delay_sec > 0 for a in plan.train_actions))
        self.assertTrue(env.evaluate(plan).feasible)

    def test_step_uses_truth_and_conserves_passengers(self):
        """实际执行推进一个窗口且乘客总数守恒。"""
        env = ALNSEnv(config())
        env.reset()
        result = env.step(env.initial_plan())
        self.assertEqual(result.actual_metrics["conservation_gap"], 0)
        self.assertEqual(result.next_snapshot.time, 2400)
        self.assertTrue(result.next_snapshot.blocked)

    def test_reopening_interrupts_window_without_revealing_future_time(self):
        """实际开放发生在窗口中途时立即反馈，但开放前快照不泄露该时刻。"""
        env = ALNSEnv(config(actual_duration=1050, predicted_duration=1500))
        snapshot = env.reset()
        for _ in range(3):
            snapshot = env.step(env.initial_plan(snapshot)).next_snapshot
        self.assertEqual(snapshot.time, 3000)
        self.assertTrue(snapshot.blocked)
        self.assertNotIn("actual_duration", str(asdict(snapshot)))
        reopened = env.step(env.initial_plan(snapshot)).next_snapshot
        self.assertEqual(reopened.time, 3150)
        self.assertFalse(reopened.blocked)

    def test_recovery_can_advance_unexecuted_held_train(self):
        """提前开放后，恢复方案能提前放行待令车，且不改写已执行时刻。"""
        env = ALNSEnv(config(actual_duration=900, predicted_duration=1500))
        snapshot = env.reset()
        for _ in range(3):
            snapshot = env.step(env.initial_plan(snapshot)).next_snapshot
        self.assertEqual(snapshot.time, 3000)
        self.assertFalse(snapshot.blocked)
        plan = env.initial_plan(snapshot)
        releases = [action for action in plan.train_actions if action.action == "release"]
        self.assertTrue(releases)
        self.assertTrue(env.evaluate(plan, snapshot).feasible)
        old_departures = {action.train_id: int(snapshot.schedule[action.train_id, action.station, 1])
                          for action in releases}
        feedback = env.step(plan)
        self.assertEqual(feedback.actual_metrics["conservation_gap"], 0)
        self.assertTrue(any(env.schedule[action.train_id, action.station, 1] < old_departures[action.train_id]
                            for action in releases))
        for train_id in old_departures:
            happened = snapshot.schedule[train_id, :, 1] <= snapshot.time
            self.assertTrue((env.schedule[train_id, happened, 1] ==
                             snapshot.schedule[train_id, happened, 1]).all())

    def test_recovery_search_uses_release_operators_and_rejects_past_departure(self):
        """恢复搜索应生成逐车放行候选，且不能修改已发生或早于开放的发车。"""
        env = ALNSEnv(config(actual_duration=900, predicted_duration=1500, alns_iterations=60))
        snapshot = env.reset()
        for _ in range(3):
            snapshot = env.step(env.initial_plan(snapshot)).next_snapshot
        search = ALNSSolver().search(snapshot, env.public_config(), env.evaluate)
        self.assertIn("release_train", search.operator_weights["repair"])
        self.assertGreater(search.evaluated, 1)
        self.assertTrue(search.best_eval.feasible)
        self.assertTrue(any(a.action == "release" for a in search.best_plan.train_actions))
        train_id, _ = snapshot.held_departures[0]
        forbidden = TrainPlan((TrainAction(train_id, snapshot.break_from, "release", 0,
                                           snapshot.time),))
        self.assertFalse(env.evaluate(forbidden, snapshot).feasible)
        wrong_station = TrainPlan((TrainAction(train_id, snapshot.break_to,
                                               "release", 0,
                                               int(snapshot.schedule[train_id, snapshot.break_to, 1]) - 1),))
        self.assertIn("恢复放行时刻无效", env.evaluate(wrong_station, snapshot).violations[0])

    def test_recovery_cannot_release_following_train_ahead_of_queued_train(self):
        """没有越行设施假设时，恢复候选不得让后到车越过前方待令车。"""
        env = ALNSEnv(config(actual_duration=900, predicted_duration=1500))
        snapshot = env.reset()
        for _ in range(3):
            snapshot = env.step(env.initial_plan(snapshot)).next_snapshot
        following, _ = snapshot.held_departures[1]
        plan = TrainPlan((TrainAction(following, snapshot.break_from, "release", 0,
                                     snapshot.time + 1),))
        self.assertFalse(env.evaluate(plan, snapshot).feasible)

    def test_run_records_reopening_and_recovery_scope(self):
        """结果应标出开放时刻及恢复范围，短观察不能误报完整恢复。"""
        from tempfile import TemporaryDirectory
        from run_alns import run_case

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            summary = run_case(config(actual_duration=900, predicted_duration=1500),
                               Path(directory), 4, "rule", 0)
            self.assertEqual(summary["reopened_at"], 3000)
            self.assertIsNone(summary["release_completed_at"])
            self.assertEqual(summary["recovery_scope"], "held_train_release")
            self.assertNotEqual(summary["recovery_status"], "full_route_recovered")
            record = json.loads(Path(summary["decisions"][-1]["record"]).read_text(encoding="utf-8"))
            self.assertEqual(record["snapshot"]["phase"], "recovery")
            self.assertEqual(record["feedback"]["next_snapshot"]["phase"], "recovery")
            self.assertEqual(record["station_control_window"],
                             {"start": 3000, "end": 3300, "controls": []})
            self.assertAlmostEqual(summary["extra_wait_pax_sec"],
                                   summary["total_wait_pax_sec"] - summary["normal_wait_pax_sec"])

    def test_recovery_uses_its_configured_weights_at_reopening(self):
        """区间实际开放后的首轮采用恢复权重，不引入固定过渡分钟数。"""
        from tempfile import TemporaryDirectory
        from run_alns import run_case

        recovery = Weights(wait=0.2, end_waiting=0.2, delay=0.4,
                           crowd=0.15, change=0.05)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            summary = run_case(config(actual_duration=900,
                                      recovery_weight_sets=(recovery, recovery)),
                               Path(directory), 4, "rule", 0)
            record = json.loads(Path(summary["decisions"][-1]["record"]).read_text(encoding="utf-8"))
            self.assertEqual(record["weight_set"], asdict(recovery))

    def test_release_completion_time_is_separate_from_reopening(self):
        """待令车清空的时刻应晚于区间开放，且只代表放行范围完成。"""
        from tempfile import TemporaryDirectory
        from run_alns import run_case

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            summary = run_case(config(actual_duration=900, predicted_duration=1500),
                               Path(directory), 8, "rule", 0)
            self.assertGreater(summary["release_completed_at"], summary["reopened_at"])
            self.assertEqual(summary["recovery_status"], "held_trains_released")

    def test_run_can_stop_when_held_trains_are_released(self):
        """完整案例到待令车放完时停，并标明实际决策轮数。"""
        from tempfile import TemporaryDirectory
        from run_alns import run_case

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            summary = run_case(config(actual_duration=900, predicted_duration=1500),
                               Path(directory), 20, "rule", 0, until_recovered=True)
            self.assertEqual(summary["recovery_status"], "held_trains_released")
            self.assertLess(len(summary["decisions"]), 20)

    def test_forecast_evaluation_ignores_unobserved_actual_demand(self):
        """修改未来实际需求不能改变已经生成的候选预测。"""
        env = ALNSEnv(config())
        snapshot = env.reset()
        plan = env.initial_plan(snapshot)
        before = env.evaluate(plan, snapshot).metrics
        env.demand[2160, 0, 20, 21] += 100
        after = env.evaluate(plan, snapshot).metrics
        self.assertEqual(before, after)
        actual = env.step(plan)
        self.assertEqual(actual.next_snapshot.arrivals_total - snapshot.arrivals_total,
                         100 + sum(int(env.demand[t].sum()) - (100 if t == 2160 else 0)
                                   for t in env.demand_times if 2100 < t <= 2400))

    def test_admission_control_moves_waiting_outside(self):
        """限流会把等待乘客保留在站外而不凭空丢失。"""
        env = ALNSEnv(config())
        snapshot = env.reset()
        basic = env.initial_plan(snapshot)
        limited = TrainPlan(basic.train_actions, (StationControl(20, 0, 0),))
        result = env.evaluate(limited, snapshot)
        self.assertTrue(result.feasible, result.violations)
        feedback = env.step(limited)
        self.assertGreater(feedback.next_snapshot.outside[0, 20].sum(),
                           snapshot.outside[0, 20].sum())

    def test_search_and_json_record(self):
        """搜索返回可行候选并能写出完整决策记录。"""
        env = ALNSEnv(config())
        snapshot = env.reset()
        search = ALNSSolver().search(snapshot, env.public_config(), env.evaluate)
        self.assertTrue(search.best_eval.feasible)
        self.assertLessEqual(search.elapsed_sec, 5.5)
        self.assertTrue(all(c.eval_result.feasible for c in search.candidates))
        self.assertTrue(any(a.action == "turnback" for c in search.candidates
                            for a in c.plan.train_actions))
        feedback = env.step(search.best_plan)
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            recorder = ResultRecorder(Path(directory))
            path = recorder.record_decision(snapshot, Weights(), search, feedback)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("candidates", data)
            self.assertIn("feedback", data)
            self.assertNotIn("actual_duration", data["snapshot"])
            self.assertEqual(len({route["station"] for route in
                                  data["snapshot"]["available_turnback_routes"]}), 7)

    def test_turnback_is_searched_when_initial_plan_has_no_train_actions(self):
        """下一轮无需新增待令时，仍应试算本窗口到站的折返车。"""
        env = ALNSEnv(config(alns_iterations=5))
        env.reset()
        env.step(env.initial_plan())
        snapshot = env.observe()
        self.assertEqual(snapshot.time, 2400)
        self.assertEqual(env.initial_plan(snapshot).train_actions, ())
        self.assertTrue(env.evaluate(TrainPlan(train_actions=(
            TrainAction(18, 19, "turnback", 0),)), snapshot).feasible)

        search = ALNSSolver().search(snapshot, env.public_config(), env.evaluate)
        self.assertTrue(any(action.action == "turnback"
                            for candidate in search.candidates
                            for action in candidate.plan.train_actions))
        self.assertTrue(any(trial["train_id"] == 18 and trial["station"] == 19
                            and trial["feasible"]
                            for trial in search.to_dict()["turnback_trials"]))

    def test_held_train_remains_in_delay_account_after_window_boundary(self):
        """原计划已到但仍被扣住的车，下一轮仍有剩余晚点。"""
        env = ALNSEnv(config(evaluation_horizon=300))
        first = env.reset()
        plan = env.initial_plan(first)
        forecast = env.evaluate(plan, first)
        self.assertGreaterEqual(forecast.metrics["remaining_delay_sec"], 882)
        feedback = env.step(plan)
        self.assertEqual(feedback.actual_metrics["realized_delay_sec"], 0)
        next_forecast = env.evaluate(env.initial_plan(feedback.next_snapshot),
                                     feedback.next_snapshot)
        self.assertGreaterEqual(next_forecast.metrics["remaining_delay_sec"], 600)

    def test_score_uses_remaining_delay_and_window_end_waiting(self):
        """候选评分使用待发车剩余晚点和窗口末等待人数。"""
        metrics = {"wait_pax_sec": 0.0, "end_waiting_pax": 20.0,
                   "remaining_delay_sec": 300.0,
                   "crowd_excess_pax_sec": 0.0, "change_cost": 0.0,
                   "future_timetable_delay_sec": 99999.0}
        scales = {name: 100.0 for name in
                  ("wait_pax_sec", "end_waiting_pax", "remaining_delay_sec",
                   "crowd_excess_pax_sec", "change_cost")}
        weights = Weights(wait=0, end_waiting=0.25, delay=0.75,
                          crowd=0, change=0)
        self.assertAlmostEqual(EvalResult(True, (), metrics).score(weights, scales), 2.30)

    def test_search_uses_frozen_calibrated_scales(self):
        """本轮基础方案的零拥挤值不能重新决定评分尺度。"""
        env = ALNSEnv(config(alns_iterations=1))
        snapshot = env.reset()
        settings = env.public_config()
        search = ALNSSolver().search(snapshot, settings, env.evaluate)
        self.assertEqual(search.scales, settings.metric_scales)
        self.assertGreater(search.scales["crowd_excess_pax_sec"], 1)

    def test_calibration_rejects_unobserved_metrics(self):
        """校准样本始终为零的指标不得悄悄使用分母一。"""
        rows = [{name: 10.0 for name in METRIC_NAMES}]
        rows[0]["crowd_excess_pax_sec"] = 0.0
        with self.assertRaisesRegex(ValueError, "crowd_excess_pax_sec"):
            derive_scales(rows)

    def test_turnback_reports_cancelled_original_stops(self):
        """折返取消的原方向停站须可核查，暂不参与评分。"""
        env = ALNSEnv(config())
        snapshot = env.reset()
        base = env.initial_plan(snapshot)
        actions = tuple(a for a in base.train_actions if a.train_id != 16)
        plan = TrainPlan(actions + (TrainAction(16, 19, "turnback", 0),))
        result = env.evaluate(plan, snapshot)
        self.assertTrue(result.feasible, result.violations)
        self.assertGreater(result.metrics["cancelled_original_stops"], 0)
        self.assertTrue(any(stop["train_id"] == 16 and stop["station"] == 19
                            for stop in result.audit["cancelled_stops"]))

    def test_empty_routes_forbid_turnback(self):
        """显式关闭折返进路时，折返动作必须不可行。"""
        env = ALNSEnv(config(turnback_routes=()))
        env.reset()
        result = env.evaluate(TrainPlan(train_actions=(TrainAction(14, 20, "turnback", 0),)))
        self.assertFalse(result.feasible)

    def test_assumed_route_turns_train_and_conserves_passengers(self):
        """历史候选站按仿真假设开放后，逐车折返并清客。"""
        env = ALNSEnv(config())
        snapshot = env.reset()
        self.assertEqual(len({route.station for route in env.config.turnback_routes}), 7)
        base = env.initial_plan(snapshot)
        actions = tuple(a for a in base.train_actions if a.train_id != 16)
        plan = TrainPlan(actions + (TrainAction(16, 19, "turnback", 0),))
        result = env.evaluate(plan, snapshot)
        self.assertTrue(result.feasible, result.violations[:5])
        self.assertEqual(result.metrics["turnback_count"], 1)
        feedback = env.step(plan)
        self.assertEqual(feedback.actual_metrics["conservation_gap"], 0)
        self.assertGreater(feedback.actual_metrics["cleared_pax"], 0)
        self.assertEqual(len(feedback.next_snapshot.schedule), len(snapshot.schedule) + 1)
        self.assertEqual(int(feedback.next_snapshot.schedule[-1, 19, 2]), 1)
        self.assertIn((16, len(snapshot.schedule)), feedback.next_snapshot.turnback_links)
        continuation = env.initial_plan(feedback.next_snapshot)
        self.assertTrue(env.evaluate(continuation).feasible)
        self.assertEqual(env.step(continuation).actual_metrics["conservation_gap"], 0)


if __name__ == "__main__":
    unittest.main()
