"""ALNS 接入 V1 后需要保持的决策边界。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from Env import (ActionPlan, Config, DisruptionTruth, EntryControlAction, Env,
                 HoldingAction, InfeasibleAction, ReleaseAction, TurnbackAction)
from Env.Evaluator import Evaluator
from Env.ALNS import ALNS, candidate_actions
from run_alns import run_one


class ALNSIntegrationTests(unittest.TestCase):
    def test_solver_config_contains_no_actual_recovery_truth(self):
        public = Config().public()
        self.assertFalse(hasattr(public, "disruption"))
        self.assertFalse(hasattr(public, "actual_duration"))

    def test_snapshot_exposes_known_arrival_without_future_truth(self):
        snapshot = Env().observe()
        arriving = next(row for row in snapshot.trains if row["train_id"] == 16)
        self.assertEqual(arriving["next_event_time"], 2195)
        self.assertTrue(arriving["crosses_blocked_edge"])
        self.assertFalse(hasattr(snapshot.disruption, "actual_end"))

    def test_terminal_arrival_cannot_be_ordered_to_turnback(self):
        env = Env()
        with self.assertRaises(InfeasibleAction):
            env.step(ActionPlan((TurnbackAction(12, 22),)))

    def test_evaluation_uses_prediction_and_preserves_real_environment(self):
        first = Env(Config(disruption=DisruptionTruth(20, 21, 0, 2100, 1200, 900)))
        second = Env(Config(disruption=DisruptionTruth(20, 21, 0, 2100, 1800, 900)))
        second.demand[2160:] = 0  # 只改决策时刻之后的真实客流。
        one = Evaluator(first).evaluate(ActionPlan())
        two = Evaluator(second).evaluate(ActionPlan())
        self.assertTrue(one.feasible)
        self.assertEqual(one.metrics, two.metrics)
        self.assertEqual(one.end_time, 3300)
        self.assertEqual(first.current_time, 2100)
        self.assertEqual(second.current_time, 2100)
        self.assertEqual(first.metrics()["wait_pax_sec"], second.metrics()["wait_pax_sec"])

    def test_late_actual_recovery_does_not_shift_predicted_release(self):
        first = Env(Config(disruption=DisruptionTruth(20, 21, 0, 2100, 1200, 900)))
        second = Env(Config(disruption=DisruptionTruth(20, 21, 0, 2100, 1800, 900)))
        for _ in range(3):
            first.step()
            second.step()
        self.assertEqual(first.current_time, 3000)
        self.assertEqual(first.observe().trains, second.observe().trains)
        self.assertEqual(first.metrics(), second.metrics())
        one = Evaluator(first).evaluate(ActionPlan())
        two = Evaluator(second).evaluate(ActionPlan())
        self.assertEqual(one.metrics, two.metrics)

    def test_generated_turnback_is_executable_not_terminal(self):
        env = Env()
        actions = candidate_actions(env.observe(), env._config.public())
        self.assertIn(TurnbackAction(16, 19), actions)
        self.assertNotIn(TurnbackAction(12, 22), actions)
        result = Evaluator(env).evaluate(ActionPlan((TurnbackAction(16, 19),)))
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(result.executed_actions["train_actions"][0]["status"], "executed")

    def test_joint_train_and_passenger_plan_is_evaluated_together(self):
        env = Env()
        plan = ActionPlan((TurnbackAction(16, 19),), (EntryControlAction(20, 0, 0),))
        result = Evaluator(env).evaluate(plan)
        self.assertTrue(result.feasible, result.reason)
        self.assertEqual(len(result.executed_actions["train_actions"]), 1)
        self.assertEqual(len(result.executed_actions["passenger_actions"]), 1)
        self.assertEqual(env.current_time, 2100)

    def test_search_returns_executable_plan_and_evidence(self):
        env = Env()
        evaluator = Evaluator(env)
        search = ALNS(iterations=8, seed=7).solve(env.observe(), env._config.public(), evaluator)
        self.assertTrue(search.best_evaluation.feasible)
        self.assertLessEqual(search.best_evaluation.score, search.baseline_evaluation.score)
        self.assertGreater(search.candidates_generated, 0)
        self.assertGreater(search.candidates_feasible, 0)
        self.assertEqual(env.current_time, 2100)
        env.step(search.best_plan)
        self.assertEqual(env.current_time, 2400)

    def test_recovery_search_releases_a_held_train_when_it_reduces_waiting(self):
        env = Env()
        env.step()
        env.step(ActionPlan((HoldingAction(14, 20),)))
        env.step()
        env.step()
        self.assertEqual(env.observe().phase, "recovery")
        search = ALNS(iterations=8, seed=3).solve(
            env.observe(), env._config.public(), Evaluator(env))
        self.assertEqual(search.best_plan, ActionPlan((ReleaseAction(14, 20),)))
        self.assertLess(search.best_evaluation.score, search.baseline_evaluation.score)

    def test_run_writes_separate_prediction_and_actual_feedback(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            summary = run_one("rule", Config(), rounds=1, iterations=0, seed=7,
                              output_root=Path(temp))
            record = json.loads((Path(temp) / "rule" / "round_001.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["rounds"], 1)
            self.assertIsNone(summary["rounds_with_feasible_turnback_candidate"])
            self.assertIn("selected_prediction", record)
            self.assertIn("actual_feedback", record)
            self.assertNotIn("actual_duration", record["snapshot"]["disruption"])


if __name__ == "__main__":
    unittest.main()
