"""V1 修正：计量边界、温度尺度和公平评价预算。"""
import copy
import math
import unittest

from Env import ActionPlan, Env, TurnbackAction
from Env.ALNS import ALNS
from Env.Evaluator import Evaluation, Evaluator


class CorrectionTests(unittest.TestCase):
    def test_running_train_waiting_for_station_counts_onboard_lateness(self):
        env = Env()
        env.step()
        train = env._trains[15]
        self.assertEqual(train.status.value, "RUNNING")
        self.assertGreater(int(env.onboard[15].sum()), 0)
        self.assertGreater(env.current_time, int(env.base_schedule[15, train.next_station, 0]))
        without = copy.deepcopy(env)
        passengers = int(env.onboard[15].sum())
        without.onboard[15] = 0
        env._integrate_waiting(env.current_time + 1)
        without._integrate_waiting(without.current_time + 1)
        self.assertEqual(env.onboard_late_pax_sec - without.onboard_late_pax_sec,
                         passengers)

    def test_turnback_records_cancelled_and_new_reverse_service(self):
        env = Env()
        env.step(ActionPlan((TurnbackAction(16, 19),)))
        metrics = env.metrics()
        self.assertGreater(metrics["cancelled_service_sec"], 0)
        self.assertGreater(metrics["added_reverse_service_sec"], 0)
        self.assertGreater(metrics["cancelled_stop_count"], 0)
        self.assertGreater(metrics["added_reverse_stop_count"], 0)
        result = Evaluator(Env()).evaluate(ActionPlan((TurnbackAction(16, 19),)))
        self.assertGreater(result.objective_values["train_schedule_adjustment"],
                           result.metrics["window_schedule_shift_sec"])

    def test_pending_train_delay_is_visible_before_departure(self):
        env = Env()
        env.step()
        self.assertGreater(env.metrics()["pending_schedule_shift_sec"], 0)
        result = Evaluator(env).evaluate(ActionPlan())
        self.assertIn("window_schedule_adjustment_sec", result.metrics)
        self.assertGreaterEqual(result.metrics["window_schedule_adjustment_sec"], 0)

    def test_temperature_uses_candidate_deltas_not_absolute_baseline(self):
        temperature = ALNS._temperature_from_deltas((100.0, 200.0))
        self.assertAlmostEqual(math.exp(-150.0 / temperature), 0.5)
        self.assertEqual(temperature, ALNS._temperature_from_deltas((100.0, 200.0)))

    def test_enforced_unique_evaluation_budget_for_both_profiles(self):
        class CheapEvaluator:
            def __init__(self):
                self.calls = 0
            def evaluate(self, plan):
                self.calls += 1
                return Evaluation(True, 100.0 + len(plan.train_actions)
                                  + len(plan.passenger_actions), {}, {}, 3300)
        env = Env()
        for profile in ("standard", "enhanced"):
            evaluator = CheapEvaluator()
            result = ALNS(iterations=50, seed=3, profile=profile,
                          max_evaluations=14).solve(env.observe(), env._config.public(),
                                                    evaluator)
            self.assertEqual(result.unique_candidates_evaluated, evaluator.calls)
            self.assertLessEqual(evaluator.calls, 14)
            self.assertEqual(result.evaluation_budget, 14)


if __name__ == "__main__":
    unittest.main()
