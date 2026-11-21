"""搜索预试算也应遵守评价和时间预算。"""
from dataclasses import replace
from time import sleep
import unittest

from Env import Env
from Env.ALNS import ALNS
from Env.Evaluator import Evaluation


class CheapEvaluator:
    def __init__(self, delay=0.0):
        self.calls = 0
        self.delay = delay

    def evaluate(self, plan):
        self.calls += 1
        if self.delay:
            sleep(self.delay)
        count = len(plan.train_actions) + len(plan.passenger_actions)
        return Evaluation(True, 1000.0 + 500.0 * count, {}, {}, 3300)


class SearchBudgetTests(unittest.TestCase):
    def test_small_budget_leaves_room_for_destroy_and_repair(self):
        env = Env()
        evaluator = CheapEvaluator()
        result = ALNS(iterations=20, seed=3, profile="enhanced",
                      max_evaluations=10).solve(env.observe(), env._config.public(), evaluator)
        self.assertGreater(result.iterations_completed, 0)
        self.assertGreater(result.unique_candidates_evaluated,
                           1 + result.seed_candidates_evaluated)
        self.assertLessEqual(evaluator.calls, 10)

    def test_recovery_candidates_calibrate_temperature(self):
        env = Env()
        snapshot = env.observe()
        row = next(r for r in snapshot.trains if r["current_station"] is not None)
        held_row = dict(row, status="HELD", held=True)
        recovery = replace(snapshot, phase="recovery", trains=(held_row,))
        result = ALNS(iterations=0).solve(recovery, env._config.public(), CheapEvaluator())
        self.assertGreater(result.initial_temperature, 1.0)

    def test_time_limit_applies_to_seed_evaluations(self):
        env = Env()
        evaluator = CheapEvaluator(delay=0.02)
        result = ALNS(iterations=20, max_seconds=0.005).solve(
            env.observe(), env._config.public(), evaluator)
        self.assertEqual(evaluator.calls, 1)
        self.assertEqual(result.iterations_completed, 0)


if __name__ == "__main__":
    unittest.main()
