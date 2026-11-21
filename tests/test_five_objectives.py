"""五目标设计的最小回归集；不改动V1原有约束或调度动作。"""
import math
import unittest

from Env import ActionPlan, Env
from Env.Evaluator import Evaluator, OBJECTIVE_KEYS, ObjectiveWeights
from Env.ResultRecorder import evaluation_to_dict


class FiveObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Env()
        cls.evaluator = Evaluator(cls.env)
        cls.empty = cls.evaluator.evaluate(ActionPlan())

    def test_five_components_are_present_and_reconstruct_score(self):
        r = self.empty
        self.assertTrue(r.feasible)
        self.assertEqual(set(r.objective_values), set(OBJECTIVE_KEYS))
        self.assertEqual(set(r.objective_weights), set(OBJECTIVE_KEYS))
        self.assertEqual(set(r.objective_scales), set(OBJECTIVE_KEYS))
        self.assertAlmostEqual(sum(r.objective_weights.values()), 1.0)
        unit = r.objective_scales["passenger_waiting"]
        self.assertAlmostEqual(r.score, unit * sum(r.objective_contributions.values()), places=6)
        for k in OBJECTIVE_KEYS:
            self.assertGreater(r.objective_scales[k], 0)
            self.assertGreaterEqual(r.objective_values[k], 0)

    def test_passenger_unfinished_includes_onboard(self):
        r = self.empty
        self.assertAlmostEqual(r.objective_values["unfinished_trips"],
                               r.metrics["end_waiting_pax"] + r.metrics["end_onboard_pax"])
        self.assertEqual(r.metrics["conservation_gap"], 0)

    def test_delay_metric_is_not_total_onboard_time(self):
        r = self.empty
        self.assertEqual(r.objective_values["passenger_delay_proxy"],
                         r.metrics["onboard_late_pax_sec"])
        self.assertGreater(r.objective_values["passenger_delay_proxy"], 0)

    def test_wrong_weights_fail(self):
        with self.assertRaises(ValueError):
            ObjectiveWeights.from_sequence((0.2, 0.2, 0.2, 0.2, 0.3))
        with self.assertRaises(ValueError):
            ObjectiveWeights.from_sequence((0.5, -0.1, 0.2, 0.2, 0.2))
        with self.assertRaises(ValueError):
            ObjectiveWeights.from_sequence((1, 0, 0))

    def test_changed_weights_change_score_but_not_physics(self):
        settings = ObjectiveWeights.from_sequence((1, 0, 0, 0, 0))
        r = Evaluator(self.env, weights=settings).evaluate(ActionPlan())
        self.assertEqual(r.metrics, self.empty.metrics)
        self.assertEqual(r.objective_values, self.empty.objective_values)
        self.assertAlmostEqual(r.score, r.metrics["wait_pax_sec"])
        self.assertNotEqual(r.score, self.empty.score)
        self.assertEqual(self.env.current_time, 2100)

    def test_saved_score_not_mislabelled_as_waiting(self):
        raw = evaluation_to_dict(self.empty)
        self.assertEqual(raw["score_total"], self.empty.score)
        self.assertEqual(raw["score_wait_pax_sec"], self.empty.metrics["wait_pax_sec"])
        self.assertEqual(len(raw["objective_contributions"]), 5)
        self.assertNotEqual(raw["score_total"], raw["score_wait_pax_sec"])


if __name__ == "__main__":
    unittest.main()
