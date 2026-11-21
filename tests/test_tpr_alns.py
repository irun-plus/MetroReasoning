"""新增算法升级专项测试：真实数据+公共快照，禁止手工伪造车站/列车。"""
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from Env import ActionPlan, Config, DisruptionTruth, Env
from Env.ALNS import (ALNS, TopologyContext, _coupled_plan,
                      candidate_actions, candidate_controls)
from Env.Evaluator import (DEFAULT_FORECAST_SCENARIOS, ForecastScenario,
                           RobustEvaluator)
from Env.PassengerOperator import EntryControlAction
from Env.TrainOperator import TurnbackAction
from run_alns import run_one


class TPRALNSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = Env()
        cls.snapshot = cls.env.observe()
        cls.public = cls.env._config.public()

    def test_topology_only_reads_public_snapshot(self):
        topo = TopologyContext.from_snapshot(self.snapshot, self.public)
        self.assertEqual(topo.blocked_from, 20)
        self.assertEqual(topo.direction, 0)
        self.assertEqual(topo.predicted_remaining_seconds, 900)
        self.assertIn(16, topo.affected_train_ids)
        self.assertIn(19, topo.preferred_turnback_stations)
        self.assertFalse(hasattr(topo, 'actual_end'))

    def test_related_turnback_station_added_to_search_controls(self):
        actions = candidate_actions(self.snapshot, self.public)
        self.assertIn(TurnbackAction(16, 19), actions)
        ctrl = candidate_controls(self.snapshot, actions)
        self.assertIn(EntryControlAction(19, 0, 0), ctrl)
        self.assertIn(EntryControlAction(20, 0, 300), ctrl)

    def test_coupled_repair_uses_original_train_direction_and_station_capacity(self):
        p = _coupled_plan(self.snapshot, ActionPlan(), TurnbackAction(16, 19))
        self.assertIn(TurnbackAction(16, 19), p.train_actions)
        self.assertEqual(len(p.passenger_actions), 1)
        action = p.passenger_actions[0]
        self.assertEqual((action.station, action.direction), (19, 0))
        self.assertTrue(0 <= action.max_admit <= 300)
        self.assertEqual(self.env.current_time, self.snapshot.time)

    def test_robust_uses_three_fixed_prediction_scenarios(self):
        r = RobustEvaluator(self.env).evaluate(ActionPlan())
        self.assertTrue(r.feasible, r.reason)
        d = r.robust_details
        self.assertEqual(len(d['scenario_results']), 3)
        self.assertEqual({x['name'] for x in d['scenario_results']},
                         {s.name for s in DEFAULT_FORECAST_SCENARIOS})
        self.assertGreaterEqual(d['cvar_score'], d['expected_score'])
        self.assertAlmostEqual(r.score, d['expected_score'] +
                               d['risk_aversion'] * (d['cvar_score'] - d['expected_score']))
        self.assertEqual(self.env.current_time, self.snapshot.time)
        self.assertEqual(self.env.metrics()['conservation_gap'], 0)

    def test_robust_evaluation_does_not_use_future_actual_recovery_or_demand(self):
        a = Env(Config(disruption=DisruptionTruth(20, 21, 0, 2100, 1200, 900)))
        b = Env(Config(disruption=DisruptionTruth(20, 21, 0, 2100, 1800, 900)))
        b.demand[2160:] = 0  # 真正未来不能进入评价；历史快照相同。
        r1 = RobustEvaluator(a).evaluate(ActionPlan())
        r2 = RobustEvaluator(b).evaluate(ActionPlan())
        self.assertTrue(r1.feasible and r2.feasible)
        self.assertEqual(r1.score, r2.score)
        self.assertEqual(r1.robust_details, r2.robust_details)
        self.assertEqual(a.current_time, b.current_time)

    def test_no_scenario_probability_or_cvar_parameter_hallucination(self):
        with self.assertRaises(ValueError):
            RobustEvaluator(self.env, scenarios=(ForecastScenario('only', probability=0.6),))
        with self.assertRaises(ValueError):
            RobustEvaluator(self.env, risk_aversion=-0.1)
        with self.assertRaises(ValueError):
            RobustEvaluator(self.env, alpha=1)

    def test_enhanced_search_records_real_candidate_evidence(self):
        r = ALNS(iterations=4, seed=7, profile='enhanced').solve(
            self.snapshot, self.public, RobustEvaluator(self.env))
        self.assertTrue(r.best_evaluation.feasible, r.best_evaluation.reason)
        self.assertLessEqual(r.best_evaluation.score, r.baseline_evaluation.score + 1e-9)
        self.assertEqual(r.search_profile, 'enhanced')
        self.assertIn('related_destroy', r.operator_weights)
        self.assertIn('coupled_turnback', r.operator_weights)
        self.assertGreater(r.unique_candidates_evaluated, 1)
        self.assertEqual(len(r.candidate_audit), r.unique_candidates_evaluated)
        self.assertEqual(self.env.current_time, self.snapshot.time)

    def test_result_recorder_distinguishes_forecast_risk_and_actual_execution(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            summary = run_one('alns', Config(), rounds=1, iterations=2, seed=42,
                              output_root=Path(directory), search_profile='enhanced', robust=True)
            path = Path(directory) / 'alns' / 'round_001.json'
            raw = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(summary['evaluation_mode'], 'robust_forecast_scenarios')
            self.assertEqual(raw['search_evidence']['search_profile'], 'enhanced')
            self.assertIn('candidate_audit', raw['search_evidence'])
            self.assertEqual(len(raw['selected_prediction']['robust_details']['scenario_results']), 3)
            self.assertNotIn('actual_end', str(raw['snapshot']['disruption']))
            self.assertIn('actual_feedback', raw)


if __name__ == '__main__':
    unittest.main()
