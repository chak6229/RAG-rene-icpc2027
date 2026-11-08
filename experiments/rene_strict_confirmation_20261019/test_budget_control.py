from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
from threadpoolctl import threadpool_limits

from experiments.rene_strict_confirmation_20261019.common import OUT, ARMS, load, observable, read_json, write_json, sha
from experiments.rene_strict_confirmation_20261019 import budget_control as b


class BudgetIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = load(OUT / 'development/items.pkl')
        cls.scores = load(OUT / 'development/scores.pkl')
        cls.models = b.load_models('fusion')
        cls.risks = joblib.load(OUT / 'models/budget_fusion_risk.joblib')
        cls.outputs, cls.sessions, cls.quota = b.infer_pair(cls.models, cls.risks,
                [observable(e) for e in cls.dev], cls.scores, 'fusion')

    def test_equal_actual_pairs_and_billed_slots(self):
        self.assertEqual(self.quota, 128)
        for traces in self.outputs.values():
            self.assertEqual(sum(t['actual_pairs'] for t in traces), 1024)
            self.assertEqual(sum(t['billed_slots'] for t in traces), 1024)
            self.assertEqual(sum(t['verified'] for t in traces), 128)

    def test_risk_refit_uses_only_original_separate_role(self):
        byid = {e['query_index']: e for e in self.dev}
        for arm in ARMS:
            role = self.models[arm]['roles']['risk']
            repos = {byid[q]['repo'] for q in role}
            self.assertTrue(set(self.risks[arm]['fit_repos']) <= repos)
            gain = set(self.models[arm]['role_repositories']['gain'])
            update = set(self.models[arm]['role_repositories']['update'])
            self.assertFalse(set(self.risks[arm]['fit_repos']) & (gain | update))

    def test_postgate_keeps_selected_action_or_original_and_cost(self):
        for arm in ARMS:
            before = self.outputs[f'fusion/{arm}/budget25_verified']
            after = self.outputs[f'fusion/{arm}/budget25_risk']
            for e, a, z in zip(self.dev, before, after):
                self.assertEqual(a['actual_pairs'], z['actual_pairs'])
                self.assertEqual(a['executed_ce_tokens'], z['executed_ce_tokens'])
                if z['accepted']:
                    self.assertEqual(a['ranking'], z['ranking'])
                    self.assertEqual(a['chosen'], z['chosen'])
                else:
                    self.assertEqual(z['ranking'], tuple(e['original_top100']))

    def test_supplement_was_frozen_without_outcomes_or_threshold_selection(self):
        p = read_json(OUT / 'fresh/BUDGET_PROTOCOL.json')
        self.assertFalse(p['outcomes_opened'])
        self.assertTrue(p['primary_hypotheses_unchanged'])
        self.assertEqual(p['source_sha256'], sha(b.__file__))
        self.assertEqual(p['thresholds'], {'base': .1, 'risk': .075})


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(BudgetIntegrity))
    write_json(OUT / 'development/BUDGET_TESTS.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
               failures=[str(x[0]) for x in result.failures], errors=[str(x[0]) for x in result.errors]))
    sys.exit(0 if result.wasSuccessful() else 1)
