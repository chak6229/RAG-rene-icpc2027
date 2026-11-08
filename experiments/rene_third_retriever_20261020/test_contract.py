from __future__ import annotations

import builtins
import itertools
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.rene_budget_aggregation_20261020.metrics import contrasts, exact_random_max, factorial_values, ordered_positions

from .common import OUT, deny_outcomes, now, write
from .encoder import stable_order, token_ids
from .statistics import summarize


class TokenizerStub:
    cls_token = '<s>'
    sep_token = '</s>'
    unk_token_id = 99
    pad_token_id = 1
    vocabulary = {'<s>': 0, '</s>': 2, '<encoder-only>': 7, 'one': 8, 'two': 9}

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [self.vocabulary.get(t, 99) for t in tokens]


class ContractTests(unittest.TestCase):
    def test_official_prefix_and_truncation(self):
        self.assertEqual(token_ids(TokenizerStub(), ['one two one', ''], 6), [[0, 7, 2, 8, 9, 2], [0, 7, 2, 2]])
        with self.assertRaises(ValueError):
            token_ids(TokenizerStub(), ['one'], 3)

    def test_score_ties_preserve_original_doc_id(self):
        self.assertEqual(stable_order([.5, .9, .5, .9], [8, 6, 2, 4]).tolist(), [3, 1, 2, 0])
        with self.assertRaises(ValueError):
            stable_order([float('nan')], [0])

    def test_exact_uniform_matches_exhaustive_position_sampling(self):
        # Repeated values must preserve their separate position probability mass.
        for values in ([0., .25, .25, 1.], [0., 0.], [.5]):
            for k in (1, 2, 3):
                brute = np.mean([max(c) for c in itertools.combinations(values, min(k, len(values)))])
                self.assertAlmostEqual(exact_random_max(values, k), brute, places=15)

    def test_decomposition_changes_relative_selector_advantage(self):
        scores = {1: 1., 2: .1, 3: .2, 4: 0.}
        orders = {'ENTR': [1, 2, 3, 4], 'PROB': [2, 3, 4, 1]}
        fx = contrasts(factorial_values(scores, orders, .5, exact_random_max(list(scores.values()), 3)))
        self.assertGreater(fx['ENTR_B_budget_mean_1_to_3'], 0)
        self.assertLess(fx['PROB_B_budget_mean_1_to_3'], 0)
        self.assertAlmostEqual(fx['ENTR_J_exact'], fx['ENTR_B_budget_mean_1_to_3'] + fx['ENTR_G_aggregation_at3_mean_to_max'])
        self.assertAlmostEqual(fx['ENTR_minus_PROB_B_budget_mean_1_to_3'], fx['ENTR_B_budget_mean_1_to_3'] - fx['PROB_B_budget_mean_1_to_3'])

    def test_short_pool_has_no_original_padding(self):
        fx = factorial_values({1: 0.}, {'ENTR': [1], 'PROB': [1]}, 1., 0.)
        self.assertEqual(fx['ENTR_k3_max'], 0.)
        self.assertEqual(fx['ENTR_k3_mean'], 0.)

    def test_selector_deterministic_ties_and_invalid_input(self):
        pool = [{'position': p, 'full_vocab_entropy': 1., 'label_mean_log_probability': -2.} for p in (3, 1, 2)]
        self.assertEqual(ordered_positions(pool, 'ENTR'), [1, 2, 3])
        self.assertEqual(ordered_positions(pool, 'PROB'), [1, 2, 3])
        with self.assertRaises(ValueError):
            ordered_positions(pool + [pool[0]], 'ENTR')

    def test_pairing_and_cluster_query_weighting(self):
        a = np.array([.2, .2, -.1, .4, .4])
        b = np.array([.1, .1, -.2, .3, .3])
        x = np.column_stack([a, b, a - b])
        r = summarize(x, ['a', 'b', 'paired'], ['g1', 'g1', 'g2', 'g3', 'g3'], family_names=['paired'], repetitions=200, family_size=12)
        self.assertAlmostEqual(r['a']['mean'], a.mean())
        self.assertAlmostEqual(r['paired']['mean'], .1)
        self.assertLess(r['paired']['se_cluster'], 1e-15)
        # The individual standard errors cannot be combined as independent arms.
        self.assertGreater(r['a']['se_cluster'], .01)
        self.assertEqual(r['paired']['direction'], 'unresolved')  # fewer than 20 nonzero groups
        self.assertLessEqual(r['paired']['family12_ci_cluster_t'][0], r['paired']['ci95_cluster_t'][0])

    def test_bootstrap_uses_whole_groups(self):
        x = np.array([[0.], [1.], [1.]])
        groups = ['a', 'b', 'b']
        r = summarize(x, ['x'], groups, repetitions=1000, seed=10)['x']
        rng = np.random.default_rng(10)
        draw = rng.integers(0, 2, (1000, 2))
        expected = np.array([sum([0., 2.][g] for g in d) / sum([1, 2][g] for g in d) for d in draw])
        self.assertTrue(np.allclose(r['ci95_group_bootstrap'], np.quantile(expected, [.025, .975]), atol=1e-15))

    def test_outcome_guard_all_open_apis(self):
        with tempfile.TemporaryDirectory() as tmp:
            forbidden = Path(tmp) / 'private' / 'labels.json'
            forbidden.parent.mkdir()
            forbidden.write_text('[]')
            public = Path(tmp) / 'queries.json'
            public.write_text('[]')
            with deny_outcomes():
                for op in (lambda: forbidden.read_text(), lambda: builtins.open(forbidden), lambda: (Path(tmp) / 'RESULTS.json').read_text()):
                    with self.assertRaises(PermissionError):
                        op()
                self.assertEqual(public.read_text(), '[]')
            self.assertEqual(forbidden.read_text(), '[]')


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    report = {'created_utc': now(), 'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors)}
    if (OUT / 'TEST_RESULTS.json').exists():
        if not result.wasSuccessful():
            raise SystemExit(1)
    else:
        write(OUT / 'TEST_RESULTS.json', report)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
