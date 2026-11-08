from __future__ import annotations

import itertools
import math
import unittest
from unittest.mock import patch

import numpy as np

from .metrics import exact_uniform_max, factorial, order_candidates, paired_difference, summarize


class Contracts(unittest.TestCase):
    def test_uniform_max_matches_enumeration_with_ties(self):
        values = [0., 0., .1, .5, .5, 1.]
        for k in (1, 2, 3, 9):
            brute = [max(c) for c in itertools.combinations(values, min(k, len(values)))]
            self.assertAlmostEqual(exact_uniform_max(values, k), math.fsum(brute) / len(brute), places=14)

    def test_duplicate_positions_keep_probability_mass(self):
        # The first two positions may have identical text; both must remain in the pool.
        scores = {1: 1., 2: 1., 3: 0.}
        v, _ = factorial(scores, {'ENTR': [1, 2, 3], 'PROB': [3, 2, 1]}, 0.)
        self.assertAlmostEqual(v['RAND_exact_k1_mean'], 2 / 3)
        self.assertEqual(v['RAND_exact_k2_max'], 1.)

    def test_short_pool_clips_without_original_padding(self):
        v, c = factorial({1: .2}, {'ENTR': [1], 'PROB': [1]}, 1.)
        self.assertEqual(v['ENTR_k3_max'], .2)
        self.assertEqual(v['RAND_exact_k3_max'], .2)
        self.assertEqual(c['ENTR_B'], 0.)
        self.assertEqual(c['ENTR_G'], 0.)

    def test_selection_is_score_only_and_stable_on_ties(self):
        pool = [{'position': p, 'full_vocab_entropy': e, 'label_mean_log_probability': l, 'rr': rr}
                for p, e, l, rr in [(3, 1., -.2, 1.), (1, 1., -.2, 0.), (2, 2., -.1, .5)]]
        self.assertEqual(order_candidates(pool, 'ENTR'), [1, 3, 2])
        self.assertEqual(order_candidates(pool, 'PROB'), [2, 1, 3])
        for c in pool:
            c['rr'] = 1. - c['rr']
        self.assertEqual(order_candidates(pool, 'ENTR'), [1, 3, 2])

    def test_invalid_positions_and_scores_are_rejected(self):
        pool = [{'position': 1, 'full_vocab_entropy': 0., 'label_mean_log_probability': -.1}]
        with self.assertRaises(ValueError):
            order_candidates(pool * 2, 'ENTR')
        pool[0]['full_vocab_entropy'] = float('nan')
        with self.assertRaises(ValueError):
            order_candidates(pool, 'ENTR')
        with self.assertRaises(ValueError):
            factorial({1: .1, 2: .2}, {'ENTR': [1, 1], 'PROB': [1, 2]}, 0.)

    def test_nontrivial_decomposition_has_expected_sign(self):
        v, c = factorial({1: 1., 2: 0., 3: 0., 4: 0.}, {'ENTR': [1, 2, 3, 4], 'PROB': [2, 3, 4, 1]}, 0.)
        self.assertAlmostEqual(c['ENTR_B'], 2 / 3)
        self.assertAlmostEqual(c['ENTR_G'], -1 / 6)
        self.assertAlmostEqual(c['ENTR_J'], .5)
        self.assertAlmostEqual(c['ENTR_B'] + c['ENTR_G'], c['ENTR_J'])
        self.assertAlmostEqual(v['RAND_exact_k3_max'], .75)

    def test_pairing_is_by_query_and_keeps_direction(self):
        local = {'domain': 'source', 'query_id': 'q', 'group': 'g', 'positions': 3,
                 'values': {'NoRewrite': .3, 'ENTR_k1_mean': .1}, 'contrasts': {'ENTR_B': .02}}
        base = {**local, 'values': {'NoRewrite': .3, 'ENTR_k1_mean': .2}, 'contrasts': {'ENTR_B': -.03}}
        paired = paired_difference(local, base)
        self.assertAlmostEqual(paired['values']['ENTR_k1_mean'], .1)
        self.assertAlmostEqual(paired['contrasts']['ENTR_B'], -.05)
        with self.assertRaises(ValueError):
            paired_difference(local, {**base, 'query_id': 'other'})
        with self.assertRaises(ValueError):
            paired_difference(local, {**base, 'values': {'NoRewrite': .4, 'ENTR_k1_mean': .2}})

    def test_cluster_mean_is_query_weighted_and_draws_are_shared(self):
        x = np.array([[0., 0., 0.], [0., 0., 0.], [1., 2., 1.]])
        with patch('experiments.rene_t5_weight_factors_20261020.metrics.REPETITIONS', 500):
            s = summarize(x, ['a', 'b', 'diff'], ['g', 'g', 'h'], ())
        self.assertAlmostEqual(s['a']['mean'], 1 / 3)
        self.assertEqual(s['a']['groups'], 2)
        self.assertEqual(s['a']['ci95_group_bootstrap'], s['diff']['ci95_group_bootstrap'])
        self.assertTrue(np.allclose(s['b']['ci95_group_bootstrap'], np.array(s['a']['ci95_group_bootstrap']) * 2))

    def test_zero_difference_does_not_become_equivalence(self):
        with patch('experiments.rene_t5_weight_factors_20261020.metrics.REPETITIONS', 100):
            s = summarize(np.zeros((24, 1)), ['paired'], [str(i) for i in range(24)], {'paired'})
        self.assertEqual(s['paired']['direction'], 'unresolved')
        self.assertNotIn('equivalent', s['paired'])


if __name__ == '__main__':
    unittest.main()
