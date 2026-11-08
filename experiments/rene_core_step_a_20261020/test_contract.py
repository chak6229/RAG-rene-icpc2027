"""Tests for the estimand, pairing, and exact random comparator."""
from __future__ import annotations

import itertools
import math
import unittest

import numpy as np
from scipy.stats import t

from .metrics import clustered_summary, effects, exact_random_max, rr


class ScientificContract(unittest.TestCase):
    def test_exact_maximum_matches_exhaustive_subsets_including_ties(self):
        for n in range(1, 10):
            values = [float((index * 7) % 4) / 4 for index in range(n)]
            for k in range(1, min(4, n) + 1):
                enumeration = [max(values[i] for i in subset) for subset in itertools.combinations(range(n), k)]
                self.assertAlmostEqual(exact_random_max(values, k), math.fsum(enumeration) / len(enumeration), places=14)

    def test_exact_random_small_pool_keeps_real_positions(self):
        self.assertEqual(exact_random_max([0.0], 3), 0.0)
        self.assertEqual(exact_random_max([0.0, 0.5], 3), 0.5)
        self.assertAlmostEqual(exact_random_max([0.0, 0.5, 0.5, 1.0], 1), 0.5)
        with self.assertRaises(ValueError):
            exact_random_max([], 3)

    def test_rr_missing_target_and_depth_boundary(self):
        self.assertEqual(rr(list(range(100)), {0}), 1.0)
        self.assertEqual(rr(list(range(100)), {99}), 0.01)
        self.assertEqual(rr(list(range(100)), {100}), 0.0)
        with self.assertRaises(ValueError):
            rr([0] * 100, {0})

    def test_contract_difference_cancels_when_paired_effects_match(self):
        # Large and variable marginal effects can still have identically zero j.
        rows = []
        for baseline in (0.1, 0.3, 0.8, 0.2, 0.5):
            metric = {"NoRewrite": 0.2, "ENTR_top1": baseline, "PROB_top1": baseline,
                      "ENTR_best3": baseline, "PROB_best3": baseline,
                      "Uniform_single_exact": 0.1, "RAND20_best3": 0.1, "Uniform_best3_exact": 0.1}
            rows.append([effects(metric)["j_ENTR_contract_difference"]])
        summary = clustered_summary(rows, ["a", "a", "b", "c", "c"], ["j_ENTR_contract_difference"], repetitions=300, seed=3)
        self.assertEqual(summary["j_ENTR_contract_difference"]["mean"], 0.0)
        self.assertEqual(summary["j_ENTR_contract_difference"]["standard_error_cluster"], 0.0)

    def test_query_weighting_and_cluster_se_by_independent_hand_calculation(self):
        values = [0.1, 0.3, -0.1, 0.5, 0.2, 0.4]
        labels = ["a", "a", "b", "c", "c", "c"]
        mean = math.fsum(values) / 6
        influence = [0.4 - 2 * mean, -0.1 - mean, 1.1 - 3 * mean]
        se = math.sqrt(3 / 2 * math.fsum(value * value for value in influence) / 36)
        result = clustered_summary([[value] for value in values], labels, ["metric"], repetitions=300, seed=9)["metric"]
        self.assertAlmostEqual(result["mean"], mean)
        self.assertAlmostEqual(result["standard_error_cluster"], se)
        self.assertAlmostEqual(result["ci95_cluster_t"][0], mean - t.ppf(0.975, 2) * se)
        self.assertNotAlmostEqual(result["mean"], np.mean([0.2, -0.1, 1.1 / 3]))


if __name__ == "__main__":
    unittest.main()
