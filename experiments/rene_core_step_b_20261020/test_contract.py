"""Scientific-contract checks: leakage, original-access boundaries, action budgets."""
from __future__ import annotations

import unittest

import numpy as np

from .common import BLOCKS, M0, R, T
from .metrics import clustered
from .modeling import decide, fit_fold


class Contract(unittest.TestCase):
    def test_no_probability_or_original_leaks_into_common_block(self):
        self.assertEqual(sorted(M0+T+R), list(range(71)))
        self.assertEqual(len(set(M0+T+R)), 71)
        self.assertFalse(set(M0) & set(list(range(19, 31))+list(range(43, 65))))
        self.assertFalse(set(M0) & {0, 1, 2, 3, 6, 7, 15, 16, 17, 18})
        self.assertEqual(set(BLOCKS["MRT"])-set(BLOCKS["MT"]), set(R))
        self.assertEqual(set(BLOCKS["MRT"])-set(BLOCKS["MR"]), set(T))

    def test_held_labels_and_held_covariates_do_not_fit_transform(self):
        rng = np.random.default_rng(81)
        x, y = rng.normal(size=(35, 71)), rng.normal(size=35)
        folds = np.arange(35) % 5
        model, pred = fit_fold(x, y, folds, "MRT", 2)
        y2 = y.copy()
        y2[folds == 2] = 1e9
        model2, pred2 = fit_fold(x, y2, folds, "MRT", 2)
        np.testing.assert_array_equal(pred, pred2)
        np.testing.assert_array_equal(model["coefficient"], model2["coefficient"])
        x2 = x.copy()
        x2[folds == 2] += 1000
        model3, _ = fit_fold(x2, y, folds, "MRT", 2)
        np.testing.assert_array_equal(model["scaler_mean"], model3["scaler_mean"])
        np.testing.assert_array_equal(model["scaler_scale"], model3["scaler_scale"])
        np.testing.assert_array_equal(model["coefficient"], model3["coefficient"])

    def test_quota_excludes_unavailable_and_is_per_fold_with_id_ties(self):
        ids = [f"q{i:02d}" for i in range(15, 0, -1)]
        fold = np.asarray([0]*9 + [1]*6)
        available = np.ones(15, dtype=bool)
        available[0] = False
        chosen = decide(np.zeros(15), ids, fold, available, "fixed20")
        self.assertFalse(chosen[0])
        self.assertEqual(int(chosen[:9].sum()), 1)
        self.assertEqual(int(chosen[9:].sum()), 1)
        self.assertEqual({ids[i] for i in np.flatnonzero(chosen)}, {"q07", "q01"})

    def test_fixed_budget_includes_negative_predictions_without_gate(self):
        ids = [str(i) for i in range(10)]
        folds = np.zeros(10, dtype=int)
        available = np.ones(10, dtype=bool)
        score = -np.arange(1, 11, dtype=float)
        self.assertEqual(int(decide(score, ids, folds, available, "positive").sum()), 0)
        self.assertEqual(int(decide(score, ids, folds, available, "fixed20").sum()), 2)
        self.assertEqual(set(np.flatnonzero(decide(score, ids, folds, available, "fixed20"))), {0, 1})

    def test_common_noop_gate_does_not_claim_confidence_gain(self):
        score = np.asarray([2.0, -1.0, 0.0, 3.0])
        selected = decide(score, list("abcd"), np.zeros(4), np.asarray([False, True, True, True]), "positive")
        np.testing.assert_array_equal(selected, [False, False, False, True])

    def test_query_weighting_and_paired_group_intervals(self):
        groups = np.asarray(["large", "large", "large", "small"])
        vector = np.asarray([1., 1., 1., -1.])
        result = clustered(np.column_stack([vector, np.zeros(4)]), groups, ["delta", "paired_zero"], repetitions=300)
        self.assertAlmostEqual(result["delta"]["mean"], .5)
        np.testing.assert_array_equal(result["paired_zero"]["ci95_group_bootstrap"], [0, 0])


if __name__ == "__main__":
    unittest.main()
