from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from .common import BLOCKS
from .modeling import (assert_partition, calibrated_winners, calibration_matrix, fit_calibrator,
                       inner_oof, make_decision, quota, winners)


def sample():
    rng = np.random.default_rng(31)
    rows, query_rows, qids, groups, folds, actions = [], {}, [], [], [], {}
    for q in range(100):
        qid = f"q{q:03d}"
        qids.append(qid)
        query_rows[qid] = []
        actions[qid, None] = {"ranked_doc_ids": [1, 2, 3]}
        for p in range(3):
            query_rows[qid].append(len(rows))
            rows.append({"query_id": qid, "position": p + 1, "entr_rank": p + 1, "structural_zero": False})
            groups.append(f"g{q // 2}")
            folds.append((q // 2) % 5)
            actions[qid, p + 1] = {"ranked_doc_ids": [2, 1, 3]}
    x = rng.normal(size=(len(rows), 71))
    y = .2 * x[:, 0] + rng.normal(scale=.1, size=len(rows))
    return SimpleNamespace(retriever="synthetic", qids=qids, rows=rows, query_rows=query_rows, x=x, y=y, groups=np.array(groups),
                           folds=np.array(folds), actions=actions, fold_map={q: folds[query_rows[q][0]] for q in qids})


class IsolationTests(unittest.TestCase):
    def test_outer_label_poison_does_not_change_inner_or_calibration(self):
        a = sample()
        b = copy.deepcopy(a)
        b.y[b.folds == 0] += 100000
        outer_prediction = a.x[:, 0] * .1 + .3
        with threadpool_limits(limits=1):
            ia = inner_oof(a, 0, "RIDGE", {})
            ib = inner_oof(b, 0, "RIDGE", {})
            wa, _, _ = calibrated_winners(a, 0, outer_prediction, ia)
            wb, _, _ = calibrated_winners(b, 0, outer_prediction, ib)
        np.testing.assert_array_equal(ia[a.folds != 0], ib[b.folds != 0])
        self.assertEqual(wa, wb)

    def test_inner_fit_never_uses_outer_groups(self):
        d = sample()
        observed = []
        with threadpool_limits(limits=1):
            pred = inner_oof(d, 2, "RIDGE", {}, lambda f, tr, he, m: observed.append((tr, he)))
        for tr, he in observed:
            self.assertFalse(np.any(d.folds[tr] == 2))
            self.assertFalse(np.any(d.folds[he] == 2))
            self.assertFalse(set(d.groups[tr]) & set(d.groups[he]))
        self.assertTrue(np.isnan(pred[d.folds == 2]).all())

    def test_group_overlap_is_rejected(self):
        d = sample()
        with self.assertRaises(ValueError):
            assert_partition(d, np.array([0, 1]), np.array([3, 4]))


class DecisionTests(unittest.TestCase):
    def test_native_rank_breaks_tie(self):
        d = sample()
        p = np.ones(len(d.rows))
        self.assertTrue(all(r["position"] == 1 for r in winners(d, range(len(d.rows)), p)))

    def test_known_identity_beats_negative_candidates(self):
        d = sample()
        d.rows[0]["structural_zero"] = True
        p = np.full(len(d.rows), -.1)
        p[0] = -100
        r = winners(d, [0, 1, 2], p)[0]
        self.assertEqual(r["score"], 0)
        self.assertFalse(r["eligible"])

    def test_partial_query_is_rejected(self):
        d = sample()
        with self.assertRaises(ValueError):
            winners(d, [0, 1], np.ones(len(d.rows)))

    def test_positive_winner_calibration_keeps_position(self):
        d = sample()
        p = np.full(len(d.rows), .3)
        with threadpool_limits(limits=1):
            inner = inner_oof(d, 0, "RIDGE", {})
            held, _, _ = calibrated_winners(d, 0, p, inner)
        self.assertTrue(all(r["position"] == 1 for r in held))

    def test_fallback_restores_original_complete_ranking(self):
        d = sample()
        w = winners(d, [0, 1, 2], np.ones(len(d.rows)))[0]
        row = make_decision(d, w, "CAL", -.01, False)
        self.assertIsNone(row["selected_position"])
        self.assertEqual(row["ranking_reference"]["position"], None)
        self.assertFalse(row["adopted"])

    def test_score_zero_does_not_count_as_eligible(self):
        d = sample()
        selected = winners(d, range(len(d.rows)), np.zeros(len(d.rows)))
        self.assertFalse(any(r["eligible"] for r in selected))


class CalibrationTests(unittest.TestCase):
    def test_no_eligible_training_returns_zero(self):
        d = sample()
        selected = winners(d, range(len(d.rows)), -np.ones(len(d.rows)))
        model, eligible = fit_calibrator(d, selected, "AFFINE")
        self.assertEqual(eligible, [])
        np.testing.assert_array_equal(model.predict(np.ones((4, 1))), np.zeros(4))

    def test_matrices_do_not_depend_on_gold(self):
        d = sample()
        selected = winners(d, range(len(d.rows)), np.ones(len(d.rows)))
        before = calibration_matrix(d, selected, "CONTEXT_RIDGE")
        d.y[:] = 10000
        np.testing.assert_array_equal(before, calibration_matrix(d, selected, "CONTEXT_RIDGE"))
        self.assertEqual(before.shape, (100, 55))

    def test_feature_blocks_exclude_gold_and_have_exact_dimensions(self):
        self.assertEqual({k: len(v) for k, v in BLOCKS.items()}, {"GEN": 19, "GS": 55, "GSA": 64, "GSL": 62, "FULL": 71})
        self.assertEqual(set(BLOCKS["GSA"]) & set(range(64, 71)), set())
        self.assertEqual(set(BLOCKS["GSL"]) & set(range(55, 64)), set())

    def test_equal_quota_preserves_adoption_count(self):
        d = sample()
        rows = winners(d, range(len(d.rows)), np.arange(len(d.rows), dtype=float) / 1000 + .1)
        for r in rows:
            r["alternate"] = -r["score"]
        a = quota(rows, "score", .5, eligible_only=True)
        b = quota(rows, "alternate", .5, eligible_only=True)
        self.assertEqual(len(a), 50)
        self.assertEqual(len(a), len(b))
        self.assertNotEqual(a, b)

    def test_quota_ties_are_query_id_deterministic(self):
        d = sample()
        rows = winners(d, range(len(d.rows)), np.ones(len(d.rows)))
        self.assertEqual(quota(rows, "score", .2, eligible_only=False),
                         quota(list(reversed(rows)), "score", .2, eligible_only=False))

    def test_isotonic_plateau_keeps_raw_ranking(self):
        d = sample()
        rows = winners(d, range(len(d.rows)), np.arange(len(d.rows), dtype=float) / 1000 + .1)
        for r in rows:
            r["flat_calibration"] = .01
        self.assertEqual(quota(rows, "score", .5, eligible_only=True),
                         quota(rows, "flat_calibration", .5, eligible_only=True))


if __name__ == "__main__":
    unittest.main()
