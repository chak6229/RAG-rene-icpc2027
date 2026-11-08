from __future__ import annotations

import copy
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from .common import group_folds
from .features import feature_vector, FEATURE_NAMES, GEN_NAMES, score_features
from .gain import choose, decision, train_partition
from .evaluation import rq2_rows


def sample():
    candidate = {"position": 1, "span": "file", "rewrite": "find file path", "is_identity": False,
                 "full_vocab_entropy": 1.5, "label_mean_log_probability": -.8, "label_sum_log_probability": -4.0,
                 "mean_max_vocab_probability": .6, "label_token_count": 5, "span_token_count": 2,
                 "span_full_vocab_entropy": 2.0, "span_mean_log_probability": -1.0}
    original = {"ranked_doc_ids": list(range(100)), "scores": np.linspace(4, 0, 100).tolist()}
    rewritten = {"ranked_doc_ids": [1, 0] + list(range(2, 100)), "scores": np.linspace(5, 0, 100).tolist()}
    code_words = {i: {"find", "file", "path", "python"} for i in range(100)}
    return candidate, original, rewritten, code_words


def prediction(position, rank, value):
    return {"position": position, "entr_rank": rank, "predictions": {"M": {"effective": value}}}


class FeatureTests(unittest.TestCase):
    def test_schema_and_finite(self):
        c, o, n, w = sample()
        x = feature_vector("find path", c, o, n, [c], w)
        self.assertEqual(x.shape, (len(FEATURE_NAMES),))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(len(FEATURE_NAMES), len(set(FEATURE_NAMES)))

    def test_document_ids_are_not_predictive_inputs(self):
        c, o, n, w = sample()
        first = feature_vector("find path", c, o, n, [c], w)
        transformed_o = {**o, "ranked_doc_ids": [1000 + 7 * d for d in o["ranked_doc_ids"]]}
        transformed_n = {**n, "ranked_doc_ids": [1000 + 7 * d for d in n["ranked_doc_ids"]]}
        transformed_w = {1000 + 7 * d: terms for d, terms in w.items()}
        np.testing.assert_array_equal(first, feature_vector("find path", c, transformed_o, transformed_n, [c], transformed_w))

    def test_gold_extra_fields_do_not_change_features(self):
        c, o, n, w = sample()
        first = feature_vector("find path", c, o, n, [c], w)
        c.update({"delta_rr": -1, "target_doc_id": 67, "repository": "secret"})
        n.update({"rr": 1, "gold_rank": 1})
        np.testing.assert_array_equal(first, feature_vector("find path", c, o, n, [c], w))

    def test_empty_span_is_finite(self):
        c, o, n, w = sample()
        c.update({"span": "", "span_full_vocab_entropy": None, "span_mean_log_probability": None, "span_token_count": 0})
        self.assertTrue(np.isfinite(feature_vector("find path", c, o, n, [c], w)).all())

    def test_score_scale_translation(self):
        scores = np.linspace(4, 0, 100)
        first, shifted = score_features(scores), score_features(scores + 23)
        indices = [5, 6, 8, 9, 10, 11]
        np.testing.assert_allclose(first[indices], shifted[indices], atol=1e-12, rtol=1e-12)

    def test_equal_scores_finite(self):
        values = score_features(np.ones(100))
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(values[-1], np.log(100))

    def test_incomplete_score_list_rejected(self):
        with self.assertRaises(ValueError):
            score_features([1, .5])


class IsolationTests(unittest.TestCase):
    def test_group_fold_assignment_is_complete_and_balanced(self):
        labels = [str(i) for i in range(512)]
        a = group_folds(labels)
        b = group_folds(list(reversed(labels)) + labels[:4])
        self.assertEqual(a, b)
        self.assertEqual(set(a.values()), set(range(5)))
        self.assertEqual(sorted(list(a.values()).count(i) for i in range(5)), [102, 102, 102, 103, 103])

    def test_cross_group_fit_rejects_leakage(self):
        x = np.arange(16).reshape(8, 2)
        with self.assertRaisesRegex(ValueError, "leakage"):
            train_partition("M", {"features": "all", "algorithm": "StandardScaler+Ridge", "alpha": 10}, {},
                            x, np.arange(8), np.ones(8), ["a"] * 8, np.array([0, 1]), np.array([2, 3]))

    def test_poisoning_held_labels_cannot_change_predictions(self):
        x = np.arange(16, dtype=float).reshape(8, 2)
        y = np.linspace(-.4, .6, 8)
        groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
        spec = {"features": "all", "algorithm": "StandardScaler+Ridge", "alpha": 10}
        fit = lambda labels: train_partition("M", spec, {}, x, labels, np.ones(8), groups, np.arange(4), np.arange(4, 8))[1]
        with threadpool_limits(limits=1):
            first = fit(y)
            poisoned = y.copy()
            poisoned[4:] = [9e5, -1e7, 8e9, -9e9]
            np.testing.assert_array_equal(first, fit(poisoned))
            poisoned[:4] += .3
            self.assertGreater(np.max(np.abs(first - fit(poisoned))), .1)


class DecisionTests(unittest.TestCase):
    def test_selective_negative_falls_back(self):
        rows = [prediction(2, 1, -.2), prediction(7, 2, -.01)]
        self.assertEqual(choose(rows, "M", selective=True), (None, 0))
        self.assertEqual(choose(rows, "M", selective=False), (7, -.01))

    def test_no_rewrite_wins_zero_tie(self):
        self.assertEqual(choose([prediction(1, 1, 0)], "M", selective=True), (None, 0))

    def test_positive_selected(self):
        self.assertEqual(choose([prediction(5, 1, .02), prediction(9, 2, .1)], "M", selective=True), (9, .1))

    def test_equal_scores_break_by_native_order(self):
        self.assertEqual(choose([prediction(1, 2, .1), prediction(8, 1, .1)], "M", selective=False), (8, .1))

    def test_adoption_uses_actual_rankings(self):
        _, original, changed, _ = sample()
        actions = {("q", None): original, ("q", 1): changed}
        adopted = decision("q", "M", 1, .1, actions, 0)
        self.assertTrue(adopted["adopted"])
        self.assertEqual(adopted["ranked_doc_ids"], changed["ranked_doc_ids"])
        fallback = decision("q", "M", None, 0, actions, 0)
        self.assertEqual(fallback["ranked_doc_ids"], original["ranked_doc_ids"])
        self.assertFalse(fallback["adopted"])

    def test_identical_candidate_restores_full_ranking(self):
        _, original, _, _ = sample()
        out = decision("q", "M", 1, 0, {("q", None): original, ("q", 1): copy.deepcopy(original)}, 0)
        self.assertEqual(out["action"], "fallback")
        self.assertEqual(len(out["ranked_doc_ids"]), 100)

    def fixture(self):
        protocol = {"query_ids": ["q"], "direct_gain": {"models": {"M": {}}}}
        actions = [{"query_id": "q", "position": None, "ranked_doc_ids": [0, 1]},
                   {"query_id": "q", "position": 1, "ranked_doc_ids": [1, 0]}]
        amap = {(r["query_id"], r["position"]): r for r in actions}
        policies = ["NoRewrite", "ENTR_top1", "PROB_in_ENTR3", "M_forced", "M_selective"]
        decisions = [decision("q", p, None if p == "NoRewrite" else 1, 0, amap, 0) for p in policies]
        selected = [{"query_id": "q", "policy": "ENTR", "selected": [{"position": 1, "selected_rank": 1}]}]
        return protocol, actions, decisions, selected

    def test_no_success_still_keeps_denominator(self):
        p, a, d, s = self.fixture()
        rows = rq2_rows(d, a, {"q": {999}}, p, {"q": "g"}, s)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["policies"]["M_forced"]["rr"], 0)
        self.assertTrue(rows[0]["policies"]["M_forced"]["adopted"])

    def test_missing_decision_rejected(self):
        p, a, d, s = self.fixture()
        with self.assertRaisesRegex(ValueError, "missing"):
            rq2_rows(d[:-1], a, {"q": {0}}, p, {"q": "g"}, s)

    def test_duplicate_decision_rejected(self):
        p, a, d, s = self.fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rq2_rows(d + [d[0]], a, {"q": {0}}, p, {"q": "g"}, s)

    def test_pool_escape_rejected(self):
        p, a, d, s = self.fixture()
        d[-1]["selected_position"] = 99
        with self.assertRaisesRegex(ValueError, "pool"):
            rq2_rows(d, a, {"q": {0}}, p, {"q": "g"}, s)


if __name__ == "__main__":
    unittest.main()
