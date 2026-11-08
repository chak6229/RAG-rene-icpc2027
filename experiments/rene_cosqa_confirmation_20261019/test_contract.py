from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from .common import ROOT, capped_ids, deny_relevance_reads
from .evaluation import primary_conclusion
from .policy import (RISK_NAMES, allocation, all_decisions, apply_risk, ce_features, fit_risk,
                     offsets, risk_features, risk_predict, trace, verification_plan, witness_matrix)
from .statistics import aggregate, design_factor, interval, plan, ratio_interval


def fixture(q="q"):
    original = list(range(1, 101))
    rankings = [[2, 1, *range(3, 101)], [3, 2, 1, *range(4, 101)], original]
    rows = [{"query_id": q, "position": i + 1, "entr_rank": i + 1,
             "values": [0.] * 71, "structural_zero": i == 2} for i in range(3)]
    mu = np.array([.03, .02, 0.])
    p = verification_plan(original, rankings, rows, mu)
    return {"query_id": q, "rows": rows, "original": original, "rankings": rankings, "mu": mu,
            "native_harm": np.array([.2, .3, .1]), "control": mu.copy(), "plan": p,
            "ce": {d: float(d % 3) for d in p["doc_ids"]}}


class SamplingAndStatistics(unittest.TestCase):
    def test_cap_preserves_every_group(self):
        ids = [str(i) for i in range(15)]
        groups = {q: "a" if int(q) < 10 else "b" for q in ids}
        chosen = capped_ids(ids, groups, 4)
        self.assertEqual(len(chosen), 8)
        self.assertEqual({groups[q] for q in chosen}, {"a", "b"})

    def test_sampling_is_input_order_invariant(self):
        ids = [str(i) for i in range(15)]
        groups = {q: "g" for q in ids}
        self.assertEqual(capped_ids(ids, groups, 4), capped_ids(ids[::-1], groups, 4))

    def test_sampling_does_not_duplicate_or_pad(self):
        self.assertEqual(capped_ids(["q"], {"q": "g"}, 16), ["q"])

    def test_cluster_aggregation_keeps_query_denominator(self):
        a, n = aggregate([1., 0., 0.], ["a", "a", "b"])
        self.assertAlmostEqual(a.sum() / n.sum(), 1 / 3)
        self.assertEqual(n.tolist(), [2., 1.])

    def test_nonfinite_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([np.nan], ["a"])

    def test_zero_denominator_ratio_is_undefined(self):
        self.assertIsNone(ratio_interval([0., 0.], [0., 0.], ["a", "b"])["value"])

    def test_cluster_interval_does_not_invent_independent_queries(self):
        v = [.1, -.2, .3, .1, -.1, .2]
        m = interval(v, ["a", "a", "b", "b", "c", "c"], repetitions=200)
        self.assertEqual(m["groups"], 3)
        self.assertEqual(m["queries"], 6)

    def test_group_imbalance_factor(self):
        self.assertEqual(design_factor([3, 3, 3]), 1.)
        self.assertGreater(design_factor([1, 1, 20]), 1.)

    def test_zero_variance_does_not_authorize_confirmation_power(self):
        with self.assertRaises(ValueError):
            plan([0.] * 30, [str(i) for i in range(30)], [1] * 40, repetitions=100)

    def test_cross_zero_is_not_automatically_equivalence(self):
        m = {"changed_groups": 100, "ci95_cluster_t": [-.008, .009], "ci95_group_bootstrap": [-.009, .010]}
        self.assertFalse(primary_conclusion(m)["practical_equivalence_within_prespecified_plusminus005"])

    def test_equivalence_requires_both_prespecified_intervals(self):
        m = {"changed_groups": 100, "ci95_cluster_t": [-.002, .003], "ci95_group_bootstrap": [-.003, .006]}
        self.assertFalse(primary_conclusion(m)["practical_equivalence_within_prespecified_plusminus005"])

    def test_equivalence_has_a_fixed_margin(self):
        m = {"changed_groups": 100, "ci95_cluster_t": [-.002, .003], "ci95_group_bootstrap": [-.003, .004]}
        self.assertTrue(primary_conclusion(m)["practical_equivalence_within_prespecified_plusminus005"])

    def test_sparse_groups_cannot_claim_equivalence(self):
        m = {"changed_groups": 2, "ci95_cluster_t": [0., 0.], "ci95_group_bootstrap": [0., 0.]}
        self.assertIn("unresolved", primary_conclusion(m))


class InferenceIntegrity(unittest.TestCase):
    def test_private_annotations_are_denied(self):
        with deny_relevance_reads(), self.assertRaises(ValueError):
            Path("C:/test/private/H_qrels.jsonl").read_text()

    def test_private_raw_qrels_are_denied(self):
        with deny_relevance_reads(), self.assertRaises(ValueError):
            open("C:/test/qrels.jsonl", "rb")

    def test_public_source_can_be_read(self):
        with deny_relevance_reads():
            self.assertIn("InferenceIntegrity", Path(__file__).read_text("utf-8"))

    def test_guard_is_restored_after_exit(self):
        with deny_relevance_reads():
            pass
        with self.assertRaises(FileNotFoundError):
            Path("C:/nonexistent/private/H_qrels.jsonl").read_text()


class StagePolicy(unittest.TestCase):
    def test_verification_has_eight_real_distinct_documents(self):
        item = fixture()
        self.assertEqual(len(item["plan"]["doc_ids"]), 8)
        self.assertEqual(len(set(item["plan"]["doc_ids"])), 8)
        self.assertIn(item["original"][0], item["plan"]["doc_ids"])

    def test_negative_gain_query_is_not_verified(self):
        item = fixture()
        self.assertIsNone(verification_plan(item["original"], item["rankings"], item["rows"], np.array([-.1, -.2, 0.])))

    def test_budget_never_exceeds_quarter(self):
        items = {str(i): fixture(str(i)) for i in range(11)}
        self.assertEqual(len(allocation(items)), 2)

    def test_budget_ties_are_reproducible(self):
        items = {str(i): fixture(str(i)) for i in range(8)}
        self.assertEqual(allocation(items), {"0", "1"})

    def test_CE_missing_requested_document_is_rejected(self):
        item = fixture()
        with self.assertRaises(ValueError):
            ce_features(item["original"], item["rankings"][0], item["plan"], {})

    def test_unrequested_CE_results_cannot_leak_extra_budget(self):
        item = fixture()
        a = witness_matrix(item["original"], item["rankings"], item["rows"], item["mu"], item["plan"], item["ce"])
        item["ce"][10000] = 1e10
        b = witness_matrix(item["original"], item["rankings"], item["rows"], item["mu"], item["plan"], item["ce"])
        np.testing.assert_array_equal(a, b)

    def test_identity_is_not_counted_as_adoption(self):
        item = fixture()
        item["mu"] = np.array([-.1, -.2, 0.])
        t = trace(item, {"model": None}, False)
        self.assertFalse(t["accepted"])
        self.assertEqual(t["ranking"], item["original"])

    def test_verification_cost_is_actual_and_retained(self):
        item = fixture()
        t = trace(item, {"model": None}, True)
        model = {"beta": None, "class_prior": np.array([.3, .6, .1])}
        after = apply_risk(item, t, model)
        self.assertEqual(after["ranking"], item["original"])
        self.assertEqual(after["actual_pairs"], 8)
        self.assertEqual(after["verification_doc_ids"], t["verification_doc_ids"])

    def test_risk_acceptance_preserves_same_action(self):
        item = fixture()
        t = trace(item, {"model": None}, False)
        kept = apply_risk(item, t, {"beta": None, "class_prior": np.array([.05, .8, .15])})
        self.assertEqual(kept["ranking"], t["ranking"])
        self.assertEqual(kept["position"], t["position"])

    def test_risk_rejection_does_not_reselect_rival(self):
        item = fixture()
        t = trace(item, {"model": None}, False)
        kept = apply_risk(item, t, {"beta": None, "class_prior": np.array([.2, .7, .1])})
        self.assertIsNone(kept["chosen"])
        self.assertEqual(kept["ranking"], item["original"])

    def test_risk_features_are_finite(self):
        item = fixture()
        x, native = risk_features(item, trace(item, {"model": None}, True))
        self.assertEqual(x.shape, (len(RISK_NAMES),))
        self.assertTrue(np.isfinite(x).all())
        self.assertGreater(native, 0.)

    def test_threeway_offsets_preserve_probability_semantics(self):
        eta = offsets([.2], .25)
        np.testing.assert_allclose(np.exp(eta)[0], [.2, .6, .2])

    def test_threeway_actual_fit_is_normalized_and_masks_duplicate(self):
        rng = np.random.default_rng(33)
        x = rng.normal(size=(90, len(RISK_NAMES)))
        delta = np.tile([-.5, 0., .2], 30)
        model = fit_risk(x, np.full(90, .2), delta, list(range(90)))
        prob = risk_predict(model, x, np.full(90, .2))
        np.testing.assert_allclose(prob.sum(1), 1.)
        self.assertEqual(model["mask"][9], 0.)
        self.assertTrue((prob > 0).all())

    def test_sparse_risk_fit_is_an_explicit_prior(self):
        model = fit_risk(np.zeros((3, len(RISK_NAMES))), [.2] * 3, [-.1, 0, .2], ["a", "b", "c"])
        self.assertIsNone(model["beta"])
        np.testing.assert_allclose(risk_predict(model, np.zeros((1, len(RISK_NAMES))), [.2]).sum(1), 1.)

    def test_gate_accounting_uses_all_query_denominator(self):
        pre = np.array([.5, -.25, .1, 0.])
        keep = np.array([False, False, True, False])
        post = np.where(keep, pre, 0.)
        missed = np.where(~keep, np.maximum(pre, 0.), 0.)
        avoided = np.where(~keep, np.maximum(-pre, 0.), 0.)
        self.assertAlmostEqual(post.mean(), pre.mean() - missed.mean() + avoided.mean())


if __name__ == "__main__":
    unittest.main()
