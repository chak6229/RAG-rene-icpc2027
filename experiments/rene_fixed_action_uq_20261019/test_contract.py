"""Meaningful ablation, leakage and decision-contract checks on synthetic data."""
from __future__ import annotations

import unittest

import numpy as np

from experiments.rene_fixed_action_uq_20261019 import model as m


def fixture(n=120):
    rng = np.random.default_rng(29)
    seed = rng.normal(size=(n, 12))
    common = rng.normal(size=(n, len(m.COMMON_NAMES)))
    native = np.full(n, .08)
    # The omitted dispersion carries a real independent synthetic target signal.
    delta = np.where(seed[:, 2] > .3, -.1, np.where(seed[:, 2] < -.3, .1, 0.))
    data = dict(common=common, seed=seed, native=native, q=list(range(n)))
    return data, dict(data, delta=delta, repo=[str(i) for i in range(n)])


class ContractTests(unittest.TestCase):
    def test_inference_refuses_outcome_and_repository_columns(self):
        data, training = fixture()
        model = m.fit(training, "common")
        for name in ("delta", "y", "repo", "labels", "target", "original_rr"):
            with self.assertRaises(ValueError):
                m.predict(model, dict(data, **{name: np.ones(len(data["q"]))}))

    def test_common_is_invariant_to_all_raw_seed_features(self):
        data, training = fixture()
        changed = dict(training, seed=-training["seed"] * 1000. + 17.)
        a, b = m.fit(training, "common"), m.fit(changed, "common")
        np.testing.assert_array_equal(a["beta"], b["beta"])
        np.testing.assert_array_equal(m.predict(a, data)["classes"],
                                      m.predict(b, dict(data, seed=changed["seed"]))["classes"])

    def test_support_refit_is_invariant_to_omitted_dispersion(self):
        data, training = fixture()
        seed = training["seed"].copy()
        seed[:, list(m.DISPERSION)] *= -1000.
        a = m.fit(training, "support")
        b = m.fit(dict(training, seed=seed), "support")
        np.testing.assert_allclose(a["beta"], b["beta"], atol=1e-12, rtol=0.)
        np.testing.assert_allclose(m.predict(a, data)["classes"],
                                   m.predict(b, dict(data, seed=seed))["classes"], atol=1e-12, rtol=0.)

    def test_full_refit_can_use_independent_dispersion(self):
        data, training = fixture()
        support, full = m.fit(training, "support"), m.fit(training, "full")
        p0, p1 = (m.predict(z, data)["classes"] for z in (support, full))
        labels = np.where(training["delta"] < 0, 0, np.where(training["delta"] > 0, 2, 1))
        loss = lambda p: -np.log(p[np.arange(len(labels)), labels]).mean()
        self.assertLess(loss(p1), loss(p0) - .1)
        self.assertGreater(np.linalg.norm(full["beta"][:, 1 + len(m.COMMON_NAMES) + 2]), .1)

    def test_fixed_adoption_ties_do_not_use_outcomes(self):
        q = [9, 1, 7, 3, 5]
        np.testing.assert_array_equal(m.adoption(q, np.full(5, .02), "fixed50"),
                                      [False, True, False, True, False])
        np.testing.assert_array_equal(m.adoption(q, [.075, .0750001, .1, .01, .0], "threshold075"),
                                      [True, False, False, True, True])
        self.assertEqual(int(m.adoption([], [], "fixed50").sum()), 0)
        with self.assertRaises(ValueError):
            m.adoption([1, 1], [.1, .1], "fixed50")

    def test_single_class_role_uses_finite_identical_prior(self):
        data, training = fixture(8)
        training["delta"] = np.zeros(8)
        ps = []
        for h in m.HEADS:
            model = m.fit(training, h)
            self.assertTrue(model["fallback"])
            ps.append(m.predict(model, data)["classes"])
        np.testing.assert_array_equal(ps[0], ps[1])
        np.testing.assert_array_equal(ps[0], ps[2])
        np.testing.assert_allclose(ps[0].sum(1), 1.)


if __name__ == "__main__":
    unittest.main()
