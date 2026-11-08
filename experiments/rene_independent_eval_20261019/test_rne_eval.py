"""Minimal contract tests for rne_eval (no pipeline imports)."""

from rne_eval import cluster_bootstrap_ci, holm, mrr, paired_delta, rr_at_k, tost


def test_rr_at_k():
    relevant = {"d1": 1.0}
    assert rr_at_k(["d1"], relevant, 100) == 1.0
    assert rr_at_k(["x", "d1"], relevant, 100) == 0.5
    assert rr_at_k(["x", "d1"], relevant, 1) == 0.0
    assert rr_at_k([], relevant, 100) == 0.0


def test_mrr_and_delta():
    qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    a = {"q1": ["x", "d1"], "q2": ["d2"]}
    b = {"q1": ["d1"], "q2": ["x"]}
    assert abs(mrr(a, qrels, 100) - 0.75) < 1e-12
    assert abs(mrr(b, qrels, 100) - 0.5) < 1e-12
    delta = paired_delta(a, b, qrels, 100)
    assert abs(delta["q1"] + 0.5) < 1e-12
    assert abs(delta["q2"] - 1.0) < 1e-12


def test_cluster_bootstrap_and_tost():
    values = {"q1": 0.001, "q2": 0.001, "q3": 0.001, "q4": 0.001}
    groups = {"q1": "g1", "q2": "g1", "q3": "g2", "q4": "g2"}
    lo, hi = cluster_bootstrap_ci(values, groups, n_boot=2000, alpha=0.05, seed=7)
    assert lo <= 0.001 <= hi
    result = tost(values, groups, margin=0.005, n_boot=2000, seed=7)
    assert result["equivalent"] is True


def test_holm_monotone():
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


if __name__ == "__main__":
    test_rr_at_k()
    test_mrr_and_delta()
    test_cluster_bootstrap_and_tost()
    test_holm_monotone()
    print("all tests passed")
