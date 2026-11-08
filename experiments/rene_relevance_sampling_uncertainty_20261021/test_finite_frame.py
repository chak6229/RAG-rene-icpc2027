from itertools import combinations, product
from statistics import mean, variance

from experiments.rene_relevance_sampling_uncertainty_20261021.analyze import estimate_endpoint

from scipy.stats import hypergeom

from experiments.rene_relevance_sampling_uncertainty_20261021.finite_frame import (
    count_limit, outer_envelope)


def test_exact_count_inversion_contains_true_counts_at_nominal_coverage():
    alpha = .05
    for N in range(3, 9):
        for n in range(1, N):
            for true_K in range(N + 1):
                covered = 0.0
                for k in range(max(0, n - (N - true_K)), min(n, true_K) + 1):
                    low = count_limit(N, n, k, alpha / 2, "lower")
                    high = count_limit(N, n, k, alpha / 2, "upper")
                    if low <= true_K <= high:
                        covered += hypergeom.pmf(k, N, true_K, n)
                assert covered >= 1 - alpha - 1e-12


def test_stratified_outer_envelope_covers_all_tiny_fixed_populations():
    support = (-6, 0, 6)
    alpha = .2
    for pop_l in product(support, repeat=4):
        for pop_u in product(support, repeat=4):
            truth_l = sum(pop_l) / 4 / 6
            truth_u = sum(pop_u) / 4 / 6
            for chosen in combinations(range(4), 2):
                strata = {"only": {
                    "N": 4, "n": 2,
                    "l": [pop_l[i] / 6 for i in chosen],
                    "u": [pop_u[i] / 6 for i in chosen],
                }}
                lo, hi = outer_envelope(strata, support, alpha)["interval"]
                if not (lo <= truth_l + 1e-12 and hi >= truth_u - 1e-12):
                    raise AssertionError((pop_l, pop_u, chosen, (lo, hi), (truth_l, truth_u)))


def test_all_limits_are_monotone_and_bounded():
    for N in range(2, 30):
        for n in range(1, N + 1):
            for k in range(n + 1):
                lower = count_limit(N, n, k, .0125, "lower")
                upper = count_limit(N, n, k, .0125, "upper")
                assert 0 <= lower <= upper <= N


def test_stratified_design_estimator_and_variance_are_unbiased_by_enumeration():
    populations = {"a": (-1., 0., 0., 1.), "b": (-1., 0., 1.)}
    estimates, variance_estimates = [], []
    for a in combinations(populations["a"], 2):
        for b in combinations(populations["b"], 2):
            strata = {"a": {"N": 4, "n": 2, "l": list(a)},
                      "b": {"N": 3, "n": 2, "l": list(b)}}
            result = estimate_endpoint(strata, "l")
            estimates.append(result["estimate"])
            variance_estimates.append(result["variance_estimate"])
    true_mean = sum(sum(p) for p in populations.values()) / 7
    assert abs(mean(estimates) - true_mean) < 1e-12
    true_sampling_variance = sum((x - true_mean) ** 2 for x in estimates) / len(estimates)
    assert abs(mean(variance_estimates) - true_sampling_variance) < 1e-12


def test_census_and_zero_observation_behave_differently():
    census = {"a": {"N": 4, "n": 4, "l": [0., 0., 0., 1.], "u": [0., 0., 1., 1.]}}
    result = outer_envelope(census, (-6, 0, 6))
    assert max(abs(x - y) for x, y in zip(result["interval"], (.25, .5))) < 1e-12
    zeros = {"a": {"N": 100, "n": 16, "l": [0.] * 16, "u": [0.] * 16}}
    lo, hi = outer_envelope(zeros, (-6, 0, 6))["interval"]
    assert lo < 0 < hi


if __name__ == "__main__":
    tests = [v for k, v in globals().copy().items() if k.startswith("test_")]
    for test in tests:
        test()
        print(test.__name__ + ": PASS")
    print(f"{len(tests)} statistical contract tests passed")
