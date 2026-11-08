"""Separate numerical replay of finite-frame uncertainty results."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom, t

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rene_relevance_sampling_uncertainty_20261021"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    result = read(OUT / "RESULTS.json")
    source = ROOT / "results/rene_review_completion_20261021/query_bounds.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line]
    prior = read(ROOT / "results/rene_review_completion_20261021/RESULTS.json")
    groups = defaultdict(list)
    for row in rows:
        groups[row["reviewer"], row["threshold"], row["cutoff"], row["policy"]].append(row)
    checks = Counter()
    evaluated_limits = {}

    def require(value, category):
        if not value:
            raise AssertionError(category)
        checks[category] += 1

    for item in result["combinations"]:
        key = tuple(item[k] for k in ("reviewer", "threshold", "cutoff", "policy"))
        sample = groups[key]
        require(len(sample) == 128 and len({x["query_id"] for x in sample}) == 128, "complete_original_sample")
        original = next(x for x in prior["summary"] if tuple(x[k] for k in ("reviewer", "threshold", "cutoff", "policy")) == key)
        for a, b in zip(item["sample_identification_range"], original["weighted_expanded_delta_identification_bounds"]):
            require(abs(a - b) < 1e-14, "unchanged_sample_identification_bounds")
        by_stratum = defaultdict(list)
        for row in sample:
            by_stratum[row["stratum"]].append(row)
        require(len(by_stratum) == 8, "original_eight_strata")
        for side, endpoint_name in ((0, "lower_endpoint_sampling"), (1, "upper_endpoint_sampling")):
            weighted_mean, estimated_variance, denominator = 0., 0., 0.
            for name, part in sorted(by_stratum.items()):
                N, n = part[0]["frame_n"], len(part)
                exact_values = [Fraction(row["expanded_delta_bounds_fraction"][side]) for row in part]
                mean = sum(exact_values) / n
                s2 = sum((v - mean) ** 2 for v in exact_values) / (n - 1)
                component = (Fraction(N, 8530) ** 2) * (1 - Fraction(n, N)) * s2 / n
                weighted_mean += float(Fraction(N, 8530) * mean)
                estimated_variance += float(component)
                denominator += float(component) ** 2 / (n - 1)
            saved = item[endpoint_name]
            require(abs(weighted_mean - saved["estimate"]) < 1e-14, "independent_fraction_weighted_means")
            require(abs(estimated_variance - saved["variance_estimate"]) < 1e-14, "independent_fraction_neyman_variance")
            if denominator:
                df = estimated_variance ** 2 / denominator
                require(abs(df - saved["welch_satterthwaite_df"]) < 1e-9, "independent_satterthwaite_df")
                direction = -1 if side == 0 else 1
                bound = weighted_mean + direction * t.ppf(.975, df) * math.sqrt(estimated_variance)
                require(abs(bound - item["confidence_envelope"]["interval"][side]) < 1e-12, "independent_approximate_envelope")
        envelopes = [item["exact_conservative_envelope"]]
        if item["exact_four_combination_simultaneous95_envelope"]:
            envelopes.append(item["exact_four_combination_simultaneous95_envelope"])
        rr_values = (Fraction(0), Fraction(1)) if item["cutoff"] == 1 else (Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(1))
        expected_support = sorted({int(6 * (a - b)) for a in rr_values for b in rr_values})
        for envelope in envelopes:
            support = envelope["support_in_sixths"]
            require(support == expected_support, "full_analytical_rr_support")
            require(abs(envelope["one_sided_alpha"] * envelope["simultaneous_tail_events"] - (1 - envelope["level"])) < 1e-12,
                    "bonferroni_tail_allocation")
            total = [0., 0.]
            for detail in envelope["strata"]:
                N, n = detail["N_h"], detail["n_h"]
                part = by_stratum[detail["stratum"]]
                require(n == len(part) == 16 and N == part[0]["frame_n"], "frozen_stratum_counts")
                for side, name in ((0, "l"), (1, "u")):
                    mean_bound = support[0] / 6
                    for j, cdf in enumerate(detail[name]["cdf_count_limits"]):
                        threshold = Fraction(cdf["threshold_sixths"], 6)
                        k = sum(Fraction(row["expanded_delta_bounds_fraction"][side]) <= threshold for row in part)
                        require(k == cdf["sample_count_leq"], "independent_exact_threshold_counts")
                        direction = cdf["population_count_limit_direction"]
                        alpha = envelope["one_sided_alpha"]
                        limit_key = N, n, k, alpha, direction
                        if limit_key not in evaluated_limits:
                            # Exhaustive vector inversion is separate from the production binary search.
                            possible = np.arange(N + 1)
                            probs = (hypergeom.sf(k - 1, N, possible, n) if direction == "lower"
                                     else hypergeom.cdf(k, N, possible, n))
                            accepted = possible[probs >= alpha]
                            evaluated_limits[limit_key] = int(accepted[0] if direction == "lower" else accepted[-1])
                        K = evaluated_limits[limit_key]
                        require(K == cdf["population_count_limit"], "independent_exhaustive_hypergeometric_inversion")
                        mean_bound += (support[j + 1] - support[j]) / 6 * (1 - K / N)
                    require(abs(mean_bound - detail[name]["population_mean_bound"]) < 1e-12, "independent_discrete_cdf_mean_bound")
                    total[side] += N / 8530 * mean_bound
            for side in (0, 1):
                require(abs(total[side] - envelope["interval"][side]) < 1e-12, "independent_stratified_exact_envelope")
        lo, hi = item["exact_conservative_envelope"]["interval"]
        require(lo <= item["sample_identification_range"][0] + 1e-12 and hi >= item["sample_identification_range"][1] - 1e-12,
                "outer_envelope_contains_sample_range")
    primary = [x for x in result["combinations"] if x["policy"] == "RIDGE_verified" and x["cutoff"] == 3]
    require(len(primary) == 4, "all_primary_reviewer_threshold_combinations")
    for item in primary:
        for kind in ("exact_conservative_envelope", "exact_four_combination_simultaneous95_envelope", "confidence_envelope"):
            lo, hi = item[kind]["interval"]
            require(lo < 0 < hi, "primary_population_direction_unresolved")
    report = {"passed": True, "checks": dict(checks), "total_checks": sum(checks.values()),
              "unique_exhaustively_inverted_hypergeometric_limits": len(evaluated_limits),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "results_sha256": hashlib.sha256((OUT / "RESULTS.json").read_bytes()).hexdigest(),
              "all_48_combinations_replayed": True,
              "independent_numerical_implementation_not_independent_human_review": True}
    (OUT / "NUMERICAL_VALIDATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
