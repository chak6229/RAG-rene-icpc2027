"""Design-based sampling uncertainty for completed relevance identification bounds.

The estimand is the fixed H8530 frame mean of each reviewer's query-level
lower/upper RR contrast under the original eight-stratum SRSWOR design.
Primary envelopes use exact hypergeometric CDF limits. A stratified Neyman
variance estimator and Welch-Satterthwaite t approximation are secondary.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from scipy.stats import t

from experiments.rene_relevance_sampling_uncertainty_20261021.finite_frame import (
    outer_envelope, rr_support_ticks)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/rene_review_completion_20261021/query_bounds.jsonl"
OUT = ROOT / "results/rene_relevance_sampling_uncertainty_20261021"
ALPHA = 0.05
PRIMARY = {"policy": "RIDGE_verified", "cutoff": 3}


def f(value):
    return Fraction(value)


def estimate_endpoint(strata, side):
    """Return the stratified estimator and its design-based variance estimate."""
    estimate = 0.0
    variance = 0.0
    components = []
    details = []
    total_N = sum(v["N"] for v in strata.values())
    for name in sorted(strata):
        frame = strata[name]
        N, n = frame["N"], len(frame[side])
        if n != frame["n"] or not 1 <= n <= N:
            raise ValueError(f"Invalid stratum sample size for {name}")
        values = frame[side]
        mean = sum(values) / n
        s2 = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
        W = N / total_N
        component = W * W * (1 - n / N) * s2 / n
        estimate += W * mean
        variance += component
        components.append((component, n - 1))
        details.append({"stratum": name, "N_h": N, "n_h": n, "weight": W,
                        "sample_mean": mean, "sample_variance": s2,
                        "finite_population_correction": 1 - n / N,
                        "variance_component": component})
    denominator = sum(component * component / df for component, df in components if df > 0)
    df = variance * variance / denominator if denominator > 0 else None
    return {"estimate": estimate, "variance_estimate": variance,
            "standard_error": math.sqrt(variance), "welch_satterthwaite_df": df,
            "strata": details}


def main():
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite {OUT}")
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    groups = defaultdict(dict)
    for row in records:
        key = (row["reviewer"], row["threshold"], row["cutoff"], row["policy"])
        qid = row["query_id"]
        if qid in groups[key]:
            raise ValueError(f"Duplicate query in combination: {key} {qid}")
        low, high = map(f, row["expanded_delta_bounds_fraction"])
        groups[key][qid] = (row, float(low), float(high))
    outputs = []
    for key in sorted(groups):
        reviewer, threshold, cutoff, policy = key
        sample = groups[key]
        if len(sample) != 128:
            raise ValueError(f"Expected 128 sampled queries for {key}; got {len(sample)}")
        strata = defaultdict(lambda: {"l": [], "u": []})
        total_frame = 0
        for row, low, high in sample.values():
            name = row["stratum"]
            frame = strata[name]
            if "N" not in frame:
                frame.update(N=row["frame_n"], n=row["sample_n"])
                total_frame += row["frame_n"]
            elif frame["N"] != row["frame_n"] or frame["n"] != row["sample_n"]:
                raise ValueError(f"Inconsistent frozen stratum metadata: {name}")
            frame["l"].append(low)
            frame["u"].append(high)
        if len(strata) != 8 or total_frame != 8530 or any(len(x["l"]) != 16 for x in strata.values()):
            raise ValueError(f"Frozen H8530/8x16 design mismatch for {key}")
        lo = estimate_endpoint(strata, "l")
        hi = estimate_endpoint(strata, "u")
        if lo["estimate"] > hi["estimate"] + 1e-12:
            raise ValueError(f"Identification bounds crossed for {key}")
        # Two one-sided endpoint statements, each at alpha/2, give a
        # Bonferroni 1-alpha confidence envelope for the population interval.
        tail_alpha = ALPHA / 2
        critical_lo = float(t.ppf(1 - tail_alpha, lo["welch_satterthwaite_df"])) if lo["welch_satterthwaite_df"] else None
        critical_hi = float(t.ppf(1 - tail_alpha, hi["welch_satterthwaite_df"])) if hi["welch_satterthwaite_df"] else None
        envelope = [
            lo["estimate"] - critical_lo * lo["standard_error"] if critical_lo is not None else lo["estimate"],
            hi["estimate"] + critical_hi * hi["standard_error"] if critical_hi is not None else hi["estimate"],
        ]
        exact = outer_envelope(strata, rr_support_ticks(cutoff), ALPHA)
        simultaneous = (outer_envelope(strata, rr_support_ticks(cutoff), ALPHA / 4)
                        if policy == PRIMARY["policy"] and cutoff == PRIMARY["cutoff"] else None)
        outputs.append({
            "reviewer": reviewer, "threshold": threshold, "cutoff": cutoff, "policy": policy,
            "sample_queries": 128, "sample_groups": len({x[0]["group_id"] for x in sample.values()}),
            "sample_identification_range": [lo["estimate"], hi["estimate"]],
            "lower_endpoint_sampling": lo, "upper_endpoint_sampling": hi,
            "exact_conservative_envelope": exact,
            "exact_four_combination_simultaneous95_envelope": simultaneous,
            "confidence_envelope": {"level": 0.95, "kind": "pointwise approximate design-based confidence envelope for the population identification interval",
                                    "method": "stratified Neyman variance with finite-population correction; Welch-Satterthwaite t critical values; Bonferroni over the two endpoint tails",
                                    "interval": envelope,
                                    "finite_population_sampling_only": True,
                                    "annotation_or_reviewer_error_included": False,
                                    "reviewer_and_threshold_multiplicity_adjusted": False},
        })
    OUT.mkdir(parents=True)
    result = {
        "analysis": "post-hoc design-based uncertainty added to the completed fixed-sample relevance sensitivity",
        "frame": "CodeBERT H8530 query population",
        "sampling_design": "original eight strata; 16 queries sampled without replacement within each stratum; query is sampling unit",
        "sample_identification_bounds_remain_distinct_from_sampling_confidence": True,
        "estimand": "finite-frame mean of query-level lower and upper paired RR contrasts under each reviewer-specific expanded-relevance rule",
        "primary": PRIMARY,
        "alpha": ALPHA,
        "method_and_limits": {
            "variance": "stratified Neyman estimator, sum_h (N_h/N)^2 (1-n_h/N_h) s_h^2/n_h",
            "interval": "Primary: one-sided exact hypergeometric CDF count limits with Bonferroni over both endpoints, all strata and complete fixed-support thresholds. Secondary: Neyman variance and Welch-Satterthwaite t approximation.",
            "exactness": "primary finite-sample coverage is conservative under the original within-stratum SRSWOR/pseudorandom-hash interpretation; t coverage is approximate and potentially unstable with sparse nonzero observations",
            "scope": "finite H8530 frame and this fixed post-hoc relevance rule; no generalization beyond the frame",
            "omissions": ["reviewer/annotation error", "new policy or retrieval effect", "simultaneous coverage across all 48 combinations"],
            "four_combination_family": "Additional simultaneous95 bounds for RIDGE_verified RR@3, A/B by relevance thresholds 1/2; each combination uses alpha=.05/4",
            "reviewers_are_fixed": True,
            "unknown_relevance_retained_as_query_level_bounds": True,
        },
        "combinations": outputs,
    }
    (OUT / "RESULTS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    selected = [x for x in outputs if x["policy"] == PRIMARY["policy"] and x["cutoff"] == PRIMARY["cutoff"]]
    lines = ['# Sensitivity of manual relevance: sample uncertainty supplements', "",
             'Follows the original eight-storey, each 16 query without returning design and original weight. The main method corresponds to RR full-segregation support over a layer of CDF array inverse hypergeometric distribution at Bonferroni simultaneously covering both ends, eight layers and all thresholds, and synthesizing the stratification average. This is a conservative and limited sample 95% package under the design explanation; A separate column with a limited overall correction of Neyman variance/t approximates as a precision reference and is not considered to be accurately covered. Neither contains evaluation error of judgement.', "",
             'The main policy is RIDGE_verified, RR@3. Each of the following combinations covers two overall endpoints between H8530 identification areas; A/B and thresholds are reported separately. And while RESULTS.json provides these four combinations, 95% The envelope, not all of it. 48 Grouping at the same time 95%. This is a diagnosis of the old results, not confirmation of new effects.', "",
             "Thrust threshold and reviewer's sample recognition domain is conservative and limited 95% Conservative layer t Approximation 95%", "|---:|---|---|---|---|"]
    for row in selected:
        lo, hi = row["sample_identification_range"]
        ci = row["exact_conservative_envelope"]["interval"]
        approx = row["confidence_envelope"]["interval"]
        lines.append(f"|{row['threshold']}|{row['reviewer']}|[{lo:+.6f}, {hi:+.6f}]|[{ci[0]:+.6f}, {ci[1]:+.6f}]|[{approx[0]:+.6f}, {approx[1]:+.6f}]|")
    lines += ["", 'Samples are moving to the identification community and do not automatically form an overall positive utility conclusion. The two main reviewers, the precise conservative envelope of the two thresholds and t are close to zero; The overall positive direction has not been established and cannot be declared invalid or equivalent on the basis of zero. Original samples, strictly unweighted results, unknown and original qrels maintained.', "",
              'Full layer supergeometric CDF array, variance, freedom, limited overall correction and 48 See RESULTS.json; The four combinations of 95% and the main envelope also span zero. Enter as results/rene_review_completion_20261021/query_bounds.jsonl.']
    (OUT / "FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result_dir": str(OUT), "primary": [{"reviewer": x["reviewer"], "threshold": x["threshold"], "sample_identification_range": x["sample_identification_range"], "envelope": x["confidence_envelope"]["interval"], "se": [x["lower_endpoint_sampling"]["standard_error"], x["upper_endpoint_sampling"]["standard_error"]], "df": [x["lower_endpoint_sampling"]["welch_satterthwaite_df"], x["upper_endpoint_sampling"]["welch_satterthwaite_df"]]} for x in selected]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
