"""Finite-sample conservative inference for stratified discrete RR endpoints."""

from functools import lru_cache

from scipy.stats import hypergeom


@lru_cache(maxsize=None)
def count_limit(N, n, k, tail_alpha, direction):
    """Invert one hypergeometric tail; K is the fixed population count."""
    if not (0 <= k <= n <= N and n > 0 and 0 < tail_alpha < 1):
        raise ValueError("Invalid hypergeometric design or alpha")
    if direction not in ("lower", "upper"):
        raise ValueError("Invalid limit direction")
    if n == N:
        return k
    left, right = k, N - n + k
    if direction == "lower":
        while left < right:
            mid = (left + right) // 2
            if hypergeom.sf(k - 1, N, mid, n) >= tail_alpha:
                right = mid
            else:
                left = mid + 1
    else:
        while left < right:
            mid = (left + right + 1) // 2
            if hypergeom.cdf(k, N, mid, n) >= tail_alpha:
                left = mid
            else:
                right = mid - 1
    return left


def rr_support_ticks(cutoff):
    """Six times the full support of differences of RR at this cutoff."""
    if cutoff == 1:
        return (-6, 0, 6)
    if cutoff == 3:
        return (-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6)
    raise ValueError("Only the frozen RR@1 and RR@3 endpoints are supported")


def outer_envelope(strata, support_ticks, alpha=.05):
    """Cover both population identification endpoints simultaneously.

    For the lower outcome endpoint, an upper bound on each CDF gives a lower
    mean bound. For the upper outcome endpoint, lower CDF bounds give an upper
    mean bound. Bonferroni covers these two directions, all strata, and every
    threshold on the full known support, including unobserved values.
    """
    support = tuple(support_ticks)
    if len(support) < 2 or support != tuple(sorted(set(support))):
        raise ValueError("Support must be fixed, sorted and distinct")
    H, thresholds = len(strata), support[:-1]
    events = 2 * H * len(thresholds)
    tail_alpha = alpha / events
    total_N = sum(frame["N"] for frame in strata.values())
    aggregate = {"l": 0.0, "u": 0.0}
    details = []
    for name in sorted(strata):
        frame = strata[name]
        N, n = frame["N"], frame["n"]
        current = {"stratum": name, "N_h": N, "n_h": n}
        for side, count_direction in (("l", "upper"), ("u", "lower")):
            if len(frame[side]) != n:
                raise ValueError("Stratum sample size mismatch")
            ticks = [round(6 * value) for value in frame[side]]
            if any(abs(6 * value - tick) > 1e-10 or tick not in support
                   for value, tick in zip(frame[side], ticks)):
                raise ValueError("An observation falls outside the fixed support")
            limits = []
            mean_bound = support[0] / 6
            for lower_tick, upper_tick in zip(thresholds, support[1:]):
                k = sum(value <= lower_tick for value in ticks)
                K = count_limit(N, n, k, tail_alpha, count_direction)
                mean_bound += (upper_tick - lower_tick) / 6 * (1 - K / N)
                limits.append({"threshold_sixths": lower_tick, "sample_count_leq": k,
                               "population_count_limit": K,
                               "population_count_limit_direction": count_direction})
            aggregate[side] += N / total_N * mean_bound
            current[side] = {"population_mean_bound": mean_bound,
                             "bound_direction": "lower" if side == "l" else "upper",
                             "cdf_count_limits": limits}
        details.append(current)
    return {"level": 1 - alpha, "interval": [aggregate["l"], aggregate["u"]],
            "method": "one-sided exact hypergeometric CDF count limits, Bonferroni across both identification endpoints, strata and fixed support thresholds; discrete CDF mean identity",
            "simultaneous_tail_events": events, "one_sided_alpha": tail_alpha,
            "support_in_sixths": list(support), "strata": details,
            "sampling_assumption": "within-stratum simple random sampling without replacement under the original pseudorandom salted-hash interpretation",
            "annotation_error_included": False,
            "coverage_scope": "both finite-frame identification endpoints within this reviewer/threshold/cutoff/policy combination"}
