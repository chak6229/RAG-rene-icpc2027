"""Original statistical functions, extracted without changing their formulas."""
from collections import Counter, defaultdict
from fractions import Fraction
import math
from scipy.stats import hypergeom, t
from finite_frame import outer_envelope, rr_support_ticks

STRATA = ('gate_reject_positive', 'gate_reject_negative', 'final_positive', 'final_negative', 'gate_reject_zero', 'remaining_opportunity', 'remaining_neutral_changed', 'remaining_no_opportunity')

def kappa(a, b, ordinal=False):
    if not a or len(a) != len(b):
        return {'n': len(a), 'agreement': None, 'kappa': None}
    levels = sorted(set(a) | set(b))
    ca, cb, n = Counter(a), Counter(b), len(a)
    weight = (lambda x, y: abs(int(x)-int(y))/2) if ordinal else (lambda x, y: float(x != y))
    observed = sum(weight(x, y) for x, y in zip(a, b))/n
    expected = sum(ca[x]*cb[y]*weight(x, y) for x in levels for y in levels)/(n*n)
    return {'n': n, 'agreement': sum(x == y for x, y in zip(a, b))/n,
            'kappa': 1-observed/expected if expected > 0 else None,
            'type': 'linear weighted' if ordinal else 'nominal'}

def finite_interval(N, n, k, alpha):
    if not 0 <= k <= n <= N or n == 0:
        raise ValueError('invalid finite sample counts')
    if n == N:
        return k/N, k/N
    if k == 0:
        lower = 0
    else:
        lo, hi = k, N-n+k
        while lo < hi:
            mid = (lo+hi)//2
            if hypergeom.sf(k-1, N, mid, n) >= alpha/2:
                hi = mid
            else:
                lo = mid+1
        lower = lo
    if k == n:
        upper = N
    else:
        lo, hi = k, N-n+k
        while lo < hi:
            mid = (lo+hi+1)//2
            if hypergeom.cdf(k, N, mid, n) >= alpha/2:
                lo = mid
            else:
                hi = mid-1
        upper = lo
    return lower/N, upper/N

def estimate(flags, sample, population_n=8530):
    point = low = high = 0.
    strata_rows = []
    for stratum in STRATA:
        part = [r for r in sample if r['stratum'] == stratum]
        N, n = part[0]['frame_n'], len(part)
        k = sum(bool(flags[r['query_id']]) for r in part)
        l, u = finite_interval(N, n, k, .05/len(STRATA))
        point += N*k/n; low += N*l; high += N*u
        strata_rows.append({'stratum': stratum, 'N': N, 'n': n, 'positive': k, 'fraction': k/n, 'simultaneous_interval': [l,u]})
    return {'estimate': point/population_n, 'ci95_conservative_finite_frame': [low/population_n, min(1.,high/population_n)],
            'estimated_frame_count': point, 'frame_n': population_n, 'strata': strata_rows,
            'inference': 'Bonferroni 8-stratum exact hypergeometric bounds; pseudorandom SRS interpretation, finite frame only'}
def rr_at(ranking, relevant, cutoff):
    return next((Fraction(1, i) for i, doc in enumerate(ranking[:cutoff], 1) if doc in relevant), Fraction(0))

def query_bounds(query, reviewer, threshold, cutoff, supplement=None):
    target = set(query['target_doc_ids'])
    panel_by_id = {p['id']: p for p in query['panels']}
    doc_to_panel = {d: p['id'] for p in panel_by_id.values() for d in p['doc_ids']}
    lists = {k: v[:cutoff] for k, v in query['rankings'].items()}
    active = {doc_to_panel[d] for ranking in lists.values() for d in ranking}
    positive, unknown = set(target), []
    unresolved = Counter()
    for panel_id in active:
        p = panel_by_id[panel_id]
        if target & set(p['doc_ids']):
            positive.update(p['doc_ids'])
            continue
        label = p['prior_labels'][reviewer] if p['prior_labels'] is not None else (supplement or {}).get((query['query_id'], panel_id))
        if label is None:
            unknown.append(p['doc_ids'])
            unresolved['missing_panels'] += 1
        elif label['relevance'] == 'u':
            unknown.append(p['doc_ids'])
            unresolved['unknown_panels'] += 1
        elif int(label['relevance']) >= threshold:
            positive.update(p['doc_ids'])
    base = {k: rr_at(ranking, target, cutoff) for k, ranking in lists.items()}
    baseline = {policy: base[action]-base['original'] for policy, action in query['actions'].items()}
    baseline['SAFE_ENTR_best3_opportunity'] = max(base.values())-base['original']
    values = defaultdict(list)
    for mask in range(1 << len(unknown)):
        relevant = positive.copy()
        for i, docs in enumerate(unknown):
            if mask & (1 << i):
                relevant.update(docs)
        current = {k: rr_at(ranking, relevant, cutoff) for k, ranking in lists.items()}
        for policy, action in query['actions'].items():
            values[policy].append(current[action]-current['original'])
        values['SAFE_ENTR_best3_opportunity'].append(max(current.values())-current['original'])
    result = []
    for policy, outcomes in values.items():
        low, high = min(outcomes), max(outcomes)
        original = baseline[policy]
        result.append({'query_id': query['query_id'], 'group_id': query['group_id'], 'stratum': query['stratum'],
                       'frame_n': query['frame_n'], 'sample_n': query['sample_n'], 'weight': query['weight'],
                       'reviewer': reviewer, 'threshold': threshold, 'cutoff': cutoff, 'policy': policy,
                       'original_qrels_delta_fraction': str(original),
                       'expanded_delta_bounds_fraction': [str(low), str(high)],
                       'change_from_original_delta_bounds_fraction': [str(low-original), str(high-original)],
                       'expanded_delta_point_identified': low == high,
                       'unknown_text_variables': len(unknown), 'assignments_enumerated': 1 << len(unknown),
                       'missing_panels': unresolved['missing_panels'], 'unknown_panels': unresolved['unknown_panels']})
    return result

def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[row['reviewer'], row['threshold'], row['cutoff'], row['policy']].append(row)
    result = []
    for (reviewer, threshold, cutoff, policy), items in groups.items():
        if len(items) != 128:
            raise ValueError('All queries must be retained')
        weights = [Fraction(r['frame_n'], r['sample_n']) for r in items]
        if sum(weights) != 8530:
            raise ValueError('Stratum weights must sum to H8530')
        def aggregate(vals, ws):
            return float(sum(v*w for v, w in zip(vals, ws))/sum(ws))
        baseline = [Fraction(r['original_qrels_delta_fraction']) for r in items]
        bounds = [[Fraction(v) for v in r['expanded_delta_bounds_fraction']] for r in items]
        increments = [[Fraction(v) for v in r['change_from_original_delta_bounds_fraction']] for r in items]
        result.append({'reviewer': reviewer, 'threshold': threshold, 'cutoff': cutoff, 'policy': policy,
                       'queries': len(items), 'sample_groups': len({r['group_id'] for r in items}),
                       'point_identified_queries': sum(r['expanded_delta_point_identified'] for r in items),
                       'weighted_original_qrels_delta': aggregate(baseline, weights),
                       'weighted_expanded_delta_identification_bounds': [aggregate([b[i] for b in bounds], weights) for i in (0, 1)],
                       'weighted_change_from_original_delta_bounds': [aggregate([b[i] for b in increments], weights) for i in (0, 1)],
                       'unweighted_original_qrels_delta': aggregate(baseline, [Fraction(1)]*len(items)),
                       'unweighted_expanded_delta_bounds': [aggregate([b[i] for b in bounds], [Fraction(1)]*len(items)) for i in (0, 1)],
                       'interval_type': 'sample identification range over shared unknown relevance, not a confidence interval'})
    return result
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
