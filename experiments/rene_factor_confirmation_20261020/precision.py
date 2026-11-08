from __future__ import annotations

import collections
import math

import numpy as np
from scipy.stats import norm, t

from .common import ROOT, OUT, digest, now, read, record, rows, write

CONTRASTS = ('ENTR_B_budget_mean_1_to_3', 'ENTR_G_aggregation_at3_mean_to_max')


def grouped(values, groups):
    labels = sorted(set(groups))
    lookup = {g: i for i, g in enumerate(labels)}
    codes = np.array([lookup[g] for g in groups])
    sizes = np.bincount(codes, minlength=len(labels)).astype(float)
    sums = np.bincount(codes, weights=values, minlength=len(labels))
    if len(sizes) < 2 or not np.isfinite(sums).all():
        raise ValueError('At least two finite independent groups required')
    return sums, sizes


def variance(sums, sizes):
    g = len(sums)
    mu = sums.sum() / sizes.sum()
    return float(g * g / (g - 1) * np.square(sums - mu * sizes).sum() / sizes.sum()**2)


def factor(sizes):
    sizes = np.asarray(sizes, dtype=float)
    return float(len(sizes) * np.square(sizes).sum() / sizes.sum()**2)


def power(groups, sigma2, effect=.005, alpha=.025):
    if groups < 3 or sigma2 <= 0:
        return 0.
    critical = t.ppf(1 - alpha/2, groups - 1)
    ncp = effect * math.sqrt(groups / sigma2)
    return float(norm.sf(critical - ncp) + norm.cdf(-critical - ncp))


def required_groups(sigma2, effect=.005, target=.9):
    lo, hi = 3, 10_000_000
    while lo < hi:
        middle = (lo + hi) // 2
        if power(middle, sigma2, effect) >= target:
            hi = middle
        else:
            lo = middle + 1
    return lo


def development_variances():
    path = ROOT / 'results/rene_budget_aggregation_20261020/per_query.jsonl'
    rr = [r for r in rows(path) if r['retriever'] == 'CodeBERT']
    result = {}
    for domain in ('source', 'cosqa'):
        selected = [r for r in rr if r['domain'] == domain]
        result[domain] = {}
        for c in CONTRASTS:
            vals = np.array([r['contrasts'][c] for r in selected], float)
            groups = [r['group'] for r in selected]
            sums, sizes = grouped(vals, groups)
            rng = np.random.default_rng(202610201)
            boot = []
            for offset in range(0, 10000, 100):
                ix = rng.integers(0, len(sums), (100, len(sums)))
                ss, nn = sums[ix], sizes[ix]
                mm = ss.sum(1) / nn.sum(1)
                boot.extend((len(sums)**2 / (len(sums)-1) * np.square(ss-mm[:, None]*nn).sum(1) / nn.sum(1)**2).tolist())
            by_group = collections.defaultdict(list)
            for r in selected:
                by_group[r['group']].append(r)
            single = [min(v,key=lambda r:digest('factor-within-group/'+r['query_id'])) for _,v in sorted(by_group.items())]
            single_values = np.array([r['contrasts'][c] for r in single])
            rng_one = np.random.default_rng(202610203)
            single_boot = []
            query_boot = []
            for offset in range(0,10000,100):
                one_ix=rng_one.integers(0,len(single_values),(100,len(single_values)))
                single_boot.extend(np.var(single_values[one_ix],axis=1,ddof=1).tolist())
                q_ix=rng_one.integers(0,len(vals),(100,len(vals)))
                query_boot.extend(np.var(vals[q_ix],axis=1,ddof=1).tolist())
            changed = sum(any(abs(vals[i]) > 1e-12 for i, g in enumerate(groups) if g == label) for label in sorted(set(groups)))
            result[domain][c] = {'queries': len(vals), 'groups': len(sums), 'mean': float(vals.mean()),
                'influence_variance': variance(sums, sizes), 'upper95_bootstrap_variance': float(np.quantile(boot, .95)),
                'matched_one_query_per_group_variance':float(np.var(single_values,ddof=1)),
                'matched_one_query_per_group_variance_upper95':float(np.quantile(single_boot,.95)),
                'all_query_variance_descriptive':float(np.var(vals,ddof=1)),
                'all_query_variance_upper95_sensitivity':float(np.quantile(query_boot,.95)),
                'size_design_factor': factor(sizes), 'groups_with_nonzero_query': changed}
    return result, record(path)


def make_plan(development, future_sizes):
    n = len(future_sizes)
    if n < 3:
        return {'available_groups': n, 'eligible': False, 'reason': 'fewer_than_three_groups'}
    out = {}
    for c in CONTRASTS:
        vv = [max(development[d][c]['upper95_bootstrap_variance'],
                  development[d][c]['matched_one_query_per_group_variance_upper95'],
                  development[d][c]['all_query_variance_upper95_sensitivity'])
                  * max(1., factor(future_sizes)/development[d][c]['size_design_factor']) for d in development]
        sigma2 = max(vv)
        critical = float(t.ppf(1-.025/2, n-1))
        out[c] = {'planning_variance': sigma2, 'required_groups_005': required_groups(sigma2),
                  'required_groups_010': required_groups(sigma2, effect=.01),
                  'planned_power_at_005': power(n, sigma2), 'planned_power_at_010': power(n, sigma2, effect=.01),
                  'mde90': float((critical + norm.ppf(.9)) * math.sqrt(sigma2/n)),
                  'simultaneous_CI_halfwidth_planning': critical * math.sqrt(sigma2/n),
                  'development_effect_sensitivity_not_gate': {d: power(n, sigma2, effect=abs(development[d][c]['mean'])) for d in development}}
    powers = [r['planned_power_at_005'] for r in out.values()]
    return {'available_groups': n, 'queries': int(sum(future_sizes)), 'size_design_factor': factor(future_sizes),
            'contrasts': out, 'joint_power_lower_bound_planning': max(0., 1-sum(1-p for p in powers)),
            'eligible': all(p >= .9 for p in powers)}


def main():
    development, receipt = development_variances()
    local = read(OUT / 'LOCAL_ELIGIBILITY.json')
    plans = {'CSN_valid_test': make_plan(development, list(local['CSN']['group_sizes'].values()))}
    if (OUT / 'STAQC_ELIGIBILITY.json').exists():
        v = read(OUT / 'STAQC_ELIGIBILITY.json')
        plans['StaQC'] = make_plan(development, list(v['group_sizes'].values()))
        plans['StaQC_one_question_per_component'] = make_plan(development, [1] * v['groups'])
    name = 'PRECISION_PLAN.json' if 'StaQC' in plans else 'LOCAL_PRECISION_PLAN.json'
    write(OUT / name, {'created_utc': now(), 'development_source': receipt, 'development': development,
          'primary_family_size': 2, 'family_alpha_two_sided': .05, 'per_test_alpha': .025,
          'resolution': .005, 'marginal_power_target': .9, 'joint_power_target_lower_bound': .8,
          'method': 'maximum of group influence, matched one-query-per-group and all-query variance upper95 bootstraps across source/cosqa; future size-imbalance inflation; prospective correction before effects',
          'assumptions': 'representative independent connected groups, development-to-new-domain variance transport; normal power approximation with t critical; imbalance inflation is a sensitivity, not a universal variance bound',
          'new_outcomes_used': False, 'plans': plans})
    print({k: {'groups': v['available_groups'], 'eligible': v['eligible']} for k, v in plans.items()}, flush=True)


if __name__ == '__main__':
    main()
