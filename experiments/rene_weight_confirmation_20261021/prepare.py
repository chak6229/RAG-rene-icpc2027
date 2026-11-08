"""Outcome-blind exposure audit, variance planning, and matched sample construction."""
from __future__ import annotations

import collections
import json
import math
import re
import subprocess

import numpy as np
from scipy.stats import norm as normal, t

from .common import OLD, OUT, PACKAGE, ROOT, PRIMARY, digest, now, read, record, rows, write, writel
from experiments.rene_factor_confirmation_20261020.common import lexical, norm


def audit_eligibility():
    if (OUT/'ELIGIBILITY.json').exists():
        return read(OUT/'ELIGIBILITY.json')
    old = read(OLD/'STAQC_ELIGIBILITY.json')
    inherited = [OLD/'STAQC_ELIGIBILITY.json', OLD/'LOCAL_ELIGIBILITY.json', ROOT/old['metadata']['path'], ROOT/old['eligible_pairs']['path']]
    for item in old['source_files'] + [old['metadata'], old['eligible_pairs']]:
        if record(ROOT/item['path']) != item:
            raise ValueError('Inherited source changed: '+item['path'])
    local = read(OLD/'LOCAL_ELIGIBILITY.json')
    if record(OLD/'LOCAL_ELIGIBILITY.json')['sha256'] != old['reference_eligibility_sha256']:
        raise ValueError('Inherited history mismatch')
    # Check every inherited training/exposure reference, not only its recorded index.
    for item in local['references'] + [local['reference_index']]:
        if record(ROOT/item['path']) != item:
            raise ValueError('Inherited exposure reference changed: '+item['path'])
    meta = {r['unit_id']:r for r in rows(ROOT/old['metadata']['path'])}
    consumed = set()
    history = []
    for f in sorted(OLD.rglob('SAMPLE_PLAN.json')):
        v = read(f)
        units = set(v['corpus_source_unit_ids'])
        consumed.update(meta[u]['group_id'] for u in units)
        history.append({**record(f),'role':'all prior corpus components, including archived pre-inference sample',
                        'corpus_units':len(units)})
    # Inventory local outcome-bearing inputs/caches, including archived partial generation.
    listed = subprocess.check_output(['rg','--files','results','-g','*.jsonl','-g','*queries.json'], cwd=ROOT, text=True, encoding='utf-8').splitlines()
    seen_qn, seen_ql, seen_code = set(), set(), set()
    found_units = set()
    for relative in sorted(listed):
        f = ROOT/relative
        if OUT in f.parents:
            continue
        name = f.name.casefold()
        if not any(s in name for s in ('quer','candidate','action','group','corpus','qrels')):
            continue
        if name in ('staqc_eligible_pairs.jsonl','staqc_pair_metadata.jsonl'):
            continue
        n, ids = 0, set()
        stream = iter(read(f)) if f.suffix=='.json' else rows(f)
        for r in stream:
            n += 1
            if not isinstance(r,dict):
                continue
            for key in ('query_id','unit_id'):
                value = r.get(key,'')
                if isinstance(value,str) and re.fullmatch(r'staqc-\d+-\d+',value):
                    ids.add(value)
            for key in ('query','rewrite','original_query','query_text'):
                value = r.get(key,'')
                if isinstance(value,str) and value.strip():
                    seen_qn.add(digest(norm(value)));seen_ql.add(digest(lexical(value)))
            value = r.get('code_text','')
            if isinstance(value,str) and value.strip():
                seen_code.add(digest(' '.join(value.split())))
        if ids - set(meta):
            raise ValueError('Unknown StaQC identities in local exposure cache')
        found_units |= ids
        history.append({**record(f),'role':'prior query/candidate/corpus identity and lexical exposure scan',
                        'rows':n,'StaQC_units':len(ids)})
    consumed.update(meta[u]['group_id'] for u in found_units)
    lexical_hits = collections.Counter()
    for r in meta.values():
        hits = {'q_norm':bool(set(r['fingerprints']['q_norm']) & seen_qn),
                'q_lex':bool(set(r['fingerprints']['q_lex']) & seen_ql),
                'code_ws_or_token_text':bool((set(r['fingerprints']['code_ws'])|set(r['fingerprints']['code_tokens'])) & seen_code)}
        for key,hit in hits.items():
            lexical_hits[key] += hit
        if any(hits.values()):
            consumed.add(r['group_id'])
    kept = []
    for r in rows(ROOT/old['eligible_pairs']['path']):
        if r['group_id'] not in consumed:
            kept.append(r)
    sizes = collections.Counter(r['group_id'] for r in kept)
    writel(OUT/'ELIGIBLE_PAIRS.jsonl',kept)
    write(OUT/'EXPOSURE_MANIFEST.json',{'created_utc':now(),'files':history,'consumed_group_ids':sorted(consumed),
          'lexical_scan_matches_all_source_pairs':dict(lexical_hits),'new_method_effects_read':False})
    report = {'created_utc':now(),'inherited_original_eligible_groups':old['groups'],
              'remaining_groups':len(sizes),'remaining_pairs':len(kept),
              'excluded_original_eligible_groups':old['groups']-len(sizes),
              'prior_StaQC_units_seen_in_queries_or_candidates':len(found_units),
              'one_query_per_component':True,'references':[record(f) for f in inherited],
              'additional_exposure_manifest':record(OUT/'EXPOSURE_MANIFEST.json'),
              'eligible_pairs':record(OUT/'ELIGIBLE_PAIRS.jsonl'),
              'scope':'recovered local exposure, normalized query/code and SO/duplicate-connected groups; foundation pretraining, semantic clones and unlogged historical exposure unknown',
              'new_outcomes_used':False}
    write(OUT/'ELIGIBILITY.json',report)
    print({k:report[k] for k in ('remaining_groups','remaining_pairs','excluded_original_eligible_groups')},flush=True)
    return report


def upper_variance(x,seed=202610216):
    x = np.asarray(x,dtype=np.float64)
    if len(x)<20 or not np.isfinite(x).all():
        raise ValueError('Insufficient variance-planning data')
    rng = np.random.default_rng(seed)
    boot=[]
    for start in range(0,10000,32):
        ix=rng.integers(0,len(x),(min(32,10000-start),len(x)))
        boot.extend(np.var(x[ix],axis=1,ddof=1).tolist())
    return {'queries':len(x),'variance':float(np.var(x,ddof=1)),
            'variance_upper95':float(np.quantile(boot,.95)), 'mean_descriptive_only':float(np.mean(x))}


def power(n,var,effect=.005):
    critical=t.ppf(1-.05/(2*6),n-1)
    z=effect*math.sqrt(n/var)
    return float(normal.sf(critical-z)+normal.cdf(-critical-z))


def halfwidth(n,var):
    return float(t.ppf(1-.05/(2*6),n-1)*math.sqrt(var/n))


def need(var,kind):
    lo,hi=3,10000000
    while lo<hi:
        n=(lo+hi)//2
        passed=power(n,var)>=.95 if kind=='direction' else halfwidth(n,var)<=.005
        if passed:hi=n
        else:lo=n+1
    return lo


def precision():
    if (OUT/'PRECISION_PLAN.json').exists():return read(OUT/'PRECISION_PLAN.json')
    source=ROOT/'results/rene_t5_weight_factors_20261020/per_query.jsonl'
    oldrows=list(rows(source))
    data={name:[] for name in PRIMARY}
    for domain in ('source','cosqa'):
        by={view:{r['query_id']:r for r in oldrows if r['domain']==domain and r['view']==view} for view in ('local','t5base')}
        if set(by['local']) != set(by['t5base']) or len(by['local'])!=512:raise ValueError('Old matched planning scope')
        for c in ('B','G'):
            a=np.array([by['local'][q]['contrasts']['ENTR_'+c] for q in by['local']])
            b=np.array([by['t5base'][q]['contrasts']['ENTR_'+c] for q in by['local']])
            for view,x in [('local',a),('t5base',b),('base_minus_local',b-a)]:
                data[f'{view}::ENTR_{c}'].append({'source':domain+'_matched512',**upper_variance(x)})
    # Existing larger local results provide conservative sensitivity against the small paired cache.
    receipts=[record(source)]
    for path in [ROOT/'results/rene_budget_aggregation_20261020/per_query.jsonl', OLD/'per_query.jsonl']:
        receipts.append(record(path))
        rr=[r for r in rows(path) if r.get('retriever')=='CodeBERT']
        for domain in sorted({r.get('domain','staqc') for r in rr}):
            subset=[r for r in rr if r.get('domain','staqc')==domain]
            by_group=collections.defaultdict(list)
            for r in subset:by_group[r['group']].append(r)
            single=[min(v,key=lambda r:digest('weight-plan-within/'+r['query_id'])) for v in by_group.values()]
            for c,key in [('B','ENTR_B_budget_mean_1_to_3'),('G','ENTR_G_aggregation_at3_mean_to_max')]:
                if not subset or key not in subset[0]['contrasts']:continue
                # Both query-marginal and one-per-group distributions; new design is exactly one/group.
                for label,chosen in [('marginal',subset),('one_per_group',single)]:
                    data[f'local::ENTR_{c}'].append({'source':path.parent.name+'/'+domain+'/'+label,
                        **upper_variance([r['contrasts'][key] for r in chosen])})
    plan={}
    for name,v in data.items():
        var=max(r['variance_upper95'] for r in v)
        kind='difference_precision' if name.startswith('base_minus_local') else 'direction'
        plan[name]={'planning_variance':var,'role':kind,'required_groups':need(var,kind),'development':v}
    minimum=max(v['required_groups'] for v in plan.values())
    count=math.ceil((minimum/.9)/256)*256
    for item in plan.values():
        var=item['planning_variance']
        item.update(power_at_005=power(count,var),power_at_010=power(count,var,.010),
                    simultaneous_CI_halfwidth=halfwidth(count,var),
                    mde95=float((t.ppf(1-.05/12,count-1)+normal.ppf(.95))*math.sqrt(var/count)))
    eligible=read(OUT/'ELIGIBILITY.json')
    corpus=2**math.ceil(math.log2(count*1.1))
    passed=count<=eligible['remaining_groups'] and corpus<=eligible['remaining_groups']
    p={'created_utc':now(),'family_size':6,'alpha_family_two_sided':.05,'per_test_alpha':.05/6,
       'resolution_RR':.005,'direction_marginal_power_target':.95,
       'four_directions_joint_power_lower_bound':max(0.,1-sum(1-v['power_at_005'] for k,v in plan.items() if not k.startswith('base_minus_local'))),
       'direct_weight_difference_precision_target':'simultaneous CI halfwidth <=.005, not an equivalence margin',
       'minimum_groups':minimum,'planned_queries':count,'planned_corpus':corpus,'available_groups':eligible['remaining_groups'],
       'passed':passed,'contrasts':plan,'development_files':receipts,
       'method':'max bootstrap upper95 variance from consumed matched domains; extra local large-sample marginal and one/group sensitivity; normal power approximation with t critical, no finite-population correction',
       'assumptions':'independent eligible components and transport of planning variance; no power guarantee under distribution shift',
       'new_outcomes_used':False}
    write(OUT/'PRECISION_PLAN.json',p)
    print({k:p[k] for k in ('planned_queries','planned_corpus','available_groups','passed','four_directions_joint_power_lower_bound')},flush=True)
    return p


def sample():
    plan=read(OUT/'PRECISION_PLAN.json')
    if not plan['passed']:raise ValueError('Prospective capacity/precision gate failed')
    by=collections.defaultdict(list)
    for r in rows(OUT/'ELIGIBLE_PAIRS.jsonl'):by[r['group_id']].append(r)
    representatives={g:min(v,key=lambda r:digest('weight-confirm-within/'+r['unit_id'])) for g,v in by.items()}
    groups=sorted(by,key=lambda g:digest('weight-confirm-query/'+g))
    selected=groups[:plan['planned_queries']];used=set(selected)
    extras=sorted((g for g in by if g not in used),key=lambda g:digest('weight-confirm-corpus/'+g))[:plan['planned_corpus']-len(selected)]
    corpus_pairs=sorted([representatives[g] for g in selected+extras],key=lambda r:digest('weight-confirm-doc/'+r['unit_id']))
    doc={r['unit_id']:i for i,r in enumerate(corpus_pairs)}
    chosen=[representatives[g] for g in selected]
    writel(OUT/'public/queries.jsonl',[{'query_id':r['unit_id'],'query':r['query']} for r in chosen])
    writel(OUT/'public/corpus.jsonl',[{'doc_id':i,'code_text':r['code_text']} for i,r in enumerate(corpus_pairs)])
    writel(OUT/'public/groups.jsonl',[{'query_id':r['unit_id'],'group_id':r['group_id'],'question_id':r['question_id']} for r in chosen])
    writel(OUT/'private/qrels.jsonl',[{'query_id':r['unit_id'],'relevant_doc_ids':[doc[r['unit_id']]]} for r in chosen])
    write(OUT/'SAMPLE_PLAN.json',{'created_utc':now(),'queries':len(chosen),'groups':len(selected),'corpus_documents':len(corpus_pairs),
        'query_group_ids':selected,'corpus_source_unit_ids':[r['unit_id'] for r in corpus_pairs],
        'query_source_unit_ids':[r['unit_id'] for r in chosen],
        'sampling':'SHA256 outcome-blind group selection, one question per connected group',
        'known_positive':'one author-accepted snippet per single-code question; other snippets unjudged, not proven irrelevant',
        'estimand':'uniform sampled eligible connected groups, one hash-selected single-code question per group, fixed new corpus; both weights paired',
        'lexical_fallback_documents':sum(r['lexical_fallback'] for r in corpus_pairs),'new_outcomes_used':False})


def main():
    audit_eligibility();precision()
    if not (OUT/'SAMPLE_PLAN.json').exists():sample()


if __name__=='__main__':main()
