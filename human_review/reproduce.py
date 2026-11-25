'Replay the Manual evaluation statistics from anonymous labels and numeric panels.'
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from scipy.stats import t
from analysis_functions import estimate, estimate_endpoint, kappa, query_bounds, summarize
from finite_frame import outer_envelope, rr_support_ticks

HERE = Path(__file__).resolve().parent
TOLERANCE = 1e-11

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def rows(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]

def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def check(actual, expected, label='root'):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual: raise ValueError(f'Missing {label}.{key}')
            check(actual[key], value, f'{label}.{key}')
    elif isinstance(expected, list):
        if len(actual)!=len(expected): raise ValueError(f'Length mismatch: {label}')
        for i,(a,b) in enumerate(zip(actual, expected)): check(a,b,f'{label}[{i}]')
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if not isinstance(actual,(float,int)) or not math.isfinite(actual) or abs(actual-expected)>TOLERANCE:
            raise ValueError(f'Numeric mismatch {label}: {actual!r}, {expected!r}')
    elif actual!=expected: raise ValueError(f'Mismatch {label}: {actual!r}, {expected!r}')

def pairs(records, kind, field):
    items=defaultdict(dict)
    for row in records:
        if row['item_type']==kind: items[row['item_ref']][row['reviewer']]=row['judgment'][field]
    if any(set(v)!={'A','B'} for v in items.values()): raise ValueError('Incomplete label pair')
    return [v['A'] for v in items.values()],[v['B'] for v in items.values()]

def relevance_agreement(a,b):
    valid=[(x,y) for x,y in zip(a,b) if x!='u' and y!='u']
    levels=['0','1','2','u']
    return {'including_unknown':kappa(a,b),
            'jointly_judged':kappa([x for x,y in valid],[y for x,y in valid],True),
            'jointly_judged_coverage':len(valid)/len(a),
            'unknown_by_reviewer':{'A':a.count('u'),'B':b.count('u')},
            'disagreements':sum(x!=y for x,y in zip(a,b)),
            'confusion_rows_A_columns_B':levels,
            'confusion':[[sum(x==u and y==v for x,y in zip(a,b)) for v in levels] for u in levels],
            'label_counts':{who:{v:values.count(v) for v in levels} for who,values in [('A',a),('B',b)]}}

def sampling_statistics(records):
    groups=defaultdict(list)
    for row in records: groups[row['reviewer'],row['threshold'],row['cutoff'],row['policy']].append(row)
    result=[]
    from fractions import Fraction
    for key in sorted(groups):
        reviewer,threshold,cutoff,policy=key
        items=groups[key]
        strata=defaultdict(lambda:{'l':[],'u':[]})
        for row in items:
            frame=strata[row['stratum']]
            frame.update(N=row['frame_n'],n=row['sample_n'])
            low,high=map(Fraction,row['expanded_delta_bounds_fraction'])
            frame['l'].append(float(low));frame['u'].append(float(high))
        if len(strata)!=8 or sum(x['N'] for x in strata.values())!=8530 or any(len(x['l'])!=16 for x in strata.values()):
            raise ValueError('Frozen eight-stratum design mismatch')
        low,high=estimate_endpoint(strata,'l'),estimate_endpoint(strata,'u')
        critical_low=float(t.ppf(.975,low['welch_satterthwaite_df'])) if low['welch_satterthwaite_df'] else None
        critical_high=float(t.ppf(.975,high['welch_satterthwaite_df'])) if high['welch_satterthwaite_df'] else None
        interval=[low['estimate']-critical_low*low['standard_error'] if critical_low is not None else low['estimate'],
                  high['estimate']+critical_high*high['standard_error'] if critical_high is not None else high['estimate']]
        result.append({'reviewer':reviewer,'threshold':threshold,'cutoff':cutoff,'policy':policy,
                       'sample_queries':128,'sample_groups':len({r['group_id'] for r in items}),
                       'sample_identification_range':[low['estimate'],high['estimate']],
                       'lower_endpoint_sampling':low,'upper_endpoint_sampling':high,
                       'exact_conservative_envelope':outer_envelope(strata,rr_support_ticks(cutoff),.05),
                       'exact_four_combination_simultaneous95_envelope':outer_envelope(strata,rr_support_ticks(cutoff),.05/4) if policy=='RIDGE_verified' and cutoff==3 else None,
                       'confidence_envelope':{'interval':interval}})
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE.parent/'reproduced/human_review')
    args=parser.parse_args()
    out=args.output.resolve()
    # Never write into delivered data/reference files.
    if HERE==out or HERE in out.parents: raise ValueError('Choose an output folder outside human_review/')
    original=rows(HERE/'data/original_annotations.jsonl')
    supplement=rows(HERE/'data/supplement_annotations.jsonl')
    sample=rows(HERE/'data/sampling_design.jsonl')
    flags=rows(HERE/'data/query_event_flags.jsonl')
    pools=rows(HERE/'data/relevance_pools.jsonl')
    if (len(original),len(supplement),len(sample))!=(1676,688,128): raise ValueError('Annotation/sample count mismatch')
    if {r['query_id'] for r in sample}!={r['query_id'] for r in pools} or {r['query_id'] for r in sample}!={r['query_id'] for r in flags}:
        raise ValueError('Inconsistent anonymous query references')
    sem=pairs(original,'rewrite_fidelity','fidelity')
    rel=pairs(original,'code_relevance','relevance')
    clarity=pairs(original,'query_clarity','clarity')
    judged=[(x,y) for x,y in zip(*rel) if x!='u' and y!='u']
    agreement={'semantics':kappa(*sem),'relevance_including_unknown':kappa(*rel),
               'relevance_ordinal_judged_subset':{**kappa([x for x,y in judged],[y for x,y in judged],True),'coverage_of_displayed_panels':len(judged)/len(rel[0])},
               'clarity':kappa(*clarity)}
    flagpairs=defaultdict(dict)
    for r in original:
        if r['item_type']=='rewrite_fidelity': flagpairs[r['item_ref']][r['reviewer']]=sorted(r['judgment']['flags'])
    agreement['semantic_flag_exact_agreement']=sum(v['A']==v['B'] for v in flagpairs.values())/len(flagpairs)
    event_names=[k for k in flags[0] if k not in ('query_id','types')]
    rates={name:estimate({r['query_id']:r[name] for r in flags},sample) for name in event_names}
    types={name:estimate({r['query_id']:r['types'][name] for r in flags},sample) for name in flags[0]['types']}
    initial={'agreement':agreement,'weighted_query_rates':rates,'weighted_query_issue_type_rates':types}
    check(initial,read(HERE/'reference/initial_statistics.json'),'initial')
    supp_maps={who:{(r['query_ref'],r['panel_ref']):r['judgment'] for r in supplement if r['reviewer']==who} for who in ('A','B')}
    supp_pairs=defaultdict(dict)
    for r in supplement: supp_pairs[r['panel_ref']][r['reviewer']]=r['judgment']['relevance']
    supplement_agreement=relevance_agreement([v['A'] for v in supp_pairs.values()],[v['B'] for v in supp_pairs.values()])
    pooled_a,pooled_b=[],[]
    for q in pools:
        for p in q['panels']:
            pair=p['prior_labels'] if p['prior_labels'] is not None else {who:supp_maps[who][q['query_id'],p['id']] for who in ('A','B')}
            pooled_a.append(pair['A']['relevance']);pooled_b.append(pair['B']['relevance'])
    pooled_agreement=relevance_agreement(pooled_a,pooled_b)
    bounds=[row for query in pools for reviewer in ('A','B') for threshold in (2,1) for cutoff in (1,3)
            for row in query_bounds(query,reviewer,threshold,cutoff,supp_maps[reviewer])]
    key=lambda r:(r['query_id'],r['reviewer'],r['threshold'],r['cutoff'],r['policy'])
    expected=rows(HERE/'data/weighted_query_bounds.jsonl')
    check(sorted(bounds,key=key),sorted(expected,key=key),'exact_fraction_bounds')
    relevance={'summary':summarize(bounds),'supplement_agreement':supplement_agreement,'pooled_top3_agreement':pooled_agreement}
    check(relevance,read(HERE/'reference/relevance_statistics.json'),'relevance')
    sampling=sampling_statistics(bounds)
    golden=read(HERE/'reference/sampling_uncertainty.json')['combinations']
    for actual,ref in zip(sampling,golden):
        # All numeric endpoints, stratum counts, CDF limits and variances are compared.
        check(actual,{k:({'interval':v['interval']} if k=='confidence_envelope' else v) for k,v in ref.items() if k in actual},'sampling')
    if len(sampling)!=len(golden) or len(sampling)!=48: raise ValueError('Sampling comparison count mismatch')
    write(out/'INITIAL_STATISTICS.json',initial)
    write(out/'RELEVANCE_STATISTICS.json',relevance)
    write(out/'SAMPLING_UNCERTAINTY.json',{'combinations':sampling})
    out.mkdir(parents=True,exist_ok=True)
    (out/'query_bounds.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in bounds),encoding='utf-8')
    acceptance={'passed':True,'source_type':'human review','queries':128,'groups':120,
                'initial_annotations':len(original),'supplement_annotations':len(supplement),
                'recomputed_exact_fraction_bounds':len(bounds),'recomputed_relevance_combinations':48,
                'recomputed_sampling_envelopes':len(sampling),'recomputed_weighted_events':len(rates),
                'recomputed_weighted_error_types':len(types),'statistics_tolerance':TOLERANCE,
                'annotation_repeated':False,'scope':'anonymous labels, numeric panels and derived event flags; fixed H8530 frame'}
    write(out/'ACCEPTANCE.json',acceptance)
    print(json.dumps(acceptance,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
