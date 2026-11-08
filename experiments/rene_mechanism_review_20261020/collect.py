from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from scipy.stats import hypergeom

from .common import (FIDELITY, FLAGS, OUT, PARENT, STRATA, now, put, put_rows, read,
                     require_frozen, rows, sha)


def strict_pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError('duplicate JSON key: ' + k)
        out[k] = v
    return out


def load_return(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=strict_pairs)


def rewrite_valid(r):
    if not isinstance(r, dict) or r.get('fidelity') not in FIDELITY:
        return False
    flags = r.get('flags')
    if not isinstance(flags, list) or not flags or len(flags) != len(set(flags)) or any(f not in FLAGS for f in flags):
        return False
    if ('none' in flags or 'uncertain' in flags) and len(flags) != 1:
        return False
    if r['fidelity'] == 'preserved' and flags != ['none']:
        return False
    if r['fidelity'] == 'uncertain' and flags != ['uncertain']:
        return False
    if r['fidelity'] == 'drift' and any(f in flags for f in ('none', 'uncertain')):
        return False
    if r['fidelity'] == 'malformed' and 'incoherent' not in flags:
        return False
    return isinstance(r.get('note'), str) and len(r['note'].strip()) >= 6


def validate_result(p, key, reviewer):
    expected = {r['case_id']: r for r in key['cases']}
    for item, phase in ((p, 'code'), (p.get('semantic', {}), 'semantic')):
        if not isinstance(item, dict) or item.get('schema_version') != 1 or item.get('phase') != phase:
            raise ValueError('wrong response schema/phase')
        if item.get('packet_id') != key['packet_id'] or item.get('reviewer') != reviewer:
            raise ValueError('wrong reviewer/packet')
        if item.get('complete') is not True or item.get('independence_confirmed') is not True:
            raise ValueError('incomplete or missing independence confirmation')
        if set(item.get('answers', {})) != set(expected):
            raise ValueError('missing/unexpected/duplicate case')
        for field in ('started_utc', 'exported_utc'):
            datetime.fromisoformat(item[field].replace('Z', '+00:00'))
    if datetime.fromisoformat(p['semantic']['exported_utc'].replace('Z', '+00:00')) > datetime.fromisoformat(p['exported_utc'].replace('Z', '+00:00')):
        raise ValueError('semantic export occurs after code export')
    result = {}
    for case_id, case in expected.items():
        a, b = p['semantic']['answers'][case_id], p['answers'][case_id]
        if a.get('clarity') not in ('clear', 'ambiguous', 'insufficient') or set(a.get('rewrites', {})) != set(case['rewrite_map']):
            raise ValueError('invalid semantic item set')
        if not all(rewrite_valid(r) for r in a['rewrites'].values()):
            raise ValueError('invalid or inconsistent semantic labels/reasons')
        if set(b.get('documents', {})) != set(case['document_map']):
            raise ValueError('invalid code item set')
        for d in b['documents'].values():
            if d.get('relevance') not in ('0', '1', '2', 'u') or not isinstance(d.get('note'), str) or len(d['note'].strip()) < 6:
                raise ValueError('invalid code label/reason')
        rewrites = {pos: a['rewrites'][rid] for rid, r in case['rewrite_map'].items() for pos in r['positions']}
        docs = {doc: b['documents'][did] for did, ds in case['document_map'].items() for doc in ds}
        result[case['query_id']] = {'clarity': a['clarity'], 'rewrites': rewrites, 'documents': docs}
    return result


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--a', required=True); parser.add_argument('--b', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    require_frozen()
    key = read(OUT / 'private/reviewer_keys.json')
    a = validate_result(load_return(args.a), key['A'], 'A')
    b = validate_result(load_return(args.b), key['B'], 'B')
    if set(a) != set(b):
        raise ValueError('reviewer query frames differ')
    sample = list(rows(OUT / 'private/sample_manifest.jsonl'))
    cached = {r['query_id']: r for r in rows(OUT / 'private/CodeBERT_stage_rows.jsonl')}
    sample_keys = {r['query_id']: r for r in key['A']['cases']}
    sem_a, sem_b, rel_a, rel_b, ord_a, ord_b = [], [], [], [], [], []
    disagreements, query_flags = [], {}
    for q in sorted(a):
        aa, bb, c, case = a[q], b[q], cached[q], sample_keys[q]
        f = {'any_definite_semantic_issue': False, 'any_possible_semantic_issue': False,
             'any_uncertain_semantics': False, 'verified_action_definite_drift': False,
             'verified_harm_with_preserved_meaning': False, 'missed_preserved_positive_opportunity': False,
             'displayed_alternative_relevant': False, 'known_target_both_incomplete': False,
             'rater_A_any_semantic_issue': False, 'rater_B_any_semantic_issue': False}
        for item in case['rewrite_map'].values():
            pos = item['positions'][0]
            x, y = aa['rewrites'][pos]['fidelity'], bb['rewrites'][pos]['fidelity']
            sem_a.append(x); sem_b.append(y)
            issue = x in ('drift','malformed') and y in ('drift','malformed')
            f['any_definite_semantic_issue'] |= issue
            f['any_possible_semantic_issue'] |= x != 'preserved' or y != 'preserved'
            f['any_uncertain_semantics'] |= x == 'uncertain' or y == 'uncertain'
            f['rater_A_any_semantic_issue'] |= x in ('drift','malformed')
            f['rater_B_any_semantic_issue'] |= y in ('drift','malformed')
            if x != y or aa['rewrites'][pos]['flags'] != bb['rewrites'][pos]['flags']:
                disagreements.append({'query_id': q,'kind':'rewrite','positions':item['positions'],'text':item['text'],
                                      'A':aa['rewrites'][pos], 'B':bb['rewrites'][pos], 'resolution':None})
            if c['policies']['RIDGE_verified']['position'] in item['positions']:
                f['verified_action_definite_drift'] = issue
                f['verified_harm_with_preserved_meaning'] = c['policies']['RIDGE_verified']['delta'] < 0 and x == y == 'preserved'
            oracle = c['safe_ENTR_oracle_delta']
            if oracle > 0 and c['candidate_deltas'][str(pos)] == oracle and c['policies']['RIDGE_verified']['delta'] < oracle and x == y == 'preserved':
                f['missed_preserved_positive_opportunity'] = True
        for docs in case['document_map'].values():
            d = docs[0]
            x, y = aa['documents'][d]['relevance'], bb['documents'][d]['relevance']
            rel_a.append(x); rel_b.append(y)
            if x != 'u' and y != 'u':
                ord_a.append(x); ord_b.append(y)
            if not set(docs) & set(case['target_doc_ids']) and x == y == '2':
                f['displayed_alternative_relevant'] = True
            if set(docs) & set(case['target_doc_ids']) and x in ('0','1') and y in ('0','1'):
                f['known_target_both_incomplete'] = True
            if x != y:
                disagreements.append({'query_id':q,'kind':'code','doc_ids':docs,'A':aa['documents'][d],
                                      'B':bb['documents'][d],'resolution':None})
        query_flags[q] = f
    rates = {name: estimate({q: flags[name] for q,flags in query_flags.items()}, sample) for name in next(iter(query_flags.values()))}
    result = {'created_utc':now(), 'role':'posthoc_blinded_human_diagnostic', 'reviews_received':2,
              'input_files':[{'reviewer':'A','sha256':sha(args.a)},{'reviewer':'B','sha256':sha(args.b)}],
              'agreement':{'semantics':kappa(sem_a,sem_b),'relevance_including_unknown':kappa(rel_a,rel_b),
                           'relevance_ordinal_judged_subset':kappa(ord_a,ord_b,True)},
              'weighted_query_rates':rates,'unresolved_disagreements':len(disagreements),
              'causal_explanation_percentage_identified':False, 'risk_certified':False}
    dest=Path(args.out).resolve()
    if dest.exists():
        raise ValueError('use a new return-analysis directory; original judgments are immutable')
    dest.mkdir(parents=True)
    put(dest/'HUMAN_RESULTS.json',result)
    put_rows(dest/'DISAGREEMENTS.jsonl',disagreements)
    put_rows(dest/'query_flags.jsonl',({'query_id':q,**v} for q,v in query_flags.items()))
    put(dest/'reviewer_A_original.json',load_return(args.a));put(dest/'reviewer_B_original.json',load_return(args.b))
    lines=['# 双人独立审阅回收结果','','本表为预定分层抽样在既定8530条有限总体内的加权诊断。错误类型不是因果解释比例。',
           '', '|指标|总体比例估计|保守95%区间|','|---|---:|---|']
    for name,v in rates.items():
        l,u=v['ci95_conservative_finite_frame'];lines.append(f"|{name}|{v['estimate']:.2%}|[{l:.2%}, {u:.2%}]|")
    lines += ['',f'待讨论分歧{len(disagreements)}项；独立原判已保存。kappa及可判子集覆盖见HUMAN_RESULTS.json。',
              '相关性只覆盖展示代码；语义保持下的RR下降不能直接断言模型相关性能力不足，需结合未标注相关代码线索。']
    (dest/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(dest),'reviews':2,'queries':len(a),'disagreements':len(disagreements)},ensure_ascii=False))


if __name__ == '__main__':
    main()
