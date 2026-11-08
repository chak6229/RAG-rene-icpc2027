"""Additional independent input-chain and precision check outside frozen science package."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import nct,t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rene_factor_confirmation_20261020'


def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**23),b''):h.update(b)
    return h.hexdigest()


def records(p):
    with p.open(encoding='utf-8') as f:
        for l in f:
            if l.strip():yield json.loads(l)


def main():
    eligibility=load(OUT/'LOCAL_ELIGIBILITY.json');staqc=load(OUT/'STAQC_ELIGIBILITY.json')
    receipts=eligibility['references']+[eligibility['reference_index'],staqc['metadata'],staqc['eligible_pairs']]+staqc['source_files']
    receipts=list({r['path']:r for r in receipts}.values())
    for r in receipts:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('Eligibility chain input changed: '+r['path'])
    # Independent union-find over all accepted pair metadata, including nonprimary
    # multi-code data, to check that duplicate-title/code chains stay together.
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def join(x,y):
        x,y=find(x),find(y)
        if x!=y:parent[max(x,y)]=min(x,y)
    metadata=list(records(OUT/'STAQC_PAIR_METADATA.jsonl'))
    for r in metadata:
        anchor='soq:'+str(r['question_id']);find(anchor)
        for k in ('q_lex','code_ws','code_tokens'):
            for v in r['fingerprints'][k]:join(anchor,k+':'+v)
    for r in metadata:
        expected=hashlib.sha256(('staqc-connected/'+find('soq:'+str(r['question_id']))).encode()).hexdigest()[:24]
        if expected!=r['group_id']:raise ValueError('Independent connected component mismatch')
    with (OUT/'reference_fingerprints.pkl').open('rb') as h:ref=pickle.load(h)
    bad=set()
    for r in metadata:
        q=r['query'];malformed=len(q.split())<3 or '<extra_id_' in q
        hit=any(set(v)&ref[k] for k,v in r['fingerprints'].items())
        if malformed or hit:bad.add(r['group_id'])
    # Empty code is represented by empty whitespace/token fingerprint sets.
    for r in metadata:
        if not r['fingerprints']['code_ws'] or not r['fingerprints']['code_tokens'] or hashlib.sha256(b'').hexdigest() in r['fingerprints']['code_tokens']:
            bad.add(r['group_id'])
    for r in metadata:
        if r['eligible']!=(r['group_id'] not in bad):raise ValueError('Independent component eligibility mismatch')
    sample=load(OUT/'SAMPLE_PLAN.json');design=load(OUT/'DESIGN.json');plan=load(OUT/'PRECISION_PLAN.json')
    primaries=['ENTR_B_budget_mean_1_to_3','ENTR_G_aggregation_at3_mean_to_max']
    newplan=sample['exact_sample_precision'];g=sample['groups'];details={}
    for c in primaries:
        variances=[]
        for domain in ('source','cosqa'):
            r=plan['development'][domain][c]
            variances.extend(r[k] for k in ('upper95_bootstrap_variance','matched_one_query_per_group_variance_upper95','all_query_variance_upper95_sensitivity'))
        variance=max(variances)
        if abs(variance-newplan['contrasts'][c]['planning_variance'])>1e-12:raise ValueError('Planning variance mismatch')
        critical=float(t.ppf(.9875,g-1));noncentrality=.005*math.sqrt(g/variance)
        exact=float(nct.sf(critical,g-1,noncentrality)+nct.cdf(-critical,g-1,noncentrality))
        details[c]={'planning_variance':variance,'normal_approx_power':newplan['contrasts'][c]['planned_power_at_005'],
                    'noncentral_t_sensitivity_power':exact,'groups':g}
        if exact<.9:raise ValueError('Exact t power sensitivity below gate')
    smallest=max(r['required_groups_005'] for r in newplan['contrasts'].values())
    if g!=math.ceil((smallest/.9)/256)*256:raise ValueError('Fixed sample-size rule mismatch')
    if len(sample['query_group_ids'])!=g or len(set(sample['query_group_ids']))!=g:
        raise ValueError('Repeated sample component')
    output={'created_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
            'scope':'independent full eligibility input hash chain, all accepted source connected components, component exposure propagation and exact-t power sensitivity',
            'known_exposure_input_hashes_checked':len(receipts),'accepted_source_pairs_reconnected':len(metadata),
            'groups':g,'power_sensitivity':details,'new_method_effects_read':False,
            'primary_joint_power_lower_bound_exact_t_sensitivity':max(0.,sum(x['noncentral_t_sensitivity_power'] for x in details.values())-1),
            'design_sha256':sha(OUT/'DESIGN.json'),'source_script_sha256':sha(Path(__file__))}
    with (OUT/'SOURCE_PRECISION_REVIEW.json').open('x',encoding='utf-8') as h:json.dump(output,h,ensure_ascii=False,indent=2)
    print({k:v for k,v in output.items() if k!='power_sensitivity'},flush=True)


if __name__=='__main__':main()
