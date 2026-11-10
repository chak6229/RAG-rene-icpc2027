"""Freeze, seal label-free position prefixes, then replay cached RR."""
from __future__ import annotations

import argparse
import builtins
import contextlib
import hashlib
import io
import json
import math
import platform
import time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

from .metrics import AGGREGATIONS,BUDGETS,POLICIES,PRIMARY_NAMES,clustered_summary,contrasts,factorial_values,ordered_positions

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(__file__).resolve().parent
OUT=ROOT/'results/rene_budget_aggregation_20261020'
BRIDGE=ROOT/'results/rene_ssqr_bridge_20261019'
A=ROOT/'results/rene_core_step_a_20261020'
PROTECTED = []  # Unrelated author-workspace registries are not part of this release.


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**23),b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    return dict(path=Path(path).relative_to(ROOT).as_posix(),sha256=sha(path))


def write(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('x',encoding='utf-8') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')


def writel(name,records):
    with (OUT/name).open('x',encoding='utf-8') as handle:
        for row in records:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


@contextlib.contextmanager
def deny_outcomes():
    original_builtin,original_io,original_path=builtins.open,io.open,Path.open

    def guard(file,mode='r'):
        if isinstance(file,(str,Path)) and ('r' in mode or '+' in mode):
            path=Path(file)
            if 'private' in path.parts or path.name in ('per_query.jsonl','RESULTS.json','PRIMARY_RESULT.json'):
                raise PermissionError('Outcome read blocked during prefix construction')

    def guarded_builtin(file,mode='r',*args,**kwargs):
        guard(file,mode)
        return original_builtin(file,mode,*args,**kwargs)

    def guarded_io(file,mode='r',*args,**kwargs):
        guard(file,mode)
        return original_io(file,mode,*args,**kwargs)

    def guarded_path(file,mode='r',*args,**kwargs):
        guard(file,mode)
        return original_path(file,mode,*args,**kwargs)

    builtins.open,io.open=guarded_builtin,guarded_io
    Path.open=guarded_path
    try:
        yield
    finally:
        builtins.open,io.open=original_builtin,original_io
        Path.open=original_path


def freeze():
    if (OUT/'PROTOCOL.json').exists():
        raise FileExistsError('Protocol already frozen')
    bridge=read(BRIDGE/'PROTOCOL.json')
    paths={PACKAGE/'DESIGN.md',OUT/'PRE_FREEZE_TEST_REPAIR.json'}|set(PACKAGE.glob('*.py'))
    paths|={A/name for name in ('PROTOCOL.json','RESULTS.json','AUDIT.json','per_query.jsonl')}
    paths|={BRIDGE/name for name in ('PROTOCOL.json','CANDIDATES_SEALED.json','RANKINGS_SEALED.json','selections.jsonl')}
    for domain in ('source','cosqa'):
        paths|={BRIDGE/f'data/{domain}/queries.jsonl',BRIDGE/f'data/private/{domain}_qrels.jsonl',BRIDGE/f'candidates/{domain}/baseline.jsonl'}
        paths|={BRIDGE/f'rankings/{domain}/{ret}.jsonl' for ret in ('CodeBERT','BM25')}
    # Validate the previously accepted scientific input chain before freezing this diagnostic.
    old=read(A/'PROTOCOL.json')
    for item in old['bound_files']+old['protected_files']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    assert read(A/'AUDIT.json')['passed']
    samples={d:dict(query_ids=bridge['samples'][d]['main_ids'],groups=bridge['samples'][d]['groups'],
                    queries=len(bridge['samples'][d]['main_ids']),unique_groups=len(set(bridge['samples'][d]['groups'].values()))) for d in ('source','cosqa')}
    write('PROTOCOL.json',dict(created_utc=now(),status='FROZEN_BEFORE_NEW_FACTORIAL_REPLAY',
          role='post_hoc_paired_factorial_diagnostic_on_consumed_source_and_D',
          previous_A_effects_known=True,external_preregistration=False,new_confirmation=False,
          profile='greedy20_source',mask='full',depth=100,samples=samples,
          policies=list(POLICIES),budgets=list(BUDGETS),aggregations=list(AGGREGATIONS),
          main_factorial_aggregations=['mean','max'],first_is_structural_anchor=True,
          mean_definition='expected RR when choosing uniformly within the selected prefix; not rank fusion',
          max_definition='query-specific label-assisted largest RR in selected prefix',
          random_definition='exact uniform without replacement over original positions; preserve duplicate-text position mass',
          short_pool='min(k,n), no zero or original-query padding',
          grouping='query-weighted effects; whole original repository/connected group bootstrap',
          bootstrap=dict(repetitions=10000,seed=20261020,shared_draws_across_all_effects=True),
          primary_comparisons=dict(per_view=list(PRIMARY_NAMES),views=4,family_size=24,
             correction='Bonferroni 24; both group-bootstrap and cluster-t intervals reported'),
          other_effects='pointwise95 descriptive; no new equivalence or policy selection',
          decomposition='J_exact = B_mean_budget + G_max_vs_mean_at3; J20 = J_exact + RAND20_minus_exact3',
          decomposition_interpretation='prespecified reference-path arithmetic; not unique causal attribution',
          no_model_inference=True,no_new_training=True,no_H_effects=True,no_new_policy=True,risk_certified=False,
          bound_files=[record(p) for p in sorted(paths)],protected_files=[record(ROOT/p) for p in PROTECTED],
          environment=dict(python=platform.python_version(),numpy=np.__version__)))
    print(json.dumps(dict(frozen=True,samples={d:(s['queries'],s['unique_groups']) for d,s in samples.items()},family_size=24)),flush=True)


def verify():
    p=read(OUT/'PROTOCOL.json')
    for item in p['bound_files']+p['protected_files']:
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    return p


def actions():
    p=verify()
    output=[]
    with deny_outcomes():
        previous={(r['domain'],r['query_id']):r for r in rows(BRIDGE/'selections.jsonl') if r['scope']=='main' and r['profile']=='greedy20_source' and r['mask']=='full'}
        for domain,sample in p['samples'].items():
            queries={r['query_id']:r['query'] for r in rows(BRIDGE/f'data/{domain}/queries.jsonl')}
            pools=defaultdict(list)
            for r in rows(BRIDGE/f'candidates/{domain}/baseline.jsonl'):
                pools[r['query_id']].append(r)
            assert set(queries)==set(pools)==set(sample['query_ids'])
            for qid in sample['query_ids']:
                pool=sorted(pools[qid],key=lambda r:r['position'])
                assert [r['position'] for r in pool]==list(range(1,len(queries[qid].split())+1))
                order={policy:ordered_positions(pool,policy) for policy in ('ENTR','PROB')}
                for policy in order:
                    assert order[policy][:3]==previous[domain,qid][policy]
                output.append(dict(domain=domain,query_id=qid,group=sample['groups'][qid],positions=len(pool),orders=order,
                                   effective_k={str(k):min(k,len(pool)) for k in BUDGETS}))
    writel('ACTIONS.jsonl',output)
    write('ACTIONS_SEALED.json',dict(created_utc=now(),protocol_sha256=sha(OUT/'PROTOCOL.json'),
          files=[record(OUT/'ACTIONS.jsonl')],query_actions=len(output),qrels_parsed=False,
          outcome_inputs_blocked=True,new_generation=False))
    print(json.dumps(dict(actions_sealed=True,queries=len(output))),flush=True)


def calculate():
    started=time.perf_counter()
    p=verify()
    assert not (OUT/'RESULTS.json').exists()
    seal=read(OUT/'ACTIONS_SEALED.json')
    assert seal['protocol_sha256']==sha(OUT/'PROTOCOL.json')
    for item in seal['files']:
        assert sha(ROOT/item['path'])==item['sha256']
    write('OUTCOME_ACCESS.json',dict(created_utc=now(),protocol_sha256=sha(OUT/'PROTOCOL.json'),
          action_seal_sha256=sha(OUT/'ACTIONS_SEALED.json'),prior_outcomes_known=True,
          new_confirmation=False,H_qrels_or_rankings_accessed=False))
    actions={(r['domain'],r['query_id']):r for r in rows(OUT/'ACTIONS.jsonl')}
    old={(r['domain'],r['retriever'],r['query_id']):r for r in rows(A/'per_query.jsonl')}
    views=[]
    ledger=[]
    anchor_error=0.0
    for domain,sample in p['samples'].items():
        for retriever in ('CodeBERT','BM25'):
            cell=[]
            for qid in sample['query_ids']:
                action=actions[domain,qid]
                source=old[domain,retriever,qid]
                assert source['group']==action['group'] and source['position_count']==action['positions']
                scores=dict(source['candidate_RR'])
                values=factorial_values(scores,action['orders'],source['values']['NoRewrite'],source['values']['RAND20_best3'])
                fx=contrasts(values)
                for policy in ('ENTR','PROB'):
                    anchor_error=max(anchor_error,
                        abs(fx[f'{policy}_J_exact']-source['contrasts'][f'j_{policy}_contract_difference_exact_random3']),
                        abs(fx[f'{policy}_J_RAND20']-source['contrasts'][f'j_{policy}_contract_difference']),
                        abs(values[f'{policy}_k1_first']-source['values'][f'{policy}_top1']),
                        abs(values[f'{policy}_k3_max']-source['values'][f'{policy}_best3']))
                assert all(abs(fx[f'{policy}_J_RAND20']-fx[f'{policy}_J_exact']-fx['RAND20_minus_exact3'])<1e-13 for policy in ('ENTR','PROB'))
                row=dict(domain=domain,retriever=retriever,query_id=qid,group=action['group'],positions=len(scores),
                         values=values,contrasts=fx)
                cell.append(row);ledger.append(row)
            joined=[{**r['values'],**r['contrasts']} for r in cell]
            names=list(joined[0])
            summaries=clustered_summary([[r[name] for name in names] for r in joined],
                [r['group'] for r in cell],names,repetitions=p['bootstrap']['repetitions'],seed=p['bootstrap']['seed'],family_size=24)
            factorial=[]
            for policy in POLICIES:
                for k in BUDGETS:
                    for aggregation in AGGREGATIONS:
                        name=f'{policy}_k{k}_{aggregation}'
                        factorial.append(dict(policy=policy,k=k,aggregation=aggregation,MRR=summaries[name],
                           delta_NoRewrite=summaries['delta_'+name],
                           advantage_vs_random=summaries[f'S_{policy}_k{k}_{aggregation}'] if policy!='RAND_exact' else None,
                           label_assisted=(aggregation=='max')))
            view=dict(domain=domain,retriever=retriever,queries=len(cell),groups=sample['unique_groups'],
                      raw_summaries={name:summaries[name] for name in cell[0]['values']},
                      effects={name:summaries[name] for name in cell[0]['contrasts']},factorial=factorial,
                      primary_decomposition={name:summaries[name] for name in PRIMARY_NAMES},
                      pool_counts=dict(min=min(r['positions'] for r in cell),max=max(r['positions'] for r in cell),
                                       less_than_three=sum(r['positions']<3 for r in cell)),
                      max_per_query_decomposition_error=max(abs(r['contrasts'][f'{policy}_J_exact']-
                          r['contrasts'][f'{policy}_B_budget_mean_1_to_3']-
                          r['contrasts'][f'{policy}_G_aggregation_at3_mean_to_max']) for r in cell for policy in ('ENTR','PROB')))
            views.append(view)
            print(json.dumps(dict(computed=f'{domain}/{retriever}',queries=len(cell),
                components={name:summaries[name]['mean'] for name in PRIMARY_NAMES}),ensure_ascii=False),flush=True)
    assert len(ledger)==7076 and anchor_error<1e-12
    writel('per_query.jsonl',ledger)
    result=dict(created_utc=now(),role=p['role'],protocol_sha256=sha(OUT/'PROTOCOL.json'),
                action_seal_sha256=sha(OUT/'ACTIONS_SEALED.json'),views=views,
                integrity=dict(A_anchor_max_error=anchor_error,no_H_qrels_or_rankings_loaded=True,
                               no_model_inference=True,no_new_training=True,protected_state_unchanged=True),
                seconds=time.perf_counter()-started,new_confirmation=False,risk_certified=False)
    verify()
    write('RESULTS.json',result)
    print(json.dumps(dict(results_saved=True,seconds=result['seconds'],query_rows=len(ledger))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('freeze','actions','calculate'))
    args=parser.parse_args()
    globals()[args.phase]()
