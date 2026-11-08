"""Final immutable receipt for the bounded CQC training diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rene_cqc_training_fidelity_20261020'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(2**23),b''):h.update(part)
    return h.hexdigest()


def require(value,message):
    if not value:raise AssertionError(message)


def resolve(label):
    p=Path(label);return p if p.is_absolute() else ROOT/p


def extra_contrast_check():
    protocol=read(OUT/'EXTRA_CONTRAST_PROTOCOL.json');extra=read(OUT/'EXTRA_CONTRASTS.json')
    training=read(OUT/'TRAIN_PROTOCOL.json')
    require(protocol['source_sha256']==sha(ROOT/'experiments/rene_cqc_training_extra_contrasts_20261020.py'),'Extra contrast source binding')
    require(protocol['training_protocol_sha256']==sha(OUT/'TRAIN_PROTOCOL.json'),'Extra training binding')
    require(protocol['evaluation_protocol_sha256']==sha(OUT/'EVAL_PROTOCOL.json'),'Extra evaluation binding')
    require(protocol['created_utc']<read(OUT/'OUTCOME_ACCESS.json')['created_utc'],'Extra contrasts declared before effects')
    require(extra['extra_protocol_sha256']==sha(OUT/'EXTRA_CONTRAST_PROTOCOL.json'),'Extra protocol binding')
    require(extra['results_sha256']==sha(OUT/'RESULTS.json'),'Extra results binding')
    with (OUT/'per_query.jsonl').open(encoding='utf-8-sig') as f:bank=[json.loads(line) for line in f if line.strip()]
    index={(r['domain'],r['view'],r['arm'],r['query_id']):r for r in bank}
    require(len(index)==len(bank)==16384,'Complete paired records')
    coefs={'progress_scaled_minus_literal':{'PN':1,'P':-1},
        'nonadditivity':{'P':1,'O':-1,'S':-1,'N':-1,'W':-1,'L':3}}
    records=extra['contrasts'];seen=set();max_error=0.;fields=0
    require(len(records)==16,'Extra contrast family size')
    for row in records:
        domain,view,metric,contrast=(row[k] for k in ('domain','view','metric','contrast'))
        key=(domain,view,metric,contrast);require(key not in seen,'Duplicate extra contrast');seen.add(key)
        require(domain in ('source','cosqa') and view in ('own','fixed') and metric in ('S3','top1_gain') and contrast in coefs,'Extra contrast identity')
        ids=training['samples'][domain]['query_ids'];groupmap=training['samples'][domain]['groups']
        labels=sorted({groupmap[q] for q in ids});members={g:[] for g in labels}
        for q in ids:
            value=math.fsum(c*index[domain,view,a,q]['effects'][metric] for a,c in coefs[contrast].items())
            members[groupmap[q]].append(value)
        sums=np.array([math.fsum(members[g]) for g in labels]);sizes=np.array([len(members[g]) for g in labels])
        center=math.fsum(sums)/len(ids)
        se=math.sqrt(len(labels)/(len(labels)-1)*math.fsum((float(s)-center*int(n))**2 for s,n in zip(sums,sizes)))/len(ids)
        rng=np.random.default_rng(20261020);resamples=[]
        for offset in range(0,10000,64):
            sample=rng.integers(0,len(labels),(min(64,10000-offset),len(labels)))
            resamples.extend((sums[sample].sum(1)/sizes[sample].sum(1)).tolist())
        width=float(t.isf(.025,len(labels)-1))*se
        expected={'mean':center,'ci95_group_bootstrap':np.quantile(resamples,[.025,.975]),
            'ci95_cluster_t':[center-width,center+width]}
        require(row['descriptive_only'] is True,'Supplement is descriptive')
        for field,value in expected.items():
            error=float(np.max(np.abs(np.asarray(row[field])-np.asarray(value))))
            require(error<1e-11,'Independent extra contrast: '+str(key)+' '+field)
            max_error=max(max_error,error);fields+=1
    for domain in ('source','cosqa'):
        for view in ('own','fixed'):
            for q in training['samples'][domain]['query_ids']:
                left=index[domain,view,'L',q];right=index[domain,view,'N',q]
                require(left['RR']==right['RR'] and left['effects']==right['effects'],'Length inference negative control')
    return dict(passed=True,contrasts=16,independent_fields=fields,max_abs_error=max_error,
        length_control_identical_query_outcomes=True,extra_contrasts_sha256=sha(OUT/'EXTRA_CONTRASTS.json'),
        method='Independent signed-coefficient contrasts, Python group aggregation, original-group bootstrap and cluster-t')


def main():
    require(not (OUT/'FINAL_ACCEPTANCE.json').exists(),'Do not overwrite final receipt')
    train=read(OUT/'TRAIN_PROTOCOL.json');evaluation=read(OUT/'EVAL_PROTOCOL.json');result=read(OUT/'RESULTS.json')
    checked=0
    for item in train['bound_files']+train['protected_files']+evaluation['code_and_input_files']:
        require(sha(resolve(item['path']))==item['sha256'],'Bound/protected input changed: '+item['path']);checked+=1
    original_manifest=read(ROOT/'data/codenn_cqc/manifest.json')
    require(sha(ROOT/'data/codenn_cqc/train.txt')==original_manifest['sha256']['train'],'Historical training corpus provenance')
    require(sha(ROOT/'data/codenn_cqc/valid_fixed_cqc.jsonl')==original_manifest['sha256']['fixed_validation_cqc'],'Historical validation provenance')
    old_bridge=read(ROOT/'results/rene_ssqr_bridge_20261019/PROTOCOL.json')
    old_model=[r for r in old_bridge['bound_files'] if r['path']=='models/t5-cqc-codenn-1m/best/model.safetensors']
    require(len(old_model)==1 and sha(resolve(old_model[0]['path']))==old_model[0]['sha256'],'Historical formal T5 weights changed')
    for name in ('TRAIN_AUDIT.json','EFFECT_AUDIT.json','MODEL_AUDIT.json'):
        require(read(OUT/name)['passed'],'Scientific audit failed: '+name)
    for name in ('TRAIN_TESTS.json','EVAL_TESTS.json'):
        tests=read(OUT/name);require(tests['passed'] and tests['tests']==4 and not tests['errors'] and not tests['failures'],'Scientific tests')
    effect=read(OUT/'EFFECT_AUDIT.json')
    require(effect['results_sha256']==sha(OUT/'RESULTS.json'),'Audit results binding')
    require(effect['per_query_sha256']==sha(OUT/'per_query.jsonl'),'Audit per-query binding')
    require(not result['new_confirmation'] and not result['risk_certified'] and result['no_H_read'],'Evidence role')
    actions=read(OUT/'CANDIDATES_SEALED.json');ranks=read(OUT/'RANKINGS_SEALED.json');access=read(OUT/'OUTCOME_ACCESS.json')
    require(actions['created_utc']<ranks['created_utc']<access['created_utc'],'Inference-before-outcomes chronology')
    for seal in (actions,ranks):
        for item in seal['files']:
            require(sha(resolve(item['path']))==item['sha256'],'Inference output changed');checked+=1
    train_reports=result['training']
    require(len(train_reports)==8 and all(r['steps']==1024 and r['examples_seen']==32768 for r in train_reports),'Matched training budget')
    for item in read(OUT/'TRAINING_COMPLETE.json')['files']:
        require(sha(resolve(item['path']))==item['sha256'],'Completed training receipt changed');checked+=1
    weightrecords=[]
    for r in train_reports:
        require(r==read(OUT/f"training/{r['arm']}.json"),'Embedded training report changed')
        for item in r['files']+[r['train_steps_file']]:
            require(sha(resolve(item['path']))==item['sha256'],'Training artifact or update log changed');checked+=1
        path=Path(train['model_store'])/r['arm']/'model.safetensors'
        require(sha(path)==r['weights_sha256'],'Trained weight digest')
        weightrecords.append(dict(arm=r['arm'],path=path.as_posix(),sha256=r['weights_sha256']))
    exports={}
    for name,expected in [('PRIMARY.csv',result['primary_changes']),('DESCRIPTIVE.csv',result['descriptive_changes'])]:
        with (OUT/name).open(encoding='utf-8-sig',newline='') as f:actual=list(csv.DictReader(f))
        require(len(actual)==len(expected),'Export row count')
        fields=0
        for row,wanted in zip(actual,expected):
            for k,v in row.items():
                item=wanted.get(k,'');value=json.dumps(item) if isinstance(item,list) else str(item)
                require(v==value,'Export field mismatch');fields+=1
        exports[name]=dict(rows=len(actual),fields=fields)
    for name in ('READOUT.md','FINDINGS.md'):
        require((OUT/name).is_file(),'Missing deliverable: '+name)
    for name in ('cqc_configuration_effects.png','cqc_configuration_effects.pdf'):
        require((OUT/name).is_file(),'Missing scientific figure: '+name)
    supplement=extra_contrast_check()
    action=read(OUT/'ACTION_DIAGNOSTIC.json')
    require(action['candidates_seal_sha256']==sha(OUT/'CANDIDATES_SEALED.json'),'Action diagnostic candidate binding')
    require(action['source_sha256']==sha(ROOT/'experiments/rene_cqc_training_action_diagnostic_20261020.py'),'Action diagnostic code binding')
    require(action['labels_read'] is False and action['new_inference'] is False and len(action['records'])==32,'Action diagnostic scope')
    files=[p for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='RUN_STATUS.json']
    files += list((OUT/'training').glob('*.json'))
    files += [ROOT/'paper/rene2027/cqc_training_fidelity_results.md',Path(__file__).resolve(),
        ROOT/'ResearchVault/Handoffs/RENE-CQC有限训练配置对照-20261020.md',
        ROOT/'experiments/rene_cqc_training_extra_contrasts_20261020.py',
        ROOT/'experiments/rene_cqc_training_full_length_20261020.py',
        ROOT/'experiments/rene_cqc_training_action_diagnostic_20261020.py',
        ROOT/'experiments/rene_cqc_training_figure_20261020.py',
        ROOT/'experiments/rene_cqc_training_writeup_20261020.py',
        ROOT/'experiments/rene_cqc_training_fidelity_pipeline_20261020.ps1']
    require(all(p.is_file() for p in files),'Missing source/handoff')
    value=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        role=result['role'],new_confirmation=False,risk_certified=False,no_H_read=True,
        bounded_training=True,updates_per_arm=1024,training_examples_per_arm=32768,arms=8,
        source_queries=512,source_groups=512,cosqa_development_queries=512,cosqa_groups=512,
        paired_views=2,primary_comparisons=28,scientific_tests=8,hash_records_checked=checked,
        length_control_identical=read(OUT/'TRAIN_AUDIT.json')['length_control_identical_weights'],
        independent_effect_audit=effect,export_checks=exports,trained_weights=weightrecords,
        extra_contrast_audit=supplement,
        full_historical_training_effect_not_identified=True,formal_state_and_old_confirmation_unchanged=True,
        files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)) for p in files])
    with (OUT/'FINAL_ACCEPTANCE.json').open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    (OUT/'RUN_STATUS.json').write_text(json.dumps(dict(stage='complete',completed=True,passed=True,
        updated_utc=datetime.now(timezone.utc).isoformat(),receipt='FINAL_ACCEPTANCE.json'),indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,arms=8,primary_comparisons=28,hash_records_checked=checked,exports=exports),ensure_ascii=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
