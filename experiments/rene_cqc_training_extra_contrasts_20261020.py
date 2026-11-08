"""Pre-outcome declared descriptive progress and nonadditivity checks, no extra training."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rene_cqc_training_fidelity_20261020'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as h:json.dump(value,h,indent=2,ensure_ascii=False)


def freeze():
    assert not (OUT/'OUTCOME_ACCESS.json').exists()
    write(OUT/'EXTRA_CONTRAST_PROTOCOL.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(__file__),training_protocol_sha256=digest(OUT/'TRAIN_PROTOCOL.json'),
        evaluation_protocol_sha256=digest(OUT/'EVAL_PROTOCOL.json'),frozen_before_outcomes=True,
        role='descriptive adjunct, no extra training and no primary-family change',
        contrasts=dict(progress_scaled_minus_literal='PN-P',nonadditivity='(P-L)-[(O-L)+(S-L)+(N-L)+(W-L)]'),
        metrics=['S3','top1_gain'],views=['own','fixed'],domains=['source','cosqa'],
        statistic='paired pointwise95 cluster-t and10000 original-group bootstrap',seed=20261020,
        nonadditivity_interpretation='aggregate departure from sum of single changes; not individual interaction identification'))


def calculate():
    protocol=read(OUT/'EXTRA_CONTRAST_PROTOCOL.json')
    assert protocol['source_sha256']==digest(__file__)
    assert protocol['training_protocol_sha256']==digest(OUT/'TRAIN_PROTOCOL.json')
    assert protocol['evaluation_protocol_sha256']==digest(OUT/'EVAL_PROTOCOL.json')
    assert protocol['created_utc']<read(OUT/'OUTCOME_ACCESS.json')['created_utc']
    with (OUT/'per_query.jsonl').open(encoding='utf-8') as h:bank=[json.loads(x) for x in h if x.strip()]
    index={(r['domain'],r['view'],r['arm'],r['query_id']):r for r in bank}
    train=read(OUT/'TRAIN_PROTOCOL.json');outputs=[]
    for domain in ('source','cosqa'):
        ids=train['samples'][domain]['query_ids'];groupmap=train['samples'][domain]['groups']
        labels=sorted({groupmap[q] for q in ids});lookup={g:i for i,g in enumerate(labels)}
        for view in ('own','fixed'):
            for metric in ('S3','top1_gain'):
                for contrast in ('progress_scaled_minus_literal','nonadditivity'):
                    values=[]
                    for qid in ids:
                        e={a:index[domain,view,a,qid]['effects'][metric] for a in train['arms']}
                        values.append(e['PN']-e['P'] if contrast=='progress_scaled_minus_literal' else
                            (e['P']-e['L'])-sum(e[a]-e['L'] for a in ('O','S','N','W')))
                    x=np.array(values);codes=[lookup[groupmap[q]] for q in ids]
                    sums=np.bincount(codes,weights=x,minlength=len(labels));sizes=np.bincount(codes,minlength=len(labels))
                    mean=float(x.mean());se=float(np.sqrt(len(labels)/(len(labels)-1)*np.sum((sums-mean*sizes)**2))/len(x))
                    rng=np.random.default_rng(20261020);boots=[]
                    for start in range(0,10000,64):
                        draw=rng.integers(0,len(labels),(min(64,10000-start),len(labels)))
                        boots.extend((sums[draw].sum(1)/sizes[draw].sum(1)).tolist())
                    width=float(t.ppf(.975,len(labels)-1))*se
                    outputs.append(dict(domain=domain,view=view,metric=metric,contrast=contrast,mean=mean,
                        ci95_group_bootstrap=np.quantile(boots,[.025,.975]).tolist(),ci95_cluster_t=[mean-width,mean+width],
                        descriptive_only=True))
    assert len(outputs)==16
    write(OUT/'EXTRA_CONTRASTS.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),results_sha256=digest(OUT/'RESULTS.json'),
        extra_protocol_sha256=digest(OUT/'EXTRA_CONTRAST_PROTOCOL.json'),contrasts=outputs,new_training=False,new_confirmation=False))
    print(json.dumps({'extra_contrasts':outputs},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','calculate'])
    {'freeze':freeze,'calculate':calculate}[parser.parse_args().mode]()
