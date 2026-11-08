"""Descriptive candidate/selection turnover; no additional inference or outcome use."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rene_cqc_training_fidelity_20261020'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:return [json.loads(line) for line in stream if line.strip()]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol=read(OUT/'TRAIN_PROTOCOL.json')
    choices={(r['domain'],r['view'],r['arm'],r['query_id']):r for r in rows(OUT/'selections.jsonl')}
    assert (OUT/'CANDIDATES_SEALED.json').is_file(),'All candidates and choices must be sealed'
    outputs=[]
    for domain in ('source','cosqa'):
        ids=protocol['samples'][domain]['query_ids']
        for view in ('own','fixed'):
            pools={arm:{(r['query_id'],r['position']):r for r in rows(OUT/f'candidates/{arm}/{domain}_{view}.jsonl')}
                for arm in protocol['arms']}
            reference=pools['L']
            for arm,bank in pools.items():
                assert bank.keys()==reference.keys(),'Common query-position scope'
                changes=sum(r['rewrite']!=reference[key]['rewrite'] for key,r in bank.items())
                assert view!='fixed' or changes==0,'Fixed pool text must stay identical'
                item=dict(domain=domain,view=view,arm=arm,reference='L',queries=len(ids),positions=len(bank),
                    changed_rewrite_positions=changes,changed_rewrite_fraction=changes/len(bank),
                    identity_positions=sum(r['is_identity'] for r in bank.values()))
                for policy in ('ENTR','PROB'):
                    paired=[(choices[domain,view,arm,q][policy],choices[domain,view,'L',q][policy]) for q in ids]
                    item[policy+'_top1_changed_queries']=sum(a[0]!=b[0] for a,b in paired)
                    item[policy+'_top3_set_changed_queries']=sum(set(a[:3])!=set(b[:3]) for a,b in paired)
                if arm=='N':
                    for key,row in bank.items():
                        assert {k:v for k,v in row.items() if k!='rescored_arm'}=={
                            k:v for k,v in reference[key].items() if k!='rescored_arm'},'Length cap negative control: text and score fields'
                    assert item['ENTR_top1_changed_queries']==item['ENTR_top3_set_changed_queries']==0
                outputs.append(item)
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),role='descriptive action diagnostic, no causal decomposition',
        candidates_seal_sha256=sha(OUT/'CANDIDATES_SEALED.json'),source_sha256=sha(__file__),
        labels_read=False,new_inference=False,new_confirmation=False,records=outputs)
    with (OUT/'ACTION_DIAGNOSTIC.json').open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False)
    print(json.dumps(dict(action_diagnostic_records=len(outputs),labels_read=False),ensure_ascii=False))


if __name__=='__main__':main()
