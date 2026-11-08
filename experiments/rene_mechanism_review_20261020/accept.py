from __future__ import annotations

import csv
import json
import math
import zipfile
from fractions import Fraction
from html.parser import HTMLParser

from .common import (OUT, POLICIES, ROOT, STRATA, classify, now, put, read,
                     record, require_frozen, rows, sample_frame, sha)


class PacketParser(HTMLParser):
    def __init__(self):
        super().__init__();self.active=False;self.parts=[]

    def handle_starttag(self,tag,attrs):
        if tag=='script' and dict(attrs).get('id')=='packet':self.active=True

    def handle_endtag(self,tag):
        if tag=='script':self.active=False

    def handle_data(self,data):
        if self.active:self.parts.append(data)


def extract(path):
    p=PacketParser();p.feed(path.read_text(encoding='utf-8'))
    return json.loads(''.join(p.parts))


def main():
    require_frozen()
    key=read(OUT/'private/reviewer_keys.json')
    sample=list(rows(OUT/'private/sample_manifest.jsonl'))
    frame=list(rows(OUT/'private/CodeBERT_stage_rows.jsonl'))
    if sample!=sample_frame(frame):raise ValueError('independent frame/sample replay differs')
    if len({r['query_id'] for r in sample})!=128 or not math.isclose(sum(r['weight'] for r in sample),8530):
        raise ValueError('sample weights/duplicates')
    summaries=read(OUT/'STAGE_RESULTS.json')
    exact_checks=0
    for ret in ('CodeBERT','BM25'):
        fr=list(rows(OUT/f'private/{ret}_stage_rows.jsonl'))
        U=sum((Fraction(max([Fraction(0),*(Fraction(v) for v in r['candidate_fractions'].values())])) for r in fr),Fraction(0))
        for policy in POLICIES:
            ds=[Fraction(r['policies'][policy]['delta_fraction']) for r in fr]
            pos=sum((max(d,0) for d in ds),Fraction(0));neg=sum((max(-d,0) for d in ds),Fraction(0))
            m=summaries[ret]['stage_results'][policy]
            for name,value in (('net',sum(ds,Fraction(0))),('positive',pos),('harm',neg),('missed',U-pos),('regret',U-pos+neg)):
                if abs(m[name]-float(value/len(fr)))>1e-14:raise ValueError('independent accounting mismatch')
                exact_checks+=1
            if m['adopted']!=sum(r['policies'][policy]['adopted'] for r in fr):raise ValueError('adoption count mismatch')
    for reviewer in ('A','B'):
        folder=OUT/f'reviewer_{reviewer}'
        p1=extract(folder/'01_查询含义.html');p2=extract(folder/'02_代码相关性.html')
        if set(p1)!={'schema_version','reviewer','packet_id','phase','cases'} or set(p2)!={'schema_version','reviewer','packet_id','phase','cases','semantic_schema'}:
            raise ValueError('non-allowlisted packet metadata')
        one={r['case_id']:r for r in p1['cases']};two={r['case_id']:r for r in p2['cases']}
        if len(one)!=128 or set(one)!=set(two) or set(one)!={r['case_id'] for r in key[reviewer]['cases']}:
            raise ValueError('packet case identity drift')
        for case in key[reviewer]['cases']:
            i=case['case_id'];a,b=one[i],two[i]
            if set(a)!={'case_id','original','rewrites'} or set(b)!={'case_id','original','documents'}:
                raise ValueError('unblinded case metadata')
            if a['original']!=case['original'] or b['original']!=case['original']:
                raise ValueError('query changed')
            for rw in a['rewrites']:
                if set(rw)!={'id','text'} or rw['text']!=case['rewrite_map'][rw['id']]['text']:
                    raise ValueError('rewrite payload changed')
            for doc in b['documents']:
                if set(doc)!={'id','code'}:raise ValueError('unblinded document metadata')
            if not 2<=len(b['documents'])<=6:raise ValueError('code panel size')
        with zipfile.ZipFile(OUT/f'审阅者{reviewer}.zip') as z:
            if set(z.namelist())!={'01_查询含义.html','02_代码相关性.html','先读说明.txt'}:
                raise ValueError('zip contains analysis/key/unexpected paths')
            for name in z.namelist():
                if z.read(name)!=(folder/name).read_bytes():raise ValueError('zip byte mismatch')
    if {r['case_id'] for r in key['A']['cases']}&{r['case_id'] for r in key['B']['cases']}:
        raise ValueError('reviewer IDs not independently encoded')
    status=read(OUT/'PACKET_STATUS.json')
    if status['human_review_complete'] or status['human_labels']!=0:raise ValueError('unearned completion claim')
    csv_rows=list(csv.DictReader((OUT/'STAGES.csv').open(encoding='utf-8-sig',newline='')))
    if len(csv_rows)!=10:raise ValueError('stage export shape')
    for r in csv_rows:
        for field,value in summaries[r['retriever']]['stage_results'][r['stage']].items():
            if abs(float(r[field])-value)>1e-14:raise ValueError('stage CSV differs')
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='DELIVERY_ACCEPTANCE.json']
    receipt={'created_utc':now(),'packet_and_numeric_acceptance_passed':True,'human_review_complete':False,
             'human_labels':0,'sample_queries':128,'sample_weight_sum':sum(r['weight'] for r in sample),
             'independent_exact_summary_checks':exact_checks,
             'raw_RR_checks':sum(s['Fraction_RR_checks'] for s in summaries.values()),
             'blinding_audit':'allowlisted payloads; phase1 contains no code, stage, rank, score, target or source identifiers',
             'zip_contents_audited':True,'bound_inputs_verified':True,
             'bound_files':[record(p) for p in sorted(files)]}
    put(OUT/'DELIVERY_ACCEPTANCE.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='bound_files'},ensure_ascii=False))


if __name__=='__main__':main()
