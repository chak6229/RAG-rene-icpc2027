"""Export frozen analyses, with denominators and training-budget scope explicit."""
from __future__ import annotations

import csv
import json

from .common import OUT,ARM_SPECS,read,write,sha,now,require


def main():
    result=read(OUT/'RESULTS.json');audit=read(OUT/'EFFECT_AUDIT.json');train=read(OUT/'TRAIN_AUDIT.json');models=read(OUT/'MODEL_AUDIT.json')
    require(audit['passed'] and train['passed'] and models['passed'],'All scientific audits must pass')
    require(audit['results_sha256']==sha(OUT/'RESULTS.json'),'Audited result digest')
    lines=['# CQC Limited training configuration comparison: complete results','',
        'Fixed original t5-base initialization, 32768 Article shared CODEnn sample/fixed mask and sequence, 1024 Update, Valid batch32, lr Starting point. 001, fixed the end step weight. All branches have the same training data and updated budgets.',
        'The source512/512 repository and the CoSQA development 512 / 512 associated group follow the same freeze CodeBERT as the used bridge sample, complete 19210 / 6267. This round is not new; Unread H.','',
        'L Local; O Only Adafactor; S Only in writing linear100000; N Length only 128; W Only weight_decay0; P Grouping in the public table; PN Scaled according to historical progress linear1093; PS for Psee 42 Sensitivity.',
        'All branches maintain the same local continuity 15% mask; P does not restore the author CSV or complete trainer. PN Explain it to the public on paper.','',
        '## Negative training versus length','',
        'Zilong subdivision fixed initial authentication NLL 1024 Step validation NLL Time of training (seconds, process)','|---|---:|---:|---:|']
    for r in result['training']:
        lines.append(f"|{r['arm']}|{r['validation'][0]['token_nll']:.8f}|{r['validation'][-1]['token_nll']:.8f}|{r['seconds_this_process']:.1f}|")
    lines+=['','128 / 512 training input did not trigger the cut-off; L is fully consistent with the 1024 step of N loss / gradient trajectory and end-step weight bytes, which are independently negative.','',
        '## Under the same budget, each configuration generates its own candidate','',
        'S3 = ENTR best3− Accurate random best3; S1 = ENTR Top1− Accurate random Top1. best3 Optimal use of relevant labels is an opportunity indicator.','',
        'NoRewrite MRR | ENTR Top1 | query ENTR best3 | query | S1 | S3 | PROB best3− Accurate random |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    n=lambda v:f'{v:+.8f}'
    ci=lambda v:'['+', '.join(n(x) for x in v)+']'
    for v in result['views']:
        if v['view']!='own':continue
        f=v['effects']
        lines.append(f"|{v['domain']}|{v['arm']}|{v['MRR']['NoRewrite']['mean']:.8f}|{n(f['top1_gain']['mean'])}|{n(f['best3_gain']['mean'])}|{n(f['S1']['mean'])}|{n(f['S3']['mean'])}|{n(f['prob_S3']['mean'])}|")
    lines+=['','## Direct pair changes relative to budget matching L','',
        '28 Pre-sets = net proceeds of 7 branches of L branch x 2 domain x S3/Top1. The following table has been corrected by Bonferroni while retaining groups bootstrap and cluster-t. The positive value increases the corresponding indicator for this configuration L; Not a pairing effect with the historical 93750 step model.','',
        '|-L| indicator change estimated | 28 calibrates | 28 calibrates |',
        '|---|---|---|---:|---|---|']
    for r in result['primary_changes']:
        lines.append(f"|{r['domain']}|{r['arm']}−L|{r['metric']}|{n(r['mean'])}|{ci(r['family28_ci95_group_bootstrap'])}|{ci(r['family28_ci95_cluster_t'])}|")
    lines+=['','## Fixed historical candidates, changing training weights only.','',
        'The perspective is fully reworded to the same rewrite/rank. Changes relative to L are used for diagnostic choice changes; The difference in perspective is not interpreted as the only causal breakdown. The table below is a descriptive point estimate. See full area DESCRIPTIVE.csv.','',
        '|-L | S3 Change Top1 Changes in net gains','|---|---|---:|---:|']
    for domain in ('source','cosqa'):
        for arm in ARM_SPECS:
            if arm=='L':continue
            entries={r['metric']:r for r in result['descriptive_changes'] if r['domain']==domain and r['view']=='fixed' and r['arm']==arm and r['reference']=='L'}
            lines.append(f"|{domain}|{arm}−L|{n(entries['S3']['mean'])}|{n(entries['top1_gain']['mean'])}|")
    lines+=['','## Interpretation and review','',
        '- The present round quantifys all the differences between historical 93750 step models and unauthorised models because of their configuration under a fixed limited training budget.',
        '- Literally open linear100000 at 1024 step only slight decay; PN Individually check the scaling of training, which cannot be combined.',
        '- Inter-territory reflecting uncertainty in reference/relationship groups on regular training examples; PS Provides only one training of random sensitivity, not a substitute for multiple seed training overall differences.',
        '- ENTR/PROB / Accurate random pooling of budgets; Relative random advantages are separated from relative NoRewrite net gains. NLL Change is not a substitute for retrieving gains.',
        '- Original H Recognition, formal five pointers, historical weights and results maintained; Not to replace the official model with this round, risk_certified = false.','',
        'Independent audit: TRAIN_AUDIT.json Check common data/updates/lr/negative comparisons; EFFECT_AUDIT json Re-place and independently calculate ranges from the original ranking by Fraction and another CDF formula; MODEL_AUDIT.json Independent physical forward, FP64 entropy and greedy regenerated.',
        'Full result RESULTS.json; Main change PRIMARY.csv; The remaining pairs DESCRIPTIVE.csv; Each query and each candidate is per_query.jsonl, candidate_RR.jsonl.','']
    with (OUT/'READOUT.md').open('x',encoding='utf-8') as h:h.write('\n'.join(lines))
    print(dict(stage='report_exported',primary_rows=len(result['primary_changes'])),flush=True)


if __name__=='__main__':main()
