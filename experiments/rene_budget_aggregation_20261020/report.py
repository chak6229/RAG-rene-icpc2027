"""Export all frozen factorial cells and paired contrasts without re-estimation."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/rene_budget_aggregation_20261020'


def csv_write(name,records):
    columns=list(dict.fromkeys(key for r in records for key in r))
    with (OUT/name).open('x',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(dict,list)) else value for key,value in r.items()})


def main():
    result=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'))
    cells=[];effects=[];components=[]
    lines=['# The number of candidates for the pool is broken down by aggregation.','',
           'Source side 2048 query/594 repository, CoSQA development 1490 query/512 associated group; Fixed local CQC T5 candidate and double searcher cache. k For downstream candidates, all positions are generated and the scoring cost is maintained. No new generation, training, retrieval or H impact assessment. This round is followed by a pair of diagnostics.','',
           'mean is the expectation of evenly applying a single article in the selection pool RR; max Tag support best-k opportunities; first Always execute prefix one. Accurate randomly does not return points, and repeats text positions do not merge. mean Not integration, max not deployable selection.','',
           '## Quantified and fully compared to aggregated','',
           'The following table shows the relative homogeneity of the selection rule k. mean / max per cell; Full MRR, relative to original query difference and line-by-line range FACTORIAL.csv. first The advantage is the same for all k.','',
           '|Data / Retrieval |K=1 mean / max|k=2 mean / max|k=3 mean / max|',
           '|---|---|---:|---:|---:|']
    n=lambda value:f'{value:+.8f}'
    ci=lambda values:'['+', '.join(map(n,values))+']'
    for view in result['views']:
        index={(r['policy'],r['k'],r['aggregation']):r for r in view['factorial']}
        for cell in view['factorial']:
            record={key:view[key] for key in ('domain','retriever','queries','groups')}
            record.update({key:cell[key] for key in ('policy','k','aggregation','label_assisted')})
            record.update(MRR=cell['MRR']['mean'],MRR_ci95=cell['MRR']['ci95_group_bootstrap'],
                delta_NoRewrite=cell['delta_NoRewrite']['mean'],delta_NoRewrite_ci95=cell['delta_NoRewrite']['ci95_group_bootstrap'],
                advantage_vs_random=cell['advantage_vs_random']['mean'] if cell['advantage_vs_random'] is not None else 0.,
                advantage_vs_random_ci95=cell['advantage_vs_random']['ci95_group_bootstrap'] if cell['advantage_vs_random'] is not None else [0.,0.])
            cells.append(record)
        for name,effect in view['effects'].items():
            row=dict(domain=view['domain'],retriever=view['retriever'],contrast=name,**effect)
            effects.append(row)
            if name in view['primary_decomposition']:
                components.append(row)
        for p in ('ENTR','PROB'):
            values=[]
            for k in (1,2,3):
                values.append(' / '.join(n(index[p,k,a]['advantage_vs_random']['mean']) for a in ('mean','max')))
            lines.append('|'+f"{view['domain']} / {view['retriever']}|{p}|"+'|'.join(values)+'|')
    lines+=['','## Preset Path Decomposition','',
            'S is relative to k the advantage of precise randomity. B=S(1,mean)−S(3,mean)，G=S(3,mean)−S(3,max)，J_exact=B+G。 The positive values for B and G represent a weakening of the original advantage under corresponding changes, while negative values represent an increase.','',
            'Presets that 24 items are two parts of four data x searcher cells, ENTR/PROB/ENTR−PROB and B/G. The Bonferroni correction group bootstrap slots in the table; The complete result also provides cluster-t correction slots. Do not add an equivalent determination.','',
            '|Data / Retriever |1/3 change in B fixedmean | 24 correction zones |G 3 fix mean→max change | 24 correction sites |J_exact|',
            '|---|---|---:|---|---:|---|---:|']
    for view in result['views']:
        for p in ('ENTR','PROB','ENTR_minus_PROB'):
            B=view['effects'][p+'_B_budget_mean_1_to_3'];G=view['effects'][p+'_G_aggregation_at3_mean_to_max']
            total=view['effects'][p+'_J_exact']['mean'] if p!='ENTR_minus_PROB' else view['effects']['ENTR_J_exact']['mean']-view['effects']['PROB_J_exact']['mean']
            lines.append(f"|{view['domain']} / {view['retriever']}|{p}|{n(B['mean'])}|{ci(B['family24_ci95_group_bootstrap'])}|{n(G['mean'])}|{ci(G['family24_ci95_group_bootstrap'])}|{n(total)}|")
    lines+=['','## RAND20 for old A.','',
            'Original A does not cover: J20 = J_exact+ (RAND20_best3−RAND_exact_best3). The main decomposition of the wheel is based on precise randomity and a limited number of duplicate errors are shown separately.','',
            'Data / Retriever ENTR  J20| ENTR J_exact | RAND20− Accurate Random best3 |','|---|---:|---:|---:|']
    for v in result['views']:
        fx=v['effects']
        lines.append(f"|{v['domain']} / {v['retriever']}|{n(fx['ENTR_J_RAND20']['mean'])}|{n(fx['ENTR_J_exact']['mean'])}|{n(fx['RAND20_minus_exact3']['mean'])}|")
    lines+=['','## Limitation on interpretation','',
            '- k1 The three are the same aggregates; first Add k without changing the actual action. mean → max has changed the right to choose candidates using labels and cannot be written as free additional online capacity.',
            '- B/G is the arithmetical decomposition under the specified reference path, which is not an independent and manageable cause-and-effect mechanism that does not report unjustified “principal ratio”.',
            '- Relative random advantages and relative original query net proceeds are reported separately, and the former cannot be replaced by the latter; ENTR Differences with PROB need to be directly compared, and not only more prominent.',
            '- linked groups bootstrap10000 times, query weights; Data are developed/source side of consumption. Old H confirmed, fully configured and risk_certified = false maintained.','',
            'Full machine results RESULTS.json; FACTORIAL csv Yes 108 Factor unit with first anchor, main mean/max unit 72 One; CONTRASTS. csv Volume changes containing fixed aggregates and aggregate changes in fixed quantities; DECOMPOSITION csv with 24 Predefined amount of the item. Independent Fraction/CDF Reopen AUDIT.json, accept and accept FINAL_ACCEPTANCE.json.','']
    csv_write('FACTORIAL.csv',cells)
    csv_write('CONTRASTS.csv',effects)
    csv_write('DECOMPOSITION.csv',components)
    with (OUT/'READOUT.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps(dict(factorial_cells=len(cells),contrasts=len(effects),planned_components=len(components))))


if __name__=='__main__':
    main()
