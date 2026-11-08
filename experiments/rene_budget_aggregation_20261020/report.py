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
    lines=['# 同池候选数量与聚合方式分解','',
           '源侧2048查询/594仓库，CoSQA开发1490查询/512关联组；固定本地CQC T5候选和双检索器缓存。k为下游候选数，全部位置生成与打分成本保持。无新生成、训练、检索或H效果评估。此轮为事后配对诊断。','',
           'mean是在选中集合中均匀采用一条的期望RR；max为标签辅助best-k机会；first始终执行前缀第一条。精确随机按原位置无放回积分，重复文本位置不合并。mean不是融合，max不是可部署选择器。','',
           '## 同数量下聚合、同聚合下数量的完整对照','',
           '下表为选择规则相对同k精确随机的优势。每格为mean / max；完整MRR、相对原query差及逐项区间见FACTORIAL.csv。first优势对所有k相同。','',
           '|数据 / 检索器|规则|k=1 mean / max|k=2 mean / max|k=3 mean / max|',
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
    lines+=['','## 预设路径分解','',
            'S为相对同k精确随机的优势。B=S(1,mean)−S(3,mean)，G=S(3,mean)−S(3,max)，J_exact=B+G。B和G的正值表示原优势在相应变化下减弱，负值表示增强。','',
            '预设24项为四个数据×检索器格、ENTR/PROB/ENTR−PROB、B/G两个分量。表内为Bonferroni校正的组bootstrap区间；完整结果同时提供cluster-t校正区间。不新增等效判定。','',
            '|数据 / 检索器|规则|B 固定mean的1→3变化|24项校正区间|G 固定3的mean→max变化|24项校正区间|J_exact|',
            '|---|---|---:|---|---:|---|---:|']
    for view in result['views']:
        for p in ('ENTR','PROB','ENTR_minus_PROB'):
            B=view['effects'][p+'_B_budget_mean_1_to_3'];G=view['effects'][p+'_G_aggregation_at3_mean_to_max']
            total=view['effects'][p+'_J_exact']['mean'] if p!='ENTR_minus_PROB' else view['effects']['ENTR_J_exact']['mean']-view['effects']['PROB_J_exact']['mean']
            lines.append(f"|{view['domain']} / {view['retriever']}|{p}|{n(B['mean'])}|{ci(B['family24_ci95_group_bootstrap'])}|{n(G['mean'])}|{ci(G['family24_ci95_group_bootstrap'])}|{n(total)}|")
    lines+=['','## 原A的RAND20记账','',
            '原A不覆盖：J20=J_exact+(RAND20_best3−RAND_exact_best3)。此轮主分解用精确随机，有限重复误差另列。','',
            '|数据 / 检索器|ENTR J20|ENTR J_exact|RAND20−精确随机best3|','|---|---:|---:|---:|']
    for v in result['views']:
        fx=v['effects']
        lines.append(f"|{v['domain']} / {v['retriever']}|{n(fx['ENTR_J_RAND20']['mean'])}|{n(fx['ENTR_J_exact']['mean'])}|{n(fx['RAND20_minus_exact3']['mean'])}|")
    lines+=['','## 对解释的约束','',
            '- k1时三种聚合相同；first下增加k不改变实际动作。mean→max改变了利用标签选择候选的权利，不能写成免费增加的线上能力。',
            '- B/G是指定参考路径下的算术分解，不是独立可操纵的两个因果机制，不报告无依据的“主因占比”。',
            '- 相对随机优势和相对原query净收益分别报告，不能用前者代替后者；ENTR与PROB差别需直接比较，不能仅比较显著性。',
            '- 关联组bootstrap10000次、查询加权；数据为已消费开发/源侧。旧H确认、正式配置及risk_certified=false保持。','',
            '完整机器结果RESULTS.json；FACTORIAL.csv有108个含first锚点的因素单元，主mean/max单元72个；CONTRASTS.csv含固定聚合的数量变化和固定数量的聚合变化；DECOMPOSITION.csv为24项预设分量。独立Fraction/CDF重放见AUDIT.json，验收范围见FINAL_ACCEPTANCE.json。','']
    csv_write('FACTORIAL.csv',cells)
    csv_write('CONTRASTS.csv',effects)
    csv_write('DECOMPOSITION.csv',components)
    with (OUT/'READOUT.md').open('x',encoding='utf-8') as handle:
        handle.write('\n'.join(lines))
    print(json.dumps(dict(factorial_cells=len(cells),contrasts=len(effects),planned_components=len(components))))


if __name__=='__main__':
    main()
