"""Supplementary readout and figure from sealed confirmation outputs."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rene_factor_confirmation_20261020'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
    accepted=read(OUT/'FINAL_ACCEPTANCE.json')
    if not accepted['passed']:raise ValueError('Final confirmation not accepted')
    r=read(OUT/'RESULTS.json');sample=read(OUT/'SAMPLE_PLAN.json')
    dev=read(ROOT/'results/rene_budget_aggregation_20261020/RESULTS.json')
    dv=next(v for v in dev['views'] if v['domain']=='cosqa' and v['retriever']=='CodeBERT')
    contrasts=['ENTR_B_budget_mean_1_to_3','ENTR_G_aggregation_at3_mean_to_max']
    # The two datasets have different corpus/query sources; no cross-dataset causal
    # comparison or separate-significance inference is represented by this plot.
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    colours=['#58647c','#148477']
    for ax,c,title in zip(axes,contrasts,['B: candidate count at fixed mean','G: aggregation at fixed k=3']):
        old=dv['primary_decomposition'][c]
        new=r['views']['CodeBERT']['summaries'][c]
        means=[old['mean'],new['mean']]
        cis=[old['family24_ci95_cluster_t'],new['family2_ci95_cluster_t']]
        for i,(mu,ci) in enumerate(zip(means,cis)):
            ax.errorbar(mu,i,xerr=np.array([[mu-ci[0]],[ci[1]-mu]]),fmt='o',color=colours[i],capsize=4,ms=6,lw=1.8)
        ax.axvline(0,color='#888888',lw=.9,ls='--')
        ax.set_yticks([0,1],['CoSQA development\n24-comparison correction','StaQC frozen confirmation\n2-comparison correction'])
        ax.invert_yaxis();ax.set_ylim(1.6,-.6);ax.set_title(title,fontsize=11)
        ax.set_xlabel('ENTR advantage attenuation (MRR)',fontsize=9)
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'factor_confirmation.png',dpi=180)
    fig.savefig(OUT/'factor_confirmation.pdf')
    plt.close(fig)
    lines=['# 新场景预先冻结确认补充结果','',
           f'在已知开发发现之后，我们对独立来源的StaQC Python单代码答案场景预先固定B/G比较。查询与代码精确/规范化重叠、源问题和重复目标关联经审查；按哈希每组一题，{r["queries"]}题/{r["groups"]}组，固定{r["corpus_documents"]}代码片段。',
           '保留同一CQC T5与SSQR pooler CodeBERT；一次生成全位置，同池ENTR/PROB/精确随机，k=1/2/3和mean/best-k。两个主比较使用Bonferroni校正cluster-t和组bootstrap区间；PROB/BM25为预定描述性对照。',
           '这里k指从同一固定生成池中选入聚合的候选数，不等同于生成次数或生成成本；mean是所选候选RR的平均，best-k是使用相关标签取其中最高RR，均不额外包含原查询安全臂。',
           '规划曾在无标签候选生成阶段发现抽样方差匹配问题，并在开新效果前撤销初版、留存修订；最终样本基于旧开发匹配单query/边际方差上界重新封存。外部预注册未实施。','',
           '|主分量|估计|校正cluster-t区间|校正bootstrap区间|预设判断|','|---|---:|---|---|---|']
    for c in contrasts:
        v=r['views']['CodeBERT']['summaries'][c]
        fmt=lambda a:'['+', '.join(f'{x:+.6f}' for x in a)+']'
        lines.append(f'|{c}|{v["mean"]:+.8f}|{fmt(v["family2_ci95_cluster_t"])}|{fmt(v["family2_ci95_group_bootstrap"])}|{v["direction"]}|')
    lines += ['', '以下为预定描述性参照；S表示ENTR相对同池精确随机的位置选择优势。', '',
              '|检索器|S(1,mean)|S(3,mean)|S(3,best3)|Top1相对原查询|best3相对原查询|',
              '|---|---:|---:|---:|---:|---:|']
    for ret in ('CodeBERT','BM25'):
        summaries=r['views'][ret]['summaries']
        keys=['S_ENTR_k1_mean','S_ENTR_k3_mean','S_ENTR_k3_max','delta_ENTR_k1_mean','delta_ENTR_k3_max']
        lines.append('|'+ret+'|'+'|'.join(f'{summaries[k]["mean"]:+.8f}' for k in keys)+'|')
    directions=r['primary_directions']
    labels={contrasts[0]:'固定mean的候选数分量B',contrasts[1]:'固定3候选的聚合分量G'}
    readings={'positive':'校正的两种区间均位于零以上，确认相对优势减弱',
              'negative':'校正的两种区间均位于零以下，确认与开发预期相反的方向',
              'unresolved':'未满足预设方向确认条件，按未决报告'}
    lines += ['', '；'.join(labels[c]+'：'+readings[directions[c]] for c in contrasts)+'。',
              '该研究扩展到经过挖掘的SO标题/代码片段场景，不把它写成CoSQA再次独立确认；best-k保持标签辅助机会指标身份，不能据此称在线采纳正收益或新UQ有效。',
              '复核包括作者Git对象/资格关联组、旧开发功效、全量Fraction RR及CDF随机期望与校正区间、真实T5FP64熵/再生成和CodeBERT向量抽查。完整来源与验收：`results/rene_factor_confirmation_20261020/`。',
              '图中CoSQA开发与StaQC确认使用各自预定比较族校正，不凭两个独立区间是否跨零推断数据源之间的效应差异。','']
    path=ROOT/'paper/rene2027/factor_confirmation_results.md'
    with path.open('x',encoding='utf-8') as h:h.write('\n'.join(lines))
    print({'figure':str(OUT/'factor_confirmation.png'),'manuscript_supplement':str(path)},flush=True)


if __name__=='__main__':main()
