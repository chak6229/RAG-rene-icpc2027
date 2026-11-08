"""Export frozen analyses, with denominators and training-budget scope explicit."""
from __future__ import annotations

import csv
import json

from .common import OUT,ARM_SPECS,read,write,sha,now,require


def main():
    result=read(OUT/'RESULTS.json');audit=read(OUT/'EFFECT_AUDIT.json');train=read(OUT/'TRAIN_AUDIT.json');models=read(OUT/'MODEL_AUDIT.json')
    require(audit['passed'] and train['passed'] and models['passed'],'All scientific audits must pass')
    require(audit['results_sha256']==sha(OUT/'RESULTS.json'),'Audited result digest')
    lines=['# CQC有限训练配置对照：完整结果','',
        '固定原始t5-base初始化、32768条共享CODEnn样本/固定掩码及顺序、1024更新、有效batch32、lr起点.001、固定末步权重。所有分支训练数据和更新预算相同。',
        'source512/512仓库及CoSQA开发512/512关联组，均沿用旧桥接样本、完整19210/6267语料与同一冻结CodeBERT。此轮不是新确认；未读取H。','',
        'L本地；O仅Adafactor；S仅公开字面linear100000；N仅长度128；W仅weight_decay0；P公开表内组合；PN按历史进度缩放linear1093；PS为P的seed42敏感性。',
        '所有分支保持相同本地连续15%掩码；P并非恢复作者CSV或完整训练器。PN与字面公开调度分别解释。','',
        '## 训练与长度负对照','',
        '|分支|固定初始验证NLL|1024步验证NLL|训练时长（秒，本进程）|','|---|---:|---:|---:|']
    for r in result['training']:
        lines.append(f"|{r['arm']}|{r['validation'][0]['token_nll']:.8f}|{r['validation'][-1]['token_nll']:.8f}|{r['seconds_this_process']:.1f}|")
    lines+=['','128/512的训练输入均未触发截断；L与N的1024步loss/梯度轨迹及末步权重字节完全一致，独立负对照通过。','',
        '## 相同预算下，各配置生成自己的候选','',
        'S3=ENTR best3−精确随机best3；S1=ENTR Top1−精确随机Top1。best3使用相关性标签选优，属于机会指标。','',
        '|域|分支|NoRewrite MRR|ENTR Top1 Δ原query|ENTR best3 Δ原query|S1|S3|PROB best3−精确随机|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    n=lambda v:f'{v:+.8f}'
    ci=lambda v:'['+', '.join(n(x) for x in v)+']'
    for v in result['views']:
        if v['view']!='own':continue
        f=v['effects']
        lines.append(f"|{v['domain']}|{v['arm']}|{v['MRR']['NoRewrite']['mean']:.8f}|{n(f['top1_gain']['mean'])}|{n(f['best3_gain']['mean'])}|{n(f['S1']['mean'])}|{n(f['S3']['mean'])}|{n(f['prob_S3']['mean'])}|")
    lines+=['','## 相对预算匹配L的直接配对变化','',
        '28项预设族=7个非L分支×2域×S3/Top1净收益。下表区间已作Bonferroni校正，同时保留组bootstrap与cluster-t。正值为该配置相对L提高相应指标；不是与历史93750步模型的配对配置效应。','',
        '|域|分支−L|指标变化|估计|28项校正bootstrap区间|28项校正t区间|',
        '|---|---|---|---:|---|---|']
    for r in result['primary_changes']:
        lines.append(f"|{r['domain']}|{r['arm']}−L|{r['metric']}|{n(r['mean'])}|{ci(r['family28_ci95_group_bootstrap'])}|{ci(r['family28_ci95_cluster_t'])}|")
    lines+=['','## 固定历史候选，只改变训练权重评分','',
        '该视角完全复用同一改写文本/排名。相对L的变化用于诊断选择变化；不同视角的差不解释为唯一因果分解。下表为描述性点估计，完整区间见DESCRIPTIVE.csv。','',
        '|域|分支−L|S3变化|Top1净收益变化|','|---|---|---:|---:|']
    for domain in ('source','cosqa'):
        for arm in ARM_SPECS:
            if arm=='L':continue
            entries={r['metric']:r for r in result['descriptive_changes'] if r['domain']==domain and r['view']=='fixed' and r['arm']==arm and r['reference']=='L'}
            lines.append(f"|{domain}|{arm}−L|{n(entries['S3']['mean'])}|{n(entries['top1_gain']['mean'])}|")
    lines+=['','## 解释口径与复核','',
        '- 本轮量化的是固定有限训练预算下的配置影响，无法单独识别历史93750步模型与不可得作者模型之间的全部差异。',
        '- 字面公开linear100000在1024步仅轻微衰减；PN单独检查训练进度缩放，不能将两者合并。',
        '- 区间反映固定训练实例上的查询/关联组不确定性；PS只提供一次训练随机性敏感性，不能代替多seed训练总体方差。',
        '- ENTR/PROB/精确随机同池同预算；相对随机优势与相对NoRewrite净收益分开。NLL变化不能代替检索收益。',
        '- 原H确认、正式五指针、历史权重与原结果保持；不以本轮挑选替换正式模型，risk_certified=false。','',
        '独立审计：TRAIN_AUDIT.json检查共同数据/更新/lr/负对照；EFFECT_AUDIT.json从原排名以Fraction及另一CDF公式重放并独立算区间；MODEL_AUDIT.json独立实际前向、FP64熵与greedy再生成。',
        '完整结果RESULTS.json；主变化PRIMARY.csv；其余配对DESCRIPTIVE.csv；每query与每候选证据分别为per_query.jsonl、candidate_RR.jsonl。','']
    with (OUT/'READOUT.md').open('x',encoding='utf-8') as h:h.write('\n'.join(lines))
    print(dict(stage='report_exported',primary_rows=len(result['primary_changes'])),flush=True)


if __name__=='__main__':main()
