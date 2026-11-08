"""Deliver all prespecified results; preserve every earlier comparison family."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .common import OUT, PRIMARY, ROOT, WEIGHTS, now, read, record, sha, verify_design, verify_seal, write


def interval(v):return f'[{v[0]:+.6f}, {v[1]:+.6f}]'


def deliver():
    d=verify_design(private=True);verify_seal('CANDIDATES_SEALED.json');verify_seal('RANKINGS_SEALED.json')
    r=read(OUT/'RESULTS.json');plan=read(OUT/'PRECISION_PLAN.json');eligibility=read(OUT/'ELIGIBILITY.json')
    for name in ('TEST_RESULTS.json','PREOPEN_AUDIT.json','NUMERICAL_AUDIT.json','MODEL_AUDIT.json'):
        if not read(OUT/name)['passed']:raise ValueError('Required acceptance failed: '+name)
    primary=r['primary']
    export=[]
    for name in PRIMARY:
        x=primary[name];view,component=name.split('::')
        export.append({'comparison':name,'weight_condition':view,'component':component,'queries':r['queries'],'groups':r['groups'],
            'mean':x['mean'],'family6_t_lower':x['family6_ci_t'][0],'family6_t_upper':x['family6_ci_t'][1],
            'family6_bootstrap_lower':x['family6_ci_bootstrap'][0],'family6_bootstrap_upper':x['family6_ci_bootstrap'][1],
            'direction':x['direction'],'groups_with_nonzero':x['groups_with_nonzero']})
    with (OUT/'PRIMARY_RESULTS.csv').open('x',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
    curves=[]
    for arm in WEIGHTS:
        for policy in ('ENTR','PROB','RAND_exact'):
            for k in (1,2,3):
                for agg in ('mean','max'):
                    x=r['views'][arm]['values'][f'{policy}_k{k}_{agg}']
                    curves.append({'weight':arm,'selector':policy,'selected_count':k,'aggregation':agg,'mean':x['mean'],
                        'ci95_t_lower':x['ci95_t'][0],'ci95_t_upper':x['ci95_t'][1],
                        'ci95_bootstrap_lower':x['ci95_bootstrap'][0],'ci95_bootstrap_upper':x['ci95_bootstrap'][1]})
    with (OUT/'COMPLETE_CURVES.csv').open('x',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    lines=['# 两套T5权重下B/G新样本配对确认','',
        f'独立新查询/连接组：{r["queries"]:,}；固定代码语料：{r["corpus_documents"]:,}。',
        '本实验的资格、样本量、模型、完整生成/检索配置和六项统计族在新生成与效果前本地封存。不是外部预注册。',
        '', '## 全部六项事前比较','',
        '|比较|点估计|六项校正t区间|六项校正组bootstrap区间|预定方向判定|非零组|',
        '|---|---:|---|---|---|---:|']
    chinese={'positive':'正','negative':'负','unresolved':'未决'}
    for name in PRIMARY:
        x=primary[name];lines.append(f'|{name}|{x["mean"]:+.6f}|{interval(x["family6_ci_t"])}|{interval(x["family6_ci_bootstrap"])}|{chinese[x["direction"]]}|{x["groups_with_nonzero"]:,}|')
    directions=[primary[f'{w}::ENTR_{c}']['direction'] for w in WEIGHTS for c in ('B','G')]
    if all(v=='positive' for v in directions):
        interpretation='四个ENTR分量均通过预定的六项校正和双区间方向规则：在这批新StaQC查询、固定CodeBERT和指定完整生成契约下，两套T5权重的B/G正方向得到配对确认。这直接增强第二贡献的权重条件证据。'
    else:
        resolved=sum(v=='positive' for v in directions)
        interpretation=f'四个ENTR分量中{resolved}项确认正方向；未满足“四项均正”的事前规则，因此不能声称两套权重下B/G方向整体稳定延续。各分量及配对权重差保留表中完整结果；该实验把权重条件的成立范围具体化，不能用合并均值替代未决或反向项。'
    lines+=['',interpretation,'',
       '配对权重差为逐查询base减local。显著差值表示完整权重条件的差异，不隔离CQC训练本身；未决差值不表示等效。本实验没有设定或检验效果等效界。',
       '', '## 比较对象和效应量','',
       'B=S(1,mean)−S(3,mean)，G=S(3,mean)−S(3,best3)，S为ENTR相对本权重自身完整位置池精确随机的RR@100优势。正值表示沿指定契约步骤相对随机的优势减弱；不是自动执行的收益，不是生成调用预算，也不是唯一因果归因。',
       '', '|权重|S(1,mean)|S(3,mean)|S(3,best3)|NoRewrite MRR@100|','|---|---:|---:|---:|---:|']
    for w in WEIGHTS:
        v=r['views'][w];lines.append('|'+w+'|'+'|'.join(f'{v["contrasts"][key]["mean"]:+.6f}' for key in ('S_ENTR_k1_mean','S_ENTR_k3_mean','S_ENTR_k3_max'))+f'|{v["values"]["NoRewrite"]["mean"]:.6f}|')
    lines+=['','PROB、ENTR−PROB及完整k1/2/3×mean/best-k作为事前指定描述性结果，见COMPLETE_CURVES.csv和RESULTS.json；未追加确认性拒绝声明。',
       '', '## 资格与规划','',
       f'- 原历史参考排除后有84,618个StaQC单目标连接组；再排除旧完整语料、查询、历史生成及规范化暴露，剩{eligibility["remaining_groups"]:,}个合格组件。每组件固定哈希取一条、固定哈希选查询和干扰语料；新旧完整语料组件无交集。',
       f'- 只用已消费两域匹配缓存及较大local结果规划：.005 RR分辨率、每方向95%功效目标；25,600组下四方向联合规划下界为{plan["four_directions_joint_power_lower_bound"]:.4f}。这是方差可迁移假设下对.005效应的规划，不保证更小真实效应显著。',
       '- 两项权重差的校正规划半宽不超过.005；此处.005为测量精度目标，不是等效容忍界。无中途看效果停样、无加种子、无模型拟合或政策升级。',
       '- accepted single-code标题/代码是一种固定检索条件，其他代码未被全面人工判为不相关。未知基础模型预训练暴露、语义克隆和未记录历史暴露不在本地资格保证内。',
       '', '## 与既有证据的关系','',
       '旧每域512条匹配分析及其24项族保持原样；旧StaQC 13,312组/16,384代码的两项族同样不变。这批25,600组是另一个新组件样本和新固定语料，六项族不回溯改变原结论。新结果不证明跨模型家族、跨所有检索器或历史全量训练归因。',
       '', '## 验收与复核','',
       '14项契约测试；两权重旧查询360位置span及完整ENTR/PROB排序复现；全查询/语料来源、组独立性、旧暴露、候选/动作/排名覆盖审查；独立有理数/CDF公式重算完整B/G；六项t和20,000次共同组bootstrap独立数值复放；32查询和32代码实际检索器重载与完整语料排名复放。',
       '', '核验文件：DESIGN.json、ELIGIBILITY.json、PRECISION_PLAN.json、PREOPEN_AUDIT.json、NUMERICAL_AUDIT.json、MODEL_AUDIT.json、FINAL_ACCEPTANCE.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    english=['# Prospective paired T5-weight confirmation of B/G','',
      f'A separately frozen study uses {r["queries"]:,} previously unused connected StaQC components and a fixed {r["corpus_documents"]:,}-snippet corpus. Both T5 weight conditions are paired on queries, targets, corpus and the fixed CodeBERT retriever, with their own complete insertion-position pools. No retraining or parameter search is performed. The old 512-query analyses and their original 24-comparison family remain unchanged.',
      '', 'The six prespecified contrasts comprise ENTR B/G under each weight and paired base-minus-local B/G. Two-sided Bonferroni family alpha is .05. A direction requires both the adjusted t and 20,000-draw paired component-bootstrap intervals to exclude zero on the same side, with at least 20 nonzero components. PROB and the complete curves are descriptive. The .005 planning resolution and difference-precision target do not define equivalence margins.',
      '', '|Contrast|Estimate|Adjusted t interval|Adjusted bootstrap interval|Decision|','|---|---:|---|---|---|']
    for name in PRIMARY:
        x=primary[name];english.append(f'|{name}|{x["mean"]:+.6f}|{interval(x["family6_ci_t"])}|{interval(x["family6_ci_bootstrap"])}|{x["direction"]}|')
    english+=['',
      ('All four within-weight ENTR B/G components meet the prespecified positive-direction rule, confirming the direction in both tested T5 weight conditions on this new StaQC setting.' if r['both_weights_BG_positive'] else 'The four-component positive-direction criterion is not met. Individual positive, negative and unresolved components retain their separate interpretations; an overall weight-stability claim is not established.'),
      '', 'The paired differences compare complete weight conditions, not an isolated effect of CQC training. Unresolved differences do not establish equivalence. The scope is the specified StaQC title/snippet setting, fixed CodeBERT and two weights within T5-base; it does not establish cross-family universality, automatic-policy benefit or historical training attribution.',
      '', 'Sampling excludes all prior StaQC query and corpus components, archived partial generations, normalized exposed text and inherited training/reference overlaps. The outcome-blind precision plan used consumed development data only; the four-direction joint power lower bound at .005 is '+f'{plan["four_directions_joint_power_lower_bound"]:.4f}'+', conditional on variance transport. The protocol was locally sealed before new generation and effects; it was not externally preregistered. Source mappings, complete candidate/action/ranking coverage, exact-random arithmetic, paired intervals and representative model/ranking reloads passed the specified audits.']
    (OUT/'weight_confirmation_results.md').write_text('\n'.join(english)+'\n',encoding='utf-8')
    files=[OUT/n for n in ['DESIGN.json','ELIGIBILITY.json','PRECISION_PLAN.json','SAMPLE_PLAN.json','TEST_RESULTS.json',
        'BATCHING_DEVELOPMENT_CHECK.json','CANDIDATES_SEALED.json','RANKINGS_SEALED.json','PREOPEN_AUDIT.json',
        'OUTCOME_OPENING.json','RESULTS.json','NUMERICAL_AUDIT.json','MODEL_AUDIT.json','per_query.jsonl',
        'paired_per_query.jsonl','PRIMARY_RESULTS.csv','COMPLETE_CURVES.csv','REPORT.md','weight_confirmation_results.md','RUNTIME.json']]
    files += [Path(__file__),Path(__file__).with_name('README.md')]
    write(OUT/'FINAL_ACCEPTANCE.json',{'created_utc':now(),'passed':True,'scientific_result_positive_required':False,
        'queries':r['queries'],'groups':r['groups'],'corpus_documents':r['corpus_documents'],
        'both_weights_BG_positive':r['both_weights_BG_positive'],'primary_directions':r['primary_directions'],
        'all_original_protected_hashes_unchanged':True,'protected_count':len(d['protected_files']),
        'files':[record(f) for f in files],'new_results':True,'risk_certified':False,'no_equivalence_claim':True})
    (OUT/'RUN_STATUS.json').write_text(json.dumps({'updated_utc':now(),'stage':'complete','completed':True,'passed':True,
        'primary_directions':r['primary_directions'],'both_weights_BG_positive':r['both_weights_BG_positive']},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print({'accepted':True,'primary_directions':r['primary_directions']},flush=True)


if __name__=='__main__':deliver()
