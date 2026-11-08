from __future__ import annotations

import csv
import json

from .common import ROOT, OUT, now, read, record, sha, write
from .inference import verify_design,verify_seal,progress
from .precision import CONTRASTS


def interval(v):return '['+', '.join(f'{x:+.6f}' for x in v)+']'


def main():
    d=verify_design(private=True)
    for seal in ('CANDIDATES_SEALED.json','RANKINGS_SEALED.json'):verify_seal(seal)
    for name in ('PREOPEN_AUDIT.json','EFFECT_AUDIT.json','MODEL_AUDIT.json'):
        if not read(OUT/name)['passed']:raise ValueError('Acceptance missing: '+name)
    r=read(OUT/'RESULTS.json');plan=read(OUT/'SAMPLE_PLAN.json');elig=read(OUT/'STAQC_ELIGIBILITY.json')
    export=[]
    for ret in ('CodeBERT','BM25'):
        for c in CONTRASTS:
            v=r['views'][ret]['summaries'][c]
            export.append({'retriever':ret,'contrast':c,**{k:json.dumps(val) if isinstance(val,list) else val for k,val in v.items()}})
    fields=list(dict.fromkeys(k for x in export for k in x))
    with (OUT/'PRIMARY_AND_SECONDARY.csv').open('x',encoding='utf-8',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(export)
    lines=['# 新样本资格、功效与B/G冻结确认','',
       f'完成时间：{now()}。本轮是新StaQC单代码答案场景的预先冻结确认，不是重用CoSQA的确认。原H比较及正式组件保持。','',
       '## 样本与精度','',
       '- 原始CodeSearchNet valid/test经严格已知重叠排除后420查询/103关联组，未通过0.005分辨率规划，因此未计算其新方法效果。',
       f'- StaQC作者原件固定commit；单代码答案子集审查后{elig["eligible_primary_pairs"]}条/{elig["groups"]}关联组。',
       f'- 按哈希抽取每关联组一条，共{d["queries"]}查询/{d["groups"]}组；固定{d["corpus_documents"]}代码片段语料，额外语料来自其他独立关联组。',
       '- 预开效果前独立复核发现最初规划的组内多查询方差与新抽样不匹配；已撤销该功效数值，保留初版封存与无标签候选，改用匹配单查询方差/敏感性上界后重新封存。最终功效及样本以本表为准。',
       '- 两个主比较：每项双侧alpha=.025，目标边际功效90%；在方差可迁移假设下，两项同时检出的功效下界80%。不是用新样本效果反选规模。','',
       '|主比较|所需组数（0.005）|实际规划功效|估计|校正cluster-t区间|校正组bootstrap区间|判断|',
       '|---|---:|---:|---:|---|---|---|']
    for c in CONTRASTS:
        v=r['views']['CodeBERT']['summaries'][c];p=plan['exact_sample_precision']['contrasts'][c]
        lines.append(f'|{c}|{p["required_groups_005"]}|{p["planned_power_at_005"]:.3%}|{v["mean"]:+.8f}|{interval(v["family2_ci95_cluster_t"])}|{interval(v["family2_ci95_group_bootstrap"])}|{v["direction"]}|')
    lines += ['', '## 次比较与实际采纳参照','', '|检索器|ENTR Top1−原句|ENTR best3−原句|ENTR Top1−精确随机|ENTR best3−精确随机|', '|---|---:|---:|---:|---:|']
    for ret in ('CodeBERT','BM25'):
        s=r['views'][ret]['summaries'];keys=['delta_ENTR_k1_mean','delta_ENTR_k3_max','S_ENTR_k1_mean','S_ENTR_k3_max']
        lines.append('|'+ret+'|'+'|'.join(f'{s[k]["mean"]:+.8f}' for k in keys)+'|')
    lines += ['', '次比较只作描述估计，PROB、ENTR−PROB、BM25和直接检索器差区间见RESULTS.json。best3使用相关标签，不是可执行采用收益。', '',
       '## 对论文贡献的作用与范围','',
       '这次直接检验位置选择相对优势在候选数量及聚合契约改变时的变化。判断按两个预定分量的校正区间，而不是只看Top1负/best3正。',
       '两个分量若均正，支持两项发现迁移到此新场景；只有一项为正，仅确认该分量；跨零按未决报告。不能以事后新解释改变确认主比较。',
       'StaQC是经过挖掘的Stack Overflow问题标题和accepted-answer代码片段，单目标RR不同于完整函数语料及Bing查询。它补充新场景证据，不替代原CoSQA冻结确认，也不证明真实查询来源因果、熵独有机制或新UQ有效。',
       '审查覆盖已知训练/开发/确认、原始与规范化查询/代码/URL/函数/仓库。基础模型预训练暴露及语义克隆无法完全排除。单代码来源只有一个已知正例，不把其他未标注片段称为人工负例。',
       '原始标签在来源审查时可见；确认冻结的是新方法预测和效果。仅本地时间戳/哈希封存，external_preregistration=false。', '',
       '## 复核','',
       '- 预开效果前全量核对查询、源答案映射、关联组、已知重叠、候选位置、选择与导出排名结构。',
       '- 独立Fraction RR、另一CDF组合公式及校正区间复算；真实T5再生成/FP64熵和CodeBERT向量抽查。具体数量和误差见各审计收据，不把抽查写成全量模型复算。',
       '- 旧正式指针、原H确认、两个开发因素/训练对照验收哈希保持。risk_certified=false，无新训练、扩seed或策略调参。','']
    with (OUT/'FINDINGS.md').open('x',encoding='utf-8') as h:h.write('\n'.join(lines))
    artifacts=[OUT/n for n in ('DESIGN.json','SAMPLE_PLAN.json','PRECISION_PLAN.json','LOCAL_ELIGIBILITY.json','STAQC_ELIGIBILITY.json',
          'OUTCOME_OPENING.json','RESULTS.json','per_query.jsonl','PRIMARY_AND_SECONDARY.csv','FINDINGS.md',
          'PREOPEN_AUDIT.json','EFFECT_AUDIT.json','MODEL_AUDIT.json','TEST_RESULTS.json')]
    write(OUT/'FINAL_ACCEPTANCE.json',{'created_utc':now(),'passed':True,'new_setting_confirmation':True,
          'CoSQA_new_confirmation':False,'external_preregistration':False,'risk_certified':False,
          'queries':d['queries'],'groups':d['groups'],'corpus_documents':d['corpus_documents'],
          'primary_directions':r['primary_directions'],'files':[record(p) for p in artifacts],
          'protected_files':d['protected_files'],'original_confirmation_unchanged':True,'official_model_promotion':False})
    progress('complete',completed=True,passed=True,primary_directions=r['primary_directions'])


if __name__=='__main__':main()
