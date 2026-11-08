"""Generate traceable manuscript tables from audited bounded-training results."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rene_cqc_training_fidelity_20261020'


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))


def fmt(x):return f'{x:+.8f}'


def ci(bounds):return '['+', '.join(fmt(x) for x in bounds)+']'


def save(path,lines):
    with path.open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines)+'\n')


def main():
    r=read('RESULTS.json');audit=read('EFFECT_AUDIT.json');models=read('MODEL_AUDIT.json')
    assert audit['passed'] and models['passed'] and read('TRAIN_AUDIT.json')['passed']
    views={(x['domain'],x['view'],x['arm']):x for x in r['views']}
    changes={(x['domain'],x['arm'],x['metric']):x for x in r['primary_changes']}
    names={'L':'本地参照','O':'仅换Adafactor','S':'仅换linear100000','N':'仅改长度128','W':'仅改weight decay为0',
        'P':'公开超参数组合','PN':'组合+进度缩放调度','PS':'公开组合+训练seed42'}
    positive=[];negative=[];zeros=[];unresolved=[]
    for row in r['primary_changes']:
        b=row['family28_ci95_group_bootstrap'];t=row['family28_ci95_cluster_t']
        (zeros if b==t==[0.,0.] else positive if b[0]>0 and t[0]>0 else negative if b[1]<0 and t[1]<0 else unresolved).append(row)
    assert (len(positive),len(negative),len(zeros),len(unresolved))==(0,1,4,23)
    absolute=['|域|分支|ENTR Top1 Δ原query|ENTR best3 Δ原query|ENTR best3−精确随机|','|---|---|---:|---:|---:|']
    for domain in ('source','cosqa'):
        for arm in names:
            e=views[domain,'own',arm]['effects']
            absolute.append('|'+domain+'|'+arm+'|'+ '|'.join(fmt(e[k]['mean']) for k in ('top1_gain','best3_gain','S3'))+'|')
    table=['|配置−L|source：S3变化|CoSQA：S3变化|source：Top1净收益变化|CoSQA：Top1净收益变化|',
        '|---|---:|---:|---:|---:|']
    for arm in list(names)[1:]:
        values=[changes[d,arm,m]['mean'] for m in ('S3','top1_gain') for d in ('source','cosqa')]
        table.append('|'+arm+' '+names[arm]+'|'+ '|'.join(fmt(x) for x in values)+'|')
    bundle=['|域|指标变化 P−L|配对估计|28项校正bootstrap区间|28项校正cluster-t区间|',
        '|---|---|---:|---|---|']
    for domain in ('source','cosqa'):
        for metric in ('S3','top1_gain'):
            x=changes[domain,'P',metric]
            bundle.append('|'+domain+'|'+metric+'|'+fmt(x['mean'])+'|'+ci(x['family28_ci95_group_bootstrap'])+'|'+ci(x['family28_ci95_cluster_t'])+'|')
    fixed=['|域|固定候选P−L的S3变化|固定候选P−L的Top1净收益变化|','|---|---:|---:|']
    for domain in ('source','cosqa'):
        a={x['metric']:x for x in r['descriptive_changes'] if x['domain']==domain and x['view']=='fixed' and x['arm']=='P' and x['reference']=='L'}
        fixed.append('|'+domain+'|'+fmt(a['S3']['mean'])+'|'+fmt(a['top1_gain']['mean'])+'|')
    extra=['|视角|域|PN−P指标|配对估计|描述性95%组bootstrap区间|','|---|---|---|---:|---|']
    for row in read('EXTRA_CONTRASTS.json')['contrasts']:
        if row['contrast']=='progress_scaled_minus_literal':
            extra.append('|'+row['view']+'|'+row['domain']+'|'+row['metric']+'|'+fmt(row['mean'])+'|'+ci(row['ci95_group_bootstrap'])+'|')
    seed=['|域|PS−P指标|配对估计|描述性95%组bootstrap区间|','|---|---|---:|---|']
    for x in r['descriptive_changes']:
        if x['reference']=='P' and x['view']=='own':
            seed.append('|'+x['domain']+'|'+x['metric']+'|'+fmt(x['mean'])+'|'+ci(x['ci95_group_bootstrap'])+'|')
    lines=['# CQC有限训练配置影响：结果与判断','',
        '已完成8组等数据/等更新训练、两域两视角评测与三层独立审计。结论是：**长度上限在本轮产生严格零效应；优化器/调度/训练种子会明显改变候选与选择，但公开超参数组合没有建立对本地参照的校正后正改善，也没有消除本轮“best3有机会、Top1净收益点估计为负”的现象。**','',
        '## 共同控制','',
        '全部从同一原始t5-base开始，固定32,768条CODEnn训练行、连续15%掩码及顺序、1,024更新、batch8×4、lr起点.001、BF16计算/FP32权重、末步checkpoint，无早停。验证NLL不用于选权重。训练合计约97.7分钟，不含生成/检索/审计。配置详见METHODS.md及冻结DESIGN.md。','',
        'source与CoSQA各512查询/512原组；完整19210/6267语料，同一冻结CodeBERT。查询沿用旧桥接样本，不是新确认。每组9340个位置候选；两视角、八组共149440条候选RR和16384条query记录。','',
        'L=AdamW/constant/512/wd.01/seed101；O仅Adafactor；S仅字面公开linear100000；N仅长度128；W仅wd0；P=Adafactor/linear100000/128/wd0/seed101；PN仅将P的调度horizon按历史进度缩放为1093；PS仅将P的训练seed改42。Adafactor的relative_step与scale_parameter均False。','',
        '## 1. 各训练因素对检索指标改变多少','',
        'S3=ENTR best3−精确随机best3；Top1净收益=ENTR Top1−NoRewrite。下表是同查询直接配对“配置−L”的变化，不是与历史全量模型相减。','',*table,'',
        '**28项预设比较族中，0项校正后正、1项校正后负、4项长度结构零、23项区间跨零。**唯一校正负比较为source/PS−L的Top1净收益：−.01611474，bootstrap区间[−.03271290,−.00059109]，t区间[−.03221445,−.00001503]。PS−L同时改变了组合配置和seed，不能把这一差值全部归给seed。','',
        '其余跨零比较不建立无影响或等效。比如公开组合的source Top1变化为−.01179079，幅度不能忽略，而校正区间仍不足以精确确定其大小和方向。完整28行及两种区间见READOUT.md/PRIMARY.csv。','',
        '### 公开超参数组合的直接答案','',*bundle,'',
        '在相同有限预算下，P相对L没有恢复经校正建立的ENTR−随机优势，也没有建立Top1净收益改善；这不是“优化器和调度不重要”的等效结论。','',
        '## 2. 长度上限：可定位的严格零效应','',
        '共享训练bank的输入最长78、标签最长26；验证55/14，没有128截断。L/N的1024步loss和梯度、真实权重文件哈希完全相同；两域9340个新生成候选逐字节一致，固定池评分、选择、RR及全部效果相同。因此本轮长度差异的训练与评测效应精确为0。','',
        '完整百万条辅助扫描在源顺序/seed101、每条一次固定掩码下有22条输入>128（0.0022%），标签0条>128；max306/54。这不是历史三轮全部动态掩码，不能把本轮的零效应外推为全部历史训练的零效应。','',
        '## 3. 候选生成和同池评分都会受配置影响','',
        '公开组合相对L改变source 4413/5995=73.61%、CoSQA 1898/3345=56.74%的生成文本。在完全固定历史候选文本/排名时，ENTR Top1位置仍分别改变195/512和186/512，Top3集合改变308/512和253/512。该结果来自无标签动作审计ACTION_DIAGNOSTIC.json。','',
        '仅重新评分历史候选后的配对变化：','',*fixed,'',
        '四个P−L固定池区间均跨零。该视角与新生成视角回答不同问题，不能把二者之差称为唯一的“生成效应”因果分量。固定池及PROB/精确随机的完整描述区间见DESCRIPTIVE.csv。','',
        '## 4. 调度进度与训练种子','',
        '公开源码hard-coded horizon100000；不能在1024步预算内直接衰减到零后称作者设置。PN显式使用1093，P仍使用100000。PN末步验证NLL 1.14316240，P为1.21728411；训练目标改善需要另看检索效用。','',*extra,'',
        'PN−P在各自新生成池的两个域、两个主指标均跨零。固定历史source池的S3和Top1差为正且点式95%区间在零以上；CoSQA未建立相应改善。这8项是事先声明的描述比较，不纳入28项主族，不选PN替换正式模型。','',
        '单独改变训练随机种子（PS−P），数据、掩码、顺序固定：','',*seed,'',
        'CoSQA Top1的描述区间为负，提示当前单次训练对采纳效用的敏感性。该检查只有101/42这一对，不估计训练seed总体方差，也不从一个seed表现选最优权重。','',
        '组合效应减去四单项效应总和的非可加残差也按开效果前的协议计算；own视角4项均跨零，未单独识别某个因素交互。完整16项在EXTRA_CONTRASTS.json，交付验收另用独立带符号系数复算。','',
        '## 5. 对论文主线的作用','',*absolute,'',
        '八配置×两域中，Top1净收益点估计全部为负，best3机会收益点估计全部为正；这是同一受控设计的16格敏感性结果，不是16次独立确认。部分Top1的95%区间跨零，不写为16格均显著有害。best3使用真实相关标签，是机会评价，不是线上可执行增益。','',
        '本轮为RENE增加“训练设置受控敏感性”证据，关闭有限配置对照这一实验缺项。最明确排除的是本轮长度截断解释；其他配置的影响得到配对量化。当前没有证据表明换成公开表内超参数即可把所测自动单选变为正收益，也不能把复现差异归因给CoSQA查询来源。','',
        '**历史93750步训练与作者不可得checkpoint之间的全部差异仍未被本轮识别。**P共同保留本地训练数据、单span掩码、batch/精度和末步选择，不能称完整公开训练器重现；本轮1024步未覆盖长程训练动态。论文采用“固定有限预算下的设置影响”表述，不给历史差异解释百分比。','',
        '## 6. 独立复核与交付','',
        '- 8项科学测试通过；训练审计重新推导32768训练例、8192次更新及逐步lr，8权重哈希、L/N负对照通过。',
        '- 独立Fraction/CDF重算149440候选RR、311296查询指标字段、2432汇总统计字段和648配对字段；最大统计差4.997e−16。45305条文本排名检查完成。',
        '- 新增34764条检索文本的544003702个持久化得分对应Top100顺序逐行复核；不是独立重跑了所有点积数值。',
        '- 实际T5检查192例，其中96例独立greedy再生成；FP64熵最大误差9.618e−7、log概率8.843e−7。4批64条CodeBERT真实向量复算最大误差0。',
        '- 三层审计TRAIN_AUDIT/EFFECT_AUDIT/MODEL_AUDIT通过；额外对照、CSV字段、所有保护哈希及最终交付见FINAL_ACCEPTANCE.json。','',
        '结果文件：RESULTS.json、PRIMARY.csv、DESCRIPTIVE.csv、per_query.jsonl、candidate_RR.jsonl。图：cqc_configuration_effects.png/pdf。新权重与dense得分：external_inputs/cqc_training_model_store/。','',
        '论文补充稿：paper/rene2027/cqc_training_fidelity_results.md。旧论文源稿、历史T5/CodeBERT、正式五指针与原H确认保持；new_confirmation=false，risk_certified=false。']
    # Check narrative counts against sealed, label-free action diagnostics.
    actions=read('ACTION_DIAGNOSTIC.json')['records']
    for domain,changed in [('source',4413),('cosqa',1898)]:
        assert next(x for x in actions if x['domain']==domain and x['view']=='own' and x['arm']=='P')['changed_rewrite_positions']==changed
    save(OUT/'FINDINGS.md',lines)
    english=['# Bounded CQC training-configuration sensitivity','',
        'Supplementary manuscript material, 2026-10-20. The reconstruction is sensitive at the candidate and selection levels, while changing the tested optimization settings does not establish a positive family-adjusted improvement over the budget-matched local configuration. The length limit is an exact negative control in this experiment.','',
        '## Controlled design','',
        'We train eight CQC variants from the same original T5-base initialization on the same 32,768 CODEnn rows, one fixed contiguous 15% corruption per row, and the same order. Each variant receives 1,024 updates with microbatch 8 and accumulation 4, initial learning rate 1e-3, gradient clipping at 1, BF16 computation and FP32 parameters. The final update is always evaluated; validation NLL is descriptive and neither early stopping nor outcome-based checkpoint selection is used. The 32,768 training rows contain 26,155 distinct texts.','',
        '|Arm|Optimizer|Schedule horizon|Length cap|Weight decay|Training seed|',
        '|---|---|---|---:|---:|---:|',
        '|L|AdamW|constant|512|.01|101|','|O|Adafactor|constant|512|.01|101|',
        '|S|AdamW|linear100000|512|.01|101|','|N|AdamW|constant|128|.01|101|',
        '|W|AdamW|constant|512|0|101|','|P|Adafactor|linear100000|128|0|101|',
        '|PN|Adafactor|linear1093|128|0|101|','|PS|Adafactor|linear100000|128|0|42|','',
        'Adafactor uses explicit learning rates with relative_step=False and scale_parameter=False. Public code hard-codes a 100,000-step linear horizon and overrides the configured seed 101 with set_seed(42). We preserve the literal horizon in S/P/PS. PN is a separately named progress-scaled sensitivity, using ceil(1024*100000/93750)=1093. It does not stand for the literal public schedule. P aligns the tabulated hyperparameters while retaining the shared local corpus, single-span corruption, batch budget and precision.','',
        'We retain the existing bridge samples: 512 source queries in 512 repositories and 512 consumed CoSQA development queries in 512 connected groups, searching the full 19,210 and 6,267-document corpora with the same frozen CodeBERT pooler. For each weight, we evaluate both its own generated candidates and rescoring the same historical local candidates. Inference uses the same positions, greedy 19-new-token decoding, and FP32 full-vocabulary teacher-forcing entropy. Probability scoring and exact uniform position sampling remain controls. Repeated texts at different positions retain their probability mass.','',
        'The main paired outcomes are the change relative to L in S3 (ENTR best3 minus exact RAND best3) and in ENTR Top1 minus NoRewrite. Seven non-L arms, two domains and two outcomes form 28 predefined comparisons. We report 10,000 original-group bootstrap and cluster-t intervals with Bonferroni adjustment. Intervals condition on the trained weights, not a population of training seeds. Fixed-pool, probability, direct seed and progress-scaling contrasts are descriptive. No fresh confirmation is claimed.','',
        '## Main results','',
        '|Arm minus L|Source: change in S3|CoSQA: change in S3|Source: change in Top1 gain|CoSQA: change in Top1 gain|',
        '|---|---:|---:|---:|---:|']
    for arm in list(names)[1:]:
        english.append('|'+arm+'|'+ '|'.join(fmt(changes[d,arm,m]['mean']) for m in ('S3','top1_gain') for d in ('source','cosqa'))+'|')
    english+=['',
        'For the public hyperparameter bundle P minus L, source S3 changes by -0.00114332 (family-adjusted bootstrap interval [-0.01347887, +0.01066236]) and CoSQA S3 by -0.00396651 ([-0.02318952, +0.01510376]). Top1 gain changes by -0.01179079 on source ([-0.02949673, +0.00379373]) and +0.00148675 on CoSQA ([-0.01792189, +0.02302523]). These intervals do not establish equivalence or negligible configuration effects.','',
        'No primary contrast has positive adjusted intervals in both statistical methods. One contrast, source PS minus L in Top1 gain, is negative: -0.01611474, adjusted bootstrap [-0.03271290, -0.00059109] and cluster-t [-0.03221445, -0.00001503]. Four length-control contrasts are exactly zero; 23 other intervals cross zero. PS minus L changes the public bundle and seed together and cannot identify a seed-only effect. In the direct descriptive PS minus P comparison, CoSQA Top1 changes by -0.01575799 (pointwise bootstrap [-0.03027118, -0.00147269]); this single seed pair does not estimate seed-population variance.','',
        '## Candidate and selection sensitivity','',
        'The shared training inputs and labels never exceed 78 and 26 tokens. Caps 128 and 512 therefore produce identical tensors: L and N have identical loss/gradient trajectories, weight-file hashes, generated candidates, scores, choices and retrieval outcomes. The separate one-mask-per-row scan of one million comments finds 22 inputs above 128 and no labels above 128; it is not an audit of every historical epoch mask.','',
        'The bundle P changes 73.61% of generated source position texts and 56.74% of CoSQA texts relative to L. Even with historical candidate texts and rankings held fixed, its ENTR Top1 position changes on 195/512 source and 186/512 CoSQA queries. Fixed-pool P minus L changes S3 by -0.00222529/+0.00253087 and Top1 gain by -0.00505630/-0.00671564 (source/CoSQA); all four descriptive intervals cross zero. Differences between own-pool and fixed-pool contrasts are not a unique causal decomposition of generation and selection.','',
        'Progress scaling improves CQC validation NLL from 1.21728411 to 1.14316240, but the direct PN minus P own-pool retrieval contrasts all cross zero. Source/CoSQA S3 changes are +0.00621942/+0.00676645 and Top1 changes are +0.00485258/+0.00406629. Rescoring the fixed source pool yields positive pointwise intervals for both outcomes; the corresponding CoSQA changes are unresolved. These descriptive comparisons are not used to choose a replacement model. The predefined nonadditivity contrasts likewise do not resolve an individual interaction.','',
        'Across the eight configurations and two domains, best3-minus-original point estimates remain positive and Top1-minus-original point estimates remain negative. These are 16 related sensitivity cells, not 16 independent confirmations; several Top1 intervals cross zero. Best3 uses relevance labels and measures candidate opportunity rather than a deployable policy.','',
        '## Interpretation and reproducibility','',
        'This controlled sensitivity study quantifies optimization differences as a concrete reconstruction factor and excludes length truncation for its shared bank. None of the tested changes establishes a family-adjusted positive improvement over L or removes the observed candidate-to-Top1 separation at the point-estimate level. The study does not assign a percentage of the original replication discrepancy to training: 1,024 updates cannot identify all differences between the historical 93,750-update model and an unavailable author checkpoint. Corpus bytes, corruption conventions and the full historical trainer are not restored by the hyperparameter bundle. No query-origin causal conclusion follows.','',
        'Training records were independently rederived for all 32,768 examples and 8,192 updates. Independent Fraction/CDF calculations reproduce 149,440 candidate reciprocal ranks and all summary/paired statistics (maximum statistical discrepancy below 5e-16). The audit checks 45,305 text rankings and recomputes Top100 ordering from 544,003,702 stored new retrieval scores; it does not independently repeat every numerical dot product. Actual-model checks cover 192 T5 score cases, including 96 greedy regenerations, and 64 CodeBERT query vectors. Maximum independent FP64 entropy discrepancy is below 9.62e-7.','',
        'Evidence and complete adjusted intervals: results/rene_cqc_training_fidelity_20261020/{RESULTS.json,READOUT.md,PRIMARY.csv,DESCRIPTIVE.csv,EXTRA_CONTRASTS.json,FINAL_ACCEPTANCE.json}. Models and dense scores: external_inputs/cqc_training_model_store/. Figure: cqc_configuration_effects.pdf in the result folder. The original confirmation, historical weights, existing manuscript sources and formal deployment pointers are preserved.']
    save(ROOT/'paper/rene2027/cqc_training_fidelity_results.md',english)
    print(json.dumps(dict(written=['FINDINGS.md','paper/rene2027/cqc_training_fidelity_results.md'],primary_positive=0,primary_negative=1,length_zeros=4,unresolved=23)))


if __name__=='__main__':main()
