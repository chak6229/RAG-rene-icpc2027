from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/rene_evidence_integration_20261020'
ENV=ROOT/'results/rene_envelope_completion_20261020'


def put(path,text):
    with path.open('x',encoding='utf-8') as handle:
        handle.write(text)


def main():
    tables=json.loads((OUT/'TABLES.json').read_text(encoding='utf-8'))['tables']
    lines=['# 两套T5权重证据补齐与独立复核','',
           '日期：2026-10-20。历史四臂摘要已知后的证据补齐，不是新独立确认。复用原生成候选和原512查询/512组匹配子集，不重训、不重新生成T5。新增计算是用冻结CodeBERT为base候选补存排名与全得分。','',
           '## 主要结果','',
           '以下为相对原查询的MRR@100差，区间为按关联组重采样的逐项95%区间。每域两权重使用相同512查询与原查询对照；不与2048源侧或1490开发集拼接分母。','',
           '|数据|T5权重|Top1 Δ|95%区间|best3 Δ|95%区间|',
           '|---|---|---:|---|---:|---|']
    fmt=lambda x:f'{x:+.8f}'
    ci=lambda values:'['+', '.join(map(fmt,values))+']'
    for row in tables['T6_two_T5_weights']:
        lines.append(f"|{row['domain']}|{row['T5_weights']}|{fmt(row['top1_delta'])}|{ci(row['top1_delta_ci95'])}|{fmt(row['best3_delta'])}|{ci(row['best3_delta_ci95'])}|")
    lines += ['',
        '八项比较校正后，四个best3−原查询区间均为正；Top1的source/base和CoSQA/local两格跨0，另两格为负。因此可以报告四格点估计方向相同，不能写四格Top1均在校正后显著为负。两权重的best3直接差区间跨0，不宣布base或微调权重更优。','',
        'CoSQA/base的ENTR单选−精确均匀单选为+.03475321，95%[+.01882089,+.05044681]。所以Top1低于原查询不等于熵排序没有信息。旧ENVELOPE.md中的“信号失败”或“生成器无问题”解释不再作为本文机制结论；旧文件保留以便追踪。','',
        '## 已执行的证据闭环','',
        '- source/base 5995个候选位置、6283个唯一检索输入、19210个语料文档；CoSQA/base为3345、3855、6267。',
        '- 原查询与全部base候选重新编码：冻结pooler_output、无归一化、FP32、TF32关闭、最大长度256、batch16；代码向量复用并核对原文档ID顺序。',
        '- 保存全语料得分与Top100：按FP32原始点积分数降序、原文档ID升序打破并列。source与CoSQA检索步骤约87.46秒和50.19秒；不含原T5生成/训练及独立审计，不作为全链路在线延迟。',
        '- 独立重算10138条完整Top100、144855715个全语料得分，FP32误差0；18680个候选RR、16384个查询指标及组区间匹配。',
        '- 实际模型独立前向核对16个查询和16个语料样例，最大误差0。16次FP64点积抽查最大绝对差5.955e−5，符合1e−4检查界；不声称FP64与FP32全排序相同。',
        '- 四个旧摘要最大差3.33e−16；6项科学口径测试通过；59个冻结/保护文件与10个新增封存输出哈希通过。','',
        '## 对论文的作用','',
        '它关闭“只有base汇总、没有排名和逐query证据”的缺口，提供相同样本下两套T5-base权重的稳健性诊断。两权重属于同一家族，样本也非未见确认；不称跨模型家族泛化或四次独立确认。核心贡献仍由指定SSQR复现和A同池直接对照承担。','',
        '## 复算与文件','',
        '- `PROTOCOL.json`、`RANKINGS_SEALED.json`：冻结输入和排名口径。',
        '- `rankings/*/{Top100.jsonl,texts.json,query_vectors.npy,scores.npy}`：文本、向量、全得分和最终排名。',
        '- `per_query.jsonl`、`candidate_RR.jsonl`、`RESULTS.json`：完整逐项结果。',
        '- `AUDIT.json`、`TEST_RESULTS.json`、`FINAL_ACCEPTANCE.json`：实际验收。',
        '- 独立入口：`experiments/rene_envelope_completion_20261020/audit.py --root <工作区>`。完整模式需要原模型/向量/冻结输入；紧凑增量包使用`--cached`，其范围仅为排名到RR/区间。','',
        '本轮未重新评估H标签/排名，未修改原确认、CURRENT/STABLE或风险配置；risk_certified=false。','']
    put(ENV/'READOUT.md','\n'.join(lines))
    put(ENV/'README.md','''# Two-weight evidence completion

Start with READOUT.md. This is post-hoc evidence completion on the exact historical sensitivity queries, not fresh confirmation. T5 candidate caches are reused; the missing base-weight retrieval evidence is now persisted.

Full audit: Python 3.10 with numpy, scipy, threadpoolctl, torch and transformers; run `python experiments/rene_envelope_completion_20261020/audit.py --root <workspace>` with the original bound inputs. AUDIT.json records the completed full-score and actual-model checks. Output is exclusive, so keep existing AUDIT.json rather than overwriting it.

The compact supplement runs the same independent evaluator with `--cached`. It contains candidate/ranking/label inputs, omits model weights and full scores, and separately verifies its derived-file manifest. Its success is not a new neural-training reproduction. See the supplement README for a single entry point.
''')
    put(OUT/'README.md','''# Integrated evidence tables

MAIN_RESULTS.md is the manuscript-readable entry; paper/rene2027/integrated_results.md is the identical paper copy. TABLES.json records nine source hashes, seven tables, 38 rows, and 235 source cells with exact JSON pointers. T1–T7 CSV files preserve full precision.

The independent audit.py verifies source hashes, semantic row mappings, CSV round trips, manuscript numerical presence, citation resolution, experiment acceptance and population identities. AUDIT.json records the successful check. It does not refit or reinterpret H; it reads sealed aggregate result JSON only.

The manuscript source snapshot is paper/rene2027/history/20261020-before-evidence-integration/. Updated outline, introduction, related work, result contract and contribution_evidence.md retain the candidate-to-realized-gain mainline. The full manuscript and final typesetting remain separate deliverables.

build_history/attempt1_csv_columns retains the incomplete first CSV-export attempt. The writer was repaired to include the union of optional CSV columns; the scientific results did not change. build.py is the table exporter; audit.py is independent of it. Do not rerun exclusive writers into an existing output directory.

Compact artifact supplement and final acceptance are in results/rene_evidence_release_20261020/. No original protocol, formal component pointer or risk certification state is changed by this integration.
''')
    print('Evidence readouts written.')


if __name__=='__main__':
    main()
