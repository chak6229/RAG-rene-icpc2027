"""Publish the complete frozen confirmation after the final audit."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

from experiments.rene_uq_independent_proof_20261019.study import FRESH,OUT,heads
from experiments.rene_strict_confirmation_20261019.common import read_json,sha,utc,write_json


def publish():
    r=read_json(OUT/"RESULTS.json")
    audit=read_json(OUT/"AUDIT.json")
    assert audit["passed"] and audit["results_sha256"]==sha(OUT/"RESULTS.json")
    p=read_json(FRESH/"PROTOCOL.json")
    labels=dict(common="共同风险证据",support="共同证据＋支持/均值/质量",full="共同证据＋完整跨seed统计")
    text=[
        "# 指定跨seed统计的风险位置独立确认",
        "",
        f"2026-10-19；{r['queries']}个新查询仓库，{r['corpus_documents']}文档固定语料，{r['candidates']}候选。",
        "",
        "本轮实际检验核验后风险位置的特征增量：每个上游固定候选动作、证据和成本，三个风险头真实重训；拒绝后恢复NoRewrite，不重选候选。原512开发证据形成假设，本轮使用新查询确认；旧严格1536完整流程结果保留。",
        "",
        "## 主要比较",
        "",
        "主上游no_seed_uq；固定逐query门p_harm≤.075。阈值沿用已冻结流程，不能根据新结果移动；不同头的实际覆盖可能不同。",
        "",
        "|风险头|采用|正收益|有害|零收益采用|ΔMRR|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for h in heads.HEADS:
        m=r["methods"][f"no_seed_uq/{h}/threshold075"]
        text.append(f"|{labels[h]}|{m['adopted']}|{m['positive']}|{m['harmful']}|{m['adopted_zero']}|{m['delta_mrr']:+.9f}|")
    text += ["","|预定主要增量|均值|同时97.5%区间包络|方向|实质等效±.005|","|---|---:|---|---|---|"]
    for name,c in r["primary_contrasts"].items():
        lo,hi=c["simultaneous_envelope"]
        text.append(f"|{name}|{c['mean']:+.9f}|[{lo:+.9f}, {hi:+.9f}]|{c['verdict']['direction']}|{c['verdict']['equivalence_established']}|")
    text += [
        "",
        "区间：预定8-query生成块成对bootstrap 10000次，两个比较Bonferroni调整；报告其与生成块t区间的包络。属于指定抽样/模型下的近似统计推断。等效判定另要求≥20非零query、≥10非零生成块；零差或稀疏退化区间不能用来证明总体等效。",
        "",
        "## 同采用量诊断",
        "",
        "下表每头保留门前合格动作的一半；这是次要批次排序诊断，每次bootstrap重新排序并计算采用数。",
        "",
        "|风险头|采用|正收益|有害|ΔMRR|",
        "|---|---:|---:|---:|---:|",
    ]
    for h in heads.HEADS:
        m=r["methods"][f"no_seed_uq/{h}/fixed50"]
        text.append(f"|{labels[h]}|{m['adopted']}|{m['positive']}|{m['harmful']}|{m['delta_mrr']:+.9f}|")
    for name in ("full_minus_support","full_minus_common"):
        c=r["secondary_comparisons"][f"no_seed_uq/{name}/fixed50"]
        text += ["",f"{name}：{c['mean']:+.9f}；描述性95%区间{c['descriptive_ci95']}。"]
    text += [
        "",
        "## 概率质量和机制",
        "",
        "同一批门前动作，适当评分越低越好；概率信息与实际决策效用分别判断。",
        "",
        "|风险头|harm Brier|harm NLL|三分类NLL|harm AUC|",
        "|---|---:|---:|---:|---:|",
    ]
    for h in heads.HEADS:
        v=r["proper_scores"]["no_seed_uq"]["methods"][h]
        text.append(f"|{labels[h]}|{v['harm_brier']}|{v['harm_nll']}|{v['class_nll']}|{v['harm_auc']}|")
    for name in ("full_minus_support","full_minus_common"):
        k=r["mechanisms"][f"no_seed_uq/{name}"]
        bound=r["primary_contrasts"][name]["label_free_cohort_bounds"]
        text += [
            "",
            f"{name}：{k['adoption_disagreements']}次采用分歧，{k['nonzero_outcome_disagreements']}次RR分歧；无标签排名几何界[{bound['mean_lower']:+.9f}, {bound['mean_upper']:+.9f}]。",
            "几何界遍历每条query所有可能单一目标身份，确定性限制本批固定动作可达到的增量；不推断其他样本或模型。",
        ]
        for c in k["cases"][:6]:
            text.append(f"- query {c['query_index']}：before/full采用={c['before_adopted']}/{c['full_adopted']}，门前ΔRR={c['pre_gate_delta']:+.9f}，对均值贡献={c['mrr_contribution']:+.9f}。")
    text += [
        "",
        "完整RESULTS.json还包括seed_uq上游复核、概率差区间和候选oracle。预定次要结果不能替换主要比较。有限风险模型评分之差不等于真实条件互信息；采样分歧也不是已验证的参数后验认知不确定性。",
        "",
        "## 数据、成本与验收",
        "",
        f"- 四seed={p['seeds']}；不重训T5。新查询来自原库存未评测仓库，其代码此前是固定语料的无标签干扰项。上游CodeBERT见过source-train，结论范围是固定语料下游新查询。",
        f"- 功效规划n={r['power']['planned_queries']}，目标效应.005、目标功效.8、双比较族alpha=.05；依赖开发SD上界的近似，不自动证明实际达到功效。",
        f"- 真实共享CE={r['ce_physical_cost']['physical_unique_pairs']}对、{r['ce_physical_cost']['physical_tokens']} tokens；拒绝保留已执行成本。",
        f"- 测试{read_json(OUT/'TESTS.json')['tests']}项通过，开标签前与开标签后独立审计PASS；{audit['vector_ranking_checks']}个向量排名、{audit['independently_recomputed_probability_rows']}行概率、{audit['candidate_label_checks']}个候选标签核对。",
        "- 实际动作/概率先封存、审计通过后首次打开目标映射；本批已consumed。旧test只读取仓库元数据用于排除，未计算新指标。",
        "- 正式CURRENT/STABLE保持；risk_certified=false，部署仍为NoRewrite。本轮检验特征价值，不签发风险证书。",
        "",
        f"协议SHA-256：{sha(FRESH/'PROTOCOL.json')}",
        f"结果SHA-256：{sha(OUT/'RESULTS.json')}",
        "",
        "复现顺序：test_contract.py → stages.py --stage prepare → stages.py --stage all → audit.py --phase pre → evaluate.py → audit.py --phase final → publish.py。",
        "",
    ]
    report=OUT/"REPORT.md"
    if report.exists():
        assert report.read_text("utf8")=="\n".join(text)
    else:
        report.write_text("\n".join(text),encoding="utf8")
    write_json(OUT/"DELIVERY.json",dict(
        completed=True,completed_at_utc=utc(),results_sha256=sha(OUT/"RESULTS.json"),
        audit_sha256=sha(OUT/"AUDIT.json"),report_sha256=sha(report),
        risk_certified=False,protected_unchanged=True,
    ),immutable=True)
    print(str(report),flush=True)


if __name__=="__main__":
    publish()
