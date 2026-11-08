from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rene_strict_confirmation_20261019.common import read_json, sha, write_json

OUT = ROOT / "results/rene_fixed_action_uq_20261019"


def signed(x):
    return f"{x:+.8f}"


def percent(x):
    return "—" if x is None else f"{x * 100:.2f}%"


def main():
    r, a = (read_json(OUT / n) for n in ("RESULTS.json", "AUDIT.json"))
    assert a["passed"] and a["evidence"]["results"] == sha(OUT / "RESULTS.json")
    p = r["primary_result"]
    ci = p["descriptive_bootstrap95"]
    lines = [
        "# 固定动作与证据的 UQ 风险头实验",
        "",
        "日期：2026-10-19。512查询/512仓库、四外折、三风险头共48次真实拟合。"
        "先封存全部预测和采用名单，再计算held指标。独立重算审计通过。",
        "",
        f"主比较：固定no_seed_uq上游的融合动作，每折保留一半合格动作，full−common ΔMRR="
        f"{signed(p['delta_mrr'])}，描述性95%区间[{signed(ci[0])}, {signed(ci[1])}]。",
        "",
        "该区间以已拟合外折模型为条件；开发池已用于历史探索，不作为新的独立确认或风险证书。"
        "common仍包含共同生成熵及原生收益/损害预测；full是在风险头显式新增原始跨seed统计。"
        "本实验不直接分解旧1536完整流程增量，也不回答四seed相比单seed的成本效益。",
        "",
        "## 主分析：固定无跨seed上游，融合",
        "",
        "各头执行相同的门前动作与CE，差异只发生在后置保留。正收益保留率以该固定动作集合的正RR总量为分母。",
        "",
        "| 决策 | 风险头 | 采用数 | ΔMRR | 正/害/零 | 害率 | 正收益保留 | 负RR总量 |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for policy, label in (("all", "门前全保留"), ("fixed50", "固定一半"), ("threshold075", "固定阈值.075")):
        hs = ("common",) if policy == "all" else ("common", "support", "full")
        for h in hs:
            s = r["families"]["fusion"]["no_seed_uq"]["methods"][h][policy]
            lines.append(f"| {label} | {h} | {s['accepted']} | {signed(s['delta_mrr'])} | "
                         f"{s['positive']}/{s['harmful']}/{s['zero']} | {percent(s['harmful_rate'])} | "
                         f"{percent(s['positive_retention'])} | {s['negative_rr_mass']:.6f} |")
    lines += [
        "",
        "support增加6项支持/平均一致性/质量；full再增加6项离散度/多样性。"
        "这是操作性分组，平均一致性也可能表达UQ；有效特征数18/24/30，模型类和正则化规则一致。",
        "",
        "| 同采用量对比 | ΔMRR增量 | 描述性95%区间 | 有害动作数变化 |",
        "| --- | ---: | --- | ---: |",
    ]
    for name in ("full_minus_common", "support_minus_common", "full_minus_support"):
        c = r["families"]["fusion"]["no_seed_uq"]["contrasts"]["fixed50"][name]
        ci = c["descriptive_bootstrap95"]
        lines.append(f"| {name} | {signed(c['delta_mrr'])} | [{signed(ci[0])}, {signed(ci[1])}] | {c['harmful_count_change']:+d} |")
    lines += [
        "", "## 概率质量：主分析合格动作", "",
        "| 风险头 | harm Brier↓ | harm NLL↓ | harm AUC↑ | 三分类 NLL↓ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for h, s in r["families"]["fusion"]["no_seed_uq"]["probability_quality"].items():
        auc = "—" if s["harm_auc"] is None else f"{s['harm_auc']:.6f}"
        lines.append(f"| {h} | {s['harm_brier']:.6f} | {s['harm_nll']:.6f} | {auc} | {s['class_nll']:.6f} |")
    lines += ["", "## 固定采用量曲线与预指定复核", "",
              "| 流程/固定上游 | 保留比例 | 采用数 | common ΔMRR | support ΔMRR | full ΔMRR | full−common |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for family in ("fusion", "rewrite"):
        for upstream in ("no_seed_uq", "seed_uq"):
            for policy in ("fixed25", "fixed50", "fixed75", "all"):
                group = r["families"][family][upstream]
                ss = [group["methods"][h][policy] for h in ("common", "support", "full")]
                lines.append(f"| {family}/{upstream} | {policy} | {ss[0]['accepted']} | "
                             f"{signed(ss[0]['delta_mrr'])} | {signed(ss[1]['delta_mrr'])} | "
                             f"{signed(ss[2]['delta_mrr'])} | "
                             f"{signed(group['contrasts'][policy]['full_minus_common']['delta_mrr'])} |")
    lines += [
        "", "## 执行、成本与验收", "",
        "- 同一上游内三风险头的候选、实际动作、核验文档、CE结果和已执行成本全部固定。"
        "拒绝只回到原完整排名，没有重新选择候选。",
        "- 固定一半按每折合格动作数向下取整，包含零收益且实际改变排名的采用。"
        "这是离线排序诊断；固定.075才是逐query门，未把相同阈值当成相同风险。",
        f"- 候选真实排名标签检查{a['candidate_label_checks']}项，最大误差"
        f"{a['maximum_candidate_label_error']:.3g}；独立概率重算误差"
        f"{a['maximum_probability_recompute_error']:.3g}。",
        "- common头重新拟合后复现既有风险概率与.075采用；所有训练/评价角色按仓库隔离。"
        "每折风险拟合仅16–58个已选动作，细目与类计数见RESULTS.json。",
        "- 无新增T5、检索、CE神经推理；未读取严格1536或旧test产生新结果；"
        "正式CURRENT/STABLE保持原哈希，risk_certified=false。",
        "",
        "## 后续确认所需样本规划", "",
        f"沿用.005实质效应参考，单个双侧alpha=.05对比、80%功效，开发配对SD上界及.05下限规划为"
        f"{r['future_power_planning']['planned_queries']}个仓库。"
        "此为正态近似的样本规划；新确认仍须另冻结协议、评估样本来源并纳入多重比较及风险认证需求。",
        "",
        "复现入口：experiments/rene_fixed_action_uq_20261019/{DESIGN.md,run.py,audit.py,test_contract.py}。"
        "主要证据：PROTOCOL.json、DECISION_SEAL.json、RESULTS.json、PER_QUERY.pkl、AUDIT.json。",
        "",
    ]
    path = OUT / "REPORT.md"
    if path.exists():
        raise FileExistsError(path)
    path.write_text("\n".join(lines), encoding="utf8")
    write_json(OUT / "COMPLETION.json", dict(
        completed=True, audit_passed=True, report_sha256=sha(path),
        results_sha256=sha(OUT / "RESULTS.json"), audit_sha256=sha(OUT / "AUDIT.json"),
        protocol_sha256=sha(OUT / "PROTOCOL.json"),
    ), immutable=True)
    print(str(path), flush=True)


if __name__ == "__main__":
    main()
