from __future__ import annotations

from .common import OUT, now, read, require, sha, status, write


def put(name, lines):
    path = OUT / name
    require(not path.exists(), "report already exists")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    require(read(OUT / "FINAL_ACCEPTANCE.json")["passed"], "independent acceptance required")
    design, policy = read(OUT / "DESIGN.json"), read(OUT / "PROTOCOL.json")
    precision = read(OUT / "PRECISION_PLAN.json")
    primary = read(OUT / "PRIMARY_RESULT.json")
    ledger = read(OUT / "TRAINING_LEDGER.json")
    acceptance = read(OUT / "FINAL_ACCEPTANCE.json")
    selected = precision["selected"]
    s1 = ["# 第一步 单一采纳策略与独立确认", "",
          "确认前已冻结唯一完整采纳策略 RIDGE_risk：71维直接收益选择、收益优先真实CE核验、三分类风险回退。CodeBERT为主检索器，BM25执行同一算法作为边界分析。", "",
          "这是依据开发证据选择的新CoSQA研究适配器；原开发协议的HGB身份不改，HGB保留为同输入对照。旧strong84、融合流程和五个正式指针均未替换。", "",
          "训练角色按关联组互斥：收益953查询/307组；核验更新288查询/103组；风险249查询/102组。三个角色均来自已消费开发D，不能将其拟合结果称新确认。H只进行冻结策略推理。", "",
          "核验每检索器取最高正收益优先级的最多floor(n/4)条查询，每条8个实际query-code对；使用原用户查询评价代码。只在同ENTR候选池内更新选择。末端风险阈值固定.075，拒绝恢复原Top100，不重新挑选、不退还核验成本。", "",
          "唯一确认性比较：CodeBERT ENTR−RAND20候选best3，在冻结capped16样本上检验。PROB、BM25、HGB和阶段差为预定义描述估计，不新增确认性拒绝主张。", "",
          f"设计SHA256：`{sha(OUT / 'DESIGN.json')}`。完整策略SHA256：`{sha(OUT / 'PROTOCOL.json')}`。", "",
          "开封链已核对：设计→模型/策略→实际确认决策→独立前审计→一次开效果→干净目录复算。全部8530查询/1963组现已消费，不得再用于选择新策略。"]
    s2 = ["# 第二步 确认精度", "",
          "原四项共同主比较需要2475组；计入原D/H组大小敏感性需3662组，不能以现有1963组完成该设计。本轮明确替换确认设计，保留旧规划。", "",
          "回到原统一方案的CodeBERT ENTR−RAND主问题；全1963关联组均保留，每组至多16条查询，按预定SHA256顺序选取7718条。指标仍是样本内常规查询加权MRR；目标人群为这一冻结抽样设计，不冒称全8530查询等功效。", "",
          "组内上限依据D方差可行性选择：预列4/8/12/16/全部，选择满足保守组需求不超过容量95%的最大上限。没有使用H预测或标签，也没有按D收益正负挑选。全部8530结果另作事前指定敏感性。", "",
          f"目标差仍为.005，双侧alpha=.05，功效80%。D方差10000次组bootstrap的95%上界为{selected['variance_bootstrap_upper95']:.10f}；组大小敏感性乘数{selected['size_imbalance_multiplier_at_least1']:.6f}。保守需要{selected['required_groups']}组，容量1963组，规划功效{selected['planned_power_at_capacity']:.2%}，80%最小可检差{selected['mde80']:.6f}。", "",
          "该规划依赖D到H条件方差可迁移及组独立假设；不是对未知H方差的数学保证。不开后验功效替代实际区间。", "",
          f"实际主比较ΔMRR={primary['mean']:+.8f}；t95%区间{primary['ci95_cluster_t']}；组bootstrap95%区间{primary['ci95_group_bootstrap']}；变化组{primary['changed_groups']}。", "",
          f"按冻结规则判读：`{primary['conclusion']}`。", "",
          "区间跨0只表示方向未确定；实质等效要求两个95%区间均完全位于±.005且至少20变化组。该规则只约束指定负熵候选位置比较，不推广为全部UQ结论。"]
    s3 = ["# 第三步 CoSQA 核验与风险回退完整阶段证据", "",
          "全8530条查询、1963关联组；每个检索器使用同ENTR三候选（不足时min(3,N)）与已冻结策略。下列阶段及效应区间均为事前指定描述估计。", "",
          "| 检索器 | 阶段 | ΔMRR@100 | 采用/有害 | 采用内有害率 | 正收益保留 | CE对数 |", "|---|---|---:|---:|---:|---:|---:|"]
    for ret in ("CodeBERT", "BM25"):
        summary = read(OUT / f"metrics/{ret}_full_H.json")
        s3 += [f"| {ret} | ENTR候选安全oracle | {summary['safe_ENTR_oracle_delta']:+.8f} | — | — | 100% | 0 |"]
        for name in ("ENTR_top1", "HGB_gain_control", "RIDGE_gain", "RIDGE_verified", "RIDGE_risk"):
            m = summary["policy_results"][name]
            harm = m["harm_within_adopted"]["value"]
            retention = m["positive_gain_retention"]["value"]
            h = "未定义" if harm is None else f"{harm:.2%}"
            r = "未定义" if retention is None else f"{retention:.2%}"
            s3.append(f"| {ret} | {name} | {m['delta_MRR']['mean']:+.8f} | {m['adopted']}/{m['harmed']} | {h} | {r} | {m['CE_pairs_charged']} |")
        g = summary["gate_accounting"]
        s3 += ["", f"{ret}门前{g['before_delta']:+.8f} − 误拒正收益{g['missed_positive_gain']:.8f} + 避免损失{g['avoided_loss']:.8f} = 门后{g['after_delta']:+.8f}。恒等式误差{g['identity_max_error']:.3g}。", "",
               f"{ret}阶段配对95%描述区间："]
        for name, m in summary["stage_contrasts"].items():
            s3 += [f"- {name}：{m['mean']:+.8f}，{m['ci95_group_bootstrap']}。"]
        s3 += [""]
    s3 += ["正收益保留率以同查询安全oracle正收益总量为分母；不以接近零的净收益作比例。核验可能改变动作，单独报告其阶段差；风险记账只比较已固定的核验后动作。", "",
           "采用内有害频率按关联组bootstrap给区间；不把组内依赖查询当独立Bernoulli发风险证书。risk_certified=false，正式部署仍为NoRewrite。", "",
           "CE输入token、物理唯一对与每条策略的已付成本见INFERENCE_LEDGER.json和metrics/*_full_H.json。"]
    put("STEP1_POLICY_FREEZE.md", s1)
    put("STEP2_PRECISION.md", s2)
    put("STEP3_STAGE_RESULTS.md", s3)
    put("REPORT.md", ["# RENE 独立确认 精度与完整阶段验收", "", *s2[2:], "", *s3[2:], "",
                      f"独立检查：{acceptance['model_decisions_reloaded']}项模型决策重载；{acceptance['Fraction_RR_checks']}项Fraction RR复算；最大数值差{acceptance['max_metric_error']:.3g}。", "",
                      "干净目录验收是缓存排名/决策到主表的独立数值重建，不冒称重新训练神经模型。真实T5全词表FP64、检索排名、CE与模型推理另经前审计核对。", "",
                      "完整单一策略见STEP1_POLICY_FREEZE.md；精度见STEP2_PRECISION.md；阶段分解见STEP3_STAGE_RESULTS.md。"])
    write(OUT / "EXPOSURE_LEDGER.json", {"created_utc": now(), "former_role": "unopened_H",
                                         "current_role": "consumed_independent_confirmation", "queries": 8530, "groups": 1963,
                                         "one_shot_opening_sha256": sha(OUT / "OUTCOME_OPENING.json"),
                                         "not_available_for_future_policy_selection": True}, immutable=True)
    write(OUT / "REPORT_RECEIPT.json", {"created_utc": now(), "reports": {name: sha(OUT / name) for name in
                                         ("REPORT.md", "STEP1_POLICY_FREEZE.md", "STEP2_PRECISION.md", "STEP3_STAGE_RESULTS.md")},
                                         "final_acceptance_sha256": sha(OUT / "FINAL_ACCEPTANCE.json")}, immutable=True)
    status("complete", acceptance_passed=True)


if __name__ == "__main__":
    main()
