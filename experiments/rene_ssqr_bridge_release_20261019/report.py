"""Generate review and manuscript tables from accepted supplementary results."""
from __future__ import annotations

import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/rene_ssqr_bridge_20261019"
RELEASE=ROOT/"results/rene_ssqr_bridge_release_20261019"


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def effect(value):return f'{value:+.8f}'


def ci(value):return '['+', '.join(effect(x) for x in value)+']'


def main():
    acceptance=read(OUT/"FINAL_ACCEPTANCE.json")
    if not acceptance["passed"]:raise ValueError("supplementary experiment did not pass acceptance")
    results=read(OUT/"metrics/RESULTS.json")
    aggregation=read(ROOT/"results/rene_ssqr_aggregation_audit_20261019/RESULTS.json")
    aggregation_main=[row for row in aggregation["views"] if row["scope"]=="main"]
    protocol=read(OUT/"PROTOCOL.json")
    views={(r["domain"],r["scope"],r["profile"],r["mask"],r["retriever"]):r for r in results["views"]}
    baseline=lambda domain,retriever:views[(domain,"main","greedy20_source","full",retriever)]
    source=baseline("source","CodeBERT")["contrasts"]["ENTR_minus_RAND"]
    cosqa=baseline("cosqa","CodeBERT")["contrasts"]["ENTR_minus_RAND"]
    difference=next(r for r in results["domain_contrasts"] if r["scope"]=="main" and r["retriever"]=="CodeBERT")
    if source["ci95_group_bootstrap"][0]>0 and difference["ci95_group_bootstrap"][0]>0:
        judgment="同一重建在源数据上观察到正向ENTR−RAND优势，且源−CoSQA直接差区间高于0。桥接支持数据环境相关的效应差；语料、标注和难度同时变化，不能进一步归为查询措辞的单独因果。"
    elif source["ci95_group_bootstrap"][0]>0:
        judgment="源数据上观察到正向ENTR−RAND优势；直接域间差区间未排除0，因此不能声称它在两个数据环境中的效果确有差别。"
    else:
        judgment="源数据桥接也未形成可明确归于负熵位置选择的正向优势；不能把CoSQA确认结果归因于真实查询迁移。源侧差异与不可恢复的训练/权重因素仍须分开解释，本轮不把差异清单写成已经确定的失败原因。"
    lines=[
        "# SSQR复现公平性补强：原数据桥接与实现敏感性",
        "",
        "状态：实际生成、双检索器排名、独立RR及真实T5检查完成。证据角色为确认后补充诊断；CoSQA只用开发D，不改已有H确认。",
        "",
        "## 结果先看",
        "",
        judgment,
        "",
        "主线继续是SSQR式候选收益的外部复现与自动采纳分析。本文可以报告经过核对的重建结果、有限实现敏感性和冻结阶段效用；不以补充诊断冒充新的独立确认，也不据此推广全部UQ有效或无效。",
        "",
        "## 同一实现的源数据桥接",
        "",
        "原AdvTest固定哈希抽2048条/594仓库，搜索作者split顺序的全部19210原测试函数。查询及代码逐URL对照作者输入一致。CoSQA复用D1490/512、6267函数。两边同T5、同CodeBERT、1…n位置、greedy/总decoder20、k=3和RAND20。",
        "",
        "下表best3不包含NoRewrite；Δ是相对各自原查询，绝对MRR不跨数据直接比较。区间为10000次关联组bootstrap95%。",
        "本表沿用已冻结的逐query最大RR定义。公开评估器的重复URL处理与另外两种聚合口径，另见`../rene_ssqr_aggregation_audit_20261019/READOUT.md`；没有声称原作者的生产聚合路径已恢复。",
        "",
        "| 检索器 | 数据与查询数 | NoRewrite MRR | ENTR best3 ΔMRR | ENTR−RAND20 | 95%区间 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for retriever in ("CodeBERT","BM25"):
        for domain in ("source","cosqa"):
            view=baseline(domain,retriever)
            contrast=view["contrasts"]["ENTR_minus_RAND"]
            label="AdvTest 2048" if domain=="source" else "CoSQA D1490"
            lines.append(f'| {retriever} | {label} | {view["mrr"]["NoRewrite"]:.8f} | {effect(view["contrasts"]["ENTR_best3_gain"]["mean"])} | {effect(contrast["mean"])} | {ci(contrast["ci95_group_bootstrap"])} |')
    lines += ["","直接估计源−CoSQA的ENTR−RAND效应差，跨域共享仓库合成同一重采样组，保留协方差：","",
              "| 检索器 | 源−CoSQA效应差 | 组t95% | 组bootstrap95% | 共享组/联合组 |","|---|---:|---|---|---:|"]
    for row in results["domain_contrasts"]:
        if row["scope"]!="main":continue
        lines.append(f'| {row["retriever"]} | {effect(row["mean"])} | {ci(row["ci95_cluster_t"])} | {ci(row["ci95_group_bootstrap"])} | {row["shared_groups"]}/{row["joint_groups"]} |')
    lines += ["","原SSQR表4ENTR−RAND=.2221−.1886=+.0335；本地权重、训练细节及运行时不同，不能把原文数值当作本地对照臂。源数据桥接仍同时改变语料、查询长度、标签与难度，因此没有识别纯查询来源因果。",
              "","## 六种有限设置敏感性","",
              "以下两域均为事先固定每组一条的512查询子集，所有profile用完全相同的查询；与上表查询加权主桥接总体不同。total decoder长度包含start，10对应9个new tokens，是显式m10操作化而非声称恢复原历史计数。采样每位置1条、固定位置局部RNG，共同位置复用；不是多seed集成。以下均为full-label entropy下的ENTR−RAND20。","",
              "| 设置 | AdvTest CodeBERT | CoSQA CodeBERT | AdvTest BM25 | CoSQA BM25 |","|---|---:|---:|---:|---:|"]
    for profile in protocol["profiles"]:
        values=[effect(views[(d,"sensitivity",profile,"full",r)]["contrasts"]["ENTR_minus_RAND"]["mean"])
                for r in ("CodeBERT","BM25") for d in ("source","cosqa")]
        lines.append('| '+profile+' | '+' | '.join(values)+' |')
    lines += ["","![](implementation_sensitivity.png)","",
              "有限配对改动的CodeBERT结果（同查询、同采样共同位置；Δ指ENTR−RAND效应的改变量）：","",
              "| 改动 | 数据 | 效应改变量 | 配对bootstrap95% |","|---|---|---:|---|"]
    for row in results["sensitivity_contrasts"]:
        if row["retriever"]!="CodeBERT" or row["contrast"]=="lexical_minus_full":continue
        value=row["ENTR_minus_RAND_change"]
        lines.append(f'| {row["contrast"]} | {row["domain"]} | {effect(value["mean"])} | {ci(value["ci95_group_bootstrap"])} |')
    lines += ["","完整PROB、lexical-only及各设置区间见`metrics/RESULTS.json`。所有profile均保留；没有根据这些点估计替换确认配置或升级正式模块。",
              "","## 三候选聚合本身改变估计对象","",
              "同一批候选和排名，先逐query取最好RR再平均，与先计算各选序的整体MRR再取最好，回答的是两个问题。后一指标也不等于不看标签即可部署的政策。公开reader末行行为作为第三口径单列。","",
              "| 数据/检索器 | 逐query best3 E−R | 三个rank-wise MRR取最好 E−R | 末行 E−R |",
              "|---|---:|---:|---:|"]
    for row in aggregation_main:
        modes=row["aggregations"]
        lines.append(f'| {row["domain"]}/{row["retriever"]} | {effect(modes["per_query_best3"]["ENTR_minus_RAND"]["mean"])} | {effect(modes["best_of_three_rankwise_MRRs"]["ENTR_minus_RAND"])} | {effect(modes["last_selected_row"]["ENTR_minus_RAND"]["mean"])} |')
    rankwise_cosqa=next(row for row in aggregation_main if row["domain"]=="cosqa" and row["retriever"]=="CodeBERT")["aggregations"]["best_of_three_rankwise_MRRs"]
    lines += ["",
              f'CoSQA D/CodeBERT的rank-wise最好口径点估计为{effect(rankwise_cosqa["ENTR_minus_RAND"])}；预定66均值同时界传播后的保守95%区间为{ci(rankwise_cosqa["ENTR_minus_RAND_conservative95"])}。它不是已确认的优势，也不能被逐query best3的等效结论替代。完整三口径和区间见聚合补充报告。',
              "",
              "直接执行作者公开evaluator的同URL三行合成例，末行结果为0，逐query best3为1；这确认了公开reader覆盖行为，没有恢复作者当年生产表格的聚合路径。论文将明确写成在预先定义的逐query机会指标下进行外部重建，不把原表4的全部口径视作已经精确恢复。",
              "","## 既有独立确认如何保留","",
              "CoSQA确认H已在21:37一次消费：7718/1963主比较ENTR−RAND=−.00083652，组t95%[−.00320993,+.00153688]、bootstrap95%[−.00325479,+.00154146]均在事前±.005内。源桥接/D敏感性没有改动该指定比较，也不能把别的profile套入该H区间。",
              "","±.005事前冻结；作为量级说明，它约为原表4负熵−随机差.0335的14.9%。.005 MRR亦相当于1%的查询从rank2升到rank1所带来的平均变化。这是可读的指标尺度说明，不是用户满意度研究或对所有应用的最小重要差异证明。",
              "","全H8530的三候选best3Δ为CodeBERT+.05655770、BM25+.01358236；这些候选级机会及收益→CE→风险的阶段结果支持原文档的研究主线，不能等同于一种新UQ算法已经稳定提高自动效用。",
              "","## 已补齐的复现依据与保留的差异","",
              "1. 原数据查询、完整搜索池、源行/URL与qrels身份核对；没有用4096池或45244辅助文件替代原论文搜索池。",
              "2. 同权重双数据桥接、直接域间差、六profile及两种熵掩码；位置消融不改变共同候选或采样随机流。",
              "3. 论文实现差异表及方法文本，区分已修正的query/熵/CodeBERT目标和仍不同的CQC训练；新增公开evaluator重复URL覆盖及三口径核对。",
              "4. 独立RR、真实模型检查和可移植缓存重放；已有确认与五正式指针保持。",
              "","作者原权重不可提供、本地T5训练来源/遮盖/优化器/长度与公开trainer不同、PROB原实现未公开、历史m计数/生产tie/三候选生产聚合路径不能恢复。这些已写入Design及artifact；不得由数值审计通过推成与作者模型完全相同。",
              "","## 验收与交付","",
              f'- 协议SHA256：`7d3d4c0728aa413a21dfc748d4a01dc4f29d5d59a0b4a44591eb48793dc3d95e`；13项契约测试通过。',
              f'- 独立Fraction RR核对：{acceptance["independent_fraction_rr"]["comparisons"]}项，最大差{acceptance["independent_fraction_rr"]["maximum_error"]:.3g}。',
              f'- 真实T5采样与FP64熵：{acceptance["actual_model_checks"]["actual_candidates"]}候选，最大熵差{acceptance["actual_model_checks"]["fp64_entropy_max_error"]:.3g}；抽样生成token一致。',
              '- 可移植artifact额外验收见`../rene_ssqr_bridge_release_20261019/RELEASE_ACCEPTANCE.json`；独立脚本不导入workspace，缓存重放不冒称完整重训。',
              '- 代码：`experiments/rene_ssqr_bridge_20261019/`；报告和主数据：本目录；论文：`paper/rene2027/implementation_fidelity.md`与`replication_results.md`。',
              '- 五正式指针、旧权重与确认结果保留，risk_certified=false、NoRewrite状态保持。',
              ""]
    (OUT/"READOUT.md").write_text("\n".join(lines),encoding="utf-8")
    make_figure(views)
    paper_lines=["# Replication results: implementation fidelity and source bridge","",
                 "Supplementary source-data and sensitivity results; the primary CoSQA confirmation remains unchanged.","",
                 f'Under the same frozen greedy reconstruction, ENTR minus RAND20 best-of-three is {effect(source["mean"])} on the 2,048-query original-data bridge (group-bootstrap 95% {ci(source["ci95_group_bootstrap"])}), and {effect(cosqa["mean"])} on the 1,490-query CoSQA development arm ({ci(cosqa["ci95_group_bootstrap"])}). The directly estimated source-minus-CoSQA contrast is {effect(difference["mean"])} ({ci(difference["ci95_group_bootstrap"])}), with {difference["shared_groups"]} shared repository-connected groups retained in the covariance estimate.',"",
                 "These are post-confirmation supplementary diagnostics, selected before their new effects were opened. The estimand remains the frozen per-query maximum-RR candidate opportunity. A source-interface audit additionally found duplicate-URL overwrite in the public evaluator; a cache-only supplement compares that behavior with per-query and rank-wise aggregation, without asserting which unpublished path produced the original article's results. The published author's checkpoint was unavailable; differences in local CQC training remain documented, and the bridge changes corpus, judgments, query length, and difficulty as well as query origin. It therefore does not identify query wording as an isolated cause.","",
                 "The finite sensitivity study uses the same 512 queries per dataset across six predefined position, length, and decoding profiles. Common-position generations are identical within position comparisons, and sampled generations use position-local RNG. Full results, paired profile changes, lexical-entropy controls, and independent artifact checks are linked in the supplementary readout. No sensitivity profile replaces the already frozen CoSQA confirmatory policy.","",
                 f'Aggregation sensitivity is material. On the same CoSQA development candidates, ENTR−RAND20 is {effect(cosqa["mean"])} for the mean of per-query maxima, compared with {effect(rankwise_cosqa["ENTR_minus_RAND"])} for the maximum of the three rank-wise aggregate MRRs. The latter has a prespecified simultaneous-bound 95% interval of {ci(rankwise_cosqa["ENTR_minus_RAND_conservative95"])} and is not a confirmed advantage. These estimands cannot substitute for each other; neither the external equivalence finding nor the public duplicate-URL behavior establishes the original production table\'s aggregation.',"",
                 "The confirmatory CoSQA estimate is −0.00083652 MRR@100 for ENTR−RAND20 on the capped-16 design (7,718 queries; 1,963 groups). Both prespecified 95% intervals fit within ±0.005. As an interpretable scale, 0.005 is about 14.9% of the original Table-4 gap of 0.0335, or the average RR change produced if 1% of queries move from rank 2 to rank 1. This is a metric-scale justification, not a claim to have measured user satisfaction or a universal application threshold.","",
                 "Candidate opportunity remains distinct from automatic realized utility: on all 8,530 CoSQA confirmation queries, the ENTR best-of-three gain, without adding NoRewrite, is +0.05655770 for CodeBERT and +0.01358236 for BM25. The fixed downstream-stage decomposition quantifies what survives selection, verification, and fallback. The implementation-fidelity evidence supports interpreting these results as a controlled external reconstruction with explicit choices, rather than as an exact numerical recovery of unavailable author artifacts.",""]
    (ROOT/"paper/rene2027/replication_results.md").write_text("\n".join(paper_lines),encoding="utf-8")
    print({"report":str(OUT/"READOUT.md"),"source_CodeBERT":source["mean"],"cosqa_D_CodeBERT":cosqa["mean"],"domain_difference":difference["mean"]},flush=True)


def make_figure(views):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    profiles=["greedy20_source","greedy20_all","greedy10_source","greedy10_all","sample10_source","sample10_all"]
    labels=["Greedy / length 20 / positions 1..n","Greedy / length 20 / positions 0..n",
            "Greedy / length 10 / positions 1..n","Greedy / length 10 / positions 0..n",
            "Sample / length 10 / positions 1..n","Sample / length 10 / positions 0..n"]
    fig,axes=plt.subplots(1,2,figsize=(12,5.5),sharey=True)
    for ax,retriever in zip(axes,("CodeBERT","BM25")):
        for domain,shift,color in (("source",-.13,"#3267A8"),("cosqa",.13,"#C77624")):
            effects=[views[(domain,"sensitivity",p,"full",retriever)]["contrasts"]["ENTR_minus_RAND"] for p in profiles]
            means=np.array([v["mean"] for v in effects])
            bounds=np.array([v["ci95_group_bootstrap"] for v in effects])
            ax.errorbar(means,np.arange(6)+shift,xerr=np.stack([means-bounds[:,0],bounds[:,1]-means]),fmt="o",
                        capsize=3,color=color,label="AdvTest" if domain=="source" else "CoSQA D",markersize=4)
        ax.axvline(0,color="0.5",linewidth=.9,linestyle="--")
        ax.set_title(retriever)
        ax.set_xlabel("ENTR - RAND20 (MRR@100)")
        ax.grid(axis="x",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_yticks(np.arange(6),labels)
    axes[0].invert_yaxis()
    axes[1].legend(loc="best",frameon=False)
    fig.suptitle("Implementation sensitivity on fixed 512-query subsets",fontsize=13)
    fig.text(.5,.015,"One query per group; descriptive group-bootstrap 95% intervals. Decoder lengths include the start token.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.96))
    fig.savefig(OUT/"implementation_sensitivity.png",dpi=180,bbox_inches="tight")
    figure_dir=ROOT/"paper/rene2027/figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir/"ssqr_implementation_sensitivity.pdf",bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":main()
