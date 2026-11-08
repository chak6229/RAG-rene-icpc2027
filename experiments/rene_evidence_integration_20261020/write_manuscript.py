"""Editorial integration; frozen machine results are never rewritten."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper/rene2027'
HISTORY = PAPER / 'history/20261020-before-evidence-integration'

DOCS = {
'outline.md': r'''# RENE 2027 论文大纲（证据整合版，2026-10-20）

写作预算：10页正文、另2页参考文献；最终版式按投稿系统复核。旧稿保存在 `history/20261020-before-evidence-integration/`。本次整合不改变原确认协议或正式组件。

## 标题与主线

*From Candidate Gain to Realized Gain: An External Replication of Query Rewriting for Code Search*

主线：开发者代码搜索中，SSQR式候选收益能实现多少自动采纳收益，哪些证据帮助决定是否采用？

论证中心是指定位置选择主张的外部检验，以及同池、同查询的直接契约对照。机会指标、自动选择、融合和风险回退分别回答自己的问题，不以新增UQ算法成功为前提。

## 四个研究问题

- **RQ1：位置选择与评价契约。** 在重建实现下，ENTR相对RAND/PROB的优势如何？同池从单选改为三候选机会评价时，相对优势是否改变？
- **RQ2：自动实现。** 固定候选后，匹配监督的采用头、可执行多候选融合，以及已冻结的核验/回退流程各实现多少净收益？
- **RQ3：UQ作用位置。** 原生生成置信和新增跨seed统计在各自测试位置有什么条件增量？保留不同人群与动作下的阳性、等效和未决证据。
- **RQ4：边界。** 结果如何随检索器和两套T5权重变化？只用同样本直接效应差判断差异；两套权重不称两个模型家族。

## 页面与展示预算

|章节|页数|承担的论证|
|---|---:|---|
|Introduction|1.0|开发者问题、指定复现、三个具体贡献、核心数字|
|Background and Related Work|1.0|SSQR与六项已核全文近邻的具体差异|
|Study Design|1.8|数据角色/重叠、重建差异、源侧桥接、评价契约、冻结与组统计|
|RQ1|1.7|唯一确认、源侧桥接、A同池直接契约差；PROB与精确随机敏感性|
|RQ2|1.9|B匹配采用头、RRF3/4、全H阶段效用/损害/成本|
|RQ3|0.6|原生熵与跨seed统计分开；旧1536/1024证据各归其位|
|RQ4|0.6|双检索器直接对照、两权重匹配子集诊断|
|Validity|0.9|实现恢复程度、训练暴露、单目标标注、事后角色、范围|
|Conclusion and Artifact|0.5|实践建议与可复算交付|

表格不平均铺满正文。正文优先展示数据/实现表、RQ1主结果与A直接差、RQ2政策与阶段表；B完整四头、两权重四格、旧UQ完整区间可放附录，正文保留决定解释的数字。机器表见 `integrated_results.md`。

## 三个贡献及证据

1. **具体方法的外部复现。** 在明确重建差异和数据重叠的条件下，冻结检验SSQR的ENTR−RAND20：7718查询/1963组，Δ−.00083652，两个95%区间均位于事前±.005内。等效结论只适用于该指定比较与逐query best3契约。
2. **同池评价契约的直接对照。** CoSQA/CodeBERT上，单选相对随机优势+.02793583，best3相对RAND20优势+.00236800，同查询直接差+.02556783，四项校正区间[+.01416754,+.03661756]。PROB表现相近；预算与机会聚合共同改变，不宣称熵独有机制或纯因果归因。
3. **候选到执行的可复算受控扩展。** 同动作匹配监督、标准RRF控制及冻结H阶段将净收益、损害频率、正收益保留与成本分开计量；两权重匹配诊断及独立重放补足可核查性。此项提供具体经验知识和artifact，不将标准RRF或核验概念包装为新算法。

## 关键表格入口

|编号|内容|机器表|
|---|---|---|
|T1|唯一冻结确认|`T1_confirmation.csv`|
|T2|A同池单选/三选直接差，四个数据×检索器格|`T2_same_pool_contracts.csv`|
|T3|B固定ENTR Top1四个同监督头、固定20%采用|`T3_adoption_fixed20.csv`|
|T4|同ENTR3池的Top1/RRF3/RRF4与标签辅助best3|`T4_multicandidate_fusion.csv`|
|T5|全H8530/1963的收益选择、真实CE、风险回退|`T5_frozen_H_stages.csv`|
|T6|每域512查询/512组，两套T5-base权重|`T6_two_T5_weights.csv`|
|T7|旧1536全流程阳性、1024风险位置等效/未决|`T7_previous_seed_UQ.csv`|

CSV位于 `results/rene_evidence_integration_20261020/`，精确来源指针见同目录 `TABLES.json`。图优先表现冻结阶段正收益/避损；risk–coverage只用D诊断和H冻结工作点。

## 解释规则与完成边界

- SSQR已报告Top1/Top3差距；QueryRoute/QPP已有一般选择缺口与决策效用研究。本文的增量由具体复现和受控直接对照承担。
- CoSQA上ENTR单选优于随机且低于原查询可以同时成立，不将后者写成熵排序无信息。
- best3使用标签；它不是可执行策略，也不是RRF新排名的上界。
- H8530阶段表与7718主样本区分；D、源侧2048、匹配子集512、旧1536/1024不拼接分母。
- 新增UQ有效、风险证书或真实查询来源的因果解释不是本稿结论。若新增这些主张，再安排相应确认或配对来源实验。

本轮已整合主表、引言、近邻与贡献对应。全文方法、讨论、图表版式及最终投稿包仍按原计划推进；当前是论文源稿，不把源稿整合等同投稿完成。

入口：`integrated_results.md`、`contribution_evidence.md`、`implementation_fidelity.md`、`results_contract.md`。近邻全文表：`results/rene_multicandidate_fusion_20261020/PRIOR_WORK_COMPARISON.md`。
''',
'intro_draft.md': r'''# Introduction（证据整合稿，2026-10-20）

Developers use natural-language queries to find code that implements a task. Query rewriting can bridge differences between a developer's wording and a function's identifiers, but a search tool must also decide which rewrite to execute and whether to keep the original query. SSQR provides a concrete method for this setting: a T5 rewriter enumerates insertion positions, ranks generated spans by negative predictive entropy, and recommends three reformulations \cite{ssqr2023}. Its position-selection comparison reports CodeBERT MRR@100 of .1886 for random selection, .2220 for probability selection, and .2221 for entropy selection. These results make the relative value of the position score a specific, testable replication target.

Candidate recommendation and automatic execution answer different questions. SSQR already reports .1726 for Top-1 and .2221 for Top-3, and describes a workflow in which users select from recommended rewrites. We therefore separate the label-assisted opportunity in a candidate set from utility realized by an executable policy. We also ask a more specific question: how does the relative advantage of a position score change when the same pool is evaluated under a single-candidate contract rather than a best-of-three opportunity contract? A direct paired comparison can answer this question without treating the familiar Top-1/Top-3 gap as a new discovery.

We study these questions through an external replication on CoSQA, which pairs human-written web queries with code targets \cite{cosqa2021}. The executed reconstruction has an explicit training, decoding, representation, ranking, and aggregation contract. A same-implementation bridge to source data and finite implementation sensitivities accompany the CoSQA evaluation. The datasets also differ in corpus, labels, and difficulty; the bridge establishes the scope of the reconstruction rather than isolating query origin as a causal factor.

Related work already studies query-variant selection, original-query references, and decision-oriented evaluation. QueryRoute compares supervised and performance-prediction selectors over frozen variants and rankings \cite{queryroute2026}; Arabzadeh et al. distinguish predictive correlation from actual variant-selection utility \cite{qppvariant2026}. Code-retrieval rewriting has also been evaluated across encoders and rewriting strategies \cite{gurioli2026}. Our focus is SSQR's insertion-position mechanism and the interpretation of its specified comparison under controlled candidate and execution contracts.

We address four research questions: **RQ1**, how the reconstructed position-selection advantage behaves in the external replication and under matched evaluation contracts; **RQ2**, how much candidate gain is realized by matched adoption heads, simple multi-candidate fusion, and a frozen verification/fallback pipeline; **RQ3**, what conditional utility generation-side uncertainty provides at its tested decision positions; and **RQ4**, how these findings depend on the retriever and two T5 weight sets.

The study makes three contributions. First, it provides a frozen external test of SSQR's specified position-selection comparison, supported by data-overlap and implementation-fidelity evidence. Second, it directly contrasts position-score advantages under two evaluation contracts on the same queries and candidate pool. Third, it connects these comparisons to executable decisions through matched-supervision controls, multi-candidate fusion, and stage-level gain, harm, and cost accounting, with independently replayable outputs.

The frozen comparison uses 7,718 queries in 1,963 connected groups. CodeBERT ENTR minus twenty-repetition RAND best-of-three is −.000837 MRR@100, with group-bootstrap 95% interval [−.003255, +.001541]; both frozen 95% intervals lie inside the prespecified ±.005 equivalence region. On the 1,490-query development pool, ENTR's advantage over uniform single selection is +.027936, compared with +.002368 under best-of-three versus RAND20. Their direct paired difference is +.025568, with a four-comparison interval [+.014168, +.036618]. Probability selection behaves similarly. Thus a useful candidate-ranking signal can coexist with a small advantage in the reported opportunity comparison and with below-original automatic Top-1 utility.

Executable controls complete the interpretation. On the development pool, adding generation confidence to a matched adoption head does not establish extra adoption gain. Fixed three-list and original-plus-three-list fusion reduce harm relative to forced Top-1, especially for BM25, while neither establishes positive utility over the original query. On all 8,530 frozen H queries, CodeBERT's gain selector, verified selector, and risk-gated policy produce ΔMRR of +.000079, +.000847, and −.000178. The risk stage records 60 harmful outcomes among 1,006 adoptions and retains .77% of available positive gain. Matched two-weight diagnostics preserve positive best-of-three opportunity in both datasets. These findings motivate a concrete reporting practice for code rewriting: measure position advantage against random selection, net utility against the original query, and the gains, harms, and costs of the actual execution policy separately.
''',
'related_work_draft.md': r'''# Related Work（全文核对后整合稿，2026-10-20）

## Query rewriting for code search

SSQR learns continuous-span corruption with T5, enumerates query insertion positions, and compares entropy, probability, and random position selection \cite{ssqr2023}. Its Table 4 supplies the specified replication comparison; its Table 5 already separates Top-1 and Top-3 performance. CoSQA supplies human-written web queries paired with code targets \cite{cosqa2021}. Our external replication brings the SSQR position mechanism to this query source with an overlap audit, a source-data bridge, and an explicit reconstruction contract. We distinguish this method-specific replication from a general claim that code rewriting can hurt retrieval.

Gurioli et al. compare rewriting strategies across six CoIR tasks, five encoders, and three rewriter families, including CosQA \cite{gurioli2026}. Their setting includes query–corpus and corpus-only transformations. The ΔH diagnostic in that work is Shannon entropy of empirical token frequencies over a batch or corpus; it is not entropy of T5's predictive distribution and is not an epistemic/aleatoric decomposition. Their diagnostic is explicitly corpus-level. We keep the corpus fixed and test SSQR insertion-position scores, candidate opportunity, and specified adoption or fusion actions. The cited version is an arXiv preprint; its formal venue is not inferred.

## Query-variant selection and decision utility

QueryRoute evaluates selection over frozen original queries, variants, and retrieval rankings, using supervised routing, pre- and post-retrieval QPP, neural predictors, and LLM judges \cite{queryroute2026}. It measures help/hurt outcomes, oracle opportunity, and retriever dependence. Its retrieval-based judges can assess documents against the original query. These are direct precedents for candidate selection and original-query evidence. Our contribution is the SSQR-specific replication and the controlled single-candidate versus best-of-three comparison, rather than the general existence of a selection gap.

Arabzadeh et al. distinguish QPP correlation from query-variant selection utility in RAG pipelines, and compare original-query, pre-retrieval, and post-retrieval references \cite{qppvariant2026}. They also distinguish retrieval quality from downstream answer quality. We evaluate all compared policies using the same RR utility, so our contract contrast concerns candidate budget and opportunity aggregation rather than a change from retrieval labels to answer quality. Together, these works motivate evaluating decisions directly; they do not establish the outcome of our method-specific contrast.

## Selective processing and verification

Mothe and Ullah use baseline-relative risk and reward to select retrieval configurations and then choose configurations per query \cite{mothe2023}. Their discussion includes selective expansion, query-performance prediction, and the extra cost of fusion, which can exceed the best individual configuration. Baseline-relative loss, safe fallback, and fusion-versus-single distinctions therefore provide established tools for our controls. Standard RRF supplies an executable comparator; the best RR among individual rewrites is not an upper bound on a fused ranking.

SELF-multi-RAG learns when to retrieve and rewrite in conversational QA, then evaluates document relevance and answer groundedness and utility after retrieval \cite{roy2024}. Post-retrieval verification is thus an existing decision point. Our frozen code-search stages quantify how a specified selection and verification policy changes net gain, harmful adoption, positive-gain retention, and paid verification cost. The empirical risk counts do not constitute a population risk certificate.

## Position of this study

The closest work already covers general selection gaps, decision utility, query rewriting for code retrieval, and retrieval-based verification. This study contributes a specific external test of SSQR's position-selection comparison and a same-pool paired analysis of its evaluation contracts, followed by executable and matched-supervision controls. We retain the distinct roles of the frozen confirmation, development diagnostics, two-weight sensitivity, and previous multi-seed experiments. The result is a replication with controlled empirical extensions; single-pass predictive entropy is not relabelled as a newly validated uncertainty decomposition.

核对依据：`results/rene_multicandidate_fusion_20261020/PRIOR_WORK_COMPARISON.md`；六项全文来源及哈希：`results/rene_contribution_fulltext_20261020/REVIEW.json`。本稿删除旧版未经核实的引用占位，不以其支撑研究空白判断。
''',
'contribution_evidence.md': '''# 贡献—证据对应与本轮完成边界

日期：2026-10-20。主线保持“SSQR式候选收益在自动采纳下能实现多少”。本表是论文组织依据，不修改已冻结CLAIM或实验协议。

|论文贡献|直接证据|能够支持的表述|解释范围|
|---|---|---|---|
|指定位置选择比较的外部复现|T1；源侧桥接及实现差异表|重建实现中，CoSQA指定ENTR−RAND20 best3比较符合事前±.005等效规则|原checkpoint与生产聚合未完全恢复；不归因为真实查询来源|
|同池预算/机会契约的直接对照|T2，特别是CoSQA/CodeBERT直接差+.02556783及校正区间|位置分数的相对优势随所测契约改变；熵单选优于随机仍可低于原查询|PROB相近；预算与max聚合共同改变；不是熵独有的纯机制效应|
|候选到执行的受控经验扩展|T3同监督、T4可执行融合、T5冻结阶段；T6稳健性与独立artifact|量化具体政策如何实现收益、改变损害并付出成本；排查单纯固定Top1这一解释|不将未建立正收益推广为所有自动政策失败，不称新风险证书|
|UQ位置证据的完整定位|T3与T7|保留1536全流程正增量及1024指定等效/未决；解释各自估计对象|不同数据、动作、监督不合并；不以后一实验因果解释前一实验|

## 本次补齐

1. 两权重历史匹配样本的base排名、逐query/候选RR、组区间与全得分独立复核；旧摘要可恢复到浮点误差范围。
2. 7张机器表、38行、235个可追溯来源单元；更新大纲、引言、近邻定位和结果契约。
3. 增量缓存artifact与工作区外隔离重放；以最终验收报告记录实际完成范围，独立重放与模型训练分别记账。

## 与当前贡献相关的后续重点

- **全文交付**：方法/数据/统计说明、结果正文压缩、讨论和图表排版、匿名与许可清单、最终投稿系统要求检查。当前是有证据源稿，不是已排完的投稿全文。
- **保真度**：原权重和生产聚合缺失按实现差异表说明；两权重不能恢复不可得作者checkpoint。
- **若主张来源因果**：再做同目标、同语料的用户query/docstring配对C；当前稿不作这一因果主张，故不自动启动。
- **若主张新策略泛化或新UQ有效**：先有具体新增假设，再审新数据资格、组功效与单一策略冻结。CoSQA已消费，不能换名重做独立确认；现有冻结确认仍然有效。

RENE的贡献依托指定复现与新的受控比较，不以结果转正或风险证书签发为门槛。同行工作覆盖的一般概念不列为首创。投稿判断取决于全文论证、复现价值与会场范围，审计PASS不替代这些判断。
'''
}

CONTRACT_APPEND = '''
## 2026-10-20受控扩展及整合

|新增证据|路径|统计角色与范围|
|---|---|---|
|A同池单选/三选直接差|`results/rene_core_step_a_20261020/`|源2048/594、D1490/512，事后诊断；四j项同时校正；PROB与精确随机三选对照|
|B固定ENTR Top1的匹配监督|`results/rene_core_step_b_20261020/`|D双检索器五折四Ridge头；区间条件于固定OOF模型；零阈值及同量采用分别报告|
|多候选融合|`results/rene_multicandidate_fusion_20261020/`|D、固定ENTR3、RRF k60/等权唯一文本；四项相对NoRewrite校正；原句加入的条件对照|
|两套T5权重证据补齐|`results/rene_envelope_completion_20261020/`|每域历史匹配512/512；摘要已知后的排名补齐，非新确认；八项机会/Top1效应校正|
|整合表格|`results/rene_evidence_integration_20261020/`|9个机器来源，7表38行235来源单元；来源哈希与JSON指针；不重新计算H结果|

两套T5同属base家族。新补齐的144855715个全语料FP32得分、10138条Top100排名已独立重算；实际模型16查询及16代码样例用于验证向量与语料映射，不能将32样例写成全部编码独立重跑。旧四摘要最大差3.33e−16。FP64点积抽查是数值口径检查，不替代FP32并列排序契约。

RRF生成新排名，best3不是其上界。RRF3/4每query唯一输入2.9799/3.9799是计数而非延迟，原始生成成本按共享候选另列。直接A对照的预算和max聚合共同改变，不能宣称单一因素因果；B与A估计对象不同，不能用显著性差异证明阶段交互。

整合主表见`integrated_results.md`，贡献判定见`contribution_evidence.md`。旧稿已快照至`history/20261020-before-evidence-integration/`。本次不改正式CURRENT/STABLE、H确认或风险状态。
'''

BIB_APPEND = r'''
@inproceedings{queryroute2026,
  title = {Route Me If You Can: A Benchmark for Query Reformulation Selection},
  author = {Le, Hai Son and Arabzadeh, Negar and Bigdeli, Amin and Hamidi Rad, Radin and Ebrahimi, Sajad and Clarke, Charles L. A. and Bagheri, Ebrahim},
  booktitle = {Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
  year = {2026},
  doi = {10.1145/3799682.3840215},
  url = {https://arxiv.org/abs/2609.14885},
  note = {Author full-text version; venue and DOI stated in that version}
}

@inproceedings{qppvariant2026,
  title = {Can QPP Choose the Right Query Variant? Evaluating Query Variant Selection for RAG Pipelines},
  author = {Arabzadeh, Negar and Drozdov, Andrew and Bendersky, Michael and Zaharia, Matei},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year = {2026},
  doi = {10.1145/3805712.3808571},
  url = {https://arxiv.org/abs/2604.22661},
  note = {Author full-text version; venue and DOI stated in that version}
}

@misc{gurioli2026,
  title = {Do not copy and paste! Rewriting strategies for code retrieval},
  author = {Gurioli, Andrea and Pennino, Federico and Gabbrielli, Maurizio},
  year = {2026},
  eprint = {2605.08299},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2605.08299}
}

@misc{mothe2023,
  title = {Selective Query Processing: a Risk-Sensitive Selection of System Configurations},
  author = {Mothe, Josiane and Ullah, Md Zia},
  year = {2023},
  eprint = {2305.18311},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2305.18311}
}

@inproceedings{roy2024,
  title = {Learning When to Retrieve, What to Rewrite, and How to Respond in Conversational QA},
  author = {Roy, Nirmal and Ribeiro, Leonardo F. R. and Blloshmi, Rexhina and Small, Kevin},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2024},
  year = {2024},
  pages = {10604--10625},
  publisher = {Association for Computational Linguistics},
  doi = {10.18653/v1/2024.findings-emnlp.622},
  url = {https://aclanthology.org/2024.findings-emnlp.622/}
}
'''

def main():
    for item in json.loads((HISTORY/'SNAPSHOT.json').read_text(encoding='utf-8'))['files']:
        path = HISTORY / Path(item['path']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    for name, text in DOCS.items():
        (PAPER/name).write_text(text, encoding='utf-8')
    contract=(HISTORY/'results_contract.md').read_text(encoding='utf-8')
    contract=contract.replace('执行版本，2026-10-19','证据整合版，2026-10-20')
    (PAPER/'results_contract.md').write_text(contract+CONTRACT_APPEND, encoding='utf-8')
    (PAPER/'references_verified.bib').write_text((HISTORY/'references_verified.bib').read_text(encoding='utf-8')+BIB_APPEND, encoding='utf-8')
    prefix='''# 后继整合入口（2026-10-20）

本文件原桥接结果保留。唯一确认、A/B、可执行融合、全H阶段、两T5权重及旧UQ定位见 `integrated_results.md`；逐单元来源见 `results/rene_evidence_integration_20261020/TABLES.json`。下方桥接结果不替换原冻结确认。

'''
    (PAPER/'replication_results.md').write_text(prefix+(HISTORY/'replication_results.md').read_text(encoding='utf-8'), encoding='utf-8')
    print('Integrated manuscript sources; six originals retained and hash-verified.')

if __name__ == '__main__':
    main()
