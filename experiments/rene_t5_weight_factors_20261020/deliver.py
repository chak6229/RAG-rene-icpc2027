"""Export audited fixed results; never changes/recomputes the scientific analysis."""
from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .common import (DOMAINS, OUT, ROOT, VIEWS, WEIGHTS, now, primary_names,
                     read, record, rows, sha, verify, write)


def number(value, digits=6):
    return f'{value:+.{digits}f}'


def interval(values, digits=6):
    return '[' + ', '.join(number(v, digits) for v in values) + ']'


def csv_file(path, records):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def exports(results):
    long_rows, primary_rows = [], []
    for domain, d in results['domains'].items():
        for view, v in d['views'].items():
            for section in ('values', 'contrasts'):
                for name, r in v[section].items():
                    record_ = {'domain': domain, 'view': view, 'section': section, 'metric': name,
                               'mean': r['mean'], 'queries': r['queries'], 'groups': r['groups'],
                               'se_cluster': r['se_cluster'],
                               'ci95_t_low': r['ci95_cluster_t'][0], 'ci95_t_high': r['ci95_cluster_t'][1],
                               'ci95_boot_low': r['ci95_group_bootstrap'][0], 'ci95_boot_high': r['ci95_group_bootstrap'][1],
                               'groups_with_nonzero': r['groups_with_nonzero'],
                               'family24_t_low': r.get('family24_ci_cluster_t', ['', ''])[0],
                               'family24_t_high': r.get('family24_ci_cluster_t', ['', ''])[1],
                               'family24_boot_low': r.get('family24_ci_group_bootstrap', ['', ''])[0],
                               'family24_boot_high': r.get('family24_ci_group_bootstrap', ['', ''])[1],
                               'direction': r.get('direction', 'secondary_pointwise95')}
                    long_rows.append(record_)
                    if f'{view}::{section}::{name}' in primary_names():
                        primary_rows.append(record_)
    csv_file(OUT / 'all_summaries.csv', long_rows)
    csv_file(OUT / 'primary_contrasts.csv', primary_rows)
    cells = []
    for domain in DOMAINS:
        for arm in WEIGHTS:
            v = results['domains'][domain]['views'][arm]
            for policy in ('ENTR', 'PROB', 'RAND_exact'):
                for k in (1, 2, 3):
                    for aggregation in ('mean', 'max'):
                        metric = f'{policy}_k{k}_{aggregation}'
                        rr, delta = v['values'][metric], v['contrasts']['delta_' + metric]
                        s = v['contrasts'].get('S_' + metric)
                        cells.append({'domain': domain, 'weight': arm, 'policy': policy, 'k': k, 'aggregation': aggregation,
                                      'queries': rr['queries'], 'groups': rr['groups'], 'RR_at100': rr['mean'],
                                      'delta_NoRewrite': delta['mean'],
                                      'selection_advantage_exact_random': s['mean'] if s else 0.,
                                      'S_ci95_t_low': s['ci95_cluster_t'][0] if s else 0.,
                                      'S_ci95_t_high': s['ci95_cluster_t'][1] if s else 0.})
    csv_file(OUT / 'factorial_cells.csv', cells)
    return len(long_rows), len(primary_rows), len(cells)


def plots(results):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    colors = {'ENTR': '#1963a6', 'PROB': '#aa4e10'}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.7), sharex=True)
    for i, domain in enumerate(DOMAINS):
        for j, aggregation in enumerate(('mean', 'max')):
            ax = axes[i, j]
            for arm_i, arm in enumerate(WEIGHTS):
                for policy_i, policy in enumerate(('ENTR', 'PROB')):
                    y, lo, hi = [], [], []
                    for k in (1, 2, 3):
                        r = results['domains'][domain]['views'][arm]['contrasts'][f'S_{policy}_k{k}_{aggregation}']
                        y.append(r['mean']); lo.append(r['mean'] - r['ci95_cluster_t'][0]); hi.append(r['ci95_cluster_t'][1] - r['mean'])
                    offset = ((arm_i * 2 + policy_i) - 1.5) * .035
                    label = f'{policy}, ' + ('local CQC' if arm == 'local' else 'T5-base')
                    ax.errorbar(np.arange(1, 4) + offset, y, yerr=[lo, hi],
                                color=colors[policy], marker='o' if arm == 'local' else 's',
                                linestyle='-' if arm == 'local' else '--', linewidth=1.5,
                                capsize=2.5, markersize=4.5, label=label)
            ax.axhline(0, color='#666666', linewidth=.8)
            ax.set_title(('AdvTest' if domain == 'source' else 'CoSQA') + ' | ' + ('mean' if aggregation == 'mean' else 'best-k opportunity'))
            ax.set_xticks([1, 2, 3]); ax.set_xlim(.8, 3.2)
            ax.grid(axis='y', alpha=.16)
            if j == 0:
                ax.set_ylabel('Selector advantage over exact random\n(RR@100)')
            if i == 1:
                ax.set_xlabel('Selected candidate count k')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, .953))
    fig.suptitle('Matched T5 weights: selection advantage under each evaluation contract', y=.994, fontsize=13)
    fig.text(.5, .015, '512 matched queries / 512 groups per domain. Pointwise 95% cluster-t intervals. Best-k uses labels.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .91))
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'selection_curves.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.8))
    specifications = [(policy, view) for policy in ('ENTR', 'PROB') for view in VIEWS]
    labels = [f'{policy} | ' + {'local': 'local CQC', 't5base': 'T5-base', 'base_minus_local': 'base - local'}[view] for policy, view in specifications]
    for i, domain in enumerate(DOMAINS):
        for j, component in enumerate(('B', 'G')):
            ax = axes[i, j]
            for pos, (policy, view) in enumerate(specifications):
                r = results['domains'][domain]['views'][view]['contrasts'][policy + '_' + component]
                low, high = r['family24_ci_cluster_t']
                ax.errorbar(r['mean'], pos, xerr=[[r['mean'] - low], [high - r['mean']]],
                            color=colors[policy], capsize=3, linewidth=1.4, marker='o', markersize=5,
                            markerfacecolor=colors[policy] if r['direction'] != 'unresolved' else 'white')
            ax.axvline(0, color='#666666', linewidth=.8)
            ax.axhline(2.5, color='#cccccc', linewidth=.8)
            ax.set_yticks(range(6), labels); ax.invert_yaxis()
            ax.set_title(('AdvTest' if domain == 'source' else 'CoSQA') + ' | ' + component + (' (candidate count)' if component == 'B' else ' (aggregation)'))
            ax.set_xlabel('Difference in RR@100')
            ax.grid(axis='x', alpha=.16)
            ax.tick_params(axis='y', labelsize=9)
    fig.suptitle('B/G effects and direct paired weight differences', y=.994, fontsize=13)
    fig.text(.5, .014, '24-comparison Bonferroni cluster-t intervals. Filled: both corrected t and bootstrap intervals exclude zero.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .965))
    for extension in ('png', 'pdf'):
        fig.savefig(OUT / f'factor_contrasts.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def reports(results):
    def effect(domain, view, name):
        return results['domains'][domain]['views'][view]['contrasts'][name]

    table = ['| 数据 | 权重 | 规则 | B（校正区间） | G（校正区间） |', '|---|---|---|---|---|']
    en_table = ['| Dataset | Weight | Selector | B (adjusted CI) | G (adjusted CI) |', '|---|---|---|---|---|']
    for domain in DOMAINS:
        for arm in WEIGHTS:
            for p in ('ENTR', 'PROB'):
                b, g = effect(domain, arm, p + '_B'), effect(domain, arm, p + '_G')
                entries = [f'{number(r["mean"])} {interval(r["family24_ci_cluster_t"])}' for r in (b, g)]
                row = f'| {"AdvTest" if domain == "source" else "CoSQA"} | {arm} | {p} | {entries[0]} | {entries[1]} |'
                table.append(row); en_table.append(row)
    paired_table = ['| 数据 | 规则 | ΔB：base−local（校正区间） | ΔG：base−local（校正区间） |', '|---|---|---|---|']
    for domain in DOMAINS:
        for p in ('ENTR', 'PROB'):
            b, g = effect(domain, 'base_minus_local', p + '_B'), effect(domain, 'base_minus_local', p + '_G')
            paired_table.append(f'| {"AdvTest" if domain == "source" else "CoSQA"} | {p} | {number(b["mean"])} {interval(b["family24_ci_cluster_t"])} | {number(g["mean"])} {interval(g["family24_ci_cluster_t"])} |')
    path_table = ['| 数据 | 权重 | S(1,mean) | S(3,mean) | S(3,best3) |', '|---|---|---:|---:|---:|']
    for domain in DOMAINS:
        for arm in WEIGHTS:
            vals = [effect(domain, arm, 'S_ENTR_' + suffix)['mean'] for suffix in ('k1_mean', 'k3_mean', 'k3_max')]
            path_table.append(f'| {"AdvTest" if domain == "source" else "CoSQA"} | {arm} | ' + ' | '.join(number(v) for v in vals) + ' |')
    counts = Counter(effect(d, v, p + '_' + c)['direction'] for d in DOMAINS for v in VIEWS for p in ('ENTR', 'PROB') for c in ('B', 'G'))
    findings = '\n'.join([
        '# 两套 T5 权重的完整 B/G 匹配对照', '',
        '完整对照已补齐。CoSQA 的 ENTR 在两套权重下均呈 B、G 正向点估计，源域 B 的点估计随权重变号。24 项统一校正后，ENTR 的八项权重内 B/G 均未决；八项 ENTR/PROB 的权重直接差异也均未决。因此，这轮支持将完整分解纳入权重敏感性证据，尚未建立“B/G 不依赖权重”的结论。', '',
        'PROB 在基础 T5 下的 G 在两域均为正向且通过校正；其余 22 项主比较未决。没有校正后反向项。ENTR−PROB 的 B/G 及其权重交互均为次要描述比较，95% 区间跨零，未建立熵独有分量。', '',
        '## 匹配与估计对象', '',
        '- AdvTest、CoSQA 各 512 条查询 / 512 个仓库或关联组；两套权重共享查询、原查询 RR、语料、标注、检索器及位置池规模。仅重用原匹配 512 样本，未替换成 2,048/1,490 全量结果。',
        '- local 为已有 CQC T5，t5base 为已有基础 T5；两者都属于 T5 家族。比较的是整套权重条件，不能单独归因为 CQC 训练。',
        '- 原始位置池逐候选 RR 共 18,680 项；所有重复文本保留位置概率质量。ENTR/PROB 仅使用原生成分数排序，按原位置打破并列。',
        '- k=1/2/3 指从已有生成池选取的候选数，不是生成预算。mean 是所选候选均匀采用的期望 RR；best-k 是依赖标签选最好的候选机会，不是自动采纳或融合收益。',
        '- S(k,a)=规则 RR−同位置池精确均匀随机 RR；B=S(1,mean)−S(3,mean)，G=S(3,mean)−S(3,best3)，J=B+G。它是指定路径的算术分解。', '',
        '## 完整分量与权重差', '',
        '以下为 24 项全族双侧 Bonferroni 校正的 cluster-t 区间。机器结果同时提供独立组 bootstrap 校正区间；方向判定要求两套区间一致且至少 20 个非零组。', '',
        *table, '', *paired_table, '',
        '两权重的直接差异区间跨零，且范围宽于 ±0.005，不能解释为等效、不变或训练影响很小。CoSQA ENTR 的 ΔB 为 +0.013152，但校正区间 [−0.016157,+0.042461]，目前无法区分有意义的权重变化与抽样不确定性。', '',
        '## 同池 ENTR 优势路径', '', *path_table, '',
        'CoSQA 两权重的点估计都随候选数量及 best3 聚合而减弱。描述性 95% 区间中，两权重 G 为正、基础 T5 的 B 为正；这些不能替代 24 项校正结论。AdvTest 的局部 CQC 权重 B 点估计为负，提示数量分量应按场景与权重报告。', '',
        '## 对论文主线的作用', '',
        '本轮将“两权重 Top1/best3 包络”推进为完整、匹配查询上的数量/聚合分解，并提供直接权重交互，关闭了“完整 B/G 未覆盖”的分析缺项。它补充候选机会与自动收益之区分及评价契约敏感性；目前跨权重证据应表述为“CoSQA 点估计同向，正式校正下仍未决”。既有 StaQC 冻结确认和第三检索器结果继续承担各自明确的结论。', '',
        '这些都是已消费样本上的事后敏感性分析，方案在本轮新分解计算前本地封存；历史 Top1/best3 与全量本地分解已知，不称新独立确认。没有恢复作者 checkpoint、原训练 CSV 或生产聚合，也不支持跨改写器家族、历史全量训练归因或完整采纳政策的迁移。', '',
        '## 核验与交付', '',
        f'- 9 项契约测试通过；16,384 项旧 Top1/best3/PROB/随机/原句等逐查询锚点精确复现（最大差 0）。',
        '- 独立从已验收 Top100 排名与 qrels 复算 18,680 候选及 2,048 原查询 Fraction RR；独立 CDF 精确随机、131,072 因素字段、65,536 配对字段、384 个汇总均值、24 项主统计通过。最大因素/配对误差 2.37e−16。',
        '- 使用 20,000 次共享整组 bootstrap；两权重配对后再统计。没有新训练、生成、模型前向或检索；既有正式指针及历史交付哈希保持。',
        '- 结果：RESULTS.json、per_query.jsonl、paired_per_query.jsonl；表格：factorial_cells.csv（72 格）、primary_contrasts.csv（24 项）、all_summaries.csv（384 项）；图：selection_curves、factor_contrasts（PNG/PDF）。',
        '- 论文补充稿：paper/rene2027/t5_weight_factor_results.md；重放说明：REPLAY.md。', '',
    ])
    (OUT / 'FINDINGS.md').write_text(findings, encoding='utf-8')
    paper = '\n'.join([
        '# Matched T5-weight sensitivity of candidate-count and aggregation effects', '',
        'We extended the two-weight Top1/best3 comparison to a complete candidate-count × aggregation analysis using the existing per-candidate retrieval cache. Both the local CQC T5 and public T5-base were evaluated on exactly the same 512 AdvTest queries (512 repository groups) and 512 CoSQA queries (512 connected groups). Original-query rankings, corpus, qrels, the CodeBERT retriever, and original insertion-position counts were held fixed. No training, candidate generation, model inference, or retrieval was repeated.', '',
        'For ENTR and PROB, we selected the first k∈{1,2,3} positions by the original generation score with position-based tie breaking. Exact uniform sampling without replacement over positions retained duplicate-text probability mass. The mean aggregation is the expected RR@100 under uniform adoption within the selected prefix; best-k is a label-assisted opportunity measure. Writing S(k,a) for the selector advantage over exact random, we used B=S(1,mean)−S(3,mean), G=S(3,mean)−S(3,best3), and J=B+G. These quantities specify an arithmetic path through the evaluation contracts; k denotes selected count from an already generated pool.', '',
        'The locally sealed sensitivity design specified 24 primary comparisons: two datasets × two selectors × two components × {local, base, paired base−local}. We report two-sided Bonferroni-adjusted cluster-t intervals and 20,000 whole-group percentile bootstrap intervals, using shared draws across both weights. A direction is established only when both adjusted intervals exclude zero on the same side and at least 20 groups have nonzero contributions. Other contrasts, including ENTR−PROB and J, use descriptive pointwise 95% intervals.', '',
        *en_table, '',
        'On CoSQA, both weights yielded positive ENTR point estimates for B and G: local B=+0.005815 and G=+0.009869; base B=+0.018967 and G=+0.012087. The ENTR advantage followed +0.021066→+0.015251→+0.005381 for local and +0.034753→+0.015786→+0.003699 for base, along k1-mean→k3-mean→k3-best. None of these eight within-weight ENTR component tests across the two datasets was resolved after the full correction. AdvTest B changed point-estimate sign between weights, from −0.005239 to +0.001237. PROB G was positive after correction under T5-base in both datasets (+0.009344 and +0.018360); the remaining primary comparisons were unresolved.', '',
        'All eight paired weight differences in B/G were unresolved after correction. For example, CoSQA ENTR ΔB=+0.013152 with adjusted cluster-t CI [−0.016157,+0.042461], and ΔG=+0.002217 with CI [−0.016541,+0.020975]. These intervals do not establish equivalence or weight invariance. ENTR−PROB component contrasts and their weight interactions had pointwise intervals spanning zero, providing no established entropy-specific increment.', '',
        'This analysis completes the matched two-weight B/G coverage and characterizes its precision. It extends the descriptive CoSQA pattern to the second T5 weight, while leaving corrected cross-weight robustness unresolved. The consumed samples and known earlier endpoint results make this a post-hoc sensitivity analysis, separate from the StaQC frozen confirmation. The comparison concerns complete weight conditions within one model family; it does not isolate the causal effect of CQC training, recover unavailable author artifacts, or establish cross-family or full-policy transfer.', '',
        'Validation: nine contract tests passed. All 16,384 old per-query anchors were recovered exactly. An independent replay used persisted Top100 rankings and qrels to recompute 18,680 candidate and 2,048 original-query reciprocal ranks as exact fractions, checked random maxima using an independent CDF formula, verified 196,608 factorial/paired fields and all 384 reported means, and independently reproduced the 24 primary statistics. The scientific result and inputs are hashed in results/rene_t5_weight_factors_20261020/.', '',
        'Figures: selection_curves.pdf (pointwise cluster-t intervals) and factor_contrasts.pdf (24-comparison adjusted cluster-t intervals). Complete bootstrap intervals and all 72 factorial cells are available in the CSV exports.', '',
    ])
    paper_path = ROOT / 'paper/rene2027/t5_weight_factor_results.md'
    with paper_path.open('x', encoding='utf-8') as stream:
        stream.write(paper)
    replay = '''# Replay

Use the workspace `.venv-models` Python runtime. The analysis consumes existing candidate JSONL, candidate RR, per-query anchors, persisted Top100 rankings and qrels listed in DESIGN.json. No model weights or accelerator are loaded. The matched samples have already been consumed.

Original workflow (new output directory, only once):

1. Run the contract suite `python -m unittest experiments.rene_t5_weight_factors_20261020.test_contract` and save TEST_RESULTS.json.
2. Run `python -m experiments.rene_t5_weight_factors_20261020.run freeze`.
3. Run `python -m experiments.rene_t5_weight_factors_20261020.run evaluate`.
4. Run `python -m experiments.rene_t5_weight_factors_20261020.audit`.
5. Run `python -m experiments.rene_t5_weight_factors_20261020.deliver`.

The scientific pipeline refuses to overwrite sealed files. For an independent replay of a completed result, copy the workspace or archive the new experiment directory first; preserve bound inputs and the original scientific files. The audit independently reconstructs RR, factorial fields, pairs, all summary means, and primary intervals. Delivery code was added after the scientific design was sealed and never changes its estimands or statistics.

No additional generation, retrieval, model spot check, or historical full-training attribution is part of this cache-only task. Existing evidence completion provides the prior retrieval/model acceptance chain. FINAL_ACCEPTANCE.json and DELIVERY_ACCEPTANCE.json describe the new scientific and export checks separately.
'''
    (OUT / 'REPLAY.md').write_text(replay, encoding='utf-8')
    return paper_path, dict(counts)


def check_exports(results):
    checks = 0
    for filename in ('all_summaries.csv', 'primary_contrasts.csv'):
        with (OUT / filename).open(encoding='utf-8-sig', newline='') as stream:
            records = list(csv.DictReader(stream))
        if len(records) != (384 if filename == 'all_summaries.csv' else 24):
            raise AssertionError('Incomplete summary export')
        for r in records:
            source = results['domains'][r['domain']]['views'][r['view']][r['section']][r['metric']]
            for column, field in [('mean', 'mean'), ('se_cluster', 'se_cluster')]:
                if float(r[column]) != source[field]:
                    raise AssertionError('CSV scalar differs')
                checks += 1
            for prefix, field in [('ci95_t', 'ci95_cluster_t'), ('ci95_boot', 'ci95_group_bootstrap'), ('family24_t', 'family24_ci_cluster_t'), ('family24_boot', 'family24_ci_group_bootstrap')]:
                for suffix, j in [('low', 0), ('high', 1)]:
                    if field in source:
                        if float(r[prefix + '_' + suffix]) != source[field][j]:
                            raise AssertionError('CSV interval differs')
                        checks += 1
                    elif r[prefix + '_' + suffix] != '':
                        raise AssertionError('Secondary correction fabricated')
    with (OUT / 'factorial_cells.csv').open(encoding='utf-8-sig', newline='') as stream:
        cells = list(csv.DictReader(stream))
    if len(cells) != 72:
        raise AssertionError('Factorial cells incomplete')
    for cell in cells:
        v = results['domains'][cell['domain']]['views'][cell['weight']]
        name = cell['policy'] + '_k' + cell['k'] + '_' + cell['aggregation']
        expected = {'RR_at100': v['values'][name]['mean'], 'delta_NoRewrite': v['contrasts']['delta_' + name]['mean']}
        s = v['contrasts'].get('S_' + name)
        expected.update(selection_advantage_exact_random=s['mean'] if s else 0., S_ci95_t_low=s['ci95_cluster_t'][0] if s else 0., S_ci95_t_high=s['ci95_cluster_t'][1] if s else 0.)
        for column, number_ in expected.items():
            if float(cell[column]) != number_:
                raise AssertionError('Factorial export differs')
            checks += 1
    return checks


def deliver():
    design = verify()
    result, audit, tests = read(OUT / 'RESULTS.json'), read(OUT / 'AUDIT.json'), read(OUT / 'TEST_RESULTS.json')
    if not audit['passed'] or not tests['passed'] or audit['results_sha256'] != sha(OUT / 'RESULTS.json'):
        raise AssertionError('Scientific acceptance missing')
    if (OUT / 'DELIVERY_ACCEPTANCE.json').exists():
        raise FileExistsError('Delivery already accepted')
    dimensions = exports(result)
    plots(result)
    paper, directions = reports(result)
    export_checks = check_exports(result)
    verify()
    write(OUT / 'FINAL_ACCEPTANCE.json', {
        'created_utc': now(), 'passed': True, 'scope': 'matched cache-only B/G sensitivity of two T5 weights',
        'design_sha256': sha(OUT / 'DESIGN.json'), 'results_sha256': sha(OUT / 'RESULTS.json'),
        'audit_sha256': sha(OUT / 'AUDIT.json'), 'scientific_tests': tests['tests'], 'primary_family': 24,
        'primary_directions': directions, 'audit_counts': audit['counts'], 'audit_max_errors': audit['max_errors'],
        'protected_files_unchanged': True, 'new_confirmation': False, 'new_training': False,
        'new_model_inference': False, 'new_retrieval': False, 'risk_certified': False,
        'scientific_source_files': [r for r in design['bound_files'] if r['path'].endswith('.py')],
    })
    files = sorted(p for p in OUT.iterdir() if p.is_file()) + [paper]
    manifest = {'created_utc': now(), 'files': [record(p) for p in files] + [record(Path(__file__))]}
    write(OUT / 'DELIVERY_MANIFEST.json', manifest)
    for item in manifest['files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise AssertionError('Delivery hash mismatch')
    write(OUT / 'DELIVERY_ACCEPTANCE.json', {
        'created_utc': now(), 'passed': True, 'manifest_sha256': sha(OUT / 'DELIVERY_MANIFEST.json'),
        'files_checked': len(manifest['files']), 'summary_rows': dimensions[0], 'primary_rows': dimensions[1],
        'factorial_cells': dimensions[2], 'numeric_export_fields_checked': export_checks,
        'plots_rendered': ['selection_curves.png', 'selection_curves.pdf', 'factor_contrasts.png', 'factor_contrasts.pdf'],
        'visual_review_record': 'See VISUAL_REVIEW.json after human-visible image inspection',
        'protected_files_unchanged': True, 'no_scientific_reanalysis': True,
    })
    print({'stage': 'delivery_complete', 'primary_directions': directions, 'export_checks': export_checks,
           'summary_rows_primary_rows_factorial_cells': dimensions}, flush=True)


if __name__ == '__main__':
    deliver()
