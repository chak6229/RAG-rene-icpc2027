from __future__ import annotations

import csv
import json

import numpy as np

from .common import COMPONENTS, NAME, OUT, PACKAGE, ROOT, now, read, record, require, sha, verify_design, write


def ci_text(interval):
    return '[' + ', '.join(f'{v:+.6f}' for v in interval) + ']'


def primary_rows(result):
    output = []
    for domain, data in result['domains'].items():
        for contrast, summary in data['views'][NAME]['summaries'].items():
            if contrast in COMPONENTS:
                output.append({'domain': domain, 'system_comparison': NAME, 'contrast': contrast, **summary})
        for ret, records in data['paired_differences'].items():
            for contrast, summary in records.items():
                output.append({'domain': domain, 'system_comparison': NAME + '_minus_' + ret, 'contrast': contrast, **summary})
    return output


def export_csv(path, records):
    fields = list(dict.fromkeys(k for row in records for k in row))
    with path.open('x', encoding='utf-8-sig', newline='') as h:
        writer = csv.DictWriter(h, fields)
        writer.writeheader()
        for row in records:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def main():
    require(read(OUT / 'FINAL_ACCEPTANCE.json')['passed'], 'Scientific audit not passed')
    design = verify_design(private=True)
    result = read(OUT / 'RESULTS.json')
    primary = primary_rows(result)
    require(len(primary) == 12, 'Primary export family mismatch')
    export_csv(OUT / 'PRIMARY_EFFECTS.csv', primary)
    all_effects = []
    for domain, data in result['domains'].items():
        for ret, view in data['views'].items():
            all_effects += [{'domain': domain, 'system_comparison': ret, 'contrast': c, **s} for c, s in view['summaries'].items()]
        for ret, records in data['paired_differences'].items():
            all_effects += [{'domain': domain, 'system_comparison': NAME + '_minus_' + ret, 'contrast': c, **s} for c, s in records.items()]
    export_csv(OUT / 'ALL_EFFECTS.csv', all_effects)
    lines = ['# 第三个检索器：UniXcoder 固定候选扩展', '',
        '本轮检验数量分量B、聚合分量G如何迁移到一个额外的代码检索系统，并直接比较与CodeBERT、BM25的分量差。',
        '采用作者公开的UniXcoder-base零样本检索权重和官方code256/nl128、encoder-only、归一化均值池化。没有本地训练或新增T5候选。', '',
        '## 主要比较', '',
        'B为ENTR相对精确随机的优势从单候选mean到三候选mean的减少量；G为三候选mean到标签辅助best3的进一步减少量。正值表示所测相对优势减弱，负值表示反向。',
        '两个场景各自包含新模型的B/G，以及新模型分别减CodeBERT/BM25的B/G，共12项，统一Bonferroni校正；以下两套区间同时用于方向判断。', '',
        '|场景|比较|分量|估计|校正cluster-t区间|校正bootstrap区间|方向|',
        '|---|---|---|---:|---|---|---|']
    for r in primary:
        lines.append(f"|{r['domain']}|{r['system_comparison']}|{'B' if '_B_' in r['contrast'] else 'G'}|{r['mean']:+.8f}|{ci_text(r['family12_ci_cluster_t'])}|{ci_text(r['family12_ci_group_bootstrap'])}|{r['direction']}|")
    lines += ['', '## 同预算选择优势与原查询参照', '',
        '下表为描述性点估计，逐项95%区间见ALL_EFFECTS.csv/RESULTS.json。best3依赖相关标签，不是可执行部署收益。', '',
        '|场景|检索器|ENTR−随机：1 mean|3 mean|3 best3|Top1−原查询|best3−原查询|',
        '|---|---|---:|---:|---:|---:|---:|']
    for domain, data in result['domains'].items():
        for ret, view in data['views'].items():
            s = view['summaries']
            vals = [s[c]['mean'] for c in ('S_ENTR_k1_mean', 'S_ENTR_k3_mean', 'S_ENTR_k3_max', 'delta_ENTR_k1_mean', 'delta_ENTR_k3_max')]
            lines.append('|'+domain+'|'+ret+'|'+'|'.join(f'{v:+.8f}' for v in vals)+'|')
    lines += ['', '## 对主线的作用和范围', '']
    for domain, data in result['domains'].items():
        s = data['views'][NAME]['summaries']
        directions = [s[c]['direction'] for c in COMPONENTS]
        if directions == ['positive', 'positive']:
            message = 'B/G两项均在校正后正向，支持所测数量、聚合分解扩展到这一UniXcoder检索条件。'
        elif any(d == 'negative' for d in directions):
            message = '至少一个分量在校正后反向，为所测分解的检索条件边界提供证据；按表中各项分别解释。'
        elif any(d == 'positive' for d in directions):
            message = '部分分量在校正后正向；其余仍未决，不能写成两项均已推广。'
        else:
            message = 'B/G两项在本比较族下均未决；不把区间跨零解释为无影响、等效或成功推广。'
        lines.append(f'- {domain}：{message}')
    lines += ['',
        '相对旧检索器的分量差须依据同查询配对区间，而不是比较各自显著与否。分量差不是检索器整体性能排名。PROB和ENTR−PROB为描述对照，不以B/G大小宣称某种分数总体更优。',
        '本研究固定同一改写池、代码文本和标签，比较完整检索系统。模型训练、池化、归一化和官方输入长度共同变化；不宣称单独的架构因果。UniXcoder仍是RoBERTa架构，本轮未扩展改写器家族。',
        'CoSQA和StaQC均已消费；这是新检索器效果前固定的模型扩展，不是新独立样本确认，也未做外部预注册。CoSQA为1490查询/512组；StaQC为13312查询/13312组。固定现有全样本，不根据效果扩样；功效仅有旧模型方差情景，不保证新模型精度。',
        'UniXcoder使用CodeSearchNet的NL-PL预训练，未知预训练暴露和语义克隆仍有边界。StaQC的单已知正例标签与Bing/函数场景不同。新增一个模型不建立所有检索器或所有自动采纳政策的普遍结论。', '',
        '## 复核与复算', '',
        '9项契约测试、官方例子/独立实际模型前向、完整向量→Top100审计先于新效果；随后用Fraction RR、独立CDF随机期望及配对统计复算。具体计数与误差见PREOPEN_AUDIT.json和FINAL_ACCEPTANCE.json。',
        'MODEL_PROVENANCE.json记录上游commit、Git/LFS哈希、许可及镜像传输；DESIGN.json绑定输入、代码、权重和保护状态。',
        '完整重放入口：python -m experiments.rene_third_retriever_20261020.audit --after；需要本地绑定数据/排名。新推理使用run模块，已完成效果不会再次开结果。',
        'risk_certified=false；本轮不改原确认或生产配置。', '']
    (OUT / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    method = ['# 第三检索器扩展结果', '',
        '我们固定CoSQA开发样本及StaQC既有样本的改写池、标签和代码语料，将检索系统扩展为公开UniXcoder-base的官方零样本代码检索配置。采用encoder-only前缀、code256/nl128截断、非padding均值池化和单位范数点积。该扩展没有新增训练或T5生成。', '',
        '主要比较为两场景中的B/G和相对CodeBERT、BM25的同查询分量差，共12项统一Bonferroni校正，报告组bootstrap与cluster-t区间。数据已用于前期实验，因此本轮属于事前指定比较的模型扩展，不称新样本确认。', '',
        '|场景|UniXcoder分量|估计|校正cluster-t区间|校正bootstrap区间|', '|---|---|---:|---|---|']
    for r in primary:
        if r['system_comparison'] == NAME:
            method.append(f"|{r['domain']}|{'B' if '_B_' in r['contrast'] else 'G'}|{r['mean']:+.8f}|{ci_text(r['family12_ci_cluster_t'])}|{ci_text(r['family12_ci_group_bootstrap'])}|")
    method += ['', '各分量与跨检索器直接差按完整报告解释。best3保持标签辅助机会指标身份，不能据此推断部署增益、ENTR独有机制或全部模型普遍性。',
               '完整结果及验收：results/rene_third_retriever_20261020/；公式、方向、范围见FINDINGS.md。', '']
    paper_path = ROOT / 'paper/rene2027/third_retriever_results.md'
    require(not paper_path.exists(), 'Supplement already exists')
    paper_path.write_text('\n'.join(method), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4), sharey=False, constrained_layout=True)
    for ax, (domain, data) in zip(axes, result['domains'].items()):
        for ret, view in data['views'].items():
            summaries = [view['summaries'][c] for c in ('S_ENTR_k1_mean', 'S_ENTR_k3_mean', 'S_ENTR_k3_max')]
            means = np.array([r['mean'] for r in summaries])
            intervals = np.array([r['ci95_cluster_t'] for r in summaries])
            ax.errorbar(range(3), means, yerr=np.stack([means - intervals[:, 0], intervals[:, 1] - means]),
                        marker='o', capsize=3, label=ret, linewidth=1.5)
        ax.axhline(0, color='.55', linewidth=.8)
        ax.set_xticks(range(3), ['1 candidate\nmean', '3 candidates\nmean', '3 candidates\nbest3 (labels)'])
        ax.set_title(domain.capitalize())
        ax.set_ylabel('ENTR minus exact random (MRR@100)')
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Fixed candidate pools across three retrieval systems\nPointwise 95% cluster-t intervals; primary B/G contrasts use Bonferroni12', fontsize=11)
    fig.savefig(OUT / 'third_retriever.png', dpi=180)
    fig.savefig(OUT / 'third_retriever.pdf')
    plt.close(fig)
    # Independently read exported primary rows and compare every numerical field.
    with (OUT / 'PRIMARY_EFFECTS.csv').open(encoding='utf-8-sig', newline='') as h:
        exported = list(csv.DictReader(h))
    require(len(exported) == len(primary), 'Primary export row count')
    fields_checked = 0
    for source, exported_row in zip(primary, exported):
        for field, value in source.items():
            actual = exported_row[field]
            if isinstance(value, list):
                require(json.loads(actual) == value, 'Exported interval changed')
            elif isinstance(value, (int, float)):
                require(float(actual) == float(value), 'Exported number changed')
            else:
                require(actual == str(value), 'Exported label changed')
            fields_checked += 1
    outputs = [OUT / name for name in ('DESIGN.json', 'RESULTS.json', 'per_query.jsonl', 'MODEL_PROVENANCE.json',
         'PRECISION_SCENARIOS.json', 'TEST_RESULTS.json', 'PREFLIGHT.json', 'PREOPEN_AUDIT.json', 'FINAL_ACCEPTANCE.json',
         'PRIMARY_EFFECTS.csv', 'ALL_EFFECTS.csv', 'FINDINGS.md', 'third_retriever.png', 'third_retriever.pdf')]
    outputs += [paper_path] + sorted(PACKAGE.glob('*.py')) + [PACKAGE / 'DESIGN.md']
    write(OUT / 'DELIVERY_MANIFEST.json', {'created_utc': now(), 'files': [record(p) for p in outputs]})
    manifest = read(OUT / 'DELIVERY_MANIFEST.json')
    require(all(sha(ROOT / r['path']) == r['sha256'] for r in manifest['files']), 'Delivery binding mismatch')
    verify_design(private=True)
    write(OUT / 'DELIVERY_ACCEPTANCE.json', {'created_utc': now(), 'passed': True,
          'manifest': record(OUT / 'DELIVERY_MANIFEST.json'), 'primary_export_fields_checked': fields_checked,
          'primary_rows': len(primary), 'all_effect_rows': len(all_effects), 'files_checked': len(outputs),
          'scientific_acceptance': record(OUT / 'FINAL_ACCEPTANCE.json'), 'risk_certified': False})
    print({'delivery_passed': True, 'files': len(outputs), 'primary_rows': len(primary)}, flush=True)


if __name__ == '__main__':
    main()
