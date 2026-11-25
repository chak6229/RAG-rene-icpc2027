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
    lines = ['# Third Retrieval: UniXcoder Fixed candidate extension', '',
        "How this round ' s mass weight B, polymer mass G is moved to an additional code retrieval system and directly compares the difference between the amount of CodeBERT and BM25.",
        "Use the author ' s publicly available UniXcoder-base zero sample search weight and official code256/nl128, encoder-only, and normalized average values. No local training or new T5 candidate.", '',
        '## Main comparisons', '',
        'B is a reduction in the relative precision and random advantage of ENTR from single candidates mean to three candidates mean; G further reductions for three candidates mean to label aids best3. Positive values indicate a weakening of the measured comparative advantage and negative values represent reverses.',
        'Two scenarios each containing B/G of the new model and B/G of the new model subtracted from CodeBERT/BM25, for a total of 12 items, harmonized Bonferroni correction; The following two sets are used for both directional determinations.', '',
        'Zero-scale calibration cluster-t Inter-sectional calibration bootstrap',
        '|---|---|---|---:|---|---|---|']
    for r in primary:
        lines.append(f"|{r['domain']}|{r['system_comparison']}|{'B' if '_B_' in r['contrast'] else 'G'}|{r['mean']:+.8f}|{ci_text(r['family12_ci_cluster_t'])}|{ci_text(r['family12_ci_group_bootstrap'])}|{r['direction']}|")
    lines += ['', '## Reference to budget selection advantages and original queries', '',
        'The table below is a descriptive point estimate, item by item 95% See All_EFFECTS.csv/ RESULTS.json. best3 Reliance on relevant labels is not an enforceable deployment gain.', '',
        'Scene Viewer ENTR -At random: 1  mean| 3  mean| 3 best3 | Top1− Original | best3− Original |',
        '|---|---|---:|---:|---:|---:|---:|']
    for domain, data in result['domains'].items():
        for ret, view in data['views'].items():
            s = view['summaries']
            vals = [s[c]['mean'] for c in ('S_ENTR_k1_mean', 'S_ENTR_k3_mean', 'S_ENTR_k3_max', 'delta_ENTR_k1_mean', 'delta_ENTR_k3_max')]
            lines.append('|'+domain+'|'+ret+'|'+'|'.join(f'{v:+.8f}' for v in vals)+'|')
    lines += ['', '## Role and scope of the main line', '']
    for domain, data in result['domains'].items():
        s = data['views'][NAME]['summaries']
        directions = [s[c]['direction'] for c in COMPONENTS]
        if directions == ['positive', 'positive']:
            message = 'B/G Both are in a positive direction at school, supporting the expansion of measured quantities and aggregate decomposition to this UniXcoder search condition.'
        elif any(d == 'negative' for d in directions):
            message = 'At least one of the points is positive and reverse in school, providing evidence on the boundaries of the decomposed search conditions; explained separately by the table.'
        elif any(d == 'positive' for d in directions):
            message = 'Some of the weights are positive and positive; the rest is still pending, and it cannot be written that both have been replicated.'
        else:
            message = 'B/G Both remain open under this comparator; no inter-territory Zero is interpreted as non-impact, equivalent or successful replication.'
        lines.append(f'- {domain}：{message}')
    lines += ['',
        'The difference in weight between older searchers must be based on a pairing of the same query, rather than comparing each other significantly. The difference in weight is not a total searcher performance ranking. PROB and ENTR−PROB are contrasted, not claiming some sort of overall better score by B/G size.',
        'The study fixes the same rewrite pool, code text and labels and provides a more complete retrieval system. Joint changes in the length of model training, pooling, regularization and official input; No individual structural causes or consequences are claimed. UniXcoder remains the structure of RoBERTa, which does not extend to the rewrite family in this round.',
        'CoSQA and StaQC have been consumed; This is a fixed model extension before the effect of the new searcher, and it is not an independent sample confirmation or external pre-registration. CoSQA is a group of 1490 queries/512; StaQC is 13312 Query/13312 group. (a) To fix the existing whole sample and not to expand it on the basis of its effects; The effectiveness is limited to the old model differential scenarios, which do not guarantee the accuracy of the new model.',
        'UniXcoder uses the CodeSearchNet NL-PL pre-training, unknown pre-training exposure and semantic cloning still have borders. StaQC The single known formal label differs from the Bing/functional scene. The addition of a model does not create all searchers or all the general conclusions that automatically adopt policies.', '',
        '## Review and recosting', '',
        '9 Contract tests, official examples/independent physical models forward, complete vectors Top100 audits ahead of new effects; Then with Fraction RR, independent CDF randomly expected and paired statistical recalculations. Details and errors PREOPEN_AUDIT.json and FINAL_ACCEPTANCE.json.',
        'MODEL_PROVENANCE... json Records upstream commit, Git/LFS Hash, permission and mirror transmission; DESIGN...json Tie input, code, weight and state of protection.',
        'Full replay entry: python - m experiments.rene_third_retriever_20261020.audit  --after; Local binding of data/ranking is required. The new reasoning uses the run module, and the completed effects will not be repeated.',
        'risk_certified = false; this round does not change the original confirmation or production configuration.', '']
    (OUT / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    method = ['# 3rd Retrieval Extension', '',
        'We fixed CoSQA development of samples and StaQC rewrite ponds, labels and code language for existing samples, expanding the retrieval system to an open UniXcoder-base official zero sample code search configuration. Use encoder-only prefixes, code256/nl128 cut-off, nonpadding mean gestation and unit model points accumulation. This extension does not contain any new training or T5 generation.', '',
        'The main comparisons are B/G and relative CodeBERT, BM25 of the two scenes with a difference in the same search mass, for a total of 12 items harmonized Bonferroni correction, reporting groups bootstrap and cluster-t. The data have been used in previous experiments, so this round is a model extension for pre-specified comparisons and does not refer to the confirmation of new samples.', '',
        'UniXcoder Mass | correction cluster-t Inter-section correction bootstrap', '|---|---|---:|---|---|']
    for r in primary:
        if r['system_comparison'] == NAME:
            method.append(f"|{r['domain']}|{'B' if '_B_' in r['contrast'] else 'G'}|{r['mean']:+.8f}|{ci_text(r['family12_ci_cluster_t'])}|{ci_text(r['family12_ci_group_bootstrap'])}|")
    method += ['', 'The difference between weights and cross-researchers is explained by the full report. best3 Maintaining tag-aided opportunity indicator identity from which deployment gains cannot be inferred, and ENTR unique mechanisms or full model prevalence.',
               'Full result and acceptance: results/rene_third_retriever_20261020 /; Formula, direction, range FINDINGS.md.', '']
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
