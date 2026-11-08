"""Independent source-to-table and manuscript-reference validation, stdlib only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def pointer(data, path):
    for part in path.strip('/').split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def audit(root):
    out = root/'results/rene_evidence_integration_20261020'
    bundle = read(out/'TABLES.json')
    sources, names = {}, {}
    for entry in bundle['sources']:
        require(sha(root/entry['path']) == entry['sha256'], 'Source hash differs: '+entry['key'])
        sources[entry['path']] = read(root/entry['path'])
        names[entry['key']] = sources[entry['path']]
    for cell in bundle['traceable_cells']:
        require(pointer(sources[cell['source']], cell['pointer']) == cell['value'], 'Wrong source cell')
    tables = bundle['tables']
    require(len(tables) == 7 and sum(map(len, tables.values())) == 38, 'Wrong table/row count')
    require(len(bundle['traceable_cells']) == 235, 'Wrong traceable-cell count')
    checked = 0

    def fields(row, values):
        nonlocal checked
        for key, value in values.items():
            require(row[key] == value, 'Semantic table mismatch: '+key)
            checked += 1

    p = names['confirmation']
    row = tables['T1_confirmation'][0]
    fields(row, dict(queries=p['queries'], groups=p['groups'], delta=p['mean'],
                    ci95=p['ci95_group_bootstrap'], practical_margin=.005,
                    equivalence=p['conclusion']['practical_equivalence_within_prespecified_plusminus005']))
    require((row['queries'], row['groups']) == (7718, 1963), 'Wrong primary denominator')
    for row, raw in zip(tables['T2_same_pool_contracts'], names['A']['views'], strict=True):
        fields(row, {k: raw[k] for k in ('domain', 'retriever', 'queries', 'groups')})
        for key, contrast in [('single_advantage', 'a_ENTR_single_minus_uniform'),
                              ('best3_advantage', 'b_ENTR_best3_minus_RAND20'),
                              ('direct_difference', 'j_ENTR_contract_difference')]:
            effect = raw['contrasts'][contrast]
            fields(row, {key: effect['mean'], key+'_ci95': effect['ci95_group_bootstrap']})
        fields(row, {'direct_difference_family95': raw['contrasts']['j_ENTR_contract_difference']['four_cell_family95_bonferroni_group_bootstrap']})
    for row in tables['T3_adoption_fixed20']:
        view = names['B']['views'][row['retriever']]
        raw = view['policies']['fixed20'][row['model']]
        fields(row, dict(delta=raw['delta_mrr'], adopted=raw['adopted'], harmful=raw['harmful'],
                        ci95=view['estimates']['fixed20:'+row['model']+'-NoRewrite']['ci95_group_bootstrap']))
    for row in tables['T4_multicandidate_fusion']:
        view = names['fusion']['views'][row['retriever']]
        raw, effect = view['policies'][row['policy']], view['estimates'][row['policy']+'-NoRewrite']
        fields(row, dict(delta=raw['delta_mrr'], harmful=raw['harmful_queries'],
                        ci95=effect['ci95_group_bootstrap'], executable=row['policy'] != 'ENTRBest3'))
        if row['policy'] in ('RRF3', 'RRF4'):
            fields(row, dict(family95=effect['four_effect_family95_bonferroni_group_bootstrap']))
    for row in tables['T5_frozen_H_stages']:
        view = names['H_'+row['retriever']]
        raw = view['policy_results'][row['policy']]
        fields(row, dict(queries=view['queries'], groups=view['groups'], delta=raw['delta_MRR']['mean'],
                        ci95=raw['delta_MRR']['ci95_group_bootstrap'], adopted=raw['adopted'],
                        harmful=raw['harmed'], positive_gain_retention=raw['positive_gain_retention']['value'],
                        CE_pairs=raw['CE_pairs_charged']))
        require((row['queries'], row['groups']) == (8530, 1963), 'H stage denominator differs')
    for row in tables['T6_two_T5_weights']:
        raw = names['envelope']['views'][row['domain']+'/'+row['T5_weights']]
        fields(row, dict(queries=raw['queries'], groups=raw['groups']))
        require((row['queries'], row['groups']) == (512, 512), 'Sensitivity denominator differs')
        for key, effect_name in [('top1_delta', 'ENTR_top1-NoRewrite'), ('best3_delta', 'ENTR_best3-NoRewrite')]:
            effect = raw['effects'][effect_name]
            fields(row, {key: effect['mean'], key+'_ci95': effect['ci95_group_bootstrap'],
                         key+'_family95': effect['eight_effect_family95_bonferroni_bootstrap']})
        effect = raw['effects']['ENTR_top1-Uniform_single_exact']
        fields(row, dict(entropy_single_vs_uniform=effect['mean'], entropy_single_vs_uniform_ci95=effect['ci95_group_bootstrap']))
    for row in tables['T7_previous_seed_UQ']:
        source, interval_key = ('seed1536', 'simultaneous_ci97_5') if row['comparison'] == 'uq_increment' else ('risk1024', 'simultaneous_envelope')
        raw = names[source]['primary_contrasts'][row['comparison']]
        fields(row, dict(delta=raw['mean'], ci97_5=raw[interval_key]))

    # CSV round-trip includes every text, boolean, scalar, and interval field.
    csv_fields = 0
    for name, records in tables.items():
        with (out/(name+'.csv')).open(encoding='utf-8-sig', newline='') as handle:
            exported = list(csv.DictReader(handle))
        require(len(exported) == len(records), 'CSV rows differ')
        for row, original in zip(exported, records, strict=True):
            for key, saved in row.items():
                value = original.get(key, '')
                expected = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
                require(saved == expected, 'CSV cell differs: '+name+'/'+key)
                csv_fields += 1
    manuscript = (root/'paper/rene2027/integrated_results.md').read_text(encoding='utf-8')
    require(manuscript == (out/'MAIN_RESULTS.md').read_text(encoding='utf-8'), 'Paper/main results copy differs')
    for records in tables.values():
        for row in records:
            for key in ('delta', 'single_advantage', 'best3_advantage', 'direct_difference', 'top1_delta', 'best3_delta'):
                if key in row:
                    require(f'{row[key]:+.6f}' in manuscript, 'Rounded manuscript estimate missing')
    bib = (root/'paper/rene2027/references_verified.bib').read_text(encoding='utf-8')
    defined = re.findall(r'@\w+\s*\{\s*([^,]+),', bib)
    require(len(defined) == len(set(defined)), 'Duplicate citation keys')
    cited = set()
    for name in ('intro_draft.md', 'related_work_draft.md'):
        text = (root/'paper/rene2027'/name).read_text(encoding='utf-8')
        for group in re.findall(r'\\cite\{([^}]+)\}', text):
            cited.update(group.split(','))
        require('[verify]' not in text and 'REPLACE_ME' not in text, 'Unverified citation placeholder')
    require(cited.issubset(set(defined)), 'Undefined citation')
    require(len(cited) == 7, 'Expected SSQR, CoSQA and five neighbours')
    require(bundle['no_cross_population_pooling'] and bundle['original_primary_unchanged'] and not bundle['risk_certified'], 'Role contract differs')
    proof_paths = [
        'results/rene_cosqa_confirmation_20261019/FINAL_ACCEPTANCE.json',
        'results/rene_core_step_a_20261020/AUDIT.json',
        'results/rene_core_step_b_20261020/FINAL_ACCEPTANCE.json',
        'results/rene_multicandidate_fusion_20261020/FINAL_ACCEPTANCE.json',
        'results/rene_envelope_completion_20261020/FINAL_ACCEPTANCE.json',
    ]
    for path in proof_paths:
        require(read(root/path)['passed'], 'Input experiment not accepted')
    return dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                independent_from_table_builder=True, source_files=len(sources),
                traceable_source_cells=len(bundle['traceable_cells']), semantic_fields=checked,
                tables=len(tables), rows=sum(map(len, tables.values())), csv_fields=csv_fields,
                resolved_citation_keys=sorted(cited), input_acceptances=len(proof_paths),
                no_new_H_estimates=True, no_cross_population_pooling=True,
                new_confirmation=False, risk_certified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    path = args.output or args.root/'results/rene_evidence_integration_20261020/AUDIT.json'
    with path.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False))
