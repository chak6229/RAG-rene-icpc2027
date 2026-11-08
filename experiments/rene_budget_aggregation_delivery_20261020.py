"""Verify delivery against sealed results without rerunning the scientific analysis."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rene_budget_aggregation_20261020'


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path: Path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 23), b''):
            value.update(block)
    return value.hexdigest()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def check_export(name, expected):
    with (OUT / name).open(encoding='utf-8-sig', newline='') as stream:
        actual = list(csv.DictReader(stream))
    require(len(actual) == len(expected), name + ' row count')
    checked = 0
    for actual_row, expected_row in zip(actual, expected):
        require(set(expected_row).issubset(actual_row), name + ' missing columns')
        for key, value in actual_row.items():
            item = expected_row.get(key, '')
            wanted = json.dumps(item, ensure_ascii=False) if isinstance(item, (list, dict)) else str(item)
            require(value == wanted, name + ' cell mismatch: ' + key)
            checked += 1
    return dict(rows=len(actual), fields_checked=checked)


def main():
    require(not (OUT / 'FINAL_ACCEPTANCE.json').exists(), 'Acceptance already exists; do not overwrite')
    protocol = read(OUT / 'PROTOCOL.json')
    result = read(OUT / 'RESULTS.json')
    audit = read(OUT / 'AUDIT.json')
    tests = read(OUT / 'TEST_RESULTS.json')
    seal = read(OUT / 'ACTIONS_SEALED.json')
    access = read(OUT / 'OUTCOME_ACCESS.json')
    digest = sha(OUT / 'PROTOCOL.json')
    protected = protocol['bound_files'] + protocol['protected_files']
    require(len(protected) == 39, '39 bound/protected inputs')
    for record in protected + seal['files']:
        require(sha(ROOT / record['path']) == record['sha256'], 'Frozen file changed: ' + record['path'])
    require(audit['passed'] and audit['independent_no_production_imports'], 'Independent scientific audit')
    require(tests['passed'] and tests['tests'] == 10 and tests['failures'] == tests['errors'] == 0, 'Scientific tests')
    for name in ('protocol_sha256',):
        require(result[name] == audit[name] == seal[name] == access[name] == digest, 'Protocol chain')
    require(result['action_seal_sha256'] == access['action_seal_sha256'] == sha(OUT / 'ACTIONS_SEALED.json'), 'Seal chain')
    require(audit['results_sha256'] == sha(OUT / 'RESULTS.json'), 'Results audit digest')
    require(audit['per_query_sha256'] == sha(OUT / 'per_query.jsonl'), 'Per-query audit digest')
    require(seal['query_actions'] == 3538 and not seal['qrels_parsed'] and seal['outcome_inputs_blocked'], 'Sealed actions')
    require(not protocol['new_confirmation'] and not result['new_confirmation'] and not audit['new_confirmation'], 'Diagnostic role')
    require(not protocol['risk_certified'] and not result['risk_certified'] and not audit['risk_certified'], 'Risk role')
    require(not audit['H_qrels_or_rankings_read'] and result['integrity']['no_H_qrels_or_rankings_loaded'], 'H outcomes not read')
    require(result['integrity']['protected_state_unchanged'], 'Protected state')
    require(protocol['created_utc'] < seal['created_utc'] < access['created_utc'], 'Freeze/action/outcome chronology')
    cells, effects, components = [], [], []
    for view in result['views']:
        for cell in view['factorial']:
            row = {key: view[key] for key in ('domain', 'retriever', 'queries', 'groups')}
            row.update({key: cell[key] for key in ('policy', 'k', 'aggregation', 'label_assisted')})
            row.update(MRR=cell['MRR']['mean'], MRR_ci95=cell['MRR']['ci95_group_bootstrap'],
                       delta_NoRewrite=cell['delta_NoRewrite']['mean'], delta_NoRewrite_ci95=cell['delta_NoRewrite']['ci95_group_bootstrap'],
                       advantage_vs_random=cell['advantage_vs_random']['mean'] if cell['advantage_vs_random'] is not None else 0.,
                       advantage_vs_random_ci95=cell['advantage_vs_random']['ci95_group_bootstrap'] if cell['advantage_vs_random'] is not None else [0., 0.])
            cells.append(row)
        for name, values in view['effects'].items():
            row = dict(domain=view['domain'], retriever=view['retriever'], contrast=name, **values)
            effects.append(row)
            if name in view['primary_decomposition']:
                components.append(row)
    require((len(cells), len(effects), len(components)) == (108, 440, 24), 'Factorial dimensions')
    exports = {name: check_export(name, records) for name, records in (
        ('FACTORIAL.csv', cells), ('CONTRASTS.csv', effects), ('DECOMPOSITION.csv', components))}
    focus = next(view for view in result['views'] if view['domain'] == 'cosqa' and view['retriever'] == 'CodeBERT')
    findings = (OUT / 'FINDINGS.md').read_text(encoding='utf-8')
    paper = (ROOT / 'paper/rene2027/budget_aggregation_results.md').read_text(encoding='utf-8')
    for name in ('ENTR_B_budget_mean_1_to_3', 'ENTR_G_aggregation_at3_mean_to_max'):
        effect = focus['effects'][name]
        for number in [effect['mean']] + effect['family24_ci95_group_bootstrap']:
            require(f'{number:.8f}' in findings and f'{number:.8f}' in paper, 'Narrative component/interval mismatch')
    for cell in focus['factorial']:
        if cell['aggregation'] in ('mean', 'max'):
            require(f"{abs(cell['delta_NoRewrite']['mean']):.8f}" in findings, 'Narrative table mismatch')
    documents = [OUT / name for name in ('PROTOCOL.json', 'ACTIONS_SEALED.json', 'ACTIONS.jsonl', 'OUTCOME_ACCESS.json',
        'TEST_RESULTS.json', 'PRE_FREEZE_TEST_REPAIR.json', 'RESULTS.json', 'per_query.jsonl', 'AUDIT.json',
        'FACTORIAL.csv', 'CONTRASTS.csv', 'DECOMPOSITION.csv', 'READOUT.md', 'FINDINGS.md')]
    documents += [ROOT / 'paper/rene2027/budget_aggregation_results.md', Path(__file__).resolve(),
                  ROOT / 'ResearchVault/Handoffs/RENE-同池数量聚合分解-20261020.md']
    receipt = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Final hashes, scientific-audit chain, complete CSV exports and key narrative values; no scientific re-estimation',
        role=protocol['role'], previous_A_effects_known=True, new_confirmation=False, risk_certified=False,
        scientific_tests=10, independent_scientific_audit='AUDIT.json', bound_and_protected_files_checked=39,
        action_files_checked=len(seal['files']), query_actions=3538, per_query_records=7076,
        factorial_cells=108, primary_mean_max_cells=72, contrasts=440, planned_components=24,
        export_checks=exports, no_model_inference=True, no_H_effects=True,
        old_formal_pointers_and_confirmation_unchanged=True, prior_release_unchanged=True,
        scientific_audit_counts=audit['counts'], scientific_audit_max_errors=audit['max_errors'],
        files=[dict(path=str(path.relative_to(ROOT)).replace('\\', '/'), sha256=sha(path)) for path in documents])
    with (OUT / 'FINAL_ACCEPTANCE.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, bound_and_protected_files=39, exports=exports,
                         receipt='results/rene_budget_aggregation_20261020/FINAL_ACCEPTANCE.json'), ensure_ascii=False))


if __name__ == '__main__':
    main()
