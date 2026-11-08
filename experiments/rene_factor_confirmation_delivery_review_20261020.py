"""Check delivery bindings without rerunning or changing the frozen analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/rene_factor_confirmation_20261020'
PRIMARY = ('ENTR_B_budget_mean_1_to_3', 'ENTR_G_aggregation_at3_mean_to_max')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    accepted = read(OUT / 'FINAL_ACCEPTANCE.json')
    require(accepted['passed'], 'Main acceptance did not pass')
    design = read(OUT / 'DESIGN.json')
    hashes_checked = 0
    for item in accepted['files'] + design['bound_files'] + design['protected_files']:
        require(digest(ROOT / item['path']) == item['sha256'], 'Changed binding: ' + item['path'])
        hashes_checked += 1
    require(accepted['queries'] == accepted['groups'] == design['queries'] == 13312, 'Sample identity')
    require(accepted['corpus_documents'] == design['corpus_documents'] == 16384, 'Corpus identity')
    require(not accepted['CoSQA_new_confirmation'] and not accepted['external_preregistration'], 'Wrong evidence role')
    require(not accepted['risk_certified'] and not accepted['official_model_promotion'], 'Wrong promotion role')

    reviews = ['SOURCE_PRECISION_REVIEW.json', 'UPSTREAM_TREE_ANCHOR_REVIEW.json',
               'PREOPEN_AUDIT.json', 'EFFECT_AUDIT.json', 'MODEL_AUDIT.json', 'TEST_RESULTS.json']
    for name in reviews:
        value = read(OUT / name)
        require(value['passed'], 'Review did not pass: ' + name)
    source_review = read(OUT / reviews[0])
    require(source_review['design_sha256'] == digest(OUT / 'DESIGN.json'), 'Source review design mismatch')
    require(source_review['known_exposure_input_hashes_checked'] == 31, 'Exposure input scope')
    require(source_review['accepted_source_pairs_reconnected'] == 147546, 'Source regrouping scope')
    local_eligibility = read(OUT / 'LOCAL_ELIGIBILITY.json')
    source_bindings = local_eligibility['references'] + local_eligibility['CSN']['files']
    source_bindings += read(OUT / 'sources/staqc_DATA_RECEIPTS.json')['files']
    for item in source_bindings:
        require(digest(ROOT / item['path']) == item['sha256'], 'Changed source/reference: ' + item['path'])
        hashes_checked += 1

    def utc(text):
        return datetime.fromisoformat(text)

    chain = ['DESIGN.json', 'CANDIDATES_SEALED.json', 'RANKINGS_SEALED.json',
             'PREOPEN_AUDIT.json', 'OUTCOME_OPENING.json', 'RESULTS.json',
             'EFFECT_AUDIT.json', 'MODEL_AUDIT.json', 'FINAL_ACCEPTANCE.json']
    times = [utc(read(OUT / name)['created_utc']) for name in chain]
    require(all(a <= b for a, b in zip(times, times[1:])), 'Freeze/opening/audit order mismatch')
    for name in ('PRE_INFERENCE_REPAIR.json', 'PRECISION_AMENDMENT_BEFORE_EFFECTS.json'):
        require((OUT / name).is_file(), 'Missing preserved amendment: ' + name)
    description = read(OUT / 'INPUT_DESCRIPTION.json')
    require(description['T5_query_plus_sentinel']['rows'] == 13312, 'T5 input coverage')
    require(description['CodeBERT_code']['rows'] == 16384, 'Code input coverage')
    require(description['CodeBERT_code']['over_declared_limit'] == 1785, 'Code truncation disclosure')
    candidate_description = read(OUT / 'CANDIDATE_DESCRIPTION.json')
    require(candidate_description['candidate_seal_sha256'] == digest(OUT / 'CANDIDATES_SEALED.json'), 'Candidate description binding')
    require(candidate_description['queries'] == 13312 and candidate_description['positions'] == 116319, 'Candidate coverage description')
    require(candidate_description['queries_with_less_than3positions'] == 0, 'Short candidate pools')
    require(not candidate_description['new_method_effects_read'], 'Candidate description outcome role')

    result = read(OUT / 'RESULTS.json')
    with (OUT / 'PRIMARY_AND_SECONDARY.csv').open(encoding='utf-8', newline='') as stream:
        exported = list(csv.DictReader(stream))
    require(len(exported) == 4, 'Expected two factors for each retriever')
    fields_checked = 0
    for row in exported:
        target = result['views'][row['retriever']]['summaries'][row['contrast']]
        for key, wanted in target.items():
            actual = json.loads(row[key]) if isinstance(wanted, list) else row[key]
            if isinstance(wanted, (float, int)):
                require(math.isclose(float(actual), wanted, rel_tol=0, abs_tol=1e-14), 'CSV numeric mismatch')
            else:
                require(actual == wanted, 'CSV text/list mismatch')
            fields_checked += 1

    supplement = ROOT / 'paper/rene2027/factor_confirmation_results.md'
    text = supplement.read_text(encoding='utf-8')
    for contrast in PRIMARY:
        target = result['views']['CodeBERT']['summaries'][contrast]
        require(contrast in text and f'{target["mean"]:+.8f}' in text, 'Supplement estimate mismatch')
        for key in ('family2_ci95_cluster_t', 'family2_ci95_group_bootstrap'):
            expected = '[' + ', '.join(f'{x:+.6f}' for x in target[key]) + ']'
            require(expected in text, 'Supplement interval mismatch')
    for name in ('factor_confirmation.png', 'factor_confirmation.pdf'):
        require((OUT / name).stat().st_size > 1000, 'Figure missing/empty')
    visual = read(OUT / 'FIGURE_VISUAL_QA.json')
    require(visual['passed'], 'Visual inspection did not pass')
    for item in visual['files']:
        require(digest(ROOT / item['path']) == item['sha256'], 'Visual QA figure changed')

    files = [OUT / name for name in reviews + ['FINAL_ACCEPTANCE.json', 'INPUT_DESCRIPTION.json', 'CANDIDATE_DESCRIPTION.json', 'RUNTIME_ENVIRONMENT.json', 'FIGURE_VISUAL_QA.json',
             'UPSTREAM_COMMIT_TREE.json', 'factor_confirmation.png', 'factor_confirmation.pdf']]
    files += [supplement, Path(__file__), ROOT / 'experiments/rene_factor_confirmation_delivery_20261020.py']
    if (OUT / 'DELIVERY_LAYOUT_REVISION.json').exists():
        files.append(OUT / 'DELIVERY_LAYOUT_REVISION.json')
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
        'scope': 'Delivery bindings, extra source/power reviews, chronological seals, exported fields and supplement; no new outcomes or statistical rerun',
        'binding_hashes_checked': hashes_checked, 'CSV_fields_checked': fields_checked,
        'scientific_acceptance': record(OUT / 'FINAL_ACCEPTANCE.json'),
        'design_sha256': digest(OUT / 'DESIGN.json'),
        'primary_directions': result['primary_directions'],
        'files': [record(path) for path in files],
        'figure_visual_QA': record(OUT / 'FIGURE_VISUAL_QA.json'),
    }
    with (OUT / 'DELIVERY_ACCEPTANCE.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print({'passed': True, 'binding_hashes_checked': hashes_checked, 'CSV_fields_checked': fields_checked})


if __name__ == '__main__':
    main()
