from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).parent
OUT = ROOT / 'results/rene_mechanism_review_20261020'
PARENT = ROOT / 'results/rene_cosqa_confirmation_20261019'
CORPUS = ROOT / 'results/rene_ssqr_alignment_20261019/cosqa_adapter/public/corpus.jsonl'
RAW_MAP = ROOT / 'results/rene_cosqa_ssqr_audit_20261019/selected_source/coclr/data/search/code_idx_map.txt'
RAW_SOURCE = ROOT / 'imports/CodeXGLUE-main/Text-Code/NL-code-search-WebQuery/CoSQA/cosqa-all.json'
POLICIES = ('ENTR_top1', 'HGB_gain_control', 'RIDGE_gain', 'RIDGE_verified', 'RIDGE_risk')
STRATA = ('gate_reject_positive', 'gate_reject_negative', 'final_positive', 'final_negative',
          'gate_reject_zero', 'remaining_opportunity', 'remaining_neutral_changed', 'remaining_no_opportunity')
STRATUM_ZH = ("It's not the right thing to do.", 'Door refusals and negative returns', 'Final positive gain.', 'Final negative gain', 'Door denial zeroes.',
              "There's still a good chance.", 'The rest is zero, but the ranking changes.', 'The remaining opportunities were not positive and had not been implemented')
PER_STRATUM = 16
SALT = 'rene-mechanism-review-20261020-v1'
FIDELITY = ('preserved', 'drift', 'malformed', 'uncertain')
FLAGS = ('unsupported_constraint', 'lost_constraint', 'operation_change', 'entity_api_change',
         'polarity_change', 'incoherent', 'none', 'uncertain')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def put_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def order_key(*parts):
    return hashlib.sha256(('\0'.join((SALT, *map(str, parts)))).encode()).hexdigest()


def record(path):
    return {'path': Path(path).relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def rr(ranking, positives):
    return next((Fraction(1, i) for i, d in enumerate(ranking[:100], 1) if d in positives), Fraction(0))


def classify(row):
    pre = row['policies']['RIDGE_verified']
    post = row['policies']['RIDGE_risk']
    if pre['adopted'] and not post['adopted']:
        return 'gate_reject_positive' if pre['delta'] > 1e-10 else 'gate_reject_negative' if pre['delta'] < -1e-10 else 'gate_reject_zero'
    if post['delta'] > 1e-10:
        return 'final_positive'
    if post['delta'] < -1e-10:
        return 'final_negative'
    if row['safe_ENTR_oracle_delta'] > 1e-10:
        return 'remaining_opportunity'
    return 'remaining_neutral_changed' if post['adopted'] else 'remaining_no_opportunity'


def sample_frame(frame):
    selected = []
    for stratum in STRATA:
        current = [r for r in frame if classify(r) == stratum]
        chosen = sorted(current, key=lambda r: order_key('sample', stratum, r['query_id']))[:PER_STRATUM]
        if not chosen:
            raise ValueError('empty planned stratum: ' + stratum)
        for r in chosen:
            selected.append({'query_id': r['query_id'], 'group_id': r['group_id'], 'stratum': stratum,
                             'frame_n': len(current), 'sample_n': len(chosen),
                             'inclusion_probability': len(chosen) / len(current), 'weight': len(current) / len(chosen)})
    return selected


def require_frozen():
    p = read(OUT / 'PROTOCOL.json')
    for item in p['bound_files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('bound input/source drift: ' + item['path'])
    return p
