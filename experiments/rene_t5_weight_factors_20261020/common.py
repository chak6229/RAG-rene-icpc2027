from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / 'results/rene_t5_weight_factors_20261020'
OLD = ROOT / 'results/rene_envelope_completion_20261020'
BRIDGE = ROOT / 'results/rene_ssqr_bridge_20261019'
DOMAINS = ('source', 'cosqa')
WEIGHTS = ('local', 't5base')
VIEWS = (*WEIGHTS, 'base_minus_local')
POLICIES = ('ENTR', 'PROB', 'RAND_exact')
SEED = 202610204
REPETITIONS = 20000
FAMILY_SIZE = 24


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**23), b''):
            h.update(block)
    return h.hexdigest()


def text_id(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def record(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def writel(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def paths(domain):
    if domain not in DOMAINS:
        raise ValueError('Unknown domain')
    return {
        'queries': BRIDGE / f'data/{domain}/queries.jsonl',
        'qrels': BRIDGE / f'data/private/{domain}_qrels.jsonl',
        'local_candidates': BRIDGE / 'candidates/source/baseline.jsonl' if domain == 'source' else ROOT / 'results/rene_cosqa_development_20261019/candidates.jsonl',
        't5base_candidates': ROOT / f'results/rene_ssqr_rewriter_envelope_20261020/candidates/{domain}/t5base.jsonl',
        'local_rankings': BRIDGE / f'rankings/{domain}/CodeBERT.jsonl',
        't5base_rankings': OLD / f'rankings/{domain}/Top100.jsonl',
    }


def primary_names():
    return [f'{view}::contrasts::{policy}_{component}' for view in VIEWS
            for policy in ('ENTR', 'PROB') for component in ('B', 'G')]


def verify():
    design = read(OUT / 'DESIGN.json')
    for item in design['bound_files'] + design['protected_files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise AssertionError('Frozen file changed: ' + item['path'])
    return design
