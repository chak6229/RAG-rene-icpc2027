from __future__ import annotations

from pathlib import Path
from experiments.rene_factor_confirmation_20261020.common import (ROOT, digest, now, read, record, rows, sha, write, writel)

PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / 'results/rene_weight_confirmation_20261021'
OLD = ROOT / 'results/rene_factor_confirmation_20261020'
WEIGHTS = ('local', 't5base')
MODEL_PATHS = {'local': 'models/t5-cqc-codenn-1m/best', 't5base': 'models/t5-base'}
PRIMARY = [f'{v}::ENTR_{c}' for v in (*WEIGHTS, 'base_minus_local') for c in ('B', 'G')]
FAMILY = 6
SEED = 202610215
REPETITIONS = 20000


def verify_design(private=False):
    d = read(OUT / 'DESIGN.json')
    for item in d['bound_files'] + d['protected_files']:
        if not private and '/private/' in item['path']:
            continue
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Frozen input changed: ' + item['path'])
    return d


def verify_seal(name):
    s = read(OUT / name)
    if s['design_sha256'] != sha(OUT / 'DESIGN.json'):
        raise ValueError('Different design seal')
    for item in s['files']:
        if sha(ROOT / item['path']) != item['sha256']:
            raise ValueError('Sealed file changed: ' + item['path'])
    return s


def progress(stage, **kwargs):
    import json
    v = {'updated_utc': now(), 'stage': stage, 'completed': False, **kwargs}
    (OUT / 'RUN_STATUS.json').write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(v, flush=True)
