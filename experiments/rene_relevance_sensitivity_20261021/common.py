from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).parent
OUT = ROOT / 'results/rene_relevance_sensitivity_20261021'
PARENT = ROOT / 'results/rene_cosqa_confirmation_20261019'
REVIEW = ROOT / 'results/rene_review_returns_20261020'
OLD = ROOT / 'results/rene_mechanism_review_20261020'
POLICIES = ('ENTR_top1', 'HGB_gain_control', 'RIDGE_gain', 'RIDGE_verified', 'RIDGE_risk')
SALT = 'rene-relevance-sensitivity-20261021-v1'


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


def put_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def key(*parts):
    return hashlib.sha256('\0'.join((SALT, *map(str, parts))).encode()).hexdigest()


def bound(path):
    return {'path': Path(path).relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def frozen():
    protocol = read(OUT / 'PROTOCOL.json')
    for record in protocol['bound_inputs_and_design']:
        if sha(ROOT / record['path']) != record['sha256']:
            raise ValueError('Frozen input changed: ' + record['path'])
    return protocol
