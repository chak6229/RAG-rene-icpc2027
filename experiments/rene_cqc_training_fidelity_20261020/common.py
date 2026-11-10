from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / 'results/rene_cqc_training_fidelity_20261020'
STORE = Path('external_inputs/cqc_training_model_store')
BASE = ROOT / 'models/t5-base'
BRIDGE = ROOT / 'results/rene_ssqr_bridge_20261019'
ARM_SPECS = {
    'L': dict(optimizer='AdamW', schedule='constant', horizon=None, max_length=512, weight_decay=.01, seed=101),
    'O': dict(optimizer='Adafactor', schedule='constant', horizon=None, max_length=512, weight_decay=.01, seed=101),
    'S': dict(optimizer='AdamW', schedule='linear', horizon=100000, max_length=512, weight_decay=.01, seed=101),
    'N': dict(optimizer='AdamW', schedule='constant', horizon=None, max_length=128, weight_decay=.01, seed=101),
    'W': dict(optimizer='AdamW', schedule='constant', horizon=None, max_length=512, weight_decay=0., seed=101),
    'P': dict(optimizer='Adafactor', schedule='linear', horizon=100000, max_length=128, weight_decay=0., seed=101),
    'PN': dict(optimizer='Adafactor', schedule='linear', horizon=1093, max_length=128, weight_decay=0., seed=101),
    'PS': dict(optimizer='Adafactor', schedule='linear', horizon=100000, max_length=128, weight_decay=0., seed=42),
}
PROTECTED = []  # Unrelated author-workspace registries are not part of this release.


def require(value, message):
    if not value:
        raise AssertionError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def writel(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 23), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path).resolve()
    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:
        label = path.as_posix()
    return dict(path=label, sha256=sha(path))


def text_id(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def verify():
    p = read(OUT / 'TRAIN_PROTOCOL.json')
    for item in p['bound_files'] + p['protected_files']:
        path = Path(item['path'])
        if not path.is_absolute():
            path = ROOT / path
        require(sha(path) == item['sha256'], 'Frozen input changed: ' + str(path))
    return p


def lr_at(spec, completed):
    return .001 if spec['schedule'] == 'constant' else .001 * max(0., 1. - completed / spec['horizon'])
