from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / 'results/rene_third_retriever_20261020'
MODEL = ROOT / 'models/unixcoder-base-pinned-20261020'
SOURCES = OUT / 'sources'
NAME = 'UniXcoder'
COMPONENTS = ('ENTR_B_budget_mean_1_to_3', 'ENTR_G_aggregation_at3_mean_to_max')
PROTECTED = []  # Unrelated author-workspace registries are not part of this release.


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as h:
        for block in iter(lambda: h.read(2**23), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def write(path, value, *, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if immutable else 'w', encoding='utf-8') as h:
        json.dump(value, h, ensure_ascii=False, indent=2, allow_nan=False)
        h.write('\n')


def writel(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as h:
        for row in data:
            h.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def progress(stage, **kwargs):
    value = {'updated_utc': now(), 'stage': stage, 'completed': False, **kwargs}
    write(OUT / 'RUN_STATUS.json', value, immutable=False)
    print(value, flush=True)


def verify_design(*, private=False):
    design = read(OUT / 'DESIGN.json')
    for item in design['bound_files'] + design['protected_files']:
        if not private and '/private/' in item['path']:
            continue
        require(sha(ROOT / item['path']) == item['sha256'], 'Frozen file changed: ' + item['path'])
    return design


def verify_seal(name):
    seal = read(OUT / name)
    require(seal['design_sha256'] == sha(OUT / 'DESIGN.json'), 'Design identity mismatch: ' + name)
    for item in seal['files']:
        require(sha(ROOT / item['path']) == item['sha256'], 'Sealed file changed: ' + item['path'])
    return seal


@contextlib.contextmanager
def deny_outcomes():
    original_builtin, original_io, original_path = builtins.open, io.open, Path.open
    blocked = {'per_query.jsonl', 'RESULTS.json', 'PRIMARY_RESULT.json', 'STAQC_ELIGIBLE_PAIRS.jsonl'}

    def check(file, mode='r'):
        if isinstance(file, (str, Path)) and ('r' in mode or '+' in mode):
            path = Path(file).resolve()
            if 'private' in path.parts or path.name in blocked:
                raise PermissionError('Labels/results blocked during inference: ' + path.name)

    def b(file, mode='r', *a, **kw):
        check(file, mode)
        return original_builtin(file, mode, *a, **kw)

    def i(file, mode='r', *a, **kw):
        check(file, mode)
        return original_io(file, mode, *a, **kw)

    def p(file, mode='r', *a, **kw):
        check(file, mode)
        return original_path(file, mode, *a, **kw)

    builtins.open, io.open, Path.open = b, i, p
    try:
        yield
    finally:
        builtins.open, io.open, Path.open = original_builtin, original_io, original_path
