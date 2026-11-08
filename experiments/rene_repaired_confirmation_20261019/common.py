from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/rene_repaired_confirmation_20261019'
DATA = ROOT / 'results/uq_pilot_20261014/action_witness_uq/data'
SEEDS = (101, 202, 303, 404)
SEED = 261019
TOL = 1e-10
FAMILIES = ('rewrite', 'fusion')
ARMS = ('no_seed_uq', 'seed_uq')
PROTECTED = (
    'configs/vac_mqr_gain_model/CURRENT.json',
    'configs/vac_mqr_verification_policy/CURRENT.json',
    'configs/vac_mqr_risk_module/CURRENT.json',
    'configs/uq_candidate_selection/CURRENT.json',
    'results/vac_mqr_v5/STABLE.json',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf8')


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return x.as_posix()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_bytes(jsonable(value))
    if immutable:
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError('Immutable file differs: ' + str(path))
            return
        with path.open('xb') as f:
            f.write(content)
        return
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(content)
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text('utf8'))


def load(path):
    with Path(path).open('rb') as f:
        return pickle.load(f)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Refusing to replace evidence: ' + str(path))
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        pickle.dump(value, f, protocol=5)
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def check_hashes(mapping):
    for name, expected in mapping.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Missing or changed bound file: ' + str(path))


def protected_hashes():
    return {p: sha(ROOT / p) for p in PROTECTED}


def require_runtime(gpu=False):
    import torch
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError('Use .venv-models/Scripts/python.exe (Python 3.10)')
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is forbidden')
    torch.set_num_threads(2)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def rr(ranking, target):
    where = np.flatnonzero(np.asarray(ranking, int)[:100] == int(target))
    return float(1 / (int(where[0]) + 1)) if len(where) else 0.


def stable_top(scores, k=100):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Ranking requires a finite score vector')
    return np.lexsort((np.arange(len(scores)), -scores))[:k]


def assert_disjoint(roles):
    names = list(roles)
    for i, a in enumerate(names):
        if len(roles[a]) != len(set(roles[a])):
            raise ValueError('Duplicate repository in role: ' + a)
        for b in names[i + 1:]:
            if set(roles[a]) & set(roles[b]):
                raise ValueError('Repository overlap: ' + a + '/' + b)


def reject_gold(value):
    forbidden = {'y', 'target', 'target_doc_id', 'source_idx', 'source_index',
                 'original_rr', 'delta_rr', 'harmful', 'true_rr', 'rr_at_100',
                 'docstring', 'docstring_tokens', 'function', 'raw_code'}
    if isinstance(value, dict):
        bad = forbidden & set(value)
        if bad:
            raise ValueError('Outcome/source fields at inference boundary: ' + str(sorted(bad)))
        for v in value.values():
            reject_gold(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            reject_gold(v)


def observable(item):
    keys = ('query_index', 'query', 'x', 'seed_x', 'original_top100',
            'candidate_top100', 'strong_evidence', 'candidates', 'seed_appearances',
            'code_word_counts', 'ce_code_lengths', 'ce_query_length')
    result = {k: item[k] for k in keys if k in item}
    reject_gold(result)
    return result
