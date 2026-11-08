from __future__ import annotations

import gzip
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
OUT = ROOT / 'results/rene_factor_confirmation_20261020'
SOURCES = OUT / 'sources'
PROTECTED = [
    'configs/vac_mqr_gain_model/CURRENT.json',
    'configs/vac_mqr_verification_policy/CURRENT.json',
    'configs/vac_mqr_risk_module/CURRENT.json',
    'configs/uq_candidate_selection/CURRENT.json',
    'results/vac_mqr_v5/STABLE.json',
    'results/rene_cosqa_confirmation_20261019/PROTOCOL.json',
    'results/rene_cosqa_confirmation_20261019/PRIMARY_RESULT.json',
    'results/rene_budget_aggregation_20261020/FINAL_ACCEPTANCE.json',
    'results/rene_cqc_training_fidelity_20261020/FINAL_ACCEPTANCE.json',
]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(2**23), b''):
            h.update(b)
    return h.hexdigest()


def digest(s):
    return hashlib.sha256(str(s).encode('utf-8')).hexdigest()


def record(path):
    return {'path': Path(path).relative_to(ROOT).as_posix(), 'sha256': sha(path)}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    path = Path(path)
    op = gzip.open if path.suffix == '.gz' else open
    with op(path, 'rt', encoding='utf-8-sig') as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as h:
        json.dump(value, h, ensure_ascii=False, indent=2, allow_nan=False)
        h.write('\n')


def writel(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as h:
        for r in records:
            h.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')


def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', str(s)).casefold().split())


def lexical(s):
    return ' '.join(re.findall(r'\w+', norm(s)))


def token_text(v):
    return ' '.join(v.split() if isinstance(v, str) else map(str, v))


def repo(r):
    s = r.get('repo') or r.get('nwo')
    if not s:
        m = re.match(r'https?://github\.com/([^/]+/[^/]+)', str(r.get('url', '')), re.I)
        s = m.group(1) if m else ''
    return str(s).strip('/').removesuffix('.git').casefold()


def fingerprints(r):
    q = r.get('query', r.get('doc', ''))
    queries = [q, r.get('docstring', ''), r.get('docstring_summary', ''), token_text(r.get('docstring_tokens', []))]
    code = r.get('code') or r.get('function') or r.get('original_string') or r.get('code_text', '')
    tokens = token_text(r.get('code_tokens', r.get('function_tokens', [])))
    fn = r.get('func_name') or r.get('identifier') or ''
    function = f'{repo(r)}:{r.get("path", "")}:{fn}' if repo(r) and fn else ''
    return {
        'q_norm': {digest(norm(q)) for q in queries if str(q).strip()},
        'q_lex': {digest(lexical(q)) for q in queries if lexical(q)},
        'code_raw': {digest(str(code))} if code else set(),
        'code_ws': {digest(' '.join(str(code).split()))} if code else set(),
        'code_tokens': {digest(tokens)} if tokens else set(),
        'url': {str(r['url']).casefold()} if r.get('url') else set(),
        'function': {function} if function else set(),
        'repo': {repo(r)} if repo(r) else set(),
    }


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)
