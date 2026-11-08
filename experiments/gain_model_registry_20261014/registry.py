"""Versioned, label-free VAC-MQR gain-component inference.

The CURRENT here selects a gain estimator. It does not select an adoption policy
or grant risk certification. Feature extraction and retrieval remain upstream.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/vac_mqr_gain_model'
ALIASES = {'old': 'legacy-v5', 'weak46': 'weak46-20261014'}
FORMAT = 'vac-mqr-gain-inference-v1'
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def resolve(version='current'):
    if version == 'current':
        pointer = read_json(CONFIG / 'CURRENT.json')
        path = ROOT / pointer['manifest']
        if sha(path) != pointer['manifest_sha256']:
            raise ValueError('CURRENT manifest hash mismatch')
        manifest = read_json(path)
        if manifest['version'] != pointer['version']:
            raise ValueError('CURRENT version mismatch')
        return manifest
    version = ALIASES.get(version, version)
    if Path(version).name != version:
        raise ValueError('Version must be a registered name')
    return read_json(CONFIG / 'versions' / (version + '.json'))


def _flow():
    path = ROOT / 'experiments/strong_flow_uq_20261014/flow.py'
    spec = importlib.util.spec_from_file_location('vac_mqr_registered_gain_flow', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _matrix(value, shape, label):
    x = np.asarray(value, dtype=float)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f'{label}: expected finite {shape}, received {x.shape}')
    return x.copy()


def observable(item, kind='strong84'):
    """Copy only inference fields; gold, role, repository and labels are ignored."""
    x = np.asarray(item['x'], float)
    if x.ndim != 2 or x.shape[1] != 34:
        raise ValueError('x must contain the registered 34 base columns')
    n = len(x)
    view = {'x': _matrix(x, (n, 34), 'x')}
    if kind == 'legacy-v5':
        return view
    view['seed_x'] = _matrix(item['seed_x'], (n, 12), 'seed_x')
    if kind == 'weak46':
        return view
    original = np.asarray(item['original_top100'], dtype=int)
    candidates = np.asarray(item['candidate_top100'], dtype=int).reshape(n, 100)
    if original.shape != (100,) or len(np.unique(original)) != 100:
        raise ValueError('Original ranking must have 100 unique document IDs')
    if any(len(np.unique(top)) != 100 for top in candidates):
        raise ValueError('Each candidate ranking must have 100 unique document IDs')
    z = item['strong_evidence']
    ids = np.asarray(z['doc_ids'], dtype=int)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Evidence IDs must be a nonempty unique vector')
    needed = set(map(int, original[:10])) | set(map(int, candidates[:, :10].ravel()))
    if not needed.issubset(set(map(int, ids))):
        raise ValueError('Evidence must cover all original/candidate Top10 documents')
    evidence = dict(doc_ids=ids.copy(),
        doc_signals=_matrix(z['doc_signals'], (len(ids), 4), 'doc_signals'),
        x={f: _matrix(z['x'][f], (n, 28), f'x28.{f}') for f in ('rewrite', 'fusion')})
    view.update(original_top100=original.copy(), candidate_top100=candidates.copy(),
                strong_evidence=evidence)
    return view


class GainModel:
    def __init__(self, version='current', family='rewrite', fold=None):
        if family not in ('rewrite', 'fusion'):
            raise ValueError('family must be rewrite or fusion')
        self.manifest = resolve(version)
        self.version = self.manifest['version']
        self.kind = self.manifest['kind']
        self.family = family
        self.fold = fold
        if self.manifest['format'] != FORMAT:
            raise ValueError('Unsupported gain manifest format')
        if family not in self.manifest['families']:
            raise ValueError(f'{self.version} has no {family} model; legacy V5 models fusion only')
        if fold is not None and fold not in range(4):
            raise ValueError('fold must be 0..3')
        for p, expected in self.manifest['inference_source_hashes'].items():
            if sha(ROOT / p) != expected:
                raise ValueError(f'Inference source changed: {p}')
        if self.kind == 'legacy-v5':
            if fold is not None:
                raise ValueError('legacy-v5 has no fold variants')
            from vac_mqr import UtilityRanker, HarmfulRiskModel
            files = self.manifest['files']
            for record in files.values():
                if sha(ROOT / record['path']) != record['sha256']:
                    raise ValueError('Legacy artifact hash mismatch')
            self.ranker = UtilityRanker.load(ROOT / files['ranker']['path'])
            self.risk = HarmfulRiskModel.load(ROOT / files['risk_model']['path'])
            if list(self.ranker.feature_names) != self.manifest['base_feature_names']:
                raise ValueError('Legacy feature schema mismatch')
            return
        key = f'{family}/full' if fold is None else f'{family}/fold{fold}'
        if key not in self.manifest['models']:
            raise ValueError('This version requires an explicit --fold 0..3')
        record = self.manifest['models'][key]
        path = ROOT / record['path']
        if sha(path) != record['sha256']:
            raise ValueError('Inference model hash mismatch')
        bundle = joblib.load(path)
        if (bundle['format'], bundle['version'], bundle['kind'], bundle['family'], bundle['fold']) != (
                FORMAT, self.version, self.kind, family, fold):
            raise ValueError('Bundle/version/family/fold mismatch')
        if bundle['feature_names'] != self.manifest['feature_names']:
            raise ValueError('Bundle feature schema mismatch')
        self.model = bundle['model']
        self.flow = _flow()

    def predict(self, items):
        views = [observable(item, self.kind) for item in items]
        outputs = [None] * len(views)
        nonempty = [i for i, v in enumerate(views) if len(v['x'])]
        for i, v in enumerate(views):
            if not len(v['x']):
                outputs[i] = dict(mu=np.empty(0), harm=np.empty(0))
                if self.kind != 'legacy-v5':
                    outputs[i].update(member_mu=np.empty((5, 0)), member_harm=np.empty((5, 0)),
                                      empirical_member_sd=np.empty(0))
                else:
                    outputs[i].update(lcb=np.empty(0), legacy_harm=np.empty(0))
        if not nonempty:
            return outputs
        active = [views[i] for i in nonempty]
        if self.kind == 'legacy-v5':
            for i in nonempty:
                rows = [dict(zip(self.manifest['base_feature_names'], row)) for row in views[i]['x']]
                mu, lcb, legacy_harm = self.ranker.predict_with_lcb(rows)
                outputs[i] = dict(mu=mu, lcb=lcb, harm=self.risk.predict_proba(rows), legacy_harm=legacy_harm)
        else:
            predictions = (self.flow.predict_strong(self.model, active, self.family)
                           if self.kind == 'strong84' else self.flow.predict_weak(self.model, active))
            for i, prediction in zip(nonempty, predictions):
                outputs[i] = {k: prediction[k] for k in ('mu', 'harm', 'member_mu', 'member_harm')}
                outputs[i]['empirical_member_sd'] = prediction['member_mu'].std(axis=0)
        return outputs


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='current')
    parser.add_argument('--family', choices=('rewrite', 'fusion'), default='rewrite')
    parser.add_argument('--fold', type=int)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--input', type=Path, help='JSON list of observable query records')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(resolve(args.version), ensure_ascii=False, indent=2))
        return
    if args.input is None or args.output is None:
        parser.error('Prediction requires --input and --output')
    model = GainModel(args.version, args.family, args.fold)
    result = dict(version=model.version, family=args.family, fold=args.fold,
                  risk_certified=False, adoption_policy_included=False,
                  predictions=jsonable(model.predict(read_json(args.input))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
