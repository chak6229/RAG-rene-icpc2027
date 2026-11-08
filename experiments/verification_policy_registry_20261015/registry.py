"""Versioned, gold-free gain-priority verification and candidate adoption.

The registered strategy is a formal component. Risk certification remains a
separate procedure and is not implied by this registry's CURRENT pointer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.gain_model_registry_20261014.registry import GainModel, observable as gain_observable, sha

CONFIG = ROOT / 'configs/vac_mqr_verification_policy'
FORMAT = 'vac-mqr-verification-policy-v1'
ROUTER = 'sha256-query-v1'
ROUTE_PREFIX = 'gain-risk-route-261015:'
ALIASES = {'old': 'legacy-hash25'}


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


def route_query(query):
    """Route the complete original query text, without canonicalization."""
    if not isinstance(query, str):
        raise ValueError('query must be a string')
    return int(hashlib.sha256((ROUTE_PREFIX + query).encode('utf8')).hexdigest()[:16], 16) % 4


def observable(item):
    view = gain_observable(item, 'strong84')
    if not isinstance(item.get('query'), str):
        raise ValueError('query must be a string')
    view['query'] = item['query']
    return view


def score_view(score, planned_docs):
    if score is None:
        raise ValueError('CE scores are required for this planned verification')
    ids = np.asarray(score['doc_ids'], int)
    logits = np.asarray(score['logits'], float)
    cls = np.asarray(score['cls_projection'], float)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError('CE document IDs must be a unique vector')
    if logits.shape != (len(ids),) or cls.shape != (len(ids), 32):
        raise ValueError('CE scores must have logits[n] and cls_projection[n,32]')
    if not np.isfinite(logits).all() or not np.isfinite(cls).all():
        raise ValueError('CE scores must be finite')
    lookup = {int(doc): i for i, doc in enumerate(ids)}
    if not set(planned_docs).issubset(lookup):
        raise ValueError('CE scores do not cover the actual planned documents')
    ix = [lookup[int(doc)] for doc in planned_docs]
    # Unplanned CE documents never enter the update feature surface.
    return dict(doc_ids=ids[ix], logits=logits[ix], cls_projection=cls[ix])


class VerificationPolicy:
    def __init__(self, version='current', family='fusion', fold=None, router=ROUTER):
        if family not in ('rewrite', 'fusion'):
            raise ValueError('family must be rewrite or fusion')
        if fold is not None and (not isinstance(fold, int) or fold not in range(4)):
            raise ValueError('fold must be 0..3')
        if router != ROUTER:
            raise ValueError('Unsupported router')
        self.manifest = resolve(version)
        if self.manifest['format'] != FORMAT:
            raise ValueError('Unsupported manifest format')
        for path, expected in self.manifest['inference_source_hashes'].items():
            if sha(ROOT / path) != expected:
                raise ValueError('Inference source changed: ' + path)
        dependency = self.manifest['gain_manifest']
        if sha(ROOT / dependency['path']) != dependency['sha256']:
            raise ValueError('Bound gain manifest changed')
        self.version = self.manifest['version']
        self.family = family
        self.fold = fold
        self.router = router
        self.models = {}
        self.arm = self.manifest['policy']
        for f in range(4) if fold is None else (fold,):
            record = self.manifest['models'][f'{family}/fold{f}']
            if sha(ROOT / record['path']) != record['sha256']:
                raise ValueError('Verification bundle hash mismatch')
            bundle = joblib.load(ROOT / record['path'])
            if (bundle['format'], bundle['family'], bundle['fold']) != (FORMAT, family, f):
                raise ValueError('Verification bundle identity mismatch')
            gain = GainModel(version='strong84-20261014', family=family, fold=f)
            self.models[f] = dict(gain=gain, update=bundle['update'], record=record)

    def prepare(self, items):
        """Plan CE calls before consuming any CE scores or outcomes."""
        views = [observable(item) for item in items]
        folds = [self.fold if self.fold is not None else route_query(v['query']) for v in views]
        sessions = [None] * len(views)
        for f, bound in self.models.items():
            indices = [i for i, route in enumerate(folds) if route == f]
            if not indices:
                continue
            flow = bound['gain'].flow
            predictions = [dict(x84=np.empty((0, 84)), mu=np.empty(0), harm=np.empty(0),
                                member_mu=np.empty((5, 0)), member_harm=np.empty((5, 0))) for i in indices]
            active = [k for k, i in enumerate(indices) if len(views[i]['x'])]
            if active:
                values = flow.predict_strong(bound['gain'].model, [views[indices[k]] for k in active], self.family)
                for k, value in zip(active, values):
                    predictions[k] = value
            record = bound['record']
            for i, prediction in zip(indices, predictions):
                short = sorted(np.flatnonzero(prediction['mu'] > 0).tolist(),
                               key=lambda j: (-prediction['mu'][j], j))[:5]
                priority = float(prediction['mu'][short[0]]) if short else 0.
                if self.arm == 'source_hash25' and short:
                    js = np.asarray(short, int)
                    sd = prediction['member_harm'][:, js].std(0)
                    priority = float(np.max(prediction['mu'][js] * sd /
                        (.01 + np.abs(prediction['harm'][js] - record['tau']) + sd)))
                trigger = (flow.query_hash(views[i]['query']) < .25 if self.arm == 'source_hash25'
                           else priority >= record['cut'] and priority > 0)
                verified = bool(trigger and short)
                plan = None
                if verified:
                    winner = short[0]
                    rival = flow.choose_rival(prediction, winner, short, 'point')
                    plan = flow.plan_pair(views[i], prediction, self.family, winner, rival, short)
                sessions[i] = dict(fold=f, verified=verified, plan=plan, before=prediction,
                                   priority=priority, tau=float(record['tau']), cut=float(record['cut']),
                                   _item=views[i], _version=self.version, _family=self.family)
        return sessions

    def finalize(self, sessions, scores=None):
        """Apply the frozen CE update and gate to the previously planned actions."""
        if scores is None:
            scores = [None] * len(sessions)
        if len(scores) != len(sessions):
            raise ValueError('One aligned CE score record per query is required')
        outputs = []
        for session, score in zip(sessions, scores):
            if (session['_version'], session['_family']) != (self.version, self.family):
                raise ValueError('Session belongs to a different policy')
            f = session['fold']; bound = self.models[f]; flow = bound['gain'].flow
            item = session['_item']; prediction = session['before']
            mu = prediction['mu'].copy(); harm = prediction['harm'].copy()
            allowed = list(range(len(mu))); updates = []
            if session['verified']:
                plan = session['plan']; allowed = list(plan['candidates'])
                consumed = score_view(score, plan['doc_ids'])
                x = np.stack([flow.witness_x(item, prediction, self.family, consumed, plan, j) for j in allowed])
                residual = np.stack([m['mean'].predict(x) for m in bound['update']['members']])
                direct = np.stack([np.full(len(x), m['harm']['constant']) if m['harm']['booster'] is None
                                   else np.clip(m['harm']['booster'].predict(x), 0., 1.)
                                   for m in bound['update']['members']])
                for k, j in enumerate(allowed):
                    member_mu = prediction['member_mu'][:, j] + .5 * residual[:, k]
                    mu[j] = member_mu.mean(); harm[j] = direct[:, k].mean()
                    updates.append(dict(candidate=j, mu=float(mu[j]), harm=float(harm[j]),
                                        member_mu=member_mu, member_harm=direct[:, k]))
            eligible = [j for j in allowed if mu[j] > 0 and harm[j] <= session['tau']]
            chosen = min(eligible, key=lambda j: (-mu[j], j)) if eligible else -1
            original = tuple(map(int, item['original_top100']))
            ranking = original if chosen < 0 else tuple(map(int, flow.c.action_ranking(item, chosen, self.family)))
            identity = chosen >= 0 and ranking == original
            outputs.append(dict(version=self.version, family=self.family, fold=f, ranking=ranking,
                chosen=chosen if not identity else -1, accepted=bool(chosen >= 0 and not identity),
                identity_fallback=bool(identity), verified=session['verified'], plan=session['plan'],
                priority=session['priority'], tau=session['tau'], cut=session['cut'],
                mu=float(mu[chosen]) if chosen >= 0 else 0., harm=float(harm[chosen]) if chosen >= 0 else 0.,
                final_mu=mu, final_harm=harm, updated=updates, risk_certified=False))
        return outputs

    def decide(self, items, scores=None):
        return self.finalize(self.prepare(items), scores)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items() if not k.startswith('_')}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default='current')
    parser.add_argument('--family', default='fusion', choices=('rewrite', 'fusion'))
    parser.add_argument('--fold', type=int)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--input'); parser.add_argument('--scores'); parser.add_argument('--output')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(resolve(args.version), ensure_ascii=False, indent=2)); return
    if not args.input or not args.output:
        parser.error('--input and --output are required unless --describe is set')
    policy = VerificationPolicy(args.version, args.family, args.fold)
    outputs = policy.decide(read_json(args.input), read_json(args.scores) if args.scores else None)
    Path(args.output).write_text(json.dumps(jsonable(outputs), ensure_ascii=False, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
