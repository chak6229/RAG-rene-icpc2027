"""Actually refitted base34/base46 diagnostic, distinct from the main strong flow."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from run_uq_pilot import booster
from experiments.strong_flow_uq_20261014.flow import member_weights
from experiments.rene_repaired_confirmation_20261019.common import (
    OUT, SEED, FAMILIES, TOL, load, read_json, write_json, sha, require_runtime, reject_gold,
)
from experiments.rene_repaired_confirmation_20261019.model import fit_binary, binary_predict, choose_trace


def fit(items, family, dimension):
    x = np.concatenate([e['x'] if dimension == 34 else np.c_[e['x'], e['seed_x']] for e in items])
    q = np.concatenate([np.full(len(e['x']), e['query_index']) for e in items])
    repo = np.concatenate([np.full(len(e['x']), e['repo'], object) for e in items])
    y = np.concatenate([e['y'][family] for e in items])
    members = []
    for m in range(5):
        w = member_weights(q, repo, m)
        members.append(dict(mean=booster(x, y, SEED + 2000 + m, weight=w),
                            harm=fit_binary(x, (y < -TOL).astype(int), SEED + 2000 + m, w)))
    return dict(dimension=dimension, family=family, members=members, fit_repos=sorted(set(repo)),
                fit_queries=[e['query_index'] for e in items], refitted=True, inference_zeroing=False)


def predict(model, items):
    for e in items:
        reject_gold(e)
    arrays = [e['x'] if model['dimension'] == 34 else np.c_[e['x'], e['seed_x']] for e in items]
    x = np.concatenate(arrays)
    mm = np.stack([m['mean'].predict(x) for m in model['members']]) if len(x) else np.empty((5, 0))
    hh = np.stack([binary_predict(m['harm'], x) for m in model['members']]) if len(x) else np.empty((5, 0))
    offset = np.r_[0, np.cumsum([len(a) for a in arrays])]
    output = []
    for i, e in enumerate(items):
        p = dict(member_mu=mm[:, offset[i]:offset[i + 1]], member_harm=hh[:, offset[i]:offset[i + 1]],
                 mu=mm[:, offset[i]:offset[i + 1]].mean(0), harm=hh[:, offset[i]:offset[i + 1]].mean(0))
        session = dict(before=p, family=model['family'], verified=False, plan=None,
                       priority=float(max([0.] + p['mu'].tolist())))
        output.append(choose_trace(e, session, tau=1., use_ce=False))
    return output


def main():
    require_runtime()
    items = load(OUT / 'development/items.pkl')
    ids = set(read_json(OUT / 'development/DESIGN.json')['final_roles']['gain'])
    items = [e for e in items if e['query_index'] in ids]
    paths = []
    with threadpool_limits(limits=2):
        for family in FAMILIES:
            for dimension in (34, 46):
                path = OUT / f'diagnostic_models/{family}_base{dimension}.joblib'
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    joblib.dump(fit(items, family, dimension), path)
                paths.append(path)
    write_json(OUT / 'development/WEAK_CONTROL.json', dict(refitted=True, inference_zeroing=False,
               development_only_fit=True, dimensions=[34, 46], main_comparison='strong72 versus strong84 full flow',
               models={str(p.relative_to(ROOT)): sha(p) for p in paths}), immutable=True)
    print('Four actual base34/base46 diagnostic models trained', flush=True)


if __name__ == '__main__':
    main()
