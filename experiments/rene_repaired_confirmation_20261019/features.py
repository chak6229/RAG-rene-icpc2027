from __future__ import annotations

from collections import Counter

import numpy as np

from build_enhanced_retrieval_feedback_dataset import weighted_rank_agreement
from vac_mqr import CandidateQualityValidator

from .common import SEEDS


def union_draws(row, seeds=SEEDS):
    byseed = {int(d['seed']): d for d in row['draws']}
    if len(byseed) != len(row['draws']) or not set(seeds) <= set(byseed):
        raise ValueError('Missing/duplicate decoding seed')
    draws = [byseed[s] for s in seeds]
    candidates, appearances, lookup = [], [], {}
    for draw in draws:
        for candidate in draw['candidates']:
            key = CandidateQualityValidator.canonical(candidate['text'])
            if key not in lookup:
                lookup[key] = len(candidates)
                candidates.append(dict(candidate))
                appearances.append([])
            j = lookup[key]
            if draw['seed'] not in appearances[j]:
                appearances[j].append(int(draw['seed']))
    return dict(row, draws=draws), candidates, appearances


def sampling_features(row, vectors, tops, appearances, seeds=SEEDS):
    """Parametrized version of the registered 12-feature estimator.

    Draw probabilities and empty-draw quality counts use the actual schema.
    The eight-seed implementation is checked independently during preparation.
    """
    n = len(vectors)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Need at least two distinct seeds')
    if len(row['draws']) != len(seeds) or tuple(d['seed'] for d in row['draws']) != tuple(seeds):
        raise ValueError('Draw order/schema mismatch')
    if len(tops) != n or len(appearances) != n:
        raise ValueError('Candidate alignment mismatch')
    if not n:
        return np.empty((0, 12))
    if any(not a or not set(a) <= set(seeds) or len(a) != len(set(a)) for a in appearances):
        raise ValueError('Invalid candidate seed appearances')
    cos = np.clip(np.asarray(vectors) @ np.asarray(vectors).T, -1., 1.)
    a10, a100 = np.eye(n), np.eye(n)
    for i in range(n):
        for j in range(i):
            a10[i, j] = a10[j, i] = weighted_rank_agreement(tops[i][:10], tops[j][:10])
            a100[i, j] = a100[j, i] = weighted_rank_agreement(tops[i], tops[j])
    groups = [[i for i, a in enumerate(appearances) if s in a] for s in seeds]
    weights = np.zeros(n)
    counts = Counter()
    for group in groups:
        if not group:
            continue
        weights[group] += 1 / (len(seeds) * len(group))
        for i in group:
            for d in tops[i][:10]:
                counts[int(d)] += 1 / (len(seeds) * len(group))
    weights /= max(weights.sum(), 1e-12)
    semdisp = float(1 - weights @ cos @ weights)
    retdisp = float(1 - weights @ a10 @ weights)
    raw = sum(len(d['candidates']) for d in row['draws'])
    quality = float(np.mean([d['quality_pass_count'] / max(d['raw_count'], 1) for d in row['draws']]))
    output = []
    for i in range(n):
        origin = next(s for s in seeds if s in appearances[i])
        semantic, top10, top100 = [], [], []
        for seed, group in zip(seeds, groups):
            if seed == origin:
                continue
            semantic.append(float(cos[i, group].max()) if group else 0.)
            top10.append(float(a10[i, group].max()) if group else 0.)
            top100.append(float(a100[i, group].max()) if group else 0.)
        output.append([len(appearances[i]) / len(seeds), np.mean(semantic), np.std(semantic),
                       np.mean(top10), np.std(top10), np.mean(top100), np.std(top100),
                       np.mean([counts[int(d)] for d in tops[i][:10]]), semdisp, retdisp,
                       n / max(raw, 1), quality])
    result = np.asarray(output)
    if result.shape != (n, 12) or not np.isfinite(result).all():
        raise ValueError('Invalid seed feature matrix')
    return result
