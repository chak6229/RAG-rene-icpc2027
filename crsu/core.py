"""Exact label-free ranking geometry and probability pushforward for RR@k."""
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np


def stable_topk(scores, k=100):
    """Score descending, corpus-position ascending ties; full sort for prototype."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or k < 1:
        raise ValueError('finite 1-D scores and positive k required')
    return np.lexsort((np.arange(len(x)), -x))[:k]


def _ranking(values):
    ids = tuple(values)
    if any(not isinstance(x, (int, np.integer)) or x < 0 for x in ids):
        raise ValueError('document IDs must be nonnegative corpus positions')
    ids = tuple(int(x) for x in ids)
    if len(set(ids)) != len(ids):
        raise ValueError('a ranking cannot contain duplicate document IDs')
    return ids


def rrf(original, rewrite, retrieve_k=100, evaluate_k=100, offset=60):
    """Equal-weight RRF with the existing fusion tie policy, without gold labels."""
    if retrieve_k < 1 or evaluate_k < 1 or offset < 0:
        raise ValueError('invalid RRF limits')
    scores = {}
    for ranking in (original, rewrite):
        for rank, doc in enumerate(_ranking(ranking)[:retrieve_k], 1):
            scores[doc] = scores.get(doc, 0.) + 1. / (offset + rank)
    return tuple(sorted(scores, key=lambda doc: (-scores[doc], doc))[:evaluate_k])


@dataclass(frozen=True)
class RankingStates:
    """Docs + one OUTSIDE state; column zero is always the original action.

    Construct after the existing quality gate. Corpus IDs never enter the network.
    Every supplied action must be an available, actual ordered result list.
    """
    document_ids: tuple[int, ...]
    action_names: tuple[str, ...]
    ranks: np.ndarray                 # [D+1,A+1], zero = outside action Top-k
    cutoff: int

    @classmethod
    def from_rankings(cls, original: Sequence[int], actions: dict[str, Sequence[int]], cutoff=100):
        if cutoff < 1 or not isinstance(cutoff, int):
            raise ValueError('positive integer cutoff required')
        if any(not name or name == 'original' for name in actions):
            raise ValueError('non-reference actions need distinct nonempty names')
        lists = [_ranking(original)[:cutoff]] + [_ranking(v)[:cutoff] for v in actions.values()]
        documents = tuple(sorted(set().union(*map(set, lists))))
        index = {doc: i for i, doc in enumerate(documents)}
        ranks = np.zeros((len(documents)+1, len(lists)), dtype=np.int64)
        for a, ranking in enumerate(lists):
            for rank, doc in enumerate(ranking, 1):
                ranks[index[doc], a] = rank
        ranks.setflags(write=False)
        return cls(documents, ('original', *actions.keys()), ranks, cutoff)

    @property
    def reciprocal_ranks(self):
        result = np.zeros_like(self.ranks, dtype=np.float64)
        np.divide(1., self.ranks, out=result, where=self.ranks > 0)
        return result

    @property
    def gains(self):
        rr = self.reciprocal_ranks
        return rr[:, 1:] - rr[:, :1]

    def target_index(self, gold_document_id):
        """Label-side operation only; OUTSIDE if gold is absent from this union."""
        try:
            return self.document_ids.index(gold_document_id)
        except ValueError:
            return len(self.document_ids)

    def audit_labels(self, gold_document_id, original_rr, action_deltas, atol=1e-10):
        """Stop on historical rank/tie/corpus/policy mismatch; never repair silently."""
        if tuple(action_deltas) != self.action_names[1:]:
            raise ValueError('label action names/order mismatch')
        i = self.target_index(gold_document_id)
        actual = np.r_[self.reciprocal_ranks[i, 0], self.gains[i]]
        expected = np.r_[original_rr, list(action_deltas.values())]
        if not np.isfinite(expected).all() or not np.allclose(actual, expected, atol=atol, rtol=0):
            raise ValueError(f'ranking/label mismatch: computed={actual}, cached={expected}')

    def signature_groups(self):
        """Exact rational gain equivalence; original ranks may differ within a group."""
        groups = {}
        for i, row in enumerate(self.ranks):
            reciprocal = [Fraction(1, int(r)) if r else Fraction(0) for r in row]
            key = tuple(v-reciprocal[0] for v in reciprocal[1:])
            groups.setdefault(key, []).append(i)
        return tuple(tuple(v) for v in groups.values())


@dataclass(frozen=True)
class GainDistribution:
    mu: np.ndarray
    covariance: np.ndarray
    scale: np.ndarray
    harm: np.ndarray
    zero: np.ndarray
    positive: np.ndarray
    expected_loss: np.ndarray
    outside_probability: float
    support: np.ndarray
    probabilities: np.ndarray

    def marginal_quantile(self, level):
        if not 0 < level < 1:
            raise ValueError('quantile level must lie in (0,1)')
        result = []
        for a in range(self.support.shape[1]):
            order = np.argsort(self.support[:, a], kind='stable')
            cdf = np.cumsum(self.probabilities[order])
            i = min(int(np.searchsorted(cdf, level)), len(order)-1)
            result.append(self.support[order[i], a])
        return np.asarray(result)

    def crps(self, observation):
        """Exact marginal CRPS in O(A*S*log S), retaining duplicate atoms."""
        y = np.asarray(observation, dtype=np.float64)
        if y.shape != self.mu.shape or not np.isfinite(y).all():
            raise ValueError('one finite observed gain per action required')
        result = []
        for a in range(len(y)):
            order = np.argsort(self.support[:, a], kind='stable')
            x, p = self.support[order, a], self.probabilities[order]
            before_p = np.cumsum(p)-p
            before_px = np.cumsum(p*x)-p*x
            half_pair_distance = np.sum(p*(x*before_p-before_px))
            result.append(np.sum(p*np.abs(x-y[a]))-half_pair_distance)
        return np.asarray(result)

    def gap_variance(self, a, b):
        return max(float(self.covariance[a, a]+self.covariance[b, b]
                         -2*self.covariance[a, b]), 0.)


def summarize(states: RankingStates, probabilities):
    p = np.asarray(probabilities, dtype=np.float64)
    g = states.gains
    if p.shape != (len(g),) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('finite nonnegative probability per state required')
    if not np.isclose(p.sum(), 1., atol=1e-8, rtol=0):
        raise ValueError('probabilities must sum to one; no silent renormalization')
    mu = p @ g
    centered = g-mu
    covariance = centered.T @ (p[:, None]*centered)
    return GainDistribution(mu=mu, covariance=covariance,
                            scale=np.sqrt(np.maximum(np.diag(covariance), 0.)),
                            harm=p @ (g < 0), zero=p @ (g == 0), positive=p @ (g > 0),
                            expected_loss=p @ np.maximum(-g, 0),
                            outside_probability=float(p[-1]), support=g, probabilities=p.copy())
