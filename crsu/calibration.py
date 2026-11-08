"""Query-based one-sided residual calibration; no conditional risk guarantee."""
from dataclasses import dataclass
import math
import numpy as np


def _aligned(mu, scale, mask):
    mu, scale, mask = np.asarray(mu, float), np.asarray(scale, float), np.asarray(mask)
    if mu.ndim != 2 or mu.shape != scale.shape or mu.shape != mask.shape or mask.dtype != bool:
        raise ValueError('aligned [query,action] arrays and boolean action mask required')
    if not np.isfinite(mu).all() or not np.isfinite(scale).all() or np.any(scale < 0):
        raise ValueError('finite mean and nonnegative scale required')
    return mu, scale, mask


@dataclass(frozen=True)
class LowerBoundCalibrator:
    correction: float
    alpha: float
    scale_floor: float
    mode: str
    eligible_query_count: int
    policy_id: str

    @classmethod
    def fit(cls, mu, scale, observed, available, *, policy_id, alpha=.1,
            scale_floor=.001, mode='first_choice'):
        """first_choice uses mean argmax with input action order for exact ties.

        simultaneous takes the max across the entire frozen available action set.
        The caller must pre-freeze this set and use the exact same inference policy.
        Empty-action queries are excluded from the calibration estimand, counted
        separately in evaluation adoption (which uses ALL queries).
        """
        if not 0 < alpha < 1 or scale_floor <= 0 or not np.isfinite(scale_floor):
            raise ValueError('invalid calibration hyperparameters')
        if mode not in ('first_choice', 'simultaneous') or not policy_id:
            raise ValueError('valid mode and explicit frozen policy ID required')
        mu, scale, mask = _aligned(mu, scale, available)
        y = np.asarray(observed, float)
        if y.shape != mu.shape or not np.isfinite(y).all():
            raise ValueError('finite labels aligned with predicted actions required')
        eligible = mask.any(axis=1)
        if not eligible.any():
            raise ValueError('no eligible calibration queries')
        errors = (mu-y) / np.maximum(scale, scale_floor)
        if mode == 'simultaneous':
            scores = np.where(mask, errors, -np.inf).max(1)[eligible]
        else:
            first = np.argmax(np.where(mask, mu, -np.inf), axis=1)
            scores = errors[np.arange(len(mu)), first][eligible]
        n = len(scores)
        rank = math.ceil((n+1)*(1-alpha))
        correction = max(float(np.sort(scores)[rank-1]), 0.) if rank <= n else float('inf')
        return cls(correction, alpha, scale_floor, mode, n, policy_id)

    def lower(self, mu, scale, available, *, policy_id):
        if policy_id != self.policy_id:
            raise ValueError('inference policy differs from calibrated policy')
        mu, scale, mask = _aligned(mu, scale, available)
        if math.isinf(self.correction):
            return np.full(mu.shape, -np.inf)
        return np.where(mask, mu-self.correction*np.maximum(scale, self.scale_floor), -np.inf)
