from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .belief import BeliefState
from .item_bank import Item


def category_probabilities(
    belief: BeliefState,
    item: Item,
) -> NDArray[np.float64]:
    """
    Predictive category probabilities under the ordinal probit / GRM model.

    If theta ~ N(mu, Sigma) and Z = a^T theta + eps with eps ~ N(0, sigma_obs^2),
    then marginally Z ~ N(a^T mu, a^T Sigma a + sigma_obs^2). The observed category
    is determined by thresholding Z.

    Here sigma_obs^2 is linked to item.behavioral_sensitivity via
    item.observation_noise_variance = base + scale * behavioral_sensitivity.
    """
    mu = belief.mu
    Sigma = belief.Sigma
    a = item.a

    if a.shape[0] != belief.dim:
        raise ValueError("Dimensions of belief and item.a must align.")

    mu_eta = float(a @ mu)
    var_eta = max(float(a @ Sigma @ a), 0.0)
    obs_noise_var = item.observation_noise_variance
    denom = np.sqrt(var_eta + obs_noise_var)

    thr = item.thresholds
    cdf = stats.norm.cdf((thr - mu_eta) / denom)

    probs = np.empty(item.n_categories, dtype=float)
    probs[0] = cdf[0]
    if thr.size > 1:
        probs[1:-1] = cdf[1:] - cdf[:-1]
    probs[-1] = 1.0 - cdf[-1]

    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum()
    return probs
