from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .belief import BeliefState
from .item_bank import Item


def projected_mean_variance(belief: BeliefState, item: Item) -> tuple[float, float]:
    """Takes current belief and item, returns mean and variance of a^T theta under the current belief. 
    Since theta is Gaussian, the mean and variance of a^T theta are also Gaussian."""
    mu_eta = float(item.a @ belief.mu) # mean of a^T theta under the current belief
    var_eta = max(float(item.a @ belief.Sigma @ item.a), 0.0) # variance of a^T theta under the current belief
    return mu_eta, var_eta


def normalized_update_direction(belief: BeliefState, item: Item) -> NDArray[np.float64]:
    """
    Calculates direction in the latent space along which the belief is updated.
    Return q = Sigma a / sqrt(a^T Sigma a), the rank-one covariance-weighted update direction
    in the ellipsoidal / Gaussian projection formulas.
    This is the same for all response categories of a given item. This is used in the update
    mechanism after observing a response and for the myopic score calculation. The surrogate does not use this. 

    """
    _, var_eta = projected_mean_variance(belief, item)
    if var_eta <= 0:
        raise ValueError("Projected variance must be positive.")
    return belief.Sigma @ item.a / np.sqrt(var_eta) # question direction is adjusted by the current uncertainty geometry


def response_interval_bounds(
    belief: BeliefState,
    item: Item,
    response: int,
) -> tuple[float, float, float]:
    """
    This function turns the response category into a interval in the latent space:
    Once the ordinal response tells us that the latent score lies in an interval, 
    the Bayesian update becomes a truncated-normal conditioning problem. 
    This function computes the scalar quantities needed for that.
    Return (alpha, beta, gamma), where
    gamma = sqrt(a^T Sigma a + sigma_obs^2), 
    alpha = (tau_r   - a^T mu) / gamma,
    beta  = (tau_r+1 - a^T mu) / gamma.

    tau_r = -inf for response == 0, tau_r+1 = +inf for response == n_categories - 1.
    sigma_obs^2 is linked to item.behavioral_sensitivity through
    item.observation_noise_variance = base + scale * behavioral_sensitivity.
    """
    if not (0 <= response < item.n_categories):
        raise ValueError(f"response must be in {{0, ..., {item.n_categories - 1}}}.")
    mu_eta, var_eta = projected_mean_variance(belief, item)
    obs_noise_var = item.observation_noise_variance
    gamma = float(np.sqrt(var_eta + obs_noise_var))
    
    if response == 0:
        alpha = -np.inf # -inf because the lowest response category has no lower bound
    else:
        alpha = (float(item.thresholds[response - 1]) - mu_eta) / gamma

    if response >= item.n_categories - 1:
        beta = np.inf # +inf because the highest response category has no upper bound
    else:
        beta = (float(item.thresholds[response]) - mu_eta) / gamma

    return alpha, beta, gamma


def _truncated_normal_coefficients(
    alpha: float,
    beta: float,
) -> tuple[float, float, float]:
    """
    Once the ordinal response tells us that the latent score lies in certain interval, 
    the Bayesian update becomes a truncated-normal conditioning problem. 
    This function computes the scalar quantities needed for that.
    For a standard normal truncated to [alpha, beta), compute:
      p   = Phi(beta) - Phi(alpha)                           (probability mass)
      lam = [phi(alpha) - phi(beta)] / p                     (truncated mean)
      eta = 1 + [alpha*phi(alpha) - beta*phi(beta)] / p - lam^2   (truncated variance)

    Handles infinite bounds: phi(+-inf) = 0, Phi(-inf) = 0, Phi(+inf) = 1.
    """
    phi_a = 0.0 if np.isinf(alpha) else float(stats.norm.pdf(alpha))
    phi_b = 0.0 if np.isinf(beta)  else float(stats.norm.pdf(beta))
    cdf_a = 0.0 if alpha == -np.inf else float(stats.norm.cdf(alpha))
    cdf_b = 1.0 if beta  ==  np.inf else float(stats.norm.cdf(beta))

    p = max(cdf_b - cdf_a, 1e-300)

    lam = (phi_a - phi_b) / p

    # x * phi(x) -> 0 as x -> +-inf
    alpha_phi = 0.0 if np.isinf(alpha) else alpha * phi_a
    beta_phi  = 0.0 if np.isinf(beta)  else beta  * phi_b
    eta = 1.0 + (alpha_phi - beta_phi) / p - lam ** 2
    eta = max(eta, 0.0)
    return p, lam, eta


def one_step_posterior_coefficients(
    belief: BeliefState,
    item: Item,
    response: int,
) -> tuple[float, float, float]:
    """

    Exact one-step posterior-moment coefficients (Prop. 1, Corollary 1).

    Returns (m, v, p_r) where:
      rho = sqrt(a^T Sigma a) / sqrt(a^T Sigma a + sigma_obs^2)
      m   = rho * lambda          (E[U | response])
      v   = 1 - rho^2 + rho^2 * eta  (Var[U | response])
      p_r = predictive probability of the realized category
    """
    _, var_eta = projected_mean_variance(belief, item)
    alpha, beta, gamma = response_interval_bounds(belief, item, response)

    rho = float(np.sqrt(var_eta)) / gamma  # s_{k,i} / sqrt(s_{k,i}^2 + sigma_obs^2)

    p_r, lam, eta = _truncated_normal_coefficients(alpha, beta)

    m = rho * lam
    v = 1.0 - rho ** 2 + rho ** 2 * eta
    v = max(v, 0.0)

    return m, v, p_r


def update_belief(
    belief: BeliefState,
    item: Item,
    response: int,
) -> BeliefState:
    """
    Rank-one Gaussian moment-matching update after observing a response category to a given item.
    (Proposition 1, Equations mu-update and sigma-update).

    mu_{k+1}    = mu_k    + q * m
    Sigma_{k+1} = Sigma_k + (v - 1) * outer(q, q)

    where q = Sigma a / sqrt(a^T Sigma a) is the covariance-weighted update direction. 
    Geometrically, the posterior mean shift and covariance correction act only in this one direction.
    """
    m, v, _ = one_step_posterior_coefficients(belief, item, response)
    q = normalized_update_direction(belief, item)

    mu_new = belief.mu + q * m # mean shift in the update direction (Proposition 1, Eq. mu-update)
    Sigma_new = belief.Sigma + (v - 1.0) * np.outer(q, q) # covariance correction in the update direction (Proposition 1, Eq. sigma-update) - because this is a rank-one update, the covariance contracriction acts only in the update direction
    asym = np.linalg.norm(Sigma_new - Sigma_new.T, ord="fro")
    if asym > 1e-10:
        raise ValueError(f"Unusually large asymmetry {asym:.3e} in the covariance matrix.")
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T) # ensure symmetry of the covariance matrix in floating point arithmetic

    return BeliefState(mu=mu_new, Sigma=Sigma_new)


def damped_update_belief(
    belief: BeliefState,
    item: Item,
    response: int,
) -> BeliefState:
    """
    Response-adaptive damped Gaussian projection
    = If the observed response was very unlikely under the current belief, 
    we may want to avoid shrinking uncertainty too aggressively after just one surprising answer.
    (Section "Response-adaptive damping variant", Eqs. damped_m, damped_v,
    damped_mu, damped_sigma).

    This starts off with the one-step posterior coefficients, dampens them and then uses the sam
    update mechanism as in the one-step posterior coefficients.

    The step size epsilon = p_r (predictive probability of the realized category).
    Damped scalar moments:
      bar_m = epsilon * m / (v + epsilon * (1 - v))
      bar_v =           v / (v + epsilon * (1 - v))
    """
    m, v, p_r = one_step_posterior_coefficients(belief, item, response)
    epsilon = p_r

    denom = v + epsilon * (1.0 - v)
    bar_m = epsilon * m / denom
    bar_v = v / denom

    q = normalized_update_direction(belief, item)

    mu_new = belief.mu + q * bar_m
    Sigma_new = belief.Sigma + (bar_v - 1.0) * np.outer(q, q)
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)

    return BeliefState(mu=mu_new, Sigma=Sigma_new)
