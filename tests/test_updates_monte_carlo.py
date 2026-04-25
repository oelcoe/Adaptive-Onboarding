import numpy as np
import pytest

from src.belief import BeliefState
from src.grm import category_probabilities
from src.item_bank import Item
from src.updates import response_interval_bounds, update_belief


@pytest.mark.slow
def test_update_belief_matches_monte_carlo_conditional_posterior() -> None:
    """
    This test verifies if our analytic update routine coincides with a brute force simulation of the posterior.
    We start with the prior, simulate many fake users from that prior, simulate what answer
    each fake user would have given to this one question, then for a fixed realized category, 
    keep only the fake users who gave the realized category response and look at the average and covariance of those 
    kept users, then check that our analytic update routine gives the same answer.

    This test is slow because it involves simulating many fake users from the prior.
    """
    rng = np.random.default_rng(12345)

    # Fixed test instance: moderate acceptance probability, interior category
    belief = BeliefState(
        mu=np.array([0.3, -0.4]),
        Sigma=np.array([[1.2, 0.25], [0.25, 0.9]]),
    )
    item = Item(
        item_id="mc_q1",
        a=np.array([1.0, -0.7]),
        thresholds=np.array([-0.6, 0.4]),
        sensitivity=0.0,
    )
    response = 1  # interior category for a 3-category item, fixed for the test, we keep only the fake users who gave this response

    # Analytic quantities
    analytic_probs = category_probabilities(belief, item)
    analytic_belief = update_belief(belief, item, response)

    # Monte Carlo draw from the prior - this is what makes the test slow
    n = 300_000
    theta = rng.multivariate_normal(mean=belief.mu, cov=belief.Sigma, size=n)
    eps = rng.standard_normal(size=n) # generate observation noise
    z = theta @ item.a + eps

    # Convert the chosen response (Zq) into a realized category interval
    alpha, beta, gamma = response_interval_bounds(belief, item, response)
    mu_eta = float(item.a @ belief.mu)

    lower = -np.inf if np.isneginf(alpha) else mu_eta + gamma * alpha
    upper = np.inf if np.isposinf(beta) else mu_eta + gamma * beta

    # We only accept samples consistent with the response
    mask = (z >= lower) & (z < upper)
    accepted = theta[mask]

    # Estimate the predictive probability empirically, giving us an estimate of the
    #  response probability of a category.
    empirical_prob = accepted.shape[0] / n
    assert accepted.shape[0] > 5_000  # stability check

    # Compute the empirical posterior mean and covariance.
    empirical_mu = accepted.mean(axis=0)
    empirical_Sigma = np.cov(accepted, rowvar=False, ddof=0)

    # Check that the empirical quantities agree with the analytic quantities
    # 1) Predictive response probability
    assert np.isclose(empirical_prob, analytic_probs[response], atol=0.01)

    # 2) Posterior mean
    assert np.allclose(empirical_mu, analytic_belief.mu, atol=0.03)

    # 3) Posterior covariance
    assert np.allclose(empirical_Sigma, analytic_belief.Sigma, atol=0.04)

    # 4) Queried-direction variance should also agree
    empirical_proj_var = float(item.a @ empirical_Sigma @ item.a)
    analytic_proj_var = float(item.a @ analytic_belief.Sigma @ item.a)
    assert np.isclose(empirical_proj_var, analytic_proj_var, atol=0.03)