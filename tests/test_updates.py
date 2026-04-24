import numpy as np
import pytest

from src.belief import BeliefState
from src.item_bank import Item
from src.updates import (
    projected_mean_variance,
    normalized_update_direction,
    response_interval_bounds,
    one_step_posterior_coefficients,
    update_belief,
    damped_update_belief,
)


def make_test_belief() -> BeliefState:
    return BeliefState(
        mu=np.array([1.0, 2.0]),
        Sigma=np.array([[2.0, 0.3], [0.3, 1.5]]),
    )


def make_test_item() -> Item:
    return Item(
        item_id="q1",
        a=np.array([1.0, -1.0]),
        thresholds=np.array([-1.0, 0.5]),
        sensitivity=0.0,
    )


def test_projected_mean_variance_matches_manual_calculation() -> None:
    belief = make_test_belief()
    item = make_test_item()

    mu_eta, var_eta = projected_mean_variance(belief, item)

    manual_mu = float(item.a @ belief.mu)
    manual_var = float(item.a @ belief.Sigma @ item.a)

    assert np.isclose(mu_eta, manual_mu)
    assert np.isclose(var_eta, manual_var)
    assert var_eta > 0.0


def test_normalized_update_direction_matches_manual_calculation() -> None:
    belief = make_test_belief()
    item = make_test_item()

    q = normalized_update_direction(belief, item)
    denom = np.sqrt(float(item.a @ belief.Sigma @ item.a))
    manual_q = belief.Sigma @ item.a / denom

    assert np.allclose(q, manual_q)


@pytest.mark.parametrize(
    "response, expected_alpha_inf, expected_beta_inf",
    [
        (0, True, False),
        (1, False, False),
        (2, False, True),
    ],
)
def test_response_interval_bounds_edge_cases(
    response: int,
    expected_alpha_inf: bool,
    expected_beta_inf: bool,
) -> None:
    belief = make_test_belief()
    item = make_test_item()

    alpha, beta, gamma = response_interval_bounds(belief, item, response)

    assert gamma > 0.0
    assert np.isneginf(alpha) == expected_alpha_inf
    assert np.isposinf(beta) == expected_beta_inf
    assert alpha < beta


def test_conditional_moments_are_well_formed() -> None:
    belief = make_test_belief()
    item = make_test_item()

    for response in range(item.n_categories):
        m, v, p_r = one_step_posterior_coefficients(belief, item, response)
        assert np.isfinite(m)
        assert np.isfinite(v)
        assert np.isfinite(p_r)
        assert 0.0 < p_r <= 1.0
        assert 0.0 < v <= 1.0 + 1e-10


def test_update_belief_preserves_symmetry_and_spd() -> None:
    belief = make_test_belief()
    item = make_test_item()

    updated = update_belief(belief, item, response=1)

    assert np.allclose(updated.Sigma, updated.Sigma.T)
    eigvals = np.linalg.eigvalsh(updated.Sigma)
    assert np.all(eigvals > 0.0)


def test_update_reduces_variance_in_queried_direction() -> None:
    belief = make_test_belief()
    item = make_test_item()

    before = float(item.a @ belief.Sigma @ item.a)
    updated = update_belief(belief, item, response=1)
    after = float(item.a @ updated.Sigma @ item.a)

    assert after <= before + 1e-10


def test_damped_update_is_less_aggressive_than_undamped_in_queried_direction() -> None:
    belief = make_test_belief()
    item = make_test_item()

    undamped = update_belief(belief, item, response=1)
    damped = damped_update_belief(belief, item, response=1)

    var_before = float(item.a @ belief.Sigma @ item.a)
    var_undamped = float(item.a @ undamped.Sigma @ item.a)
    var_damped = float(item.a @ damped.Sigma @ item.a)

    # Both should contract relative to the prior
    assert var_undamped <= var_before + 1e-10
    assert var_damped <= var_before + 1e-10

    # Damped should usually contract less than undamped
    assert var_damped >= var_undamped - 1e-10


def test_invalid_response_raises() -> None:
    belief = make_test_belief()
    item = make_test_item()

    with pytest.raises(ValueError):
        response_interval_bounds(belief, item, response=-1)

    with pytest.raises(ValueError):
        response_interval_bounds(belief, item, response=item.n_categories)