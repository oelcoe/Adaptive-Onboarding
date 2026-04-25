import numpy as np

from src.belief import BeliefState
from src.item_bank import Item
from src.policies import (
    make_linear_sensitivity_stay_prob,
    score_bank,
    score_surrogate_weighted,
)


def make_test_belief() -> BeliefState:
    return BeliefState(
        mu=np.array([0.0, 0.0]),
        Sigma=np.array([[1.5, 0.2], [0.2, 1.0]]),
    )


def test_weighted_surrogate_matches_log1p_pstay_times_projected_variance() -> None:
    belief = make_test_belief()
    item = Item(
        item_id="q",
        a=np.array([1.0, 0.0]),
        thresholds=np.array([0.0]),
        behavioral_sensitivity=0.4,
    )
    stay_prob_fn = make_linear_sensitivity_stay_prob(gamma0=0.8, gamma_step=0.1)

    score = score_surrogate_weighted(belief, item, step=2, stay_prob_fn=stay_prob_fn)
    p_stay = stay_prob_fn(item, 2)
    projected_var = float(item.a @ belief.Sigma @ item.a)
    expected = float(np.log1p(p_stay * projected_var))

    assert np.isclose(score, expected)


def test_linear_sensitivity_penalty_weakens_over_time() -> None:
    item = Item(
        item_id="q",
        a=np.array([1.0]),
        thresholds=np.array([0.0]),
        behavioral_sensitivity=1.0,
    )
    stay_prob_fn = make_linear_sensitivity_stay_prob(gamma0=0.8, gamma_step=0.2)

    p0 = stay_prob_fn(item, 0)
    p1 = stay_prob_fn(item, 1)
    p3 = stay_prob_fn(item, 3)

    assert p0 <= p1 <= p3


def test_weighted_policy_ranking_uses_inside_log_weighting() -> None:
    belief = make_test_belief()
    item_high_var = Item(
        item_id="high_var",
        a=np.array([1.0, 0.0]),
        thresholds=np.array([0.0]),
        behavioral_sensitivity=1.0,
    )
    item_low_var = Item(
        item_id="low_var",
        a=np.array([0.0, 1.0]),
        thresholds=np.array([0.0]),
        behavioral_sensitivity=0.0,
    )
    stay_prob_fn = make_linear_sensitivity_stay_prob(gamma0=0.5, gamma_step=0.0)

    scored = score_bank(
        belief=belief,
        item_bank=[item_high_var, item_low_var],
        step=0,
        weighted=True,
        stay_prob_fn=stay_prob_fn,
    )

    scores = {row.item.item_id: row.score for row in scored}
    expected_high = np.log1p(stay_prob_fn(item_high_var, 0) * float(item_high_var.a @ belief.Sigma @ item_high_var.a))
    expected_low = np.log1p(stay_prob_fn(item_low_var, 0) * float(item_low_var.a @ belief.Sigma @ item_low_var.a))

    assert np.isclose(scores["high_var"], expected_high)
    assert np.isclose(scores["low_var"], expected_low)
