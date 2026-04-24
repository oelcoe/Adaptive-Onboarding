import numpy as np

from src.belief import BeliefState
from src.grm import category_probabilities
from src.item_bank import Item


def test_category_probabilities_sum_to_one_and_nonnegative() -> None:
    belief = BeliefState(
        mu=np.array([0.1, -0.2]),
        Sigma=np.eye(2) * 0.25,
    )
    item = Item(
        item_id="q1",
        a=np.array([1.0, 0.5]),
        thresholds=np.array([-0.5, 0.0, 0.8]),
        sensitivity=1.0,
    )

    p = category_probabilities(belief, item)

    assert p.shape == (4,)
    assert np.all(p >= -1e-12)
    assert np.isclose(p.sum(), 1.0)


def test_binary_item_probabilities_valid() -> None:
    belief = BeliefState(
        mu=np.array([0.0]),
        Sigma=np.array([[0.5]]),
    )
    item = Item(
        item_id="b",
        a=np.array([1.0]),
        thresholds=np.array([0.0]),
        sensitivity=0.0,
    )

    p = category_probabilities(belief, item)

    assert p.shape == (2,)
    assert np.all(p >= 0.0)
    assert np.isclose(p.sum(), 1.0)


def test_binary_item_higher_projected_mean_increases_upper_category_probability() -> None:
    Sigma = np.array([[0.5]])
    item = Item(
        item_id="b",
        a=np.array([1.0]),
        thresholds=np.array([0.0]),
        sensitivity=0.0,
    )

    belief_low = BeliefState(mu=np.array([-1.0]), Sigma=Sigma)
    belief_high = BeliefState(mu=np.array([1.0]), Sigma=Sigma)

    p_low = category_probabilities(belief_low, item)
    p_high = category_probabilities(belief_high, item)

    assert p_high[1] > p_low[1]
    assert p_high[0] < p_low[0]