import numpy as np

from src.belief import BeliefState
from src.grm import category_probabilities
from src.item_bank import Item


def test_category_probabilities_sum_to_one_and_nonnegative() -> None:
    # test that the category probabilities sum to one, forming a valid probability distribution
    # test that 3 thresholds means 4 categories
    # test that the category probabilities are non-negative 
    belief = BeliefState(
        mu=np.array([0.1, -0.2]),
        Sigma=np.eye(2) * 0.25,
    )
    item = Item(
        item_id="q1",
        a=np.array([1.0, 0.5]),
        thresholds=np.array([-0.5, 0.0, 0.8]),
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
    )

    p = category_probabilities(belief, item)

    assert p.shape == (2,)
    assert np.all(p >= 0.0)
    assert np.isclose(p.sum(), 1.0)


def test_binary_item_higher_projected_mean_increases_upper_category_probability() -> None:
    # creates a binary item with a single threshold at 0.0
    # tests that the upper category probability is higher for the belief state with the higher projected mean
    Sigma = np.array([[0.5]])
    item = Item(
        item_id="b",
        a=np.array([1.0]),
        thresholds=np.array([0.0]),
    )

    belief_low = BeliefState(mu=np.array([-1.0]), Sigma=Sigma)
    belief_high = BeliefState(mu=np.array([1.0]), Sigma=Sigma)

    p_low = category_probabilities(belief_low, item)
    p_high = category_probabilities(belief_high, item)

    assert p_high[1] > p_low[1]
    assert p_high[0] < p_low[0]


def test_sensitivity_level_does_not_change_category_probabilities() -> None:
    """
    sensitivity_level is metadata for the dropout model. It must not affect
    category_probabilities — two items that differ only in sensitivity_level
    must produce identical predictive distributions.
    """
    belief = BeliefState(
        mu=np.array([0.6]),
        Sigma=np.array([[0.5]]),
    )
    low_level_item = Item(
        item_id="b_low",
        a=np.array([1.0]),
        thresholds=np.array([0.0]),
        is_sensitive=True,
        sensitivity_level=0.0,
    )
    high_level_item = Item(
        item_id="b_high",
        a=np.array([1.0]),
        thresholds=np.array([0.0]),
        is_sensitive=True,
        sensitivity_level=1.0,
    )

    p_low  = category_probabilities(belief, low_level_item)
    p_high = category_probabilities(belief, high_level_item)

    assert high_level_item.observation_noise_variance == low_level_item.observation_noise_variance
    assert np.allclose(p_high, p_low)


def test_higher_response_noise_flattens_category_probabilities() -> None:
    belief = BeliefState(
        mu=np.array([0.6]),
        Sigma=np.array([[0.5]]),
    )
    low_noise_item = Item(
        item_id="b_low",
        a=np.array([1.0]),
        thresholds=np.array([0.0]),
        response_noise_variance=1.0,
    )
    high_noise_item = Item(
        item_id="b_high",
        a=np.array([1.0]),
        thresholds=np.array([0.0]),
        response_noise_variance=3.0,
    )

    p_low_noise = category_probabilities(belief, low_noise_item)
    p_high_noise = category_probabilities(belief, high_noise_item)

    assert high_noise_item.observation_noise_variance > low_noise_item.observation_noise_variance
    # More observation noise should flatten probabilities toward 0.5 in binary case.
    assert abs(p_high_noise[1] - 0.5) < abs(p_low_noise[1] - 0.5)
