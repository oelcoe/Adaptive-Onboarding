import numpy as np
import pytest

from src.belief import BeliefState
from src.item_bank import Item
from src.policies import (
    make_sensitive_constant_stay_prob,
    make_sensitive_level_stay_prob,
    no_dropout_stay_prob,
    score_bank_myopic_exact,
    score_myopic_exact,
    score_bank,
    score_surrogate_unweighted,
    score_surrogate_weighted,
    select_next_item,
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
        is_sensitive=True,
        sensitivity_level=0.4,
    )
    stay_prob_fn = make_sensitive_level_stay_prob(gamma0=0.8, gamma_step=0.1)

    score = score_surrogate_weighted(belief, item, step=2, stay_prob_fn=stay_prob_fn)
    p_stay = stay_prob_fn(item, 2)
    projected_var = float(item.a @ belief.Sigma @ item.a)
    expected = float(np.log1p(p_stay * projected_var))

    assert np.isclose(score, expected)


def test_sensitive_level_penalty_weakens_over_time() -> None:
    item = Item(
        item_id="q",
        a=np.array([1.0]),
        thresholds=np.array([0.0]),
        is_sensitive=True,
        sensitivity_level=1.0,
    )
    stay_prob_fn = make_sensitive_level_stay_prob(gamma0=0.8, gamma_step=0.2)

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
        is_sensitive=True,
        sensitivity_level=1.0,
    )
    item_low_var = Item(
        item_id="low_var",
        a=np.array([0.0, 1.0]),
        thresholds=np.array([0.0]),
        is_sensitive=False,
    )
    stay_prob_fn = make_sensitive_level_stay_prob(gamma0=0.5, gamma_step=0.0)

    scored = score_bank(
        belief=belief,
        item_bank=[item_high_var, item_low_var],
        step=0,
        weighted=True,
        stay_prob_fn=stay_prob_fn,
    )

    scores = {row.item.item_id: row.score for row in scored}
    expected_high = np.log1p(stay_prob_fn(item_high_var, 0) * float(item_high_var.a @ belief.Sigma @ item_high_var.a))
    expected_low  = np.log1p(stay_prob_fn(item_low_var,  0) * float(item_low_var.a  @ belief.Sigma @ item_low_var.a))

    assert np.isclose(scores["high_var"], expected_high)
    assert np.isclose(scores["low_var"],  expected_low)


def test_myopic_exact_score_matches_definition() -> None:
    belief = make_test_belief()
    item = Item(
        item_id="q",
        a=np.array([1.0, -0.4]),
        thresholds=np.array([-0.2, 0.5]),
        is_sensitive=True,
        sensitivity_level=0.3,
    )
    stay_prob_fn = make_sensitive_level_stay_prob(gamma0=0.7, gamma_step=0.1)

    score = score_myopic_exact(belief, item, step=1, stay_prob_fn=stay_prob_fn)

    from src.grm import category_probabilities
    from src.updates import one_step_posterior_coefficients

    probs = category_probabilities(belief, item)
    expected_inner = 0.0
    for r, p_r in enumerate(probs):
        _, v_r, _ = one_step_posterior_coefficients(belief, item, r)
        expected_inner += float(p_r) * (-np.log(float(np.clip(v_r, 1e-300, 1.0))))
    expected = stay_prob_fn(item, 1) * expected_inner

    assert np.isclose(score, expected)


def test_select_next_item_myopic_exact_returns_valid_item() -> None:
    belief = make_test_belief()
    item_a = Item(
        item_id="a",
        a=np.array([1.0, 0.0]),
        thresholds=np.array([0.0]),
        is_sensitive=True,
        sensitivity_level=0.1,
    )
    item_b = Item(
        item_id="b",
        a=np.array([0.0, 1.0]),
        thresholds=np.array([0.0]),
        is_sensitive=True,
        sensitivity_level=0.8,
    )
    stay_prob_fn = make_sensitive_level_stay_prob(gamma0=0.6, gamma_step=0.0)

    selected = select_next_item(
        belief=belief,
        item_bank=[item_a, item_b],
        step=0,
        strategy="myopic_exact",
        stay_prob_fn=stay_prob_fn,
    )

    assert selected.item_id in {"a", "b"}


def test_score_bank_myopic_exact_top_item_matches_selector() -> None:
    belief = make_test_belief()
    items = [
        Item("a", np.array([1.0, 0.0]), np.array([0.0]),       is_sensitive=True, sensitivity_level=0.1),
        Item("b", np.array([0.4, 0.9]), np.array([-0.2, 0.6]), is_sensitive=True, sensitivity_level=0.7),
        Item("c", np.array([0.0, 1.0]), np.array([0.1]),        is_sensitive=True, sensitivity_level=0.2),
    ]
    stay_prob_fn = make_sensitive_level_stay_prob(gamma0=0.6, gamma_step=0.05)

    scored = score_bank_myopic_exact(
        belief=belief,
        item_bank=items,
        step=1,
        stay_prob_fn=stay_prob_fn,
    )
    selected = select_next_item(
        belief=belief,
        item_bank=items,
        step=1,
        strategy="myopic_exact",
        stay_prob_fn=stay_prob_fn,
    )

    assert len(scored) == 3
    assert selected.item_id == scored[0].item.item_id


def test_sensitivity_flag_only_affects_stay_weighted_scores() -> None:
    """
    Two items with identical measurement direction and noise but different
    is_sensitive flags should score the same on unweighted surrogate, but
    differently on weighted surrogate and myopic exact (because sensitive
    items have a lower stay probability under the constant model).
    """
    belief = make_test_belief()
    stay_fn = make_sensitive_constant_stay_prob(p_stay_sensitive=0.4, p_stay_normal=1.0)

    safe_item = Item(
        item_id="safe",
        a=np.array([1.0, -0.4]),
        thresholds=np.array([-0.2, 0.5]),
        is_sensitive=False,
    )
    sensitive_item = Item(
        item_id="sensitive",
        a=np.array([1.0, -0.4]),
        thresholds=np.array([-0.2, 0.5]),
        is_sensitive=True,
        sensitivity_level=1.0,
    )

    unweighted_safe = score_surrogate_unweighted(belief, safe_item)
    unweighted_sens = score_surrogate_unweighted(belief, sensitive_item)
    weighted_safe   = score_surrogate_weighted(belief, safe_item,      step=0, stay_prob_fn=stay_fn)
    weighted_sens   = score_surrogate_weighted(belief, sensitive_item, step=0, stay_prob_fn=stay_fn)
    exact_safe      = score_myopic_exact(belief, safe_item,      step=0, stay_prob_fn=stay_fn)
    exact_sens      = score_myopic_exact(belief, sensitive_item, step=0, stay_prob_fn=stay_fn)

    assert np.isclose(unweighted_safe, unweighted_sens)
    assert weighted_sens < weighted_safe
    assert exact_sens    < exact_safe


def test_response_noise_affects_exact_myopic_but_not_surrogate() -> None:
    belief = make_test_belief()
    low_noise_item = Item(
        item_id="low_noise",
        a=np.array([1.0, -0.4]),
        thresholds=np.array([-0.2, 0.5]),
        response_noise_variance=1.0,
    )
    high_noise_item = Item(
        item_id="high_noise",
        a=np.array([1.0, -0.4]),
        thresholds=np.array([-0.2, 0.5]),
        response_noise_variance=5.0,
    )

    surrogate_low  = score_surrogate_weighted(belief, low_noise_item,  step=0, stay_prob_fn=no_dropout_stay_prob)
    surrogate_high = score_surrogate_weighted(belief, high_noise_item, step=0, stay_prob_fn=no_dropout_stay_prob)
    exact_low      = score_myopic_exact(belief, low_noise_item,  step=0, stay_prob_fn=no_dropout_stay_prob)
    exact_high     = score_myopic_exact(belief, high_noise_item, step=0, stay_prob_fn=no_dropout_stay_prob)

    assert np.isclose(surrogate_low, surrogate_high)
    assert exact_high < exact_low


def test_already_asked_filtering_applies_to_scores_and_selection() -> None:
    belief = make_test_belief()
    items = [
        Item("asked",     np.array([1.0, 0.0]), np.array([0.0])),
        Item("available", np.array([0.0, 1.0]), np.array([0.0])),
    ]

    surrogate_scored = score_bank(
        belief=belief,
        item_bank=items,
        step=0,
        already_asked=["asked"],
        weighted=False,
    )
    exact_scored = score_bank_myopic_exact(
        belief=belief,
        item_bank=items,
        step=0,
        already_asked=["asked"],
    )
    selected = select_next_item(
        belief=belief,
        item_bank=items,
        step=0,
        strategy="surrogate_unweighted",
        already_asked=["asked"],
    )

    assert [row.item.item_id for row in surrogate_scored] == ["available"]
    assert [row.item.item_id for row in exact_scored]     == ["available"]
    assert selected.item_id == "available"


def test_tied_scores_preserve_original_item_order() -> None:
    belief = make_test_belief()
    first  = Item("first",  np.array([1.0, 0.0]), np.array([0.0]))
    second = Item("second", np.array([1.0, 0.0]), np.array([0.0]))
    items  = [first, second]

    surrogate_scored = score_bank(belief=belief, item_bank=items, step=0, weighted=False)
    exact_scored     = score_bank_myopic_exact(belief=belief, item_bank=items, step=0)
    selected         = select_next_item(belief=belief, item_bank=items, step=0, strategy="surrogate_unweighted")

    assert [row.item.item_id for row in surrogate_scored] == ["first", "second"]
    assert [row.item.item_id for row in exact_scored]     == ["first", "second"]
    assert selected.item_id == "first"


@pytest.mark.parametrize("bad_stay_prob", [np.nan, np.inf, -np.inf])
def test_nonfinite_stay_probability_raises_for_direct_scores(bad_stay_prob: float) -> None:
    belief = make_test_belief()
    item   = Item("q", np.array([1.0, 0.0]), np.array([0.0]))

    def stay_prob_fn(_: Item, __: int) -> float:
        return float(bad_stay_prob)

    with pytest.raises(ValueError, match="finite"):
        score_surrogate_weighted(belief, item, step=0, stay_prob_fn=stay_prob_fn)

    with pytest.raises(ValueError, match="finite"):
        score_myopic_exact(belief, item, step=0, stay_prob_fn=stay_prob_fn)


def test_stay_probability_is_clipped_for_direct_scores() -> None:
    belief = make_test_belief()
    item   = Item("q", np.array([1.0, 0.0]), np.array([0.0]))

    zero_surrogate   = score_surrogate_weighted(belief, item, step=0, stay_prob_fn=lambda _i, _s: -2.0)
    zero_exact       = score_myopic_exact(      belief, item, step=0, stay_prob_fn=lambda _i, _s: -2.0)
    capped_surrogate = score_surrogate_weighted(belief, item, step=0, stay_prob_fn=lambda _i, _s: 3.0)
    capped_exact     = score_myopic_exact(      belief, item, step=0, stay_prob_fn=lambda _i, _s: 3.0)
    unit_surrogate   = score_surrogate_weighted(belief, item, step=0, stay_prob_fn=lambda _i, _s: 1.0)
    unit_exact       = score_myopic_exact(      belief, item, step=0, stay_prob_fn=lambda _i, _s: 1.0)

    assert np.isclose(zero_surrogate, 0.0)
    assert np.isclose(zero_exact,     0.0)
    assert np.isclose(capped_surrogate, unit_surrogate)
    assert np.isclose(capped_exact,     unit_exact)


def test_very_high_response_noise_drives_exact_myopic_score_down() -> None:
    belief = make_test_belief()
    informative_item = Item(
        item_id="informative",
        a=np.array([1.0, -0.4]),
        thresholds=np.array([-0.2, 0.5]),
        response_noise_variance=1.0,
    )
    noisy_item = Item(
        item_id="noisy",
        a=np.array([1.0, -0.4]),
        thresholds=np.array([-0.2, 0.5]),
        response_noise_variance=1e9,
    )

    informative_score = score_myopic_exact(
        belief=belief, item=informative_item, step=0,
        stay_prob_fn=no_dropout_stay_prob,
    )
    noisy_score = score_myopic_exact(
        belief=belief, item=noisy_item, step=0,
        stay_prob_fn=no_dropout_stay_prob,
    )

    assert noisy_score < informative_score
    assert noisy_score < 1e-6


# ---------------------------------------------------------------------------
# make_sensitive_constant_stay_prob
# ---------------------------------------------------------------------------


def _make_sensitive_item(level: float = 0.5) -> Item:
    return Item("s", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                is_sensitive=True, sensitivity_level=level)


def _make_normal_item() -> Item:
    return Item("n", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                is_sensitive=False)


class TestSensitiveConstantStayProb:
    def test_sensitive_item_gets_low_stay_prob(self) -> None:
        fn = make_sensitive_constant_stay_prob(p_stay_sensitive=0.6)
        assert fn(_make_sensitive_item(), 0) == pytest.approx(0.6)
        assert fn(_make_sensitive_item(), 5) == pytest.approx(0.6)

    def test_normal_item_gets_normal_stay_prob(self) -> None:
        fn = make_sensitive_constant_stay_prob(p_stay_sensitive=0.6, p_stay_normal=0.95)
        assert fn(_make_normal_item(), 0) == pytest.approx(0.95)

    def test_normal_item_defaults_to_no_dropout(self) -> None:
        fn = make_sensitive_constant_stay_prob(p_stay_sensitive=0.5)
        assert fn(_make_normal_item(), 3) == pytest.approx(1.0)

    def test_step_has_no_effect(self) -> None:
        fn = make_sensitive_constant_stay_prob(p_stay_sensitive=0.7)
        probs = [fn(_make_sensitive_item(), k) for k in range(10)]
        assert all(p == pytest.approx(0.7) for p in probs)

    def test_invalid_p_stay_sensitive_raises(self) -> None:
        with pytest.raises(ValueError, match="p_stay_sensitive"):
            make_sensitive_constant_stay_prob(p_stay_sensitive=1.5)
        with pytest.raises(ValueError, match="p_stay_sensitive"):
            make_sensitive_constant_stay_prob(p_stay_sensitive=-0.1)

    def test_invalid_p_stay_normal_raises(self) -> None:
        with pytest.raises(ValueError, match="p_stay_normal"):
            make_sensitive_constant_stay_prob(p_stay_sensitive=0.5, p_stay_normal=1.1)


# ---------------------------------------------------------------------------
# make_sensitive_level_stay_prob
# ---------------------------------------------------------------------------


class TestSensitiveLevelStayProb:
    def test_non_sensitive_item_gets_max_stay(self) -> None:
        fn = make_sensitive_level_stay_prob(gamma0=0.5)
        assert fn(_make_normal_item(), 0) == pytest.approx(1.0)

    def test_non_sensitive_item_gets_custom_max_stay(self) -> None:
        fn = make_sensitive_level_stay_prob(gamma0=0.5, max_stay=0.9)
        assert fn(_make_normal_item(), 0) == pytest.approx(0.9)

    def test_sensitive_item_penalty_scales_with_level(self) -> None:
        fn = make_sensitive_level_stay_prob(gamma0=0.5, gamma_step=0.0)
        low  = fn(Item("l", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                        is_sensitive=True, sensitivity_level=0.2), 0)
        high = fn(Item("h", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                        is_sensitive=True, sensitivity_level=0.8), 0)
        assert low > high

    def test_penalty_decreases_over_steps(self) -> None:
        fn   = make_sensitive_level_stay_prob(gamma0=1.0, gamma_step=0.2)
        item = _make_sensitive_item(level=0.5)
        probs = [fn(item, k) for k in range(6)]
        assert probs == sorted(probs)

    def test_gamma_step_zero_is_constant(self) -> None:
        fn   = make_sensitive_level_stay_prob(gamma0=0.4, gamma_step=0.0)
        item = _make_sensitive_item(level=0.5)
        probs = [fn(item, k) for k in range(10)]
        assert all(p == pytest.approx(probs[0]) for p in probs)

    def test_high_level_clamps_to_min_stay(self) -> None:
        fn   = make_sensitive_level_stay_prob(gamma0=10.0, gamma_step=0.0, min_stay=0.05)
        item = _make_sensitive_item(level=1.0)
        assert fn(item, 0) == pytest.approx(0.05)

    def test_formula_at_step_zero(self) -> None:
        gamma0, level = 0.4, 0.6
        fn   = make_sensitive_level_stay_prob(gamma0=gamma0, gamma_step=0.0)
        item = _make_sensitive_item(level=level)
        expected = float(np.clip(1.0 - gamma0 * level, 0.0, 1.0))
        assert fn(item, 0) == pytest.approx(expected)

    def test_invalid_gamma0_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma0"):
            make_sensitive_level_stay_prob(gamma0=-1.0)

    def test_invalid_min_max_stay_raises(self) -> None:
        with pytest.raises(ValueError):
            make_sensitive_level_stay_prob(gamma0=0.5, min_stay=0.8, max_stay=0.3)
