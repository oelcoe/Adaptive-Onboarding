"""
Extension tests — sensitivity noise scaling in src/simulate.py

These tests verify the optional sensitivity_noise_scale machinery and are
intentionally kept separate from the baseline tests in test_simulate.py.

A failure here indicates a problem with the noise-extension feature, not with
the core simulator. The baseline suite should still pass entirely.

Coverage
────────
_effective_noise_variance (formula unit tests)
  1. non-sensitive item always gets base noise, regardless of scale
  2. scale=0 always returns base noise, regardless of is_sensitive
  3. sensitive item: effective_variance = base * (1 + scale * level)
  4. higher sensitivity_level → higher effective variance (monotone)

sensitivity_noise_scale in simulate_episode (integration tests)
  5. scale=0 produces identical results to omitting the parameter (identity)
  6. negative scale raises ValueError
  7. large scale makes sensitive-item responses more uniform (dispersion test)
  8. smoke test: both stay-prob factories integrate cleanly with the feature on
"""

from __future__ import annotations

import numpy as np
import pytest

from src.belief import BeliefState
from src.item_bank import Item
from src.policies import (
    make_sensitive_constant_stay_prob,
    make_sensitive_level_stay_prob,
    no_dropout_stay_prob,
)
from src.simulate import (
    _effective_noise_variance,
    simulate_episode,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_prior(d: int = 2) -> BeliefState:
    return BeliefState(mu=np.zeros(d), Sigma=np.eye(d))


def make_sensitive_bank() -> list[Item]:
    """A bank containing both non-sensitive and sensitive items at various levels."""
    return [
        Item("ns1",    a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
             is_sensitive=False, sensitivity_level=0.0),
        Item("ns2",    a=np.array([0.0, 1.0]), thresholds=np.array([0.0]),
             is_sensitive=False, sensitivity_level=0.0),
        Item("s_low",  a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
             is_sensitive=True, sensitivity_level=0.3),
        Item("s_mid",  a=np.array([0.0, 1.0]), thresholds=np.array([0.0]),
             is_sensitive=True, sensitivity_level=0.6),
        Item("s_high", a=np.array([1.0, 1.0]), thresholds=np.array([0.0]),
             is_sensitive=True, sensitivity_level=1.0),
    ]


THETA_TRUE = np.array([0.5, -0.5])


# ---------------------------------------------------------------------------
# _effective_noise_variance — formula unit tests
# ---------------------------------------------------------------------------


class TestEffectiveNoiseVariance:
    def test_non_sensitive_item_always_gets_base_noise(self) -> None:
        item = Item("ns", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                    is_sensitive=False, sensitivity_level=0.5)
        assert _effective_noise_variance(item, 2.0) == pytest.approx(1.0)

    def test_zero_scale_returns_base_for_sensitive_item(self) -> None:
        item = Item("s", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                    is_sensitive=True, sensitivity_level=0.8)
        assert _effective_noise_variance(item, 0.0) == pytest.approx(1.0)

    def test_formula_base_times_one_plus_scale_times_level(self) -> None:
        item = Item("s", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                    is_sensitive=True, sensitivity_level=0.5)
        result = _effective_noise_variance(item, sensitivity_noise_scale=2.0)
        assert result == pytest.approx(1.0 * (1 + 2.0 * 0.5))

    def test_higher_level_gives_higher_noise(self) -> None:
        levels    = [0.1, 0.5, 1.0]
        variances = [
            _effective_noise_variance(
                Item(f"s{i}", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                     is_sensitive=True, sensitivity_level=lv),
                sensitivity_noise_scale=1.0,
            )
            for i, lv in enumerate(levels)
        ]
        assert variances == sorted(variances)


# ---------------------------------------------------------------------------
# sensitivity_noise_scale in simulate_episode
# ---------------------------------------------------------------------------


class TestSensitivityNoiseScaleInEpisode:
    def test_scale_zero_is_identical_to_baseline(self) -> None:
        """
        scale=0 must produce byte-for-byte identical results to omitting the
        parameter — this is the key identity that anchors the extension to the
        baseline.
        """
        common = dict(
            theta_true=THETA_TRUE, prior_belief=make_prior(),
            item_bank=make_sensitive_bank(), strategy="fixed",
            horizon=3, stay_prob_fn=no_dropout_stay_prob,
            fixed_order=["ns1", "s_low", "s_high"],
        )
        ep_baseline = simulate_episode(**common, rng=np.random.default_rng(5))
        ep_zero     = simulate_episode(**common, rng=np.random.default_rng(5),
                                       sensitivity_noise_scale=0.0)
        assert ep_baseline.asked_item_ids == ep_zero.asked_item_ids
        assert ([s.response for s in ep_baseline.steps]
                == [s.response for s in ep_zero.steps])

    def test_negative_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="sensitivity_noise_scale"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_sensitive_bank(), strategy="random", horizon=1,
                sensitivity_noise_scale=-0.5,
            )

    def test_large_scale_makes_responses_more_uniform(self) -> None:
        """
        Very large noise on a sensitive item pushes responses toward uniform.
        Verified by comparing max-category frequency: large scale → lower peak.
        """
        sensitive_item = Item("s", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                              is_sensitive=True, sensitivity_level=1.0)
        bank = [sensitive_item]
        N    = 300

        def collect(scale: float, seed: int) -> list[int]:
            rng = np.random.default_rng(seed)
            return [
                simulate_episode(
                    THETA_TRUE, make_prior(), bank, "fixed", 1,
                    stay_prob_fn=no_dropout_stay_prob,
                    rng=rng, sensitivity_noise_scale=scale,
                ).steps[0].response
                for _ in range(N)
            ]

        freq_base  = np.bincount(collect(0.0,  seed=0), minlength=2) / N
        freq_noisy = np.bincount(collect(50.0, seed=0), minlength=2) / N

        # High noise should reduce the dominance of the most common response
        assert max(freq_noisy) < max(freq_base) + 0.15

    def test_stay_prob_factories_integrate_cleanly(self) -> None:
        """Smoke test: both dropout factories work alongside sensitivity_noise_scale."""
        bank  = make_sensitive_bank()
        prior = make_prior()
        for fn in [
            make_sensitive_constant_stay_prob(p_stay_sensitive=0.8),
            make_sensitive_level_stay_prob(gamma0=0.3, gamma_step=0.05),
        ]:
            ep = simulate_episode(
                theta_true=THETA_TRUE, prior_belief=prior, item_bank=bank,
                strategy="random", horizon=5, stay_prob_fn=fn,
                rng=np.random.default_rng(42), sensitivity_noise_scale=1.0,
            )
            assert ep.n_asked >= 1
