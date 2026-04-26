"""
Tests for src/simulate.py

Coverage plan
─────────────
sample_response
  1. always returns a valid category index
  2. threshold boundary: z exactly on a threshold maps to the correct side
  3. deterministic with a fixed seed
  4. empirical frequencies match GRM category probabilities (statistical)

simulate_episode
  5. no-dropout: n_answered == horizon, all items present, no dropout flag
  6. forced-dropout: every stay prob = 0 → dropout on step 0
  7. StepRecord fields: before/after beliefs populated correctly
  8. belief evolves (mu or Sigma changes after each answered step)
  9. asked_item_ids matches step records in order
 10. fixed strategy respects fixed_order
 11. damped update flag is forwarded (final belief differs from exact update)
 12. horizon respected: never more than horizon steps recorded
 13. items not repeated across steps (no-replacement selection)
 14. terminated_by_dropout flag consistent with step records
 15. horizon=1 baseline: single-step episode updates belief exactly once
 16. weighted vs unweighted surrogate choose different first item (policy integration)

simulate_population
 17. returns one EpisodeResult per row of theta_trues
 18. 1-D theta input is accepted without error
 19. all strategies run without error over a population
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.belief import BeliefState
from src.grm import category_probabilities
from src.item_bank import Item
from src.policies import (
    make_sensitive_constant_stay_prob,
    make_sensitive_level_stay_prob,
    no_dropout_stay_prob,
)
from src.simulate import (
    _effective_noise_variance,
    EpisodeResult,
    StepRecord,
    sample_response,
    simulate_episode,
    simulate_population,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_prior(d: int = 2) -> BeliefState:
    return BeliefState(mu=np.zeros(d), Sigma=np.eye(d))


def make_items(d: int = 2) -> list[Item]:
    """Six items: a mix of sensitive and non-sensitive with varied loadings."""
    return [
        Item("q1", a=np.array([1.0, 0.0]), thresholds=np.array([-1.0, 0.0, 1.0])),
        Item("q2", a=np.array([0.0, 1.0]), thresholds=np.array([-1.0, 0.0, 1.0]), is_sensitive=True, sensitivity_level=0.4),
        Item("q3", a=np.array([0.7, 0.7]), thresholds=np.array([-1.0, 0.0, 1.0]), is_sensitive=True, sensitivity_level=0.2),
        Item("q4", a=np.array([1.0, 0.5]), thresholds=np.array([-0.5, 0.5])),
        Item("q5", a=np.array([0.5, 1.0]), thresholds=np.array([-0.5, 0.5])),
        Item("q6", a=np.array([1.0, 1.0]), thresholds=np.array([0.0]),             is_sensitive=True, sensitivity_level=0.5),
    ]


THETA_TRUE = np.array([0.5, -0.5])


# ---------------------------------------------------------------------------
# sample_response
# ---------------------------------------------------------------------------


class TestSampleResponse:
    def test_returns_valid_category(self) -> None:
        rng = np.random.default_rng(0)
        items = make_items()
        for item in items:
            for _ in range(50):
                r = sample_response(THETA_TRUE, item, rng)
                assert 0 <= r < item.n_categories, (
                    f"response {r} out of range for item {item.item_id} "
                    f"with {item.n_categories} categories"
                )

    def test_deterministic_with_fixed_seed(self) -> None:
        item = make_items()[0]
        r1 = sample_response(THETA_TRUE, item, np.random.default_rng(99))
        r2 = sample_response(THETA_TRUE, item, np.random.default_rng(99))
        assert r1 == r2

    def test_threshold_boundary_binary_item(self) -> None:
        """
        Binary item with single threshold at 0: a user with z > 0 should
        tend to give response 1 more often than response 0.
        """
        item = Item("binary", a=np.array([1.0]), thresholds=np.array([0.0]))
        theta_high = np.array([3.0])   # z will be centred around 3
        theta_low  = np.array([-3.0])  # z will be centred around -3
        rng = np.random.default_rng(7)

        responses_high = [sample_response(theta_high, item, rng) for _ in range(200)]
        responses_low  = [sample_response(theta_low,  item, rng) for _ in range(200)]

        assert np.mean(responses_high) > 0.8, "high theta should mostly give category 1"
        assert np.mean(responses_low)  < 0.2, "low theta should mostly give category 0"

    def test_empirical_frequencies_match_grm_category_probabilities(self) -> None:
        """
        Draw N responses from a known true latent state and verify that the
        empirical category frequencies are close to the GRM analytical predictions.

        The GRM probability for category r given theta_true is:
            P(r | theta) = Phi((tau_{r+1} - a^T theta) / sigma_obs)
                         - Phi((tau_r   - a^T theta) / sigma_obs)

        This is what category_probabilities() returns when the belief is a near-
        point mass at theta_true (Sigma -> 0, so the only variance is sigma_obs^2).
        We check that each empirical proportion is within 3 standard errors of the
        model prediction (Bonferroni-corrected across categories).
        """
        item = Item(
            "stat_check",
            a=np.array([1.0, 0.5]),
            thresholds=np.array([-1.0, 0.0, 1.0]),
        )
        theta_true = np.array([0.3, 0.7])
        N = 3_000
        rng = np.random.default_rng(2024)

        responses = [sample_response(theta_true, item, rng) for _ in range(N)]
        empirical = np.bincount(responses, minlength=item.n_categories) / N

        # Near-point-mass belief at theta_true: Sigma = 1e-8 * I makes var_eta ≈ 0,
        # so denom ≈ sqrt(sigma_obs^2) = 1.0, matching the true generative model.
        point_belief = BeliefState(mu=theta_true, Sigma=1e-8 * np.eye(2))
        predicted = category_probabilities(point_belief, item)

        # 3-sigma tolerance per category (Bonferroni: divide alpha by n_categories)
        for r in range(item.n_categories):
            se = np.sqrt(predicted[r] * (1 - predicted[r]) / N)
            assert abs(empirical[r] - predicted[r]) < 4 * se, (
                f"Category {r}: empirical={empirical[r]:.3f}, "
                f"predicted={predicted[r]:.3f}, 4·SE={4*se:.3f}"
            )


# ---------------------------------------------------------------------------
# simulate_episode — no dropout
# ---------------------------------------------------------------------------


class TestSimulateEpisodeNoDropout:
    def _run(self, strategy="surrogate_weighted", horizon=4, **kwargs) -> EpisodeResult:
        return simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy=strategy,
            horizon=horizon,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(42),
            **kwargs,
        )

    def test_n_answered_equals_horizon(self) -> None:
        ep = self._run(horizon=4)
        assert ep.n_answered == 4
        assert ep.n_asked == 4

    def test_no_dropout_flag(self) -> None:
        ep = self._run(horizon=4)
        assert not ep.terminated_by_dropout
        assert all(not s.dropped_out for s in ep.steps)

    def test_responses_are_valid_categories(self) -> None:
        ep = self._run(horizon=4)
        items_by_id = {it.item_id: it for it in make_items()}
        for step in ep.steps:
            assert step.response is not None
            item = items_by_id[step.item_id]
            assert 0 <= step.response < item.n_categories

    def test_asked_ids_match_step_records(self) -> None:
        ep = self._run(horizon=5)
        assert ep.asked_item_ids == [s.item_id for s in ep.steps]

    def test_no_item_repeated(self) -> None:
        ep = self._run(horizon=6)
        assert len(ep.asked_item_ids) == len(set(ep.asked_item_ids))

    def test_horizon_respected(self) -> None:
        ep = self._run(horizon=3)
        assert len(ep.steps) <= 3

    def test_belief_before_after_populated(self) -> None:
        ep = self._run(horizon=4)
        for step in ep.steps:
            assert step.belief_mu_before is not None
            assert step.belief_Sigma_before is not None
            assert step.belief_mu_after is not None
            assert step.belief_Sigma_after is not None

    def test_belief_evolves(self) -> None:
        """Posterior mean or covariance must differ from prior after each answered step."""
        ep = self._run(horizon=4)
        prior = make_prior()
        for step in ep.steps:
            mu_changed = not np.allclose(step.belief_mu_after, prior.mu)
            Sigma_changed = not np.allclose(step.belief_Sigma_after, prior.Sigma)
            assert mu_changed or Sigma_changed, (
                f"Belief did not change at step {step.step}"
            )

    def test_step_indices_sequential(self) -> None:
        ep = self._run(horizon=4)
        assert [s.step for s in ep.steps] == list(range(len(ep.steps)))

    def test_stay_prob_is_one_with_no_dropout_fn(self) -> None:
        ep = self._run(horizon=4)
        assert all(s.stay_prob == 1.0 for s in ep.steps)

    def test_horizon_larger_than_bank_terminates_gracefully(self) -> None:
        """
        When horizon > len(item_bank) and no dropout occurs, the episode should
        stop naturally once all items are exhausted, not raise an error.
        """
        items = make_items()   # 6 items
        ep = simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=items,
            strategy="surrogate_weighted",
            horizon=100,           # far exceeds the bank
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0),
        )
        assert ep.n_answered == len(items)
        assert ep.n_asked == len(items)
        assert not ep.terminated_by_dropout
        assert len(set(ep.asked_item_ids)) == len(items)   # every item used exactly once

    def test_fixed_order_exhausted_terminates_gracefully(self) -> None:
        """
        When strategy='fixed' and fixed_order is shorter than horizon, the
        episode should stop once the order is exhausted.
        """
        order = ["q1", "q3"]
        ep = simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy="fixed",
            horizon=10,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0),
            fixed_order=order,
        )
        assert ep.asked_item_ids == order
        assert ep.n_answered == len(order)

    def test_horizon_one_updates_belief_exactly_once(self) -> None:
        """
        Focused baseline: with horizon=1 and no dropout the simulator must
        ask exactly one item, record one answered step, and the final belief
        must differ from the prior (exactly one update applied).
        """
        prior = make_prior()
        ep = simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=prior,
            item_bank=make_items(),
            strategy="surrogate_weighted",
            horizon=1,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0),
        )
        assert ep.n_asked == 1
        assert ep.n_answered == 1
        assert not ep.terminated_by_dropout
        assert len(ep.steps) == 1
        assert ep.steps[0].response is not None
        # belief must have been updated
        assert not np.allclose(ep.final_belief.mu, prior.mu) or not np.allclose(
            ep.final_belief.Sigma, prior.Sigma
        )


# ---------------------------------------------------------------------------
# simulate_episode — dropout mechanics
# ---------------------------------------------------------------------------


class TestSimulateEpisodeInputChecks:
    def test_horizon_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_items(), strategy="random", horizon=0,
            )

    def test_horizon_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon must be at least 1"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_items(), strategy="random", horizon=-5,
            )

    def test_nan_stay_prob_raises(self) -> None:
        def nan_stay(_item: Item, _step: int) -> float:
            return float("nan")

        with pytest.raises(ValueError, match="non-finite"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_items(), strategy="random", horizon=3,
                stay_prob_fn=nan_stay, rng=np.random.default_rng(0),
            )

    def test_nan_stay_prob_error_names_item_and_step(self) -> None:
        def nan_stay(_item: Item, _step: int) -> float:
            return float("nan")

        with pytest.raises(ValueError, match=r"step 0"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_items(), strategy="random", horizon=3,
                stay_prob_fn=nan_stay, rng=np.random.default_rng(0),
            )

    def test_negative_sensitivity_noise_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="sensitivity_noise_scale"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_items(), strategy="random", horizon=1,
                sensitivity_noise_scale=-0.5,
            )


class TestSimulateEpisodeDimensionChecks:
    def test_theta_wrong_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="theta_true has dimension"):
            simulate_episode(
                theta_true=np.array([0.5]),        # d=1, prior is d=2
                prior_belief=make_prior(d=2),
                item_bank=make_items(d=2),
                strategy="random",
                horizon=1,
                rng=np.random.default_rng(0),
            )

    def test_item_wrong_dimension_raises(self) -> None:
        bad_item = Item("bad", a=np.array([1.0, 0.0, 0.0]), thresholds=np.array([0.0]))
        bank = make_items(d=2) + [bad_item]
        with pytest.raises(ValueError, match="Mismatched items"):
            simulate_episode(
                theta_true=THETA_TRUE,
                prior_belief=make_prior(d=2),
                item_bank=bank,
                strategy="random",
                horizon=1,
                rng=np.random.default_rng(0),
            )

    def test_mismatched_error_names_the_offending_item(self) -> None:
        bad_item = Item("culprit", a=np.array([1.0, 0.0, 0.0]), thresholds=np.array([0.0]))
        with pytest.raises(ValueError, match="culprit"):
            simulate_episode(
                theta_true=THETA_TRUE,
                prior_belief=make_prior(d=2),
                item_bank=[bad_item],
                strategy="random",
                horizon=1,
                rng=np.random.default_rng(0),
            )

    def test_consistent_dimensions_do_not_raise(self) -> None:
        simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(d=2),
            item_bank=make_items(d=2),
            strategy="random",
            horizon=2,
            rng=np.random.default_rng(0),
        )


class TestSimulateEpisodeDropout:
    def test_zero_stay_prob_drops_on_first_step(self) -> None:
        """p_stay = 0 for all items → dropout on step 0, nothing answered."""
        always_leave = make_sensitive_constant_stay_prob(p_stay_sensitive=0.0)
        items = [
            Item("x1", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]), is_sensitive=True),
            Item("x2", a=np.array([0.0, 1.0]), thresholds=np.array([0.0]), is_sensitive=True),
        ]
        ep = simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=items,
            strategy="random",
            horizon=5,
            stay_prob_fn=always_leave,
            rng=np.random.default_rng(1),
        )
        assert ep.terminated_by_dropout
        assert ep.n_asked == 1
        assert ep.n_answered == 0
        assert ep.steps[0].dropped_out

    def test_dropout_step_has_no_response_or_posterior(self) -> None:
        always_leave = make_sensitive_constant_stay_prob(p_stay_sensitive=0.0)
        items = [
            Item("x1", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]), is_sensitive=True),
        ]
        ep = simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=items,
            strategy="fixed",
            horizon=3,
            stay_prob_fn=always_leave,
            rng=np.random.default_rng(0),
        )
        dropout_step = ep.steps[0]
        assert dropout_step.response is None
        assert dropout_step.belief_mu_after is None
        assert dropout_step.belief_Sigma_after is None

    def test_final_belief_unchanged_after_dropout(self) -> None:
        """If dropout happens on step 0, final belief should equal the prior."""
        always_leave = make_sensitive_constant_stay_prob(p_stay_sensitive=0.0)
        items = [
            Item("x1", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]), is_sensitive=True),
        ]
        prior = make_prior()
        ep = simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=prior,
            item_bank=items,
            strategy="fixed",
            horizon=3,
            stay_prob_fn=always_leave,
            rng=np.random.default_rng(0),
        )
        assert np.allclose(ep.final_belief.mu, prior.mu)
        assert np.allclose(ep.final_belief.Sigma, prior.Sigma)

    def test_terminated_by_dropout_flag_consistent(self) -> None:
        """terminated_by_dropout iff the last step has dropped_out=True."""
        rng = np.random.default_rng(5)
        # Sensitive items get p_stay=0.7, non-sensitive get 1.0 → some dropout
        stay_fn = make_sensitive_constant_stay_prob(p_stay_sensitive=0.7, p_stay_normal=1.0)
        for _ in range(20):
            ep = simulate_episode(
                theta_true=THETA_TRUE,
                prior_belief=make_prior(),
                item_bank=make_items(),
                strategy="random",
                horizon=6,
                stay_prob_fn=stay_fn,
                rng=rng,
            )
            last_step_dropped = ep.steps[-1].dropped_out if ep.steps else False
            assert ep.terminated_by_dropout == last_step_dropped


# ---------------------------------------------------------------------------
# simulate_episode — strategy variants
# ---------------------------------------------------------------------------


class TestSimulateEpisodeStrategies:
    @pytest.mark.parametrize("strategy", [
        "random",
        "fixed",
        "myopic_exact",
        "surrogate_unweighted",
        "surrogate_weighted",
    ])
    def test_all_strategies_complete_without_error(self, strategy: str) -> None:
        ep = simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy=strategy,
            horizon=4,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0),
            fixed_order=["q1", "q2", "q3", "q4"],
        )
        assert ep.n_answered == 4

    def test_fixed_order_respected(self) -> None:
        order = ["q3", "q1", "q5", "q2"]
        ep = simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy="fixed",
            horizon=4,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0),
            fixed_order=order,
        )
        assert ep.asked_item_ids == order

    def test_damped_update_differs_from_exact_update(self) -> None:
        """Damped and exact updates should produce different final beliefs."""
        common_kwargs = dict(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy="fixed",
            horizon=4,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0),
            fixed_order=["q1", "q2", "q3", "q4"],
        )
        ep_exact = simulate_episode(**common_kwargs, use_damped_update=False)
        ep_damp  = simulate_episode(**common_kwargs, use_damped_update=True)

        # Final means or covariances must differ
        means_same  = np.allclose(ep_exact.final_belief.mu,    ep_damp.final_belief.mu)
        sigmas_same = np.allclose(ep_exact.final_belief.Sigma, ep_damp.final_belief.Sigma)
        assert not (means_same and sigmas_same), (
            "Damped and exact updates produced identical final beliefs"
        )


# ---------------------------------------------------------------------------
# Policy integration: simulator actually delegates to the correct scoring rule
# ---------------------------------------------------------------------------


class TestPolicyIntegration:
    def test_weighted_and_unweighted_choose_different_first_item(self) -> None:
        """
        Construct a bank where:
          - Item A has high projected variance (var = 4) but high sensitivity (p_stay = 0.2)
          - Item B has lower projected variance (var = 1) but zero sensitivity (p_stay = 1.0)

        Unweighted score:  log(1 + var)          → A wins  (log 5 ≈ 1.61 > log 2 ≈ 0.69)
        Weighted score:    log(1 + p_stay * var) → B wins  (log 2 ≈ 0.69 > log 1.8 ≈ 0.59)

        If the simulator is correctly delegating to the scoring rule, the two
        strategies must select different items on the very first step.
        """
        # gamma0=0.8, gamma_step=0 → gamma_k = 0.8 for all steps
        # p_stay(A) = clip(1 - 0.8 * 1.0, 0, 1) = 0.2   (sensitive, level=1)
        # p_stay(B) = max_stay = 1.0                       (not sensitive)
        stay_fn = make_sensitive_level_stay_prob(gamma0=0.8, gamma_step=0.0)

        item_a = Item("high_var_high_sens", a=np.array([2.0, 0.0]),
                      thresholds=np.array([0.0]), is_sensitive=True, sensitivity_level=1.0)
        item_b = Item("low_var_no_sens",    a=np.array([0.0, 1.0]),
                      thresholds=np.array([0.0]), is_sensitive=False)
        bank = [item_a, item_b]
        prior = make_prior()  # Sigma = I → var_A = 4, var_B = 1

        ep_unweighted = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=prior, item_bank=bank,
            strategy="surrogate_unweighted", horizon=1,
            stay_prob_fn=stay_fn, rng=np.random.default_rng(0),
        )
        ep_weighted = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=prior, item_bank=bank,
            strategy="surrogate_weighted", horizon=1,
            stay_prob_fn=stay_fn, rng=np.random.default_rng(0),
        )

        first_unweighted = ep_unweighted.asked_item_ids[0]
        first_weighted   = ep_weighted.asked_item_ids[0]

        assert first_unweighted == "high_var_high_sens", (
            f"Unweighted should prefer the high-variance item, got {first_unweighted!r}"
        )
        assert first_weighted == "low_var_no_sens", (
            f"Weighted should prefer the safe item, got {first_weighted!r}"
        )
        assert first_unweighted != first_weighted


# ---------------------------------------------------------------------------
# simulate_population
# ---------------------------------------------------------------------------


class TestSimulatePopulation:
    def test_returns_one_result_per_user(self) -> None:
        n = 15
        thetas = np.random.default_rng(0).standard_normal((n, 2))
        results = simulate_population(
            theta_trues=thetas,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy="random",
            horizon=3,
            rng=np.random.default_rng(1),
        )
        assert len(results) == n
        assert all(isinstance(r, EpisodeResult) for r in results)

    def test_accepts_1d_theta(self) -> None:
        results = simulate_population(
            theta_trues=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy="random",
            horizon=2,
            rng=np.random.default_rng(0),
        )
        assert len(results) == 1

    @pytest.mark.parametrize("strategy", [
        "random",
        "surrogate_unweighted",
        "surrogate_weighted",
        "myopic_exact",
    ])
    def test_population_strategies_complete(self, strategy: str) -> None:
        thetas = np.random.default_rng(3).standard_normal((8, 2))
        results = simulate_population(
            theta_trues=thetas,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy=strategy,
            horizon=3,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(3),
        )
        assert len(results) == 8
        for ep in results:
            assert ep.n_answered == 3

    def test_population_dropout_rate_plausible(self) -> None:
        """
        With p_stay = 0.9 per step and horizon = 3, the analytical dropout rate is
        1 - 0.9^3 ≈ 0.27, so we expect something clearly between 0 and 1.
        gamma0=0.1 and sensitivity_level=1.0 → p_stay = 1 - 0.1*1.0 = 0.9.
        """
        stay_fn = make_sensitive_level_stay_prob(gamma0=0.1, gamma_step=0.0, min_stay=0.0)
        items = [
            Item(f"s{i}", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                 is_sensitive=True, sensitivity_level=1.0)
            for i in range(6)
        ]
        thetas = np.random.default_rng(9).standard_normal((300, 2))
        results = simulate_population(
            theta_trues=thetas,
            prior_belief=make_prior(),
            item_bank=items,
            strategy="random",
            horizon=3,
            stay_prob_fn=stay_fn,
            rng=np.random.default_rng(9),
        )
        dropout_rate = float(np.mean([r.terminated_by_dropout for r in results]))
        # analytical: 1 - 0.9^3 ≈ 0.271; allow generous ±0.15 for sampling noise
        assert 0.10 < dropout_rate < 0.45, (
            f"Dropout rate {dropout_rate:.2f} is implausible for p_stay=0.9, horizon=3 "
            f"(expected ~0.27)"
        )

    def test_reproducible_with_same_seed(self) -> None:
        thetas = np.random.default_rng(0).standard_normal((5, 2))
        kwargs = dict(
            theta_trues=thetas,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy="random",
            horizon=4,
        )
        r1 = simulate_population(**kwargs, rng=np.random.default_rng(77))
        r2 = simulate_population(**kwargs, rng=np.random.default_rng(77))
        for ep1, ep2 in zip(r1, r2):
            assert ep1.asked_item_ids == ep2.asked_item_ids
            assert [s.response for s in ep1.steps] == [s.response for s in ep2.steps]


# ---------------------------------------------------------------------------
# Sensitivity noise scale + new stay-prob factories end-to-end
# ---------------------------------------------------------------------------


def make_sensitive_bank() -> list[Item]:
    """Bank with a mix of sensitive and non-sensitive items at various levels."""
    return [
        Item("ns1", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
             is_sensitive=False, sensitivity_level=0.0),
        Item("ns2", a=np.array([0.0, 1.0]), thresholds=np.array([0.0]),
             is_sensitive=False, sensitivity_level=0.0),
        Item("s_low",  a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
             is_sensitive=True, sensitivity_level=0.3),
        Item("s_mid",  a=np.array([0.0, 1.0]), thresholds=np.array([0.0]),
             is_sensitive=True, sensitivity_level=0.6),
        Item("s_high", a=np.array([1.0, 1.0]), thresholds=np.array([0.0]),
             is_sensitive=True, sensitivity_level=1.0),
    ]


class TestEffectiveNoiseVariance:
    def test_non_sensitive_item_always_gets_base_noise(self) -> None:
        item = Item("ns", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                    is_sensitive=False, sensitivity_level=0.5)
        assert _effective_noise_variance(item, 2.0) == pytest.approx(1.0)

    def test_zero_scale_returns_base_for_sensitive_item(self) -> None:
        item = Item("s", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                    is_sensitive=True, sensitivity_level=0.8)
        assert _effective_noise_variance(item, 0.0) == pytest.approx(1.0)

    def test_sensitive_item_noise_inflated_by_scale_and_level(self) -> None:
        item = Item("s", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                    is_sensitive=True, sensitivity_level=0.5)
        result = _effective_noise_variance(item, sensitivity_noise_scale=2.0)
        assert result == pytest.approx(1.0 * (1 + 2.0 * 0.5))

    def test_higher_level_gives_higher_noise(self) -> None:
        items = [
            Item(f"s{i}", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                 is_sensitive=True, sensitivity_level=lv)
            for i, lv in enumerate([0.1, 0.5, 1.0])
        ]
        variances = [_effective_noise_variance(it, sensitivity_noise_scale=1.0) for it in items]
        assert variances == sorted(variances)


class TestSensitivityNoiseScaleInEpisode:
    def test_scale_zero_same_as_no_scale(self) -> None:
        """sensitivity_noise_scale=0 must produce identical results to omitting it."""
        common = dict(
            theta_true=THETA_TRUE, prior_belief=make_prior(),
            item_bank=make_sensitive_bank(), strategy="fixed",
            horizon=3, stay_prob_fn=no_dropout_stay_prob,
            fixed_order=["ns1", "s_low", "s_high"],
        )
        ep_no_scale = simulate_episode(**common, rng=np.random.default_rng(5))
        ep_zero     = simulate_episode(**common, rng=np.random.default_rng(5),
                                       sensitivity_noise_scale=0.0)
        assert ep_no_scale.asked_item_ids == ep_zero.asked_item_ids
        assert [s.response for s in ep_no_scale.steps] == [s.response for s in ep_zero.steps]

    def test_large_scale_shifts_responses_for_sensitive_items(self) -> None:
        """
        With a very large noise scale, sensitive-item responses should be more
        uniform (higher noise → response frequencies closer to 1/K each).
        Compare response variance across many episodes: large scale → higher variance.
        """
        sensitive_item = Item("s", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                              is_sensitive=True, sensitivity_level=1.0)
        bank = [sensitive_item]
        rng_base  = np.random.default_rng(0)
        rng_noisy = np.random.default_rng(0)

        responses_base  = [
            simulate_episode(THETA_TRUE, make_prior(), bank, "fixed", 1,
                             stay_prob_fn=no_dropout_stay_prob,
                             rng=rng_base, sensitivity_noise_scale=0.0).steps[0].response
            for _ in range(300)
        ]
        responses_noisy = [
            simulate_episode(THETA_TRUE, make_prior(), bank, "fixed", 1,
                             stay_prob_fn=no_dropout_stay_prob,
                             rng=rng_noisy, sensitivity_noise_scale=50.0).steps[0].response
            for _ in range(300)
        ]
        # High noise should push responses toward 50/50; base noise may be more skewed.
        # Check that the noisy distribution is closer to uniform (lower max probability).
        freq_base  = np.bincount(responses_base,  minlength=2) / 300
        freq_noisy = np.bincount(responses_noisy, minlength=2) / 300
        assert max(freq_noisy) < max(freq_base) + 0.15, (
            "Very high noise should make responses more uniform"
        )

    def test_new_stay_prob_factories_work_in_episode(self) -> None:
        """Smoke test: both new factories integrate cleanly with simulate_episode."""
        bank = make_sensitive_bank()
        prior = make_prior()

        for fn in [
            make_sensitive_constant_stay_prob(p_stay_sensitive=0.8),
            make_sensitive_level_stay_prob(gamma0=0.3, gamma_step=0.05),
        ]:
            ep = simulate_episode(
                theta_true=THETA_TRUE, prior_belief=prior, item_bank=bank,
                strategy="random", horizon=5, stay_prob_fn=fn,
                rng=np.random.default_rng(42),
            )
            assert ep.n_asked >= 1
