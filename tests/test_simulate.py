"""
Tests for src/simulate.py — baseline behaviour

Coverage
────────
sample_response
  1. always returns a valid category index
  2. deterministic with a fixed seed
  3. threshold direction: high theta → high category more often
  4. empirical frequencies match GRM category probabilities (statistical)

simulate_episode — core rollout
  5.  no-dropout: n_answered == horizon, no dropout flag
  6.  responses are valid category indices
  7.  asked_item_ids matches step records in order
  8.  no item is repeated within one episode
  9.  fixed order is respected exactly
  10. horizon is never exceeded
  11. StepRecord before/after beliefs are populated
  12. belief evolves at every answered step
  13. step indices are sequential
  14. stay_prob is 1.0 for every step under no_dropout_stay_prob

simulate_episode — graceful termination
  15. horizon > bank size stops at bank exhaustion, not an error
  16. fixed_order shorter than horizon stops at order exhaustion

simulate_episode — single-step baseline
  17. horizon=1 records exactly one answered step and updates belief

simulate_episode — input guards
  18. horizon=0 raises
  19. horizon<0 raises
  20. NaN stay_prob raises with item id and step in the message

simulate_episode — dimension guards
  21. wrong theta dimension raises
  22. wrong item dimension raises with offending item id
  23. correct dimensions do not raise

simulate_episode — dropout mechanics
  24. p_stay=0 → dropout on step 0, nothing answered
  25. dropout step has no response or posterior fields
  26. final belief is unchanged after step-0 dropout
  27. terminated_by_dropout flag equals last-step dropped_out

simulate_episode — strategy variants
  28. all five strategy names complete without error
  29. damped update produces a different final belief than exact update

policy integration
  30. weighted and unweighted surrogate choose different first items

simulate_population
  31. returns one EpisodeResult per user
  32. accepts a 1-D theta without error
  33. all strategies complete for a population
  34. empirical dropout rate is plausible for known p_stay
  35. same seed → identical results
"""

from __future__ import annotations

import numpy as np
import pytest

from src.belief import BeliefState
from src.grm import category_probabilities
from src.item_bank import Item
from src.policies import (
    make_sensitive_constant_stay_prob,
    make_sensitive_level_stay_prob,
    no_dropout_stay_prob,
)
from src.simulate import (
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
    """
    Six items with varied loadings. Some are flagged sensitive so that dropout
    tests using make_sensitive_constant_stay_prob can produce variable p_stay.
    No test in this file passes sensitivity_noise_scale > 0.
    """
    return [
        Item("q1", a=np.array([1.0, 0.0]), thresholds=np.array([-1.0, 0.0, 1.0])),
        Item("q2", a=np.array([0.0, 1.0]), thresholds=np.array([-1.0, 0.0, 1.0]), is_sensitive=True, sensitivity_level=0.4),
        Item("q3", a=np.array([0.7, 0.7]), thresholds=np.array([-1.0, 0.0, 1.0]), is_sensitive=True, sensitivity_level=0.2),
        Item("q4", a=np.array([1.0, 0.5]), thresholds=np.array([-0.5, 0.5])),
        Item("q5", a=np.array([0.5, 1.0]), thresholds=np.array([-0.5, 0.5])),
        Item("q6", a=np.array([1.0, 1.0]), thresholds=np.array([0.0]),            is_sensitive=True, sensitivity_level=0.5),
    ]


THETA_TRUE = np.array([0.5, -0.5])


# ---------------------------------------------------------------------------
# sample_response
# ---------------------------------------------------------------------------


class TestSampleResponse:
    def test_returns_valid_category(self) -> None:
        rng = np.random.default_rng(0)
        for item in make_items():
            for _ in range(50):
                r = sample_response(THETA_TRUE, item, rng)
                assert 0 <= r < item.n_categories

    def test_deterministic_with_fixed_seed(self) -> None:
        item = make_items()[0]
        r1 = sample_response(THETA_TRUE, item, np.random.default_rng(99))
        r2 = sample_response(THETA_TRUE, item, np.random.default_rng(99))
        assert r1 == r2

    def test_threshold_direction(self) -> None:
        """High theta → high category more often than low theta."""
        item  = Item("binary", a=np.array([1.0]), thresholds=np.array([0.0]))
        rng   = np.random.default_rng(7)
        high  = [sample_response(np.array([ 3.0]), item, rng) for _ in range(200)]
        low   = [sample_response(np.array([-3.0]), item, rng) for _ in range(200)]
        assert np.mean(high) > 0.8
        assert np.mean(low)  < 0.2

    def test_wrong_dimension_raises(self) -> None:
        item        = Item("q", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]))  # dim=2
        theta_wrong = np.array([0.5])  # dim=1
        with pytest.raises(ValueError, match="dimension"):
            sample_response(theta_wrong, item, np.random.default_rng(0))

    def test_empirical_frequencies_match_grm_category_probabilities(self) -> None:
        """
        Empirical category frequencies from many draws must be within 4 SE of the
        GRM analytical prediction (near-point-mass belief at theta_true).
        """
        item       = Item("stat_check", a=np.array([1.0, 0.5]), thresholds=np.array([-1.0, 0.0, 1.0]))
        theta_true = np.array([0.3, 0.7])
        N          = 3_000
        rng        = np.random.default_rng(2024)

        responses = [sample_response(theta_true, item, rng) for _ in range(N)]
        empirical = np.bincount(responses, minlength=item.n_categories) / N

        point_belief = BeliefState(mu=theta_true, Sigma=1e-8 * np.eye(2))
        predicted    = category_probabilities(point_belief, item)

        for r in range(item.n_categories):
            se = np.sqrt(predicted[r] * (1 - predicted[r]) / N)
            assert abs(empirical[r] - predicted[r]) < 4 * se


# ---------------------------------------------------------------------------
# simulate_episode — core rollout (no dropout)
# ---------------------------------------------------------------------------


class TestSimulateEpisodeNoDropout:
    def _run(self, strategy: str = "surrogate_weighted", horizon: int = 4, **kw) -> EpisodeResult:
        return simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy=strategy,
            horizon=horizon,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(42),
            **kw,
        )

    def test_n_answered_equals_horizon(self) -> None:
        ep = self._run(horizon=4)
        assert ep.n_answered == 4
        assert ep.n_asked   == 4

    def test_no_dropout_flag(self) -> None:
        ep = self._run(horizon=4)
        assert not ep.terminated_by_dropout
        assert all(not s.dropped_out for s in ep.steps)

    def test_responses_are_valid_categories(self) -> None:
        ep       = self._run(horizon=4)
        by_id    = {it.item_id: it for it in make_items()}
        for step in ep.steps:
            assert step.response is not None
            assert 0 <= step.response < by_id[step.item_id].n_categories

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
        for step in self._run(horizon=4).steps:
            assert step.belief_mu_before    is not None
            assert step.belief_Sigma_before is not None
            assert step.belief_mu_after     is not None
            assert step.belief_Sigma_after  is not None

    def test_belief_evolves(self) -> None:
        """Each answered step must change the belief relative to its own before-snapshot."""
        for step in self._run(horizon=4).steps:
            changed = (
                not np.allclose(step.belief_mu_after,    step.belief_mu_before)
                or not np.allclose(step.belief_Sigma_after, step.belief_Sigma_before)
            )
            assert changed, f"Belief did not change at step {step.step}"

    def test_step_indices_sequential(self) -> None:
        ep = self._run(horizon=4)
        assert [s.step for s in ep.steps] == list(range(len(ep.steps)))

    def test_stay_prob_is_one_with_no_dropout_fn(self) -> None:
        assert all(s.stay_prob == 1.0 for s in self._run(horizon=4).steps)

    @pytest.mark.parametrize("strategy", ["random", "surrogate_weighted"])
    def test_horizon_larger_than_bank_terminates_gracefully(self, strategy: str) -> None:
        """When horizon > bank size the episode stops at bank exhaustion, not an error."""
        items = make_items()
        ep = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=make_prior(), item_bank=items,
            strategy=strategy, horizon=100,
            stay_prob_fn=no_dropout_stay_prob, rng=np.random.default_rng(0),
        )
        assert ep.n_answered == len(items)
        assert not ep.terminated_by_dropout
        assert len(set(ep.asked_item_ids)) == len(items)

    def test_fixed_order_exhausted_terminates_gracefully(self) -> None:
        order = ["q1", "q3"]
        ep = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=make_prior(), item_bank=make_items(),
            strategy="fixed", horizon=10, stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0), fixed_order=order,
        )
        assert ep.asked_item_ids == order
        assert ep.n_answered == len(order)

    def test_horizon_one_updates_belief_exactly_once(self) -> None:
        prior = make_prior()
        ep = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=prior, item_bank=make_items(),
            strategy="surrogate_weighted", horizon=1,
            stay_prob_fn=no_dropout_stay_prob, rng=np.random.default_rng(0),
        )
        assert ep.n_asked    == 1
        assert ep.n_answered == 1
        assert len(ep.steps) == 1
        assert ep.steps[0].response is not None
        assert (
            not np.allclose(ep.final_belief.mu,    prior.mu)
            or not np.allclose(ep.final_belief.Sigma, prior.Sigma)
        )


# ---------------------------------------------------------------------------
# simulate_episode — input guards
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
        with pytest.raises(ValueError, match="non-finite"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_items(), strategy="random", horizon=3,
                stay_prob_fn=lambda _i, _s: float("nan"),
                rng=np.random.default_rng(0),
            )

    def test_nan_stay_prob_error_names_item_and_step(self) -> None:
        with pytest.raises(ValueError, match=r"step 0"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_items(), strategy="random", horizon=3,
                stay_prob_fn=lambda _i, _s: float("nan"),
                rng=np.random.default_rng(0),
            )


# ---------------------------------------------------------------------------
# simulate_episode — dimension guards
# ---------------------------------------------------------------------------


class TestSimulateEpisodeDimensionChecks:
    def test_theta_wrong_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="theta_true has dimension"):
            simulate_episode(
                theta_true=np.array([0.5]),
                prior_belief=make_prior(d=2), item_bank=make_items(d=2),
                strategy="random", horizon=1, rng=np.random.default_rng(0),
            )

    def test_item_wrong_dimension_raises(self) -> None:
        bad = Item("bad", a=np.array([1.0, 0.0, 0.0]), thresholds=np.array([0.0]))
        with pytest.raises(ValueError, match="Mismatched items"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(d=2),
                item_bank=make_items(d=2) + [bad],
                strategy="random", horizon=1, rng=np.random.default_rng(0),
            )

    def test_mismatched_error_names_the_offending_item(self) -> None:
        bad = Item("culprit", a=np.array([1.0, 0.0, 0.0]), thresholds=np.array([0.0]))
        with pytest.raises(ValueError, match="culprit"):
            simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(d=2),
                item_bank=[bad], strategy="random", horizon=1,
                rng=np.random.default_rng(0),
            )

    def test_consistent_dimensions_do_not_raise(self) -> None:
        simulate_episode(
            theta_true=THETA_TRUE, prior_belief=make_prior(d=2),
            item_bank=make_items(d=2), strategy="random", horizon=2,
            rng=np.random.default_rng(0),
        )


# ---------------------------------------------------------------------------
# simulate_episode — dropout mechanics
# ---------------------------------------------------------------------------


class TestSimulateEpisodeDropout:
    def test_zero_stay_prob_drops_on_first_step(self) -> None:
        always_leave = make_sensitive_constant_stay_prob(p_stay_sensitive=0.0)
        items = [
            Item("x1", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]), is_sensitive=True),
            Item("x2", a=np.array([0.0, 1.0]), thresholds=np.array([0.0]), is_sensitive=True),
        ]
        ep = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=make_prior(), item_bank=items,
            strategy="random", horizon=5, stay_prob_fn=always_leave,
            rng=np.random.default_rng(1),
        )
        assert ep.terminated_by_dropout
        assert ep.n_asked    == 1
        assert ep.n_answered == 0
        assert ep.steps[0].dropped_out

    def test_dropout_step_has_no_response_or_posterior(self) -> None:
        always_leave = make_sensitive_constant_stay_prob(p_stay_sensitive=0.0)
        items = [Item("x1", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]), is_sensitive=True)]
        ep = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=make_prior(), item_bank=items,
            strategy="fixed", horizon=3, stay_prob_fn=always_leave,
            rng=np.random.default_rng(0),
        )
        step = ep.steps[0]
        assert step.response         is None
        assert step.belief_mu_after  is None
        assert step.belief_Sigma_after is None

    def test_final_belief_unchanged_after_dropout(self) -> None:
        always_leave = make_sensitive_constant_stay_prob(p_stay_sensitive=0.0)
        items = [Item("x1", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]), is_sensitive=True)]
        prior = make_prior()
        ep = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=prior, item_bank=items,
            strategy="fixed", horizon=3, stay_prob_fn=always_leave,
            rng=np.random.default_rng(0),
        )
        assert np.allclose(ep.final_belief.mu,    prior.mu)
        assert np.allclose(ep.final_belief.Sigma, prior.Sigma)

    def test_terminated_by_dropout_flag_consistent(self) -> None:
        """terminated_by_dropout must equal dropped_out on the last recorded step."""
        rng     = np.random.default_rng(5)
        stay_fn = make_sensitive_constant_stay_prob(p_stay_sensitive=0.7, p_stay_normal=1.0)
        for _ in range(20):
            ep = simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_items(), strategy="random", horizon=6,
                stay_prob_fn=stay_fn, rng=rng,
            )
            last_dropped = ep.steps[-1].dropped_out if ep.steps else False
            assert ep.terminated_by_dropout == last_dropped

    def test_dropout_step_is_last_step(self) -> None:
        """
        When dropout occurs, all steps before the dropout step must be answered
        (dropped_out=False), and only the final step has dropped_out=True.
        This verifies that the episode loop does not silently swallow early dropouts.
        """
        rng     = np.random.default_rng(7)
        stay_fn = make_sensitive_constant_stay_prob(p_stay_sensitive=0.5, p_stay_normal=1.0)
        found_dropout_episode = False
        for _ in range(50):
            ep = simulate_episode(
                theta_true=THETA_TRUE, prior_belief=make_prior(),
                item_bank=make_items(), strategy="random", horizon=6,
                stay_prob_fn=stay_fn, rng=rng,
            )
            if ep.terminated_by_dropout:
                found_dropout_episode = True
                for step in ep.steps[:-1]:
                    assert not step.dropped_out, (
                        f"Step {step.step} has dropped_out=True before the final step"
                    )
                assert ep.steps[-1].dropped_out
        assert found_dropout_episode, "No dropout episode produced; increase iterations or lower p_stay"


# ---------------------------------------------------------------------------
# simulate_episode — strategy variants
# ---------------------------------------------------------------------------


class TestSimulateEpisodeStrategies:
    @pytest.mark.parametrize("strategy", [
        "random", "fixed", "myopic_exact", "surrogate_unweighted", "surrogate_weighted",
    ])
    def test_all_strategies_complete_without_error(self, strategy: str) -> None:
        ep = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=make_prior(), item_bank=make_items(),
            strategy=strategy, horizon=4, stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0), fixed_order=["q1", "q2", "q3", "q4"],
        )
        assert ep.n_answered == 4

    def test_fixed_order_respected(self) -> None:
        order = ["q3", "q1", "q5", "q2"]
        ep = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=make_prior(), item_bank=make_items(),
            strategy="fixed", horizon=4, stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0), fixed_order=order,
        )
        assert ep.asked_item_ids == order

    def test_damped_update_differs_from_exact_update(self) -> None:
        common = dict(
            theta_true=THETA_TRUE, prior_belief=make_prior(), item_bank=make_items(),
            strategy="fixed", horizon=4, stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(0), fixed_order=["q1", "q2", "q3", "q4"],
        )
        ep_exact = simulate_episode(**common, use_damped_update=False)
        ep_damp  = simulate_episode(**common, use_damped_update=True)
        assert not (
            np.allclose(ep_exact.final_belief.mu,    ep_damp.final_belief.mu)
            and np.allclose(ep_exact.final_belief.Sigma, ep_damp.final_belief.Sigma)
        )


# ---------------------------------------------------------------------------
# Policy integration
# ---------------------------------------------------------------------------


class TestPolicyIntegration:
    def test_weighted_and_unweighted_choose_different_first_item(self) -> None:
        """
        Item A: high variance (a=[2,0], var=4), high sensitivity (p_stay=0.2).
        Item B: low variance  (a=[0,1], var=1), not sensitive   (p_stay=1.0).

        Unweighted score: log(1+var) → A wins  (log5 ≈ 1.61 > log2 ≈ 0.69)
        Weighted score:   log(1+p*v) → B wins  (log2 ≈ 0.69 > log1.8 ≈ 0.59)
        """
        stay_fn = make_sensitive_level_stay_prob(gamma0=0.8, gamma_step=0.0)
        item_a  = Item("high_var_high_sens", a=np.array([2.0, 0.0]),
                       thresholds=np.array([0.0]), is_sensitive=True, sensitivity_level=1.0)
        item_b  = Item("low_var_no_sens",    a=np.array([0.0, 1.0]),
                       thresholds=np.array([0.0]), is_sensitive=False)
        bank  = [item_a, item_b]
        prior = make_prior()

        first_unweighted = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=prior, item_bank=bank,
            strategy="surrogate_unweighted", horizon=1,
            stay_prob_fn=stay_fn, rng=np.random.default_rng(0),
        ).asked_item_ids[0]

        first_weighted = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=prior, item_bank=bank,
            strategy="surrogate_weighted", horizon=1,
            stay_prob_fn=stay_fn, rng=np.random.default_rng(0),
        ).asked_item_ids[0]

        assert first_unweighted == "high_var_high_sens"
        assert first_weighted   == "low_var_no_sens"
        assert first_unweighted != first_weighted


# ---------------------------------------------------------------------------
# simulate_population
# ---------------------------------------------------------------------------


class TestSimulatePopulation:
    def test_returns_one_result_per_user(self) -> None:
        n      = 15
        thetas = np.random.default_rng(0).standard_normal((n, 2))
        results = simulate_population(
            theta_trues=thetas, prior_belief=make_prior(), item_bank=make_items(),
            strategy="random", horizon=3, rng=np.random.default_rng(1),
        )
        assert len(results) == n
        assert all(isinstance(r, EpisodeResult) for r in results)

    def test_accepts_1d_theta(self) -> None:
        results = simulate_population(
            theta_trues=THETA_TRUE, prior_belief=make_prior(), item_bank=make_items(),
            strategy="random", horizon=2, rng=np.random.default_rng(0),
        )
        assert len(results) == 1

    @pytest.mark.parametrize("strategy", [
        "random", "surrogate_unweighted", "surrogate_weighted", "myopic_exact",
    ])
    def test_population_strategies_complete(self, strategy: str) -> None:
        thetas  = np.random.default_rng(3).standard_normal((8, 2))
        results = simulate_population(
            theta_trues=thetas, prior_belief=make_prior(), item_bank=make_items(),
            strategy=strategy, horizon=3, stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(3),
        )
        assert len(results) == 8
        assert all(ep.n_answered == 3 for ep in results)

    def test_population_dropout_rate_plausible(self) -> None:
        """
        p_stay = 1 - 0.1*1.0 = 0.9 per step, horizon = 3.
        Analytical dropout rate: 1 - 0.9^3 ≈ 0.271.
        """
        stay_fn = make_sensitive_level_stay_prob(gamma0=0.1, gamma_step=0.0, min_stay=0.0)
        items   = [
            Item(f"s{i}", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]),
                 is_sensitive=True, sensitivity_level=1.0)
            for i in range(6)
        ]
        thetas  = np.random.default_rng(9).standard_normal((300, 2))
        results = simulate_population(
            theta_trues=thetas, prior_belief=make_prior(), item_bank=items,
            strategy="random", horizon=3, stay_prob_fn=stay_fn,
            rng=np.random.default_rng(9),
        )
        dropout_rate = float(np.mean([r.terminated_by_dropout for r in results]))
        assert 0.10 < dropout_rate < 0.45

    def test_reproducible_with_same_seed(self) -> None:
        thetas = np.random.default_rng(0).standard_normal((5, 2))
        kwargs = dict(
            theta_trues=thetas, prior_belief=make_prior(), item_bank=make_items(),
            strategy="random", horizon=4,
        )
        r1 = simulate_population(**kwargs, rng=np.random.default_rng(77))
        r2 = simulate_population(**kwargs, rng=np.random.default_rng(77))
        for ep1, ep2 in zip(r1, r2):
            assert ep1.asked_item_ids == ep2.asked_item_ids
            assert [s.response for s in ep1.steps] == [s.response for s in ep2.steps]
