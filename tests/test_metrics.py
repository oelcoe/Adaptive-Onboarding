"""
Tests for src/metrics.py

Coverage
────────
_logdet (internal helper)
  1.  identity matrix: log det(I) == 0
  2.  scaled identity: log det(c*I) == d * log(c)
  3.  singular matrix raises ValueError

EpisodeMetrics / episode_metrics
  4.  no-dropout episode: logdet_reduction > 0
  5.  immediate-dropout episode: total_logdet_reduction == 0
  6.  single-step episode: reduction equals before - after
  7.  initial_logdet equals log det of first step's Sigma_before
  8.  final_logdet   equals log det of final_belief.Sigma
  9.  n_answered and n_asked mirror EpisodeResult properties
  10. dropped_out mirrors EpisodeResult.terminated_by_dropout
  11. logdet_reduction == initial_logdet - final_logdet (identity check)
  12. more steps → non-negative cumulative reduction
  13. final_d_error equals det(Sigma)^(1/d)
  14. final_d_error decreases after answering questions
  15. sensitive-asked metrics are zero without item metadata
  16. sensitive-asked metrics use item-bank metadata when provided

PolicyMetrics / aggregate_policy_metrics
  17. single episode: dropout_count and rate consistent
  18. all-dropout population: dropout_rate == 1.0
  19. no-dropout population:  dropout_rate == 0.0
  20. mean_n_answered is the arithmetic mean of per-episode n_answered
  21. mean_n_asked    is the arithmetic mean of per-episode n_asked
  22. mean_final_logdet matches manual computation
  23. mean_logdet_reduction matches manual computation
  24. mean_final_d_error matches manual computation
  25. sensitivity burden means match manual computation
  26. empty results raises ValueError
  27. policy_name is preserved in output
  28. n_episodes equals len(results)

mean_estimation_error
  29. zero error when mu_final == theta_true
  30. known distance on a trivial case
  31. empty results raise ValueError
  32. mismatched lengths raise ValueError
  33. mismatched dimensions raise ValueError
  34. population-level smoke test: error decreases with more answered questions
"""

from __future__ import annotations

import numpy as np
import pytest

from src.belief import BeliefState
from src.item_bank import Item
from src.metrics import (
    _logdet,
    aggregate_policy_metrics,
    episode_metrics,
    mean_estimation_error,
)
from src.policies import (
    make_sensitive_constant_stay_prob,
    no_dropout_stay_prob,
)
from src.simulate import (
    EpisodeResult,
    simulate_episode,
    simulate_population,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_prior(d: int = 2) -> BeliefState:
    return BeliefState(mu=np.zeros(d), Sigma=np.eye(d))


def make_items(d: int = 2) -> list[Item]:
    return [
        Item("q1", a=np.array([1.0, 0.0]), thresholds=np.array([-1.0, 0.0, 1.0])),
        Item("q2", a=np.array([0.0, 1.0]), thresholds=np.array([-1.0, 0.0, 1.0]), is_sensitive=True, sensitivity_level=0.4),
        Item("q3", a=np.array([0.7, 0.7]), thresholds=np.array([-1.0, 0.0, 1.0])),
        Item("q4", a=np.array([1.0, 0.5]), thresholds=np.array([-0.5, 0.5])),
        Item("q5", a=np.array([0.5, 1.0]), thresholds=np.array([-0.5, 0.5])),
    ]


THETA_TRUE = np.array([0.5, -0.5])


def _run_no_dropout(horizon: int = 4, strategy: str = "surrogate_weighted") -> EpisodeResult:
    return simulate_episode(
        theta_true=THETA_TRUE,
        prior_belief=make_prior(),
        item_bank=make_items(),
        strategy=strategy,
        horizon=horizon,
        stay_prob_fn=no_dropout_stay_prob,
        rng=np.random.default_rng(42),
    )


def _run_immediate_dropout() -> EpisodeResult:
    """Episode where the user drops out on the very first step."""
    always_leave = make_sensitive_constant_stay_prob(p_stay_sensitive=0.0)
    items = [
        Item("s1", a=np.array([1.0, 0.0]), thresholds=np.array([0.0]), is_sensitive=True),
        Item("s2", a=np.array([0.0, 1.0]), thresholds=np.array([0.0]), is_sensitive=True),
    ]
    return simulate_episode(
        theta_true=THETA_TRUE,
        prior_belief=make_prior(),
        item_bank=items,
        strategy="random",
        horizon=5,
        stay_prob_fn=always_leave,
        rng=np.random.default_rng(1),
    )


# ---------------------------------------------------------------------------
# _logdet
# ---------------------------------------------------------------------------


class TestLogdet:
    def test_identity_matrix(self) -> None:
        for d in (1, 2, 5):
            assert _logdet(np.eye(d)) == pytest.approx(0.0)

    def test_scaled_identity(self) -> None:
        """log det(c * I_d) == d * log(c)."""
        for d in (2, 3):
            c = 3.0
            Sigma = c * np.eye(d)
            assert _logdet(Sigma) == pytest.approx(d * np.log(c))

    def test_singular_matrix_raises(self) -> None:
        Sigma = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="positive definite"):
            _logdet(Sigma)


# ---------------------------------------------------------------------------
# EpisodeMetrics / episode_metrics
# ---------------------------------------------------------------------------


class TestEpisodeMetrics:
    def test_logdet_reduction_positive_for_answered_episode(self) -> None:
        em = episode_metrics(_run_no_dropout(horizon=4))
        assert em.total_logdet_reduction > 0.0

    def test_immediate_dropout_episode_has_zero_reduction(self) -> None:
        ep = _run_immediate_dropout()
        assert ep.n_answered == 0
        em = episode_metrics(ep)
        assert em.total_logdet_reduction == pytest.approx(0.0)

    def test_single_step_reduction_matches_before_minus_after(self) -> None:
        ep = _run_no_dropout(horizon=1)
        em = episode_metrics(ep)
        step = ep.steps[0]
        expected = _logdet(step.belief_Sigma_before) - _logdet(ep.final_belief.Sigma)
        assert em.total_logdet_reduction == pytest.approx(expected)

    def test_initial_logdet_equals_first_step_sigma_before(self) -> None:
        ep = _run_no_dropout(horizon=4)
        em = episode_metrics(ep)
        assert em.initial_logdet == pytest.approx(
            _logdet(ep.steps[0].belief_Sigma_before)
        )

    def test_final_logdet_equals_final_belief_sigma(self) -> None:
        ep = _run_no_dropout(horizon=4)
        em = episode_metrics(ep)
        assert em.final_logdet == pytest.approx(_logdet(ep.final_belief.Sigma))

    def test_n_answered_mirrors_episode_result(self) -> None:
        ep = _run_no_dropout(horizon=4)
        assert episode_metrics(ep).n_answered == ep.n_answered

    def test_n_asked_mirrors_episode_result(self) -> None:
        ep = _run_no_dropout(horizon=4)
        assert episode_metrics(ep).n_asked == ep.n_asked

    def test_dropped_out_mirrors_terminated_flag(self) -> None:
        ep_ok    = _run_no_dropout(horizon=3)
        ep_drop  = _run_immediate_dropout()
        assert episode_metrics(ep_ok).dropped_out   == ep_ok.terminated_by_dropout
        assert episode_metrics(ep_drop).dropped_out == ep_drop.terminated_by_dropout

    def test_logdet_reduction_identity(self) -> None:
        """total_logdet_reduction == initial_logdet - final_logdet always."""
        ep = _run_no_dropout(horizon=4)
        em = episode_metrics(ep)
        assert em.total_logdet_reduction == pytest.approx(
            em.initial_logdet - em.final_logdet
        )

    def test_final_d_error_equals_det_to_one_over_d(self) -> None:
        ep = _run_no_dropout(horizon=4)
        em = episode_metrics(ep)
        d        = ep.final_belief.dim
        expected = float(np.linalg.det(ep.final_belief.Sigma) ** (1.0 / d))
        assert em.final_d_error == pytest.approx(expected)

    def test_final_d_error_by_dimension_matches_covariance_diagonal(self) -> None:
        ep = _run_no_dropout(horizon=4)
        em = episode_metrics(ep)
        assert em.final_d_error_by_dimension == pytest.approx(
            tuple(np.diag(ep.final_belief.Sigma))
        )

    def test_final_d_error_decreases_after_answering(self) -> None:
        prior = make_prior()
        ep    = _run_no_dropout(horizon=4)
        d     = prior.dim
        prior_d_error = float(np.linalg.det(prior.Sigma) ** (1.0 / d))
        assert episode_metrics(ep).final_d_error < prior_d_error

    def test_sensitivity_metrics_zero_without_item_bank(self) -> None:
        ep = _run_no_dropout(horizon=4)
        em = episode_metrics(ep)
        assert em.n_sensitive_asked == 0
        assert em.sensitivity_level_asked == pytest.approx(0.0)

    def test_sensitivity_metrics_use_item_bank_metadata(self) -> None:
        items = make_items()
        ep = simulate_episode(
            theta_true=THETA_TRUE,
            prior_belief=make_prior(),
            item_bank=items,
            strategy="fixed",
            horizon=3,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(42),
        )
        em = episode_metrics(ep, item_bank=items)
        assert ep.asked_item_ids == ["q1", "q2", "q3"]
        assert em.n_sensitive_asked == 1
        assert em.sensitivity_level_asked == pytest.approx(0.4)

    def test_more_steps_do_not_decrease_reduction(self) -> None:
        """
        Answering more questions should (on average) reduce uncertainty more.
        We just check that the horizon-4 run has non-negative reduction, and
        that a one-step run has a smaller or equal reduction than a four-step run
        when given the same seed and items.
        """
        ep1 = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=make_prior(), item_bank=make_items(),
            strategy="surrogate_weighted", horizon=1,
            stay_prob_fn=no_dropout_stay_prob, rng=np.random.default_rng(42),
        )
        ep4 = simulate_episode(
            theta_true=THETA_TRUE, prior_belief=make_prior(), item_bank=make_items(),
            strategy="surrogate_weighted", horizon=4,
            stay_prob_fn=no_dropout_stay_prob, rng=np.random.default_rng(42),
        )
        r1 = episode_metrics(ep1).total_logdet_reduction
        r4 = episode_metrics(ep4).total_logdet_reduction
        assert r1 >= 0.0
        assert r4 >= r1


# ---------------------------------------------------------------------------
# PolicyMetrics / aggregate_policy_metrics
# ---------------------------------------------------------------------------


class TestAggregatePolicyMetrics:
    def test_empty_results_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            aggregate_policy_metrics([], policy_name="random")

    def test_policy_name_preserved(self) -> None:
        results = [_run_no_dropout()]
        pm = aggregate_policy_metrics(results, policy_name="my_policy")
        assert pm.policy_name == "my_policy"

    def test_n_episodes_equals_len_results(self) -> None:
        results = [_run_no_dropout()] * 7
        pm = aggregate_policy_metrics(results, policy_name="x")
        assert pm.n_episodes == 7

    def test_single_episode_metrics_consistent(self) -> None:
        ep  = _run_no_dropout(horizon=4)
        pm  = aggregate_policy_metrics([ep], policy_name="single")
        em  = episode_metrics(ep)
        assert pm.dropout_count               == int(em.dropped_out)
        assert pm.dropout_rate                == float(int(em.dropped_out))
        assert pm.mean_n_answered             == pytest.approx(em.n_answered)
        assert pm.mean_n_asked                == pytest.approx(em.n_asked)
        assert pm.mean_final_logdet           == pytest.approx(em.final_logdet)
        assert pm.mean_logdet_reduction       == pytest.approx(em.total_logdet_reduction)
        assert pm.mean_sensitive_asked        == pytest.approx(em.n_sensitive_asked)
        assert pm.mean_sensitivity_level_asked == pytest.approx(
            em.sensitivity_level_asked
        )

    def test_all_dropout_population(self) -> None:
        ep_drop = _run_immediate_dropout()
        pm = aggregate_policy_metrics([ep_drop, ep_drop, ep_drop], policy_name="all_drop")
        assert pm.dropout_count == 3
        assert pm.dropout_rate  == pytest.approx(1.0)

    def test_no_dropout_population(self) -> None:
        results = [_run_no_dropout(horizon=3) for _ in range(5)]
        pm = aggregate_policy_metrics(results, policy_name="no_drop")
        assert pm.dropout_count == 0
        assert pm.dropout_rate  == pytest.approx(0.0)

    def test_mean_n_answered_arithmetic_mean(self) -> None:
        results = [_run_no_dropout(horizon=h) for h in [1, 2, 3]]
        pm      = aggregate_policy_metrics(results, policy_name="x")
        expected = np.mean([r.n_answered for r in results])
        assert pm.mean_n_answered == pytest.approx(float(expected))

    def test_mean_n_asked_arithmetic_mean(self) -> None:
        results  = [_run_no_dropout(horizon=h) for h in [1, 2, 3]]
        pm       = aggregate_policy_metrics(results, policy_name="x")
        expected = np.mean([r.n_asked for r in results])
        assert pm.mean_n_asked == pytest.approx(float(expected))

    def test_mean_final_logdet_matches_manual(self) -> None:
        results  = [_run_no_dropout(horizon=h) for h in [2, 3, 4]]
        pm       = aggregate_policy_metrics(results, policy_name="x")
        expected = np.mean([episode_metrics(r).final_logdet for r in results])
        assert pm.mean_final_logdet == pytest.approx(float(expected))

    def test_mean_logdet_reduction_matches_manual(self) -> None:
        results  = [_run_no_dropout(horizon=h) for h in [2, 3, 4]]
        pm       = aggregate_policy_metrics(results, policy_name="x")
        expected = np.mean([episode_metrics(r).total_logdet_reduction for r in results])
        assert pm.mean_logdet_reduction == pytest.approx(float(expected))

    def test_mean_final_d_error_matches_manual(self) -> None:
        results  = [_run_no_dropout(horizon=h) for h in [2, 3, 4]]
        pm       = aggregate_policy_metrics(results, policy_name="x")
        expected = np.mean([episode_metrics(r).final_d_error for r in results])
        assert pm.mean_final_d_error == pytest.approx(float(expected))

    def test_mean_final_d_error_by_dimension_matches_manual(self) -> None:
        results  = [_run_no_dropout(horizon=h) for h in [2, 3, 4]]
        pm       = aggregate_policy_metrics(results, policy_name="x")
        expected = np.mean(
            [episode_metrics(r).final_d_error_by_dimension for r in results],
            axis=0,
        )
        assert pm.mean_final_d_error_by_dimension == pytest.approx(tuple(expected))

    def test_sensitivity_burden_means_match_manual(self) -> None:
        items = make_items()
        results = [
            simulate_episode(
                theta_true=THETA_TRUE,
                prior_belief=make_prior(),
                item_bank=items,
                strategy="fixed",
                horizon=h,
                stay_prob_fn=no_dropout_stay_prob,
                rng=np.random.default_rng(42),
            )
            for h in [1, 2, 3]
        ]
        pm = aggregate_policy_metrics(results, policy_name="fixed", item_bank=items)
        expected_counts = [
            episode_metrics(r, item_bank=items).n_sensitive_asked
            for r in results
        ]
        expected_levels = [
            episode_metrics(r, item_bank=items).sensitivity_level_asked
            for r in results
        ]
        assert pm.mean_sensitive_asked == pytest.approx(float(np.mean(expected_counts)))
        assert pm.mean_sensitivity_level_asked == pytest.approx(
            float(np.mean(expected_levels))
        )

    def test_population_via_simulate_population(self) -> None:
        """End-to-end smoke test using simulate_population output."""
        thetas = np.random.default_rng(0).standard_normal((20, 2))
        results = simulate_population(
            theta_trues=thetas,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy="surrogate_weighted",
            horizon=3,
            stay_prob_fn=no_dropout_stay_prob,
            rng=np.random.default_rng(42),
        )
        pm = aggregate_policy_metrics(results, policy_name="surrogate_weighted")
        assert pm.n_episodes        == 20
        assert pm.dropout_rate      == pytest.approx(0.0)
        assert pm.mean_n_answered   == pytest.approx(3.0)
        assert pm.mean_logdet_reduction > 0.0
        assert pm.mean_final_d_error > 0.0


# ---------------------------------------------------------------------------
# mean_estimation_error
# ---------------------------------------------------------------------------


class TestMeanEstimationError:
    def test_zero_error_when_mu_equals_theta(self) -> None:
        """If the posterior mean exactly equals theta_true, error is zero."""
        theta  = np.array([1.0, 2.0])
        result = EpisodeResult(
            steps=[],
            final_belief=BeliefState(mu=theta, Sigma=np.eye(2)),
            terminated_by_dropout=False,
            asked_item_ids=[],
        )
        assert mean_estimation_error([result], np.array([theta])) == pytest.approx(0.0)

    def test_known_distance(self) -> None:
        """Single episode with known mu and theta_true produces exact L2 distance."""
        mu    = np.array([0.0, 0.0])
        theta = np.array([3.0, 4.0])   # ||theta - mu|| == 5
        result = EpisodeResult(
            steps=[],
            final_belief=BeliefState(mu=mu, Sigma=np.eye(2)),
            terminated_by_dropout=False,
            asked_item_ids=[],
        )
        assert mean_estimation_error([result], np.array([theta])) == pytest.approx(5.0)

    def test_mismatched_lengths_raise(self) -> None:
        results = [_run_no_dropout(horizon=2)] * 3
        thetas  = np.random.default_rng(0).standard_normal((5, 2))
        with pytest.raises(ValueError, match="3"):
            mean_estimation_error(results, thetas)

    def test_empty_results_raise(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            mean_estimation_error([], np.empty((0, 2)))

    def test_mismatched_dimensions_raise(self) -> None:
        result = _run_no_dropout(horizon=2)
        theta = np.zeros((1, 3))
        with pytest.raises(ValueError, match="dimension"):
            mean_estimation_error([result], theta)

    def test_more_answers_reduce_estimation_error(self) -> None:
        """
        With more questions answered the posterior mean should track theta_true
        more closely on average across a population.
        """
        thetas = np.random.default_rng(7).standard_normal((50, 2))
        common = dict(
            theta_trues=thetas,
            prior_belief=make_prior(),
            item_bank=make_items(),
            strategy="surrogate_weighted",
            stay_prob_fn=no_dropout_stay_prob,
        )
        r1 = simulate_population(**common, horizon=1, rng=np.random.default_rng(0))
        r5 = simulate_population(**common, horizon=5, rng=np.random.default_rng(0))
        err1 = mean_estimation_error(r1, thetas)
        err5 = mean_estimation_error(r5, thetas)
        assert err5 < err1
