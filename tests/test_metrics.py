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
  5.  zero-step episode (immediate dropout): total_logdet_reduction == 0
  6.  single-step episode: reduction equals before - after
  7.  initial_logdet equals log det of first step's Sigma_before
  8.  final_logdet   equals log det of final_belief.Sigma
  9.  n_answered and n_asked mirror EpisodeResult properties
  10. dropped_out mirrors EpisodeResult.terminated_by_dropout
  11. logdet_reduction == initial_logdet - final_logdet (identity check)
  12. more steps → non-negative cumulative reduction

PolicyMetrics / aggregate_policy_metrics
  13. single episode: dropout_count and rate consistent
  14. all-dropout population: dropout_rate == 1.0
  15. no-dropout population:  dropout_rate == 0.0
  16. mean_n_answered is the arithmetic mean of per-episode n_answered
  17. mean_n_asked    is the arithmetic mean of per-episode n_asked
  18. mean_final_logdet matches manual computation
  19. mean_logdet_reduction matches manual computation
  20. empty results raises ValueError
  21. policy_name is preserved in output
  22. n_episodes equals len(results)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.belief import BeliefState
from src.item_bank import Item
from src.metrics import (
    EpisodeMetrics,
    PolicyMetrics,
    _logdet,
    aggregate_policy_metrics,
    episode_metrics,
)
from src.policies import (
    make_sensitive_constant_stay_prob,
    no_dropout_stay_prob,
)
from src.simulate import (
    EpisodeResult,
    StepRecord,
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

    def test_zero_step_episode_has_zero_reduction(self) -> None:
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
