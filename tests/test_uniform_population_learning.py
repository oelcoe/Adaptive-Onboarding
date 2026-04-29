from __future__ import annotations

import numpy as np
import pytest

from experiments.uniform_population_learning import aggregate_learning_curves
from src.belief import BeliefState
from src.simulate import EpisodeResult, StepRecord


def test_learning_curves_include_traitwise_step_reduction() -> None:
    prior = BeliefState(mu=np.zeros(2), Sigma=np.diag([1.0, 2.0]))
    after_1 = np.diag([0.7, 1.5])
    after_2 = np.diag([0.4, 1.4])
    result = EpisodeResult(
        steps=[
            StepRecord(
                step=0,
                item_id="q1",
                stay_prob=1.0,
                dropped_out=False,
                response=1,
                belief_mu_before=prior.mu,
                belief_Sigma_before=prior.Sigma,
                belief_mu_after=np.array([0.1, 0.0]),
                belief_Sigma_after=after_1,
            ),
            StepRecord(
                step=1,
                item_id="q2",
                stay_prob=1.0,
                dropped_out=False,
                response=1,
                belief_mu_before=np.array([0.1, 0.0]),
                belief_Sigma_before=after_1,
                belief_mu_after=np.array([0.2, 0.1]),
                belief_Sigma_after=after_2,
            ),
        ],
        final_belief=BeliefState(mu=np.array([0.2, 0.1]), Sigma=after_2),
        terminated_by_dropout=False,
        asked_item_ids=["q1", "q2"],
    )

    rows = aggregate_learning_curves(
        results=[result],
        theta_trues=np.array([[0.0, 0.0]]),
        prior=prior,
        horizon=2,
        carry_forward=True,
    )

    assert rows[0]["mean_variance_dim_1"] == pytest.approx(1.0)
    assert rows[0]["mean_variance_step_reduction_dim_1"] == pytest.approx(0.0)
    assert rows[1]["mean_variance_dim_1"] == pytest.approx(0.7)
    assert rows[1]["mean_variance_step_reduction_dim_1"] == pytest.approx(0.3)
    assert rows[2]["mean_variance_reduction_dim_2"] == pytest.approx(0.6)
    assert rows[2]["mean_variance_step_reduction_dim_2"] == pytest.approx(0.1)
