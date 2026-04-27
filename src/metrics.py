"""
metrics.py
==========
Per-episode and per-policy aggregate metrics for adaptive questionnaire simulations.

Usage
-----
    from src.metrics import episode_metrics, aggregate_policy_metrics

    # One episode
    em = episode_metrics(result)
    print(em.total_logdet_reduction, em.n_answered, em.dropped_out)

    # Population of episodes for a single policy
    pm = aggregate_policy_metrics(results, policy_name="surrogate_weighted")
    print(pm.dropout_rate, pm.mean_n_answered, pm.mean_logdet_reduction)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .simulate import EpisodeResult


__all__ = [
    "EpisodeMetrics",
    "PolicyMetrics",
    "episode_metrics",
    "aggregate_policy_metrics",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _logdet(Sigma: NDArray[np.float64]) -> float:
    """
    Compute log det(Sigma) robustly using the sign + log-abs form of slogdet.

    Raises ValueError when the matrix is singular or has a non-positive determinant.
    """
    sign, logabsdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError(
            "Covariance matrix is not positive definite: "
            f"slogdet returned sign={sign}."
        )
    return float(logabsdet)


# ---------------------------------------------------------------------------
# Episode-level result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeMetrics:
    """
    Scalar summary of a single adaptive episode.

    Attributes
    ----------
    initial_logdet:
        log det(Sigma_0), where Sigma_0 is the covariance just before the first
        update (``steps[0].belief_Sigma_before`` when steps exist, otherwise the
        covariance of the final belief — which equals the prior when the episode
        has no steps).
    final_logdet:
        log det(Sigma_T), where Sigma_T is the covariance of ``final_belief``.
    total_logdet_reduction:
        ``initial_logdet - final_logdet``.  Positive values indicate uncertainty
        reduction.  Zero when the episode has no answered steps (e.g. immediate
        dropout).
    n_answered:
        Number of questions the user answered (mirrors ``EpisodeResult.n_answered``).
    n_asked:
        Number of questions presented, including the dropout step if applicable
        (mirrors ``EpisodeResult.n_asked``).
    dropped_out:
        Whether the episode ended due to dropout (mirrors
        ``EpisodeResult.terminated_by_dropout``).
    """

    initial_logdet: float
    final_logdet: float
    total_logdet_reduction: float
    n_answered: int
    n_asked: int
    dropped_out: bool


# ---------------------------------------------------------------------------
# Policy-level aggregate result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyMetrics:
    """
    Aggregate metrics across a population of episodes run under one policy.

    Attributes
    ----------
    policy_name:
        Human-readable label (the ``strategy`` string passed to
        ``simulate_episode``/``simulate_population``).
    n_episodes:
        Total number of episodes in the population.
    dropout_count:
        Number of episodes that ended by dropout.
    dropout_rate:
        ``dropout_count / n_episodes``.
    mean_n_answered:
        Mean number of questions answered per episode.
    mean_n_asked:
        Mean number of questions presented per episode.
    mean_final_logdet:
        Mean log det(Sigma_T) across all episodes.
    mean_logdet_reduction:
        Mean total log-determinant uncertainty reduction across all episodes.
    """

    policy_name: str
    n_episodes: int
    dropout_count: int
    dropout_rate: float
    mean_n_answered: float
    mean_n_asked: float
    mean_final_logdet: float
    mean_logdet_reduction: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def episode_metrics(result: EpisodeResult) -> EpisodeMetrics:
    """
    Compute scalar metrics for a single episode.

    Parameters
    ----------
    result:
        The ``EpisodeResult`` returned by ``simulate_episode``.

    Returns
    -------
    EpisodeMetrics
    """
    final_logdet = _logdet(result.final_belief.Sigma)

    if result.steps:
        initial_logdet = _logdet(result.steps[0].belief_Sigma_before)
    else:
        # No steps at all: prior was never updated.
        initial_logdet = final_logdet

    total_logdet_reduction = initial_logdet - final_logdet

    return EpisodeMetrics(
        initial_logdet=initial_logdet,
        final_logdet=final_logdet,
        total_logdet_reduction=total_logdet_reduction,
        n_answered=result.n_answered,
        n_asked=result.n_asked,
        dropped_out=result.terminated_by_dropout,
    )


def aggregate_policy_metrics(
    results: list[EpisodeResult],
    policy_name: str,
) -> PolicyMetrics:
    """
    Aggregate per-episode metrics into a single population-level summary.

    Parameters
    ----------
    results:
        List of ``EpisodeResult`` objects produced by a single policy, e.g.
        the output of ``simulate_population``.
    policy_name:
        Label used to identify this policy in comparison tables or plots.

    Returns
    -------
    PolicyMetrics

    Raises
    ------
    ValueError
        If ``results`` is empty.
    """
    if not results:
        raise ValueError(
            "results must contain at least one EpisodeResult; got an empty list."
        )

    per_episode = [episode_metrics(r) for r in results]

    n_episodes = len(per_episode)
    dropout_count = sum(em.dropped_out for em in per_episode)

    return PolicyMetrics(
        policy_name=policy_name,
        n_episodes=n_episodes,
        dropout_count=dropout_count,
        dropout_rate=dropout_count / n_episodes,
        mean_n_answered=float(np.mean([em.n_answered for em in per_episode])),
        mean_n_asked=float(np.mean([em.n_asked for em in per_episode])),
        mean_final_logdet=float(np.mean([em.final_logdet for em in per_episode])),
        mean_logdet_reduction=float(
            np.mean([em.total_logdet_reduction for em in per_episode])
        ),
    )
