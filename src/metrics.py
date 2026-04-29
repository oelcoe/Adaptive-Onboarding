"""
metrics.py
==========
Per-episode and per-policy aggregate metrics for adaptive questionnaire simulations.

Usage
-----
    from src.metrics import episode_metrics, aggregate_policy_metrics, mean_estimation_error

    # One episode
    em = episode_metrics(result, item_bank=item_bank)
    print(em.total_logdet_reduction, em.final_d_error, em.n_sensitive_asked)

    # Population of episodes for a single policy
    pm = aggregate_policy_metrics(
        results,
        policy_name="surrogate_weighted",
        item_bank=item_bank,
    )
    print(pm.dropout_rate, pm.mean_logdet_reduction, pm.mean_sensitive_asked)

    # Estimation error vs ground truth (requires the theta_trues used in simulation)
    mean_err = mean_estimation_error(results, theta_trues)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2 as _chi2_dist
from scipy.stats import norm as _norm_dist

from .item_bank import Item
from .simulate import EpisodeResult


__all__ = [
    "EpisodeMetrics",
    "PolicyMetrics",
    "CalibrationResult",
    "episode_metrics",
    "aggregate_policy_metrics",
    "mean_estimation_error",
    "calibration_result",
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
    final_d_error:
        det(Sigma_T)^(1/d) — the geometric mean posterior variance (D-error
        criterion from optimal design theory).  Equal to exp(log_det / d);
        smaller is better.
    n_answered:
        Number of questions the user answered (mirrors ``EpisodeResult.n_answered``).
    n_asked:
        Number of questions presented, including the dropout step if applicable
        (mirrors ``EpisodeResult.n_asked``).
    dropped_out:
        Whether the episode ended due to dropout (mirrors
        ``EpisodeResult.terminated_by_dropout``).
    n_sensitive_asked:
        Number of presented questions marked sensitive. Includes the dropout
        step when dropout occurs. Requires ``item_bank`` in ``episode_metrics``;
        otherwise set to 0.
    sensitivity_level_asked:
        Sum of sensitivity levels across presented questions. This is a
        continuous sensitivity burden per episode. Requires ``item_bank`` in
        ``episode_metrics``; otherwise set to 0.0.
    """

    initial_logdet: float
    final_logdet: float
    total_logdet_reduction: float
    final_d_error: float
    final_d_error_by_dimension: tuple[float, ...]
    n_answered: int
    n_asked: int
    dropped_out: bool
    n_sensitive_asked: int = 0
    sensitivity_level_asked: float = 0.0


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
    mean_final_d_error:
        Mean det(Sigma_T)^(1/d) across all episodes — mean D-error.
        Lower values indicate tighter beliefs.
    mean_sensitive_asked:
        Mean number of sensitive questions presented per episode.
    mean_sensitivity_level_asked:
        Mean total sensitivity-level burden presented per episode.
    """

    policy_name: str
    n_episodes: int
    dropout_count: int
    dropout_rate: float
    mean_n_answered: float
    mean_n_asked: float
    mean_final_logdet: float
    mean_logdet_reduction: float
    mean_final_d_error: float
    mean_final_d_error_by_dimension: tuple[float, ...] = field(default_factory=tuple)
    mean_sensitive_asked: float = 0.0
    mean_sensitivity_level_asked: float = 0.0
    mean_final_d_error_completed: float = float("nan")
    mean_final_d_error_by_dimension_completed: tuple[float, ...] = field(default_factory=tuple)
    sensitive_rate: float = float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _item_lookup(item_bank: list[Item] | None) -> dict[str, Item]:
    if item_bank is None:
        return {}
    return {item.item_id: item for item in item_bank}


def episode_metrics(
    result: EpisodeResult,
    item_bank: list[Item] | None = None,
) -> EpisodeMetrics:
    """
    Compute scalar metrics for a single episode.

    Parameters
    ----------
    result:
        The ``EpisodeResult`` returned by ``simulate_episode``.
    item_bank:
        Optional item bank used to look up sensitivity metadata for asked item
        IDs. When omitted, sensitivity-burden metrics are returned as zero.

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

    d = result.final_belief.dim
    final_d_error = float(np.exp(final_logdet / d))
    final_d_error_by_dimension = tuple(
        float(value) for value in np.diag(result.final_belief.Sigma)
    )

    lookup = _item_lookup(item_bank)
    asked_items = [
        lookup[item_id]
        for item_id in result.asked_item_ids
        if item_id in lookup
    ]
    n_sensitive_asked = sum(item.is_sensitive for item in asked_items)
    sensitivity_level_asked = float(
        sum(item.sensitivity_level for item in asked_items)
    )

    return EpisodeMetrics(
        initial_logdet=initial_logdet,
        final_logdet=final_logdet,
        total_logdet_reduction=total_logdet_reduction,
        final_d_error=final_d_error,
        final_d_error_by_dimension=final_d_error_by_dimension,
        n_answered=result.n_answered,
        n_asked=result.n_asked,
        dropped_out=result.terminated_by_dropout,
        n_sensitive_asked=n_sensitive_asked,
        sensitivity_level_asked=sensitivity_level_asked,
    )


def aggregate_policy_metrics(
    results: list[EpisodeResult],
    policy_name: str,
    item_bank: list[Item] | None = None,
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
    item_bank:
        Optional item bank used to look up sensitivity metadata for asked item
        IDs. When omitted, sensitivity-burden metrics are returned as zero.

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

    per_episode = [episode_metrics(r, item_bank=item_bank) for r in results]

    n_episodes = len(per_episode)
    dropout_count = sum(em.dropped_out for em in per_episode)
    completed = [em for em in per_episode if not em.dropped_out]

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
        mean_final_d_error=float(np.mean([em.final_d_error for em in per_episode])),
        mean_final_d_error_by_dimension=tuple(
            float(value)
            for value in np.mean(
                [em.final_d_error_by_dimension for em in per_episode],
                axis=0,
            )
        ),
        mean_sensitive_asked=float(
            np.mean([em.n_sensitive_asked for em in per_episode])
        ),
        mean_sensitivity_level_asked=float(
            np.mean([em.sensitivity_level_asked for em in per_episode])
        ),
        mean_final_d_error_completed=float(
            np.mean([em.final_d_error for em in completed])
        ) if completed else float("nan"),
        mean_final_d_error_by_dimension_completed=tuple(
            float(value)
            for value in np.mean(
                [em.final_d_error_by_dimension for em in completed],
                axis=0,
            )
        ) if completed else tuple(),
        sensitive_rate=float(
            np.mean([em.n_sensitive_asked for em in per_episode])
            / np.mean([em.n_asked for em in per_episode])
        ) if np.mean([em.n_asked for em in per_episode]) > 0 else float("nan"),
    )


def mean_estimation_error(
    results: list[EpisodeResult],
    theta_trues: NDArray[np.float64],
) -> float:
    """
    Mean Euclidean distance between each episode's final posterior mean and the
    corresponding true latent trait vector.

        mean_i  ||mu_T^(i) - theta_true^(i)||_2

    Parameters
    ----------
    results:
        Ordered list of ``EpisodeResult`` objects, one per user.
    theta_trues:
        Array of shape ``(n_users, dim)`` or ``(dim,)`` for a single user.
        Must be in the same order as ``results``.

    Returns
    -------
    float
        Mean L2 distance across the population.

    Raises
    ------
    ValueError
        If ``results`` is empty, or if the lengths/dimensions of ``results`` and
        ``theta_trues`` do not match.
    """
    if not results:
        raise ValueError("results must contain at least one EpisodeResult.")

    thetas = np.atleast_2d(np.asarray(theta_trues, dtype=float))
    if thetas.shape[0] != len(results):
        raise ValueError(
            f"theta_trues has {thetas.shape[0]} rows but results has "
            f"{len(results)} entries."
        )

    expected_dim = results[0].final_belief.dim
    if thetas.shape[1] != expected_dim:
        raise ValueError(
            f"theta_trues has dimension {thetas.shape[1]} but results have "
            f"dimension {expected_dim}."
        )

    mismatched_dims = [
        idx
        for idx, result in enumerate(results)
        if result.final_belief.dim != expected_dim
    ]
    if mismatched_dims:
        raise ValueError(
            "All results must have the same final belief dimension; "
            f"mismatched result indices: {mismatched_dims}."
        )

    errors = [
        float(np.linalg.norm(r.final_belief.mu - theta))
        for r, theta in zip(results, thetas)
    ]
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Posterior calibration
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """
    Calibration diagnostics for a set of final Gaussian posteriors.

    Under a perfectly calibrated Gaussian posterior, the true parameter
    ``theta_true`` is a sample from ``N(mu_T, Sigma_T)``.  Three equivalent
    tests follow:

    * **Marginal z-scores** ``(theta_i - mu_i) / sqrt(Sigma_ii)``  should be
      i.i.d. ``N(0, 1)`` across users and dimensions.
    * **Mahalanobis distances squared** ``(theta - mu)^T Sigma^{-1} (theta - mu)``
      should follow a ``chi2(d)`` distribution across users.
    * **Credible-interval coverage**: the fraction of users whose
      ``theta_true`` falls inside the alpha-level credible ellipsoid should
      equal alpha for every alpha in ``[0, 1]``.

    Attributes
    ----------
    marginal_z_scores:
        Standardized residuals ``(theta_i - mu_i) / sqrt(Sigma_ii)`` stacked
        over all included (user, dimension) pairs. Shape ``(n_users * dim,)``.
    mahalanobis_sq:
        Squared Mahalanobis distances. Shape ``(n_users,)``.
    chi2_quantiles:
        CDF of ``chi2(dim)`` evaluated at each ``mahalanobis_sq``. Shape
        ``(n_users,)``.  Under perfect calibration these are ``Uniform(0,1)``.
    alphas:
        Evenly-spaced alpha grid in ``[0, 1]``. Shape ``(n_alpha,)``.
    empirical_coverage:
        Fraction of users with ``theta_true`` inside the alpha credible
        ellipsoid for each alpha. Shape ``(n_alpha,)``.  Under perfect
        calibration this equals ``alphas`` exactly.
    dim:
        Latent dimension.
    n_users:
        Number of episodes included in the calibration calculation.
    n_dropouts_excluded:
        Number of dropout episodes skipped when ``filter_dropouts=True``.
    """

    marginal_z_scores:  NDArray[np.float64]
    mahalanobis_sq:     NDArray[np.float64]
    chi2_quantiles:     NDArray[np.float64]
    alphas:             NDArray[np.float64]
    empirical_coverage: NDArray[np.float64]
    dim:                int
    n_users:            int
    n_dropouts_excluded: int = 0


def calibration_result(
    results: list[EpisodeResult],
    theta_trues: NDArray[np.float64],
    *,
    n_alpha: int = 101,
    filter_dropouts: bool = False,
) -> CalibrationResult:
    """
    Compute posterior calibration diagnostics for a population of episodes.

    Parameters
    ----------
    results:
        Ordered list of ``EpisodeResult`` objects, one per user.
    theta_trues:
        True latent trait vectors. Shape ``(n_users, dim)`` or ``(dim,)``.
        Must be in the same order as ``results``.
    n_alpha:
        Number of evenly-spaced alpha values in ``[0, 1]`` for the coverage
        curve. Default 101 (0 %, 1 %, …, 100 %).
    filter_dropouts:
        When ``True``, episodes that ended by dropout are excluded.  Their
        posteriors are less refined (closer to the prior) and will appear
        miscalibrated even for a correct model — the ``n_dropouts_excluded``
        field records how many were removed.

    Returns
    -------
    CalibrationResult

    Raises
    ------
    ValueError
        If ``results`` is empty or dimensions are inconsistent.
    """
    if not results:
        raise ValueError("results must contain at least one EpisodeResult.")

    thetas = np.atleast_2d(np.asarray(theta_trues, dtype=float))
    if thetas.shape[0] != len(results):
        raise ValueError(
            f"theta_trues has {thetas.shape[0]} rows but results has "
            f"{len(results)} entries."
        )

    pairs = list(zip(results, thetas))

    n_dropouts_excluded = 0
    if filter_dropouts:
        before = len(pairs)
        pairs = [(r, t) for r, t in pairs if not r.terminated_by_dropout]
        n_dropouts_excluded = before - len(pairs)

    if not pairs:
        raise ValueError(
            "No episodes remain after filtering dropouts. "
            "Set filter_dropouts=False or supply more data."
        )

    dim = pairs[0][0].final_belief.dim
    n_users = len(pairs)

    # ---- marginal z-scores ------------------------------------------------
    # z_{n,i} = (theta_true_{n,i} - mu_{n,i}) / sqrt(Sigma_{n,ii})
    z_scores: list[NDArray[np.float64]] = []
    mah_sq:   list[float]               = []

    for result, theta in pairs:
        mu    = result.final_belief.mu
        Sigma = result.final_belief.Sigma
        std   = np.sqrt(np.diag(Sigma))
        z_scores.append((theta - mu) / std)

        # Mahalanobis^2 — use lstsq for numerical stability when Sigma
        # approaches singularity late in training.
        diff = (theta - mu).reshape(-1, 1)
        Sigma_inv_diff = np.linalg.lstsq(Sigma, diff, rcond=None)[0]
        mah_sq.append(float((diff.T @ Sigma_inv_diff).item()))

    marginal_z   = np.concatenate(z_scores)         # (n_users * dim,)
    mah_sq_arr   = np.array(mah_sq, dtype=float)    # (n_users,)
    chi2_quants  = _chi2_dist.cdf(mah_sq_arr, df=dim).astype(float)

    # ---- coverage curve ---------------------------------------------------
    alphas    = np.linspace(0.0, 1.0, n_alpha)
    thresholds = _chi2_dist.ppf(alphas, df=dim)     # chi2 quantiles for each alpha
    empirical_coverage = np.array(
        [float(np.mean(mah_sq_arr <= t)) for t in thresholds],
        dtype=float,
    )

    return CalibrationResult(
        marginal_z_scores=marginal_z,
        mahalanobis_sq=mah_sq_arr,
        chi2_quantiles=chi2_quants,
        alphas=alphas,
        empirical_coverage=empirical_coverage,
        dim=dim,
        n_users=n_users,
        n_dropouts_excluded=n_dropouts_excluded,
    )
