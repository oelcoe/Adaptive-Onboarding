from __future__ import annotations

from dataclasses import dataclass, replace as _dataclass_replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .belief import BeliefState
from .item_bank import Item
from .policies import PolicyName, StayProbFn, no_dropout_stay_prob, select_next_item
from .updates import damped_update_belief, update_belief


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """
    Records everything that happened at a single step of an adaptive episode.
    
    """

    step: int
    item_id: str
    stay_prob: float
    dropped_out: bool
    response: int | None
    belief_mu_before: NDArray[np.float64]
    belief_Sigma_before: NDArray[np.float64]
    belief_mu_after: NDArray[np.float64] | None
    belief_Sigma_after: NDArray[np.float64] | None


@dataclass
class EpisodeResult:
    """Complete record of one adaptive questionnaire episode."""

    steps: list[StepRecord]
    final_belief: BeliefState
    terminated_by_dropout: bool
    asked_item_ids: list[str]

    @property
    def n_answered(self) -> int:
        """Number of questions the user actually answered (no dropout at that step)."""
        return sum(1 for s in self.steps if not s.dropped_out)

    @property
    def n_asked(self) -> int:
        """Total number of items that were presented (including the one that caused dropout)."""
        return len(self.steps)


# ---------------------------------------------------------------------------
# Environment: sample a true response from a known latent state
# ---------------------------------------------------------------------------


def _effective_noise_variance(item: Item, sensitivity_noise_scale: float) -> float:
    """
    Compute the response noise variance for a sensitive item at simulation time.

        effective_variance = base * (1 + sensitivity_noise_scale * sensitivity_level)

    For non-sensitive items or when sensitivity_noise_scale == 0, the item's stored
    observation_noise_variance (default 1.0) is returned unchanged.

    This value is used in simulate_episode to create effective copies of items via
    dataclasses.replace. Both sample_response and update_belief then read the same
    inflated value from the effective item — the model is fully consistent.
    """
    if not item.is_sensitive or sensitivity_noise_scale == 0.0:
        return item.observation_noise_variance
    return item.observation_noise_variance * (1.0 + sensitivity_noise_scale * item.sensitivity_level)


def sample_response(
    theta_true: NDArray[np.float64],
    item: Item,
    rng: np.random.Generator,
) -> int:
    """
    Draw a single ordinal response from the true user latent state.

    The generative model is:
        Z = a^T theta_true + eps,   eps ~ N(0, item.observation_noise_variance)
    Response category r is returned when thresholds[r-1] <= Z < thresholds[r],
    with boundary conventions thresholds[-1] = -inf and thresholds[K] = +inf.

    When sensitivity_noise_scale > 0 is passed to simulate_episode, effective
    copies of items are created with inflated response_noise_variance before the
    episode loop begins. Both this function and the belief update therefore see
    the same noise level — the model is fully consistent.

    This is the *environment* oracle. It is intentionally separate from
    category_probabilities(), which computes probabilities under the *belief*.
    """
    theta_true = np.asarray(theta_true, dtype=float).reshape(-1)
    if theta_true.shape[0] != item.dim:
        raise ValueError(
            f"theta_true has dimension {theta_true.shape[0]} but item "
            f"'{item.item_id}' expects dimension {item.dim}."
        )
    eps = rng.normal(0.0, np.sqrt(item.observation_noise_variance))
    z = float(item.a @ theta_true) + eps

    # map z to ordinal category via item thresholds
    category = int(np.searchsorted(item.thresholds, z, side="right"))
    # searchsorted returns values in [0, len(thresholds)], which maps to [0, n_categories-1]
    return int(np.clip(category, 0, item.n_categories - 1))


# ---------------------------------------------------------------------------
# Core: simulate one episode
# ---------------------------------------------------------------------------


def simulate_episode(
    theta_true: NDArray[np.float64],
    prior_belief: BeliefState,
    item_bank: Sequence[Item],
    strategy: PolicyName,
    horizon: int,
    stay_prob_fn: StayProbFn | None = None,
    rng: np.random.Generator | None = None,
    use_damped_update: bool = False,
    fixed_order: Sequence[str] | None = None,
    sensitivity_noise_scale: float = 0.0,
) -> EpisodeResult:
    """
    Simulate one adaptive questionnaire episode under a given policy.

    Parameters
    ----------
    theta_true:
        True latent trait vector of the synthetic user.
    prior_belief:
        Initial Gaussian belief before any questions are asked.
    item_bank:
        Full pool of available items.
    strategy:
        Policy name forwarded to select_next_item().
    horizon:
        Maximum number of questions to present (T).
    stay_prob_fn:
        Callable (item, step) → p_stay in [0, 1]. Defaults to no_dropout_stay_prob
        (always 1), so no dropout occurs unless a custom function is provided.
    rng:
        NumPy random generator. A fresh default_rng() is created if None.
    use_damped_update:
        If True, uses damped_update_belief(); otherwise uses update_belief().
    fixed_order:
        Item-id sequence used only when strategy == "fixed".

    Returns
    -------
    EpisodeResult
        Full record of the episode including every StepRecord.
    """
    theta_true = np.asarray(theta_true, dtype=float).reshape(-1)
    d = prior_belief.dim

    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}.")
    if sensitivity_noise_scale < 0.0:
        raise ValueError(
            f"sensitivity_noise_scale must be nonnegative, got {sensitivity_noise_scale}."
        )

    if theta_true.shape[0] != d:
        raise ValueError(
            f"theta_true has dimension {theta_true.shape[0]} but prior belief "
            f"has dimension {d}. They must match."
        )

    mismatched = [(item.item_id, item.dim) for item in item_bank if item.dim != d]
    if mismatched:
        raise ValueError(
            f"All items must have the same dimension as the prior belief (d={d}). "
            f"Mismatched items: {mismatched}."
        )

    # Build effective item bank: inflate response_noise_variance for sensitive items
    # so that both the generative model (sample_response) and the inference model
    # (update_belief) see the same noise. This keeps the model fully consistent.
    if sensitivity_noise_scale > 0.0:
        item_bank = [
            _dataclass_replace(
                item,
                response_noise_variance=_effective_noise_variance(item, sensitivity_noise_scale),
            )
            for item in item_bank
        ]

    if rng is None:
        rng = np.random.default_rng()

    if stay_prob_fn is None:
        stay_prob_fn = no_dropout_stay_prob

    belief = prior_belief
    steps: list[StepRecord] = []
    asked_ids: list[str] = []
    terminated_by_dropout = False

    for step in range(horizon):
        # ----- a. choose item -----
        try:
            item = select_next_item(
                belief=belief,
                item_bank=item_bank,
                step=step,
                strategy=strategy,
                already_asked=asked_ids,
                stay_prob_fn=stay_prob_fn,
                fixed_order=fixed_order,
                rng=rng,
            )
        except ValueError:
            # Bank exhausted (horizon > bank size, or fixed_order fully consumed).
            break

        # ----- b. compute stay probability -----
        raw_stay = stay_prob_fn(item, step)
        if not np.isfinite(raw_stay):
            raise ValueError(
                f"stay_prob_fn returned non-finite value {raw_stay!r} "
                f"for item {item.item_id!r} at step {step}."
            )
        p_stay = float(np.clip(raw_stay, 0.0, 1.0))

        # ----- c. sample dropout -----
        dropped_out = bool(rng.random() > p_stay)

        if dropped_out:
            steps.append(
                StepRecord(
                    step=step,
                    item_id=item.item_id,
                    stay_prob=p_stay,
                    dropped_out=True,
                    response=None,
                    belief_mu_before=belief.mu.copy(),
                    belief_Sigma_before=belief.Sigma.copy(),
                    belief_mu_after=None,
                    belief_Sigma_after=None,
                )
            )
            asked_ids.append(item.item_id)
            terminated_by_dropout = True
            break

        # ----- d. sample response -----
        response = sample_response(theta_true=theta_true, item=item, rng=rng)

        # ----- e. update belief -----
        mu_before = belief.mu.copy()
        Sigma_before = belief.Sigma.copy()

        belief = (
            damped_update_belief(belief, item, response)
            if use_damped_update
            else update_belief(belief, item, response)
        )

        # ----- f. record step -----
        steps.append(
            StepRecord(
                step=step,
                item_id=item.item_id,
                stay_prob=p_stay,
                dropped_out=False,
                response=response,
                belief_mu_before=mu_before,
                belief_Sigma_before=Sigma_before,
                belief_mu_after=belief.mu.copy(),
                belief_Sigma_after=belief.Sigma.copy(),
            )
        )
        asked_ids.append(item.item_id)

    return EpisodeResult(
        steps=steps,
        final_belief=belief,
        terminated_by_dropout=terminated_by_dropout,
        asked_item_ids=asked_ids,
    )


# ---------------------------------------------------------------------------
# Population: simulate many users
# ---------------------------------------------------------------------------


def simulate_population(
    theta_trues: NDArray[np.float64],
    prior_belief: BeliefState,
    item_bank: Sequence[Item],
    strategy: PolicyName,
    horizon: int,
    stay_prob_fn: StayProbFn | None = None,
    rng: np.random.Generator | None = None,
    use_damped_update: bool = False,
    fixed_order: Sequence[str] | None = None,
    sensitivity_noise_scale: float = 0.0,
) -> list[EpisodeResult]:
    """
    Simulate one episode per row of theta_trues and return all results.

    Parameters
    ----------
    theta_trues:
        Array of shape (n_users, d). Each row is one synthetic user's true
        latent trait vector.

    All other parameters are forwarded to simulate_episode() for every user.
    The same prior_belief, item_bank, and policy are used for all users,
    reflecting the common assumption that the adaptive system shares one prior.

    Returns
    -------
    list[EpisodeResult]
        One EpisodeResult per user, in the same row order as theta_trues.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_trues = np.asarray(theta_trues, dtype=float)
    if theta_trues.ndim == 1:
        theta_trues = theta_trues[np.newaxis, :]

    results: list[EpisodeResult] = []
    for theta in theta_trues:
        result = simulate_episode(
            theta_true=theta,
            prior_belief=prior_belief,
            item_bank=item_bank,
            strategy=strategy,
            horizon=horizon,
            stay_prob_fn=stay_prob_fn,
            rng=rng,
            use_damped_update=use_damped_update,
            fixed_order=fixed_order,
            sensitivity_noise_scale=sensitivity_noise_scale,
        )
        results.append(result)

    return results
