from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np

from .belief import BeliefState
from .grm import category_probabilities
from .item_bank import Item
from .updates import one_step_posterior_coefficients


PolicyName = Literal[
    "random",
    "fixed",
    "myopic_exact",
    "surrogate_unweighted",
    "surrogate_weighted",
]

StayProbFn = Callable[[Item, int], float]


@dataclass(frozen=True)
class ScoredItem:
    """Container for policy scores, useful for debugging and plotting."""

    item: Item
    score: float
    projected_variance: float
    stay_prob: float


def projected_variance(belief: BeliefState, item: Item) -> float:
    """
    Return a^T Sigma a, the posterior variance in the item's measurement direction.
    """
    var = float(item.a @ belief.Sigma @ item.a)
    return max(var, 0.0)


def no_dropout_stay_prob(_: Item, __: int) -> float:
    """
    Baseline stay-probability model with no engagement risk.
    """
    return 1.0



def make_sensitive_constant_stay_prob(
    p_stay_sensitive: float,
    p_stay_normal: float = 1.0,
) -> StayProbFn:
    """
    Return a stay-probability function based on the binary is_sensitive flag.

    Sensitive items (item.is_sensitive == True) receive a fixed constant stay
    probability p_stay_sensitive, regardless of step.  Non-sensitive items
    receive p_stay_normal (default 1.0 = no dropout risk).

    This is the pure binary model: the flag alone determines dropout risk.
    Use make_sensitive_level_stay_prob when you need the continuous level.
    """
    if not (0.0 <= p_stay_sensitive <= 1.0):
        raise ValueError("p_stay_sensitive must be in [0, 1].")
    if not (0.0 <= p_stay_normal <= 1.0):
        raise ValueError("p_stay_normal must be in [0, 1].")

    def stay_prob(item: Item, _step: int) -> float:
        return p_stay_sensitive if item.is_sensitive else p_stay_normal

    return stay_prob


def make_sensitive_level_stay_prob(
    gamma0: float,
    gamma_step: float = 0.0,
    min_stay: float = 0.0,
    max_stay: float = 1.0,
) -> StayProbFn:
    """
    Return a stay-probability function that uses both the binary is_sensitive flag
    and the continuous sensitivity_level.

    For non-sensitive items:
        p_stay = max_stay   (no dropout risk by default)

    For sensitive items:
        gamma_k  = max(gamma0 - gamma_step * step, 0)
        p_stay   = clip(1 - gamma_k * item.sensitivity_level, min_stay, max_stay)

    The penalty is strongest at step 0 and weakens by gamma_step each step, reaching
    zero once gamma_k hits 0.  Setting gamma_step=0 gives a time-constant penalty.
    An item with sensitivity_level=0 has no dropout risk even when is_sensitive=True;
    use make_sensitive_constant_stay_prob for pure binary dropout.
    """
    if gamma0 < 0.0 or gamma_step < 0.0:
        raise ValueError("gamma0 and gamma_step must be nonnegative.")
    if min_stay < 0.0 or max_stay > 1.0 or min_stay > max_stay:
        raise ValueError("Require 0 <= min_stay <= max_stay <= 1.")

    def stay_prob(item: Item, step: int) -> float:
        if step < 0:
            raise ValueError("step must be nonnegative.")
        if not item.is_sensitive:
            return max_stay
        gamma_k = max(gamma0 - gamma_step * step, 0.0)
        p = 1.0 - gamma_k * item.sensitivity_level
        return float(np.clip(p, min_stay, max_stay))

    return stay_prob


def score_surrogate_unweighted(
    belief: BeliefState,
    item: Item,
) -> float:
    """
    Unweighted online surrogate score:

        log(1 + a^T Sigma a)

    This captures how uncertain the current belief remains in the direction probed by the item.
    """
    var = projected_variance(belief, item)
    return float(np.log1p(var))


def score_surrogate_weighted(
    belief: BeliefState,
    item: Item,
    step: int,
    stay_prob_fn: StayProbFn,
) -> float:
    """
    Weighted online surrogate score:

        log(1 + p_stay(i, k) * a^T Sigma a)

    This matches the chapter's engagement-aware surrogate:
        Delta_sur(i | s_k) = log(1 + p_stay(i,k) * a^T Sigma_k a).
    """
    p_stay = float(stay_prob_fn(item, step))
    if not np.isfinite(p_stay):
        raise ValueError("stay_prob_fn must return a finite value.")
    p_stay = float(np.clip(p_stay, 0.0, 1.0))
    var = projected_variance(belief, item)
    return float(np.log1p(p_stay * var))


def score_myopic_exact(
    belief: BeliefState,
    item: Item,
    step: int,
    stay_prob_fn: StayProbFn,
) -> float:
    """
    Exact one-step look-ahead score from Eq. (exact):

        Delta(i | s_k) = p_stay(i,k) * sum_r p_r(i) * (-log v_r(i))

    where p_r(i) is the predictive probability of category r and v_r(i) is the
    one-step variance coefficient under response r.
    """
    p_stay = float(stay_prob_fn(item, step))
    if not np.isfinite(p_stay):
        raise ValueError("stay_prob_fn must return a finite value.")
    p_stay = float(np.clip(p_stay, 0.0, 1.0))

    probs = category_probabilities(belief, item)
    expected_logdet_reduction = 0.0
    for response, p_r in enumerate(probs):
        _, v_r, _ = one_step_posterior_coefficients(belief, item, response)
        v_r = float(np.clip(v_r, 1e-300, 1.0))
        expected_logdet_reduction += float(p_r) * (-np.log(v_r))

    return p_stay * expected_logdet_reduction


def score_bank_myopic_exact(
    belief: BeliefState,
    item_bank: Sequence[Item],
    step: int,
    already_asked: Sequence[str] | None = None,
    stay_prob_fn: StayProbFn | None = None,
) -> list[ScoredItem]:
    """
    Score all currently available items using the exact one-step myopic objective:

        Delta(i | s_k) = p_stay(i,k) * sum_r p_r(i) * (-log v_r(i))

    Returns items sorted by descending score.
    """
    candidates = _available_items(item_bank, already_asked)
    if not candidates:
        return []

    if stay_prob_fn is None:
        stay_prob_fn = no_dropout_stay_prob

    scored: list[ScoredItem] = []
    for item in candidates:
        p_stay = float(np.clip(stay_prob_fn(item, step), 0.0, 1.0))
        score = score_myopic_exact(
            belief=belief,
            item=item,
            step=step,
            stay_prob_fn=stay_prob_fn,
        )
        scored.append(
            ScoredItem(
                item=item,
                score=score,
                projected_variance=projected_variance(belief, item),
                stay_prob=p_stay,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


def _available_items(
    item_bank: Sequence[Item],
    already_asked: Sequence[str] | None,
) -> list[Item]:
    asked = set(already_asked or [])
    return [item for item in item_bank if item.item_id not in asked]


def score_bank(
    belief: BeliefState,
    item_bank: Sequence[Item],
    step: int,
    already_asked: Sequence[str] | None = None,
    weighted: bool = True,
    stay_prob_fn: StayProbFn | None = None,
) -> list[ScoredItem]:
    """
    Score all currently available items and return them in descending score order.

    For weighted=False, scores are computed as log(1 + a^T Sigma a).
    For weighted=True, scores are computed as log(1 + p_stay(i,k) * a^T Sigma a).
    """
    candidates = _available_items(item_bank, already_asked)
    if not candidates:
        return []

    if stay_prob_fn is None:
        stay_prob_fn = no_dropout_stay_prob

    scored: list[ScoredItem] = []
    for item in candidates:
        var = projected_variance(belief, item)
        p_stay = float(np.clip(stay_prob_fn(item, step), 0.0, 1.0)) if weighted else 1.0
        score = float(np.log1p(p_stay * var)) if weighted else float(np.log1p(var))
        scored.append(
            ScoredItem(
                item=item,
                score=score,
                projected_variance=var,
                stay_prob=p_stay,
            )
        )

    # Stable, deterministic tie-breaking by original item order via Python's stable sort
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


def _select_fixed(
    item_bank: Sequence[Item],
    already_asked: Sequence[str] | None = None,
    fixed_order: Sequence[str] | None = None,
) -> Item:
    asked = set(already_asked or [])

    if fixed_order is not None:
        lookup = {item.item_id: item for item in item_bank}
        for item_id in fixed_order:
            if item_id not in asked:
                if item_id not in lookup:
                    raise ValueError(f"Unknown item_id in fixed_order: {item_id}")
                return lookup[item_id]
        raise ValueError("No available items remain in fixed_order.")

    for item in item_bank:
        if item.item_id not in asked:
            return item

    raise ValueError("No available items remain.")


def _select_random(
    item_bank: Sequence[Item],
    already_asked: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> Item:
    candidates = _available_items(item_bank, already_asked)
    if not candidates:
        raise ValueError("No available items remain.")

    if rng is None:
        rng = np.random.default_rng()

    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def select_next_item(
    belief: BeliefState,
    item_bank: Sequence[Item],
    step: int,
    strategy: PolicyName,
    already_asked: Sequence[str] | None = None,
    stay_prob_fn: StayProbFn | None = None,
    fixed_order: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> Item:
    """
    Select the next item according to the requested policy.

    Supported strategies:
        - "random"
        - "fixed"
        - "myopic_exact"
        - "surrogate_unweighted"
        - "surrogate_weighted"
    """
    if strategy == "random":
        return _select_random(item_bank=item_bank, already_asked=already_asked, rng=rng)

    if strategy == "fixed":
        return _select_fixed(
            item_bank=item_bank,
            already_asked=already_asked,
            fixed_order=fixed_order,
        )

    if strategy == "myopic_exact":
        scored = score_bank_myopic_exact(
            belief=belief,
            item_bank=item_bank,
            step=step,
            already_asked=already_asked,
            stay_prob_fn=stay_prob_fn,
        )
        if not scored:
            raise ValueError("No available items remain.")
        return scored[0].item

    if strategy == "surrogate_unweighted":
        scored = score_bank(
            belief=belief,
            item_bank=item_bank,
            step=step,
            already_asked=already_asked,
            weighted=False,
            stay_prob_fn=stay_prob_fn,
        )
        if not scored:
            raise ValueError("No available items remain.")
        return scored[0].item

    if strategy == "surrogate_weighted":
        if stay_prob_fn is None:
            stay_prob_fn = no_dropout_stay_prob
        scored = score_bank(
            belief=belief,
            item_bank=item_bank,
            step=step,
            already_asked=already_asked,
            weighted=True,
            stay_prob_fn=stay_prob_fn,
        )
        if not scored:
            raise ValueError("No available items remain.")
        return scored[0].item

    raise ValueError(f"Unknown strategy: {strategy}")
    