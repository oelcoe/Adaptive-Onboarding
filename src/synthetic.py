from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .item_bank import Item

SensitivityAssignment = Literal["random", "axis_aligned"]

__all__ = [
    "synthetic_item_bank",
    "generate_user_population",
    "sample_theta_true",
    "item_bank_to_records",
    "item_bank_to_dataframe",
    "select_sensitivity_examples",
    "axis_alignment_score",
    "axis_alignment_scores",
    "SensitivityAssignment",
]


def synthetic_item_bank(
    n_items: int,
    dim: int = 2,
    n_categories: int = 4,
    sensitive_fraction: float = 0.0,
    rng_seed: int | None = None,
    *,
    sensitivity_assignment: SensitivityAssignment = "random",
    vary_sensitivity_levels: bool = False,
    couple_sensitivity_level_to_axis_alignment: bool = False,
    sensitivity_level_range: tuple[float, float] = (0.25, 1.0),
    angle_jitter: float = 0.05,
    threshold_span: float = 1.0,
    threshold_perturbation: float = 0.05,
) -> list[Item]:
    """
    Build a reproducible synthetic ordinal item bank.

    For two latent dimensions, discrimination directions are placed approximately
    evenly over [0, pi), because a and -a encode the same hyperplane direction up
    to threshold reversal. For other dimensions, directions are sampled at random
    and normalized.

    Parameters
    ----------
    n_items:
        Number of items to generate.
    dim:
        Number of latent traits, i.e. len(item.a).
    n_categories:
        Number of ordinal response categories. Each item receives
        n_categories - 1 thresholds.
    sensitive_fraction:
        Fraction of items to mark sensitive. Values in [0, 1] are fractions;
        values in (1, 100] are interpreted as percentages.
    rng_seed:
        Seed forwarded to numpy.default_rng for reproducibility.
    sensitivity_assignment:
        How sensitivity labels are assigned. "random" assigns sensitivity
        independently of direction. "axis_aligned" labels the most axis-aligned
        items as sensitive.
    vary_sensitivity_levels:
        If True, sensitive items receive item-specific sensitivity levels sampled
        from sensitivity_level_range. If False, sensitive items receive level
        1.0. Non-sensitive items always receive level 0.0.
    couple_sensitivity_level_to_axis_alignment:
        If True, sensitive levels increase with axis alignment instead of being
        sampled randomly. This is most useful with sensitivity_assignment set to
        "axis_aligned".
    sensitivity_level_range:
        Inclusive range used when vary_sensitivity_levels is True or when
        coupling levels to axis alignment.
    angle_jitter:
        Fraction of the 2D angular spacing used as uniform jitter. Set to 0 for
        exactly even angles.
    threshold_span:
        Symmetric endpoint for base thresholds. For four categories, the default
        base thresholds are [-1, 0, 1].
    threshold_perturbation:
        Normal jitter standard deviation added to thresholds before sorting.
        Set to 0 for exactly symmetric thresholds.
    """
    _validate_common_inputs(
        n_items=n_items,
        dim=dim,
        n_categories=n_categories,
        sensitive_fraction=sensitive_fraction,
        angle_jitter=angle_jitter,
        threshold_span=threshold_span,
        threshold_perturbation=threshold_perturbation,
        sensitivity_assignment=sensitivity_assignment,
    )

    low_level, high_level = sensitivity_level_range
    if low_level < 0.0 or high_level < low_level:
        raise ValueError(
            "sensitivity_level_range must be a nonnegative (low, high) pair "
            "with low <= high."
        )

    rng = np.random.default_rng(rng_seed)
    directions = _sample_directions(n_items, dim, rng, angle_jitter)
    threshold_template = _base_thresholds(n_categories, threshold_span)
    alignment_scores = axis_alignment_scores(directions)
    sensitive_mask = _assign_sensitive_mask(
        directions=directions,
        sensitive_fraction=sensitive_fraction,
        rng=rng,
        sensitivity_assignment=sensitivity_assignment,
    )

    items: list[Item] = []
    for index in range(n_items):
        thresholds = _perturb_thresholds(
            threshold_template,
            rng=rng,
            threshold_perturbation=threshold_perturbation,
            threshold_span=threshold_span,
        )
        is_sensitive = bool(sensitive_mask[index])
        sensitivity_level = _sample_sensitivity_level(
            is_sensitive=is_sensitive,
            vary_sensitivity_levels=vary_sensitivity_levels,
            couple_to_axis_alignment=couple_sensitivity_level_to_axis_alignment,
            axis_alignment=float(alignment_scores[index]),
            dim=dim,
            sensitivity_level_range=sensitivity_level_range,
            rng=rng,
        )
        items.append(
            Item(
                item_id=f"item_{index:04d}",
                a=directions[index],
                thresholds=thresholds,
                is_sensitive=is_sensitive,
                sensitivity_level=sensitivity_level,
            )
        )

    return items


def generate_user_population(
    n_users: int,
    dim: int = 2,
    *,
    mean: Sequence[float] | None = None,
    covariance: Sequence[Sequence[float]] | NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
    rng_seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Generate true latent trait vectors for a synthetic user population.

    Returns an array of shape (n_users, dim), suitable for passing directly to
    simulate_population().
    """
    if n_users < 1:
        raise ValueError(f"n_users must be at least 1, got {n_users}.")
    if dim < 1:
        raise ValueError(f"dim must be at least 1, got {dim}.")
    if rng is not None and rng_seed is not None:
        raise ValueError("Pass either rng or rng_seed, not both.")

    mu = _validate_population_mean(mean, dim)
    cov = _validate_population_covariance(covariance, dim)

    if rng is None:
        rng = np.random.default_rng(rng_seed)

    return rng.multivariate_normal(mean=mu, cov=cov, size=n_users)


def sample_theta_true(
    dim: int = 2,
    *,
    mean: Sequence[float] | None = None,
    covariance: Sequence[Sequence[float]] | NDArray[np.float64] | None = None,
    rng: np.random.Generator | None = None,
    rng_seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Sample one true latent trait vector.

    Convenience wrapper for tests and single-episode simulations.
    """
    return generate_user_population(
        n_users=1,
        dim=dim,
        mean=mean,
        covariance=covariance,
        rng=rng,
        rng_seed=rng_seed,
    )[0]


def item_bank_to_records(item_bank: Sequence[Item]) -> list[dict[str, object]]:
    """
    Convert an item bank to notebook-friendly row records.

    The core representation stays as list[Item]. This helper is for inspection,
    plotting, or optional downstream conversion with pandas.DataFrame(records).
    """
    records: list[dict[str, object]] = []
    for item in item_bank:
        record: dict[str, object] = {
            "item_id": item.item_id,
            "dim": item.dim,
            "n_categories": item.n_categories,
            "is_sensitive": item.is_sensitive,
            "sensitivity_level": item.sensitivity_level,
            "axis_alignment": axis_alignment_score(item.a),
            "thresholds": item.thresholds.copy(),
            "a": item.a.copy(),
        }
        for index, value in enumerate(item.a):
            record[f"a_{index}"] = float(value)
        for index, value in enumerate(item.thresholds):
            record[f"threshold_{index}"] = float(value)
        if item.dim == 2:
            record["angle"] = float(np.mod(np.arctan2(item.a[1], item.a[0]), np.pi))
        records.append(record)
    return records


def item_bank_to_dataframe(item_bank: Sequence[Item]):
    """
    Convert an item bank to a pandas DataFrame.

    The DataFrame keeps array-valued ``a`` and ``thresholds`` columns for direct
    access, and also expands them into scalar columns such as ``a_0`` and
    ``threshold_0`` for filtering, plotting, and tabular inspection.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "item_bank_to_dataframe requires pandas. Install the project "
            "requirements or add pandas to your environment."
        ) from exc

    return pd.DataFrame(item_bank_to_records(item_bank))


def select_sensitivity_examples(item_bank: Sequence[Item]) -> tuple[Item, Item]:
    """
    Return one non-sensitive item and one sensitive item for comparison plots.

    Raises
    ------
    ValueError
        If the bank does not contain at least one item of each type.
    """
    non_sensitive_item = next((item for item in item_bank if not item.is_sensitive), None)
    sensitive_item = next((item for item in item_bank if item.is_sensitive), None)

    if non_sensitive_item is None or sensitive_item is None:
        raise ValueError(
            "item_bank must contain at least one non-sensitive item and one "
            "sensitive item."
        )

    return non_sensitive_item, sensitive_item


def axis_alignment_score(a: Sequence[float]) -> float:
    """
    Measure how close a direction is to one coordinate axis.

    The score is max(abs(a_i)) after normalizing a. A perfectly axis-aligned
    unit vector has score 1.0. In two dimensions, a diagonal unit vector has
    score 1 / sqrt(2).
    """
    vector = np.asarray(a, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError("a must be non-empty and nonzero.")
    return float(axis_alignment_scores(vector.reshape(1, -1))[0])


def axis_alignment_scores(directions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized axis-alignment scores for a matrix of directions."""
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[0] == 0 or directions.shape[1] == 0:
        raise ValueError("directions must be a non-empty 2D array.")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("directions must not contain zero rows.")
    return np.max(np.abs(directions / norms[:, np.newaxis]), axis=1)


def _validate_common_inputs(
    *,
    n_items: int,
    dim: int,
    n_categories: int,
    sensitive_fraction: float,
    angle_jitter: float,
    threshold_span: float,
    threshold_perturbation: float,
    sensitivity_assignment: SensitivityAssignment,
) -> None:
    if n_items < 1:
        raise ValueError(f"n_items must be at least 1, got {n_items}.")
    if dim < 1:
        raise ValueError(f"dim must be at least 1, got {dim}.")
    if n_categories < 2:
        raise ValueError(f"n_categories must be at least 2, got {n_categories}.")
    if sensitive_fraction < 0.0 or sensitive_fraction > 100.0:
        raise ValueError(
            "sensitive_fraction must be in [0, 1] as a fraction or in "
            "(1, 100] as a percentage."
        )
    if not np.isfinite(angle_jitter) or angle_jitter < 0.0:
        raise ValueError("angle_jitter must be finite and nonnegative.")
    if not np.isfinite(threshold_span) or threshold_span <= 0.0:
        raise ValueError("threshold_span must be finite and strictly positive.")
    if not np.isfinite(threshold_perturbation) or threshold_perturbation < 0.0:
        raise ValueError("threshold_perturbation must be finite and nonnegative.")
    if sensitivity_assignment not in {"random", "axis_aligned"}:
        raise ValueError(
            "sensitivity_assignment must be either 'random' or 'axis_aligned'."
        )


def _validate_population_mean(
    mean: Sequence[float] | None,
    dim: int,
) -> NDArray[np.float64]:
    if mean is None:
        return np.zeros(dim, dtype=float)

    mu = np.asarray(mean, dtype=float).reshape(-1)
    if mu.shape != (dim,):
        raise ValueError(f"mean must have shape ({dim},), got {mu.shape}.")
    if not np.all(np.isfinite(mu)):
        raise ValueError("mean must contain only finite values.")
    return mu


def _validate_population_covariance(
    covariance: Sequence[Sequence[float]] | NDArray[np.float64] | None,
    dim: int,
) -> NDArray[np.float64]:
    if covariance is None:
        return np.eye(dim, dtype=float)

    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (dim, dim):
        raise ValueError(f"covariance must have shape ({dim}, {dim}), got {cov.shape}.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance must contain only finite values.")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("covariance must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(cov)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("covariance must be positive semidefinite.")

    return cov


def _sample_directions(
    n_items: int,
    dim: int,
    rng: np.random.Generator,
    angle_jitter: float,
) -> NDArray[np.float64]:
    if dim == 1:
        return np.ones((n_items, 1), dtype=float)

    if dim == 2:
        spacing = np.pi / n_items
        angles = spacing * np.arange(n_items, dtype=float)
        if angle_jitter > 0.0 and n_items > 1:
            jitter = rng.uniform(
                low=-0.5 * angle_jitter * spacing,
                high=0.5 * angle_jitter * spacing,
                size=n_items,
            )
            angles = np.mod(angles + jitter, np.pi)
        return np.column_stack((np.cos(angles), np.sin(angles))).astype(float)

    directions = rng.normal(size=(n_items, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return _canonicalize_direction_signs(directions)


def _canonicalize_direction_signs(
    directions: NDArray[np.float64],
) -> NDArray[np.float64]:
    # For each row, find the first nonzero component and flip rows where it is negative.
    abs_dirs = np.abs(directions)
    first_nonzero_col = np.argmax(abs_dirs > 1e-12, axis=1)
    signs = np.sign(directions[np.arange(len(directions)), first_nonzero_col])
    signs = np.where(signs == 0, 1.0, signs)
    return directions * signs[:, np.newaxis]


def _base_thresholds(
    n_categories: int,
    threshold_span: float,
) -> NDArray[np.float64]:
    n_thresholds = n_categories - 1
    if n_thresholds == 1:
        # For binary items the base is sampled per-item in _perturb_thresholds;
        # this placeholder is never used directly.
        return np.array([0.0], dtype=float)
    return np.linspace(-threshold_span, threshold_span, n_thresholds, dtype=float)


def _perturb_thresholds(
    thresholds: NDArray[np.float64],
    *,
    rng: np.random.Generator,
    threshold_perturbation: float,
    threshold_span: float = 1.0,
) -> NDArray[np.float64]:
    if thresholds.size == 1:
        # Binary item: place the threshold uniformly across the full range so
        # that one response category can be more likely than the other, then
        # add a small normal jitter on top.
        base = rng.uniform(-threshold_span, threshold_span)
        if threshold_perturbation > 0.0:
            base += rng.normal(0.0, threshold_perturbation)
        return np.array([base], dtype=float)

    if threshold_perturbation == 0.0:
        return thresholds.copy()

    perturbed = thresholds + rng.normal(0.0, threshold_perturbation, size=thresholds.shape)
    return _strictly_increasing(perturbed)


def _strictly_increasing(values: Sequence[float]) -> NDArray[np.float64]:
    ordered = np.sort(np.asarray(values, dtype=float).reshape(-1))
    min_gap = 1e-6
    for index in range(1, ordered.size):
        if ordered[index] <= ordered[index - 1]:
            ordered[index] = ordered[index - 1] + min_gap
    return ordered


def _assign_sensitive_mask(
    *,
    directions: NDArray[np.float64],
    sensitive_fraction: float,
    rng: np.random.Generator,
    sensitivity_assignment: SensitivityAssignment,
) -> NDArray[np.bool_]:
    n_items = directions.shape[0]
    fraction = sensitive_fraction / 100.0 if sensitive_fraction > 1.0 else sensitive_fraction
    n_sensitive = int(round(n_items * fraction))

    mask = np.zeros(n_items, dtype=bool)
    if n_sensitive == 0:
        return mask

    if sensitivity_assignment == "random":
        sensitive_indices = rng.choice(n_items, size=n_sensitive, replace=False)
    else:
        alignment = axis_alignment_scores(directions)
        sensitive_indices = np.argsort(-alignment, kind="stable")[:n_sensitive]

    mask[sensitive_indices] = True
    return mask


def _sample_sensitivity_level(
    *,
    is_sensitive: bool,
    vary_sensitivity_levels: bool,
    couple_to_axis_alignment: bool,
    axis_alignment: float,
    dim: int,
    sensitivity_level_range: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    if not is_sensitive:
        return 0.0
    if couple_to_axis_alignment:
        low, high = sensitivity_level_range
        normalized_alignment = _normalized_axis_alignment(axis_alignment, dim)
        return float(low + (high - low) * normalized_alignment)
    if not vary_sensitivity_levels:
        return 1.0

    low, high = sensitivity_level_range
    return float(rng.uniform(low, high))


def _normalized_axis_alignment(axis_alignment: float, dim: int) -> float:
    min_alignment = 1.0 / np.sqrt(dim)
    if dim <= 1:
        return 1.0
    normalized = (axis_alignment - min_alignment) / (1.0 - min_alignment)
    return float(np.clip(normalized, 0.0, 1.0))
