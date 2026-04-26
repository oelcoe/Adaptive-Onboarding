from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BASE_OBSERVATION_NOISE_VARIANCE = 1.0


@dataclass(frozen=True)
class Item:
    """Ordinal question/item under the graded response model."""

    item_id: str
    a: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    response_noise_variance: float = BASE_OBSERVATION_NOISE_VARIANCE
    is_sensitive: bool = False
    sensitivity_level: float = 0.0

    def __post_init__(self) -> None:
        if not self.item_id or not self.item_id.strip():
            raise ValueError("item_id must be a non-empty string.")

        a = np.asarray(self.a, dtype=float).reshape(-1)
        thr = np.asarray(self.thresholds, dtype=float).reshape(-1)

        if a.size == 0:
            raise ValueError("a must be non-empty.")
        if np.linalg.norm(a) <= 1e-12:
            raise ValueError("a must not be numerically zero.")

        if thr.size == 0:
            raise ValueError("thresholds must be non-empty (at least two categories).")
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(thr)):
            raise ValueError("a and thresholds must be finite.")
        if np.any(np.diff(thr) <= 0):
            raise ValueError("thresholds must be strictly increasing.")
        if not np.isfinite(self.response_noise_variance) or self.response_noise_variance <= 0:
            raise ValueError("response_noise_variance must be finite and strictly positive.")
        if not np.isfinite(self.sensitivity_level) or self.sensitivity_level < 0:
            raise ValueError("sensitivity_level must be finite and nonnegative.")

        object.__setattr__(self, "a", a)
        object.__setattr__(self, "thresholds", thr)

    @property
    def dim(self) -> int:
        return self.a.shape[0]

    @property
    def n_thresholds(self) -> int:
        return self.thresholds.shape[0]

    @property
    def n_categories(self) -> int:
        return self.n_thresholds + 1

    @property
    def observation_noise_variance(self) -> float:
        """
        Observation-noise variance used by the ordinal-probit likelihood.

        Defaults to 1.0. At simulation time, simulate_episode inflates this
        for sensitive items when sensitivity_noise_scale > 0 by creating
        effective copies with adjusted response_noise_variance. Both the
        generative model (sample_response) and the inference model
        (update_belief) then see the same inflated value.
        """
        return float(self.response_noise_variance)
