from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Item:
    """Ordinal question/item under the graded response model."""

    item_id: str
    a: NDArray[np.float64]
    thresholds: NDArray[np.float64]
    sensitivity: float = 0.0

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
    if not np.isfinite(self.sensitivity) or self.sensitivity < 0:
        raise ValueError("sensitivity must be finite and nonnegative.")

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