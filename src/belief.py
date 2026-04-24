from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _is_spd(Sigma: NDArray[np.float64], rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    # check if the covariance matrix is symmetric positive definite
    # return True if the covariance matrix is symmetric positive definite, False otherwise
    # rtol: relative tolerance
    # atol: absolute tolerance
    return np.allclose(Sigma, Sigma.T, rtol=rtol, atol=atol) and np.all(np.linalg.eigvals(Sigma) > 0)


@dataclass(frozen=True)
class BeliefState:
    """
    Current Gaussian belief over the latent user state.
    mu: mean of the belief
    Sigma: covariance matrix of the belief
    """

    mu: NDArray[np.float64]
    Sigma: NDArray[np.float64]

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=float).reshape(-1)
        Sigma = np.asarray(self.Sigma, dtype=float)

        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError("Sigma must be a square matrix.")
        if mu.shape[0] != Sigma.shape[0]:
            raise ValueError("mu length must match Sigma dimension.")
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(Sigma)):
            raise ValueError("mu and Sigma must be finite.")

        Sigma = 0.5 * (Sigma + Sigma.T)

        if not _is_spd(Sigma):
            raise ValueError("Sigma must be symmetric positive definite.")

        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "Sigma", Sigma)

    @property
    def dim(self) -> int:
        return self.mu.shape[0]