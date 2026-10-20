"""A joint Gaussian over [px, py, vx, vy] on a fixed time grid."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def factor_spd(matrix: np.ndarray):
    matrix = (matrix + matrix.T) * 0.5
    scale = max(float(np.max(np.abs(np.diag(matrix)))), 1.0)
    eye = np.eye(len(matrix))
    for jitter in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        try:
            return cho_factor(matrix + jitter * scale * eye, lower=True, check_finite=True)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("covariance is not positive definite after bounded jitter")


def marginal_blocks(joint: np.ndarray) -> np.ndarray:
    count = len(joint) // 4
    block = joint.reshape(count, 4, count, 4)
    return block[np.arange(count), :, np.arange(count), :].copy()


@dataclass(frozen=True)
class Prediction:
    state: np.ndarray
    covariance: np.ndarray | None

    @property
    def position(self) -> np.ndarray:
        return self.state[:, :2]


@dataclass(frozen=True)
class Observation:
    index: int
    value: tuple[float, ...]
    dimensions: tuple[int, ...] = (0, 1)
    variance: float = 1e-5


class GaussianTrajectory:
    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        self.mean = np.asarray(mean, dtype=float)
        if self.mean.ndim != 2 or self.mean.shape[1] != 4:
            raise ValueError("mean must have shape (T, 4)")
        expected = (self.mean.size, self.mean.size)
        if covariance.shape != expected or not np.isfinite(covariance).all():
            raise ValueError(f"joint covariance must be finite with shape {expected}")
        self.covariance = (covariance + covariance.T) * 0.5
        self.marginals = marginal_blocks(self.covariance)

    def predict(self, observations: list[Observation] | None = None) -> Prediction:
        """Condition once on all observations, without touching the fitted prior.

        Repeated position constraints at the same index are replaced by the last
        value, rather than accidentally counted as independent measurements.
        """
        if not observations:
            return Prediction(self.mean.copy(), self.marginals.copy())
        entries = {}
        for obs in observations:
            if not 0 <= obs.index < len(self.mean):
                raise ValueError("observation index is outside the prediction grid")
            if len(obs.value) != len(obs.dimensions) or not obs.dimensions:
                raise ValueError("one value is required for each observed dimension")
            if not np.isfinite(obs.variance) or obs.variance <= 0:
                raise ValueError("observation variance must be positive and finite")
            for d, value in zip(obs.dimensions, obs.value):
                if d not in (0, 1, 2, 3) or not np.isfinite(value):
                    raise ValueError("invalid observed dimension/value")
                entries[4 * obs.index + d] = (value, obs.variance)
        indices = np.array(sorted(entries), dtype=int)
        values = np.array([entries[i][0] for i in indices])
        noise = np.array([entries[i][1] for i in indices])
        cross = self.covariance[:, indices]
        observed_cov = self.covariance[np.ix_(indices, indices)] + np.diag(noise)
        factor = factor_spd(observed_cov)
        residual = values - self.mean.ravel()[indices]
        mean = self.mean.ravel() + cross @ cho_solve(factor, residual)
        solved = cho_solve(factor, cross.T).T.reshape(len(self.mean), 4, -1)
        cross_blocks = cross.reshape(len(self.mean), 4, -1)
        covariance = self.marginals - np.einsum("tik,tjk->tij", cross_blocks, solved)
        covariance = (covariance + covariance.transpose(0, 2, 1)) * 0.5
        # Only roundoff-size diagonal negatives are clipped; larger problems fail.
        diagonal = np.diagonal(covariance, axis1=1, axis2=2)
        if diagonal.min() < -1e-6:
            raise FloatingPointError("conditioning produced a negative variance")
        for d in range(4):
            covariance[:, d, d] = np.maximum(covariance[:, d, d], 0.0)
        return Prediction(mean.reshape(-1, 4), covariance)
