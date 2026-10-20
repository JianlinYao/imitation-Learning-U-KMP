"""MATLAB loading and training-fold-only preprocessing.

Public arrays use (time, coordinate), unlike the historical 2-by-T notebooks.
All time derivatives use the explicitly recorded time grid, in seconds.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold


def load_letter(root: Path | str, letter: str) -> list[np.ndarray]:
    if letter not in "ABCDEFGH" or len(letter) != 1:
        raise ValueError("letter must be one of A through H")
    path = Path(root) / f"{letter}.mat"
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "demos" not in content:
        raise ValueError(f"{path} has no 'demos' array")
    result = []
    for demo in np.atleast_1d(content["demos"]).ravel():
        pos = np.asarray(demo.pos, dtype=float).squeeze()
        if pos.ndim != 2:
            raise ValueError(f"{path}: expected a 2-D position array")
        if pos.shape[0] == 2:
            pos = pos.T
        if pos.shape[1] != 2 or len(pos) < 7 or not np.isfinite(pos).all():
            raise ValueError(f"{path}: invalid trajectory shape/data {pos.shape}")
        result.append(pos.copy())
    if len(result) < 2:
        raise ValueError(f"{path}: at least two demonstrations are required")
    return result


def time_grid(points: int = 200, duration: float = 0.995) -> np.ndarray:
    if points < 11 or not np.isfinite(duration) or duration <= 0:
        raise ValueError("points must be >= 11 and duration must be positive")
    return np.linspace(0.0, duration, points)


def resample(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    if pos.ndim != 2 or pos.shape[1] != 2 or not np.isfinite(pos).all():
        raise ValueError("resample requires finite (T, 2) positions")
    old_t = np.linspace(t[0], t[-1], len(pos))
    return np.column_stack([np.interp(t, old_t, pos[:, d]) for d in range(2)])


def smooth(pos: np.ndarray, window: int = 11, order: int = 3) -> np.ndarray:
    if window == 0:
        return pos.copy()
    if window < 3 or window % 2 == 0 or order < 0 or order >= window:
        raise ValueError("smoothing needs an odd window > polynomial order (or 0)")
    usable = min(window, len(pos) if len(pos) % 2 else len(pos) - 1)
    if usable <= order:
        return pos.copy()
    return savgol_filter(pos, usable, order, axis=0, mode="interp")


def observed_segments(pos: np.ndarray, visible: np.ndarray,
                      window: int = 11, order: int = 3) -> np.ndarray:
    """Mask BEFORE filtering; neither smoothing nor derivatives bridge a gap."""
    visible = np.asarray(visible, dtype=bool)
    if visible.shape != (len(pos),):
        raise ValueError("visible must contain one flag per time step")
    result = np.full_like(pos, np.nan, dtype=float)
    edges = np.diff(np.r_[False, visible, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        values = np.asarray(pos[start:end], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("visible observations must be finite")
        result[start:end] = smooth(values, window, order)
    return result


def state_from_positions(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.column_stack([pos, np.gradient(pos, t, axis=0, edge_order=2)])


@dataclass(frozen=True)
class SpatialScaler:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training: list[np.ndarray]) -> "SpatialScaler":
        if not training:
            raise ValueError("cannot fit a scaler without training demonstrations")
        samples = np.concatenate(training)
        if not np.isfinite(samples).all():
            raise ValueError("training demonstrations must be finite")
        return cls(samples.mean(axis=0), np.maximum(samples.std(axis=0), 1e-8))

    def transform(self, pos: np.ndarray) -> np.ndarray:
        return (pos - self.mean) / self.scale

    def inverse(self, pos: np.ndarray) -> np.ndarray:
        return pos * self.scale + self.mean

    def inverse_state(self, state: np.ndarray) -> np.ndarray:
        result = state * np.tile(self.scale, 2)
        result = result.copy()
        result[:, :2] += self.mean
        return result

    def inverse_covariance(self, covariance: np.ndarray) -> np.ndarray:
        scale = np.tile(self.scale, 2)
        return covariance * scale[None, :, None] * scale[None, None, :]

    def as_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}


def folds(count: int, n_splits: int, seed: int):
    if not 2 <= n_splits <= count:
        raise ValueError("folds must be between 2 and the number of demonstrations")
    return list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.arange(count)))
