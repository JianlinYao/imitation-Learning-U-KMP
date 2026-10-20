"""Finite-horizon LQR tracking for a 2-D unit-mass double integrator.

This is a numerical control demonstration, not a physical robot safety claim.
Q_t = (predicted state covariance + variance_floor*I)^(-1) is used in the
adaptive case, following the uncertainty-aware KMP control principle.
"""
from dataclasses import dataclass

import numpy as np

from .distributions import Prediction


@dataclass(frozen=True)
class TrackingResult:
    state: np.ndarray
    acceleration: np.ndarray
    gains: np.ndarray
    disturbance: np.ndarray


def track_lqr(prediction: Prediction, t: np.ndarray, adaptive: bool = True,
              control_penalty: float = 0.001, variance_floor: float = 0.001,
              disturbance: np.ndarray | None = None) -> TrackingResult:
    if control_penalty <= 0 or variance_floor <= 0:
        raise ValueError("control penalty and variance floor must be positive")
    dt = float(t[1] - t[0])
    if not np.allclose(np.diff(t), dt) or len(prediction.state) != len(t):
        raise ValueError("LQR requires a uniform matching time grid")
    count = len(t)
    if adaptive:
        if prediction.covariance is None:
            raise ValueError("adaptive LQR requires predicted state covariance")
        covariance = prediction.covariance + variance_floor * np.eye(4)
        q = np.linalg.solve(covariance, np.broadcast_to(np.eye(4), covariance.shape))
    else:
        q = np.broadcast_to(np.diag([1.0, 1.0, 0.1, 0.1]), (count, 4, 4)).copy()
    a = np.block([[np.eye(2), dt * np.eye(2)], [np.zeros((2, 2)), np.eye(2)]])
    b = np.vstack([0.5 * dt ** 2 * np.eye(2), dt * np.eye(2)])
    r = control_penalty * np.eye(2)
    reference = prediction.state
    gains = np.empty((count - 1, 2, 4))
    offsets = np.empty((count - 1, 2))
    p_matrix = q[-1].copy()
    p_vector = -q[-1] @ reference[-1]
    for i in range(count - 2, -1, -1):
        normal = r + b.T @ p_matrix @ b
        gains[i] = np.linalg.solve(normal, b.T @ p_matrix @ a)
        offsets[i] = np.linalg.solve(normal, b.T @ p_vector)
        p_vector = -q[i] @ reference[i] + a.T @ p_vector - a.T @ p_matrix @ b @ offsets[i]
        p_matrix = q[i] + a.T @ p_matrix @ a - a.T @ p_matrix @ b @ gains[i]
        p_matrix = (p_matrix + p_matrix.T) * 0.5
    disturbance = np.zeros((count - 1, 2)) if disturbance is None else np.asarray(disturbance)
    if disturbance.shape != (count - 1, 2) or not np.isfinite(disturbance).all():
        raise ValueError("disturbance must have shape (T-1, 2) and be finite")
    state = np.empty_like(reference)
    state[0] = reference[0]
    acceleration = np.empty((count - 1, 2))
    for i in range(count - 1):
        acceleration[i] = -gains[i] @ state[i] - offsets[i]
        state[i + 1] = a @ state[i] + b @ (acceleration[i] + disturbance[i])
    if not np.isfinite(state).all():
        raise FloatingPointError("non-finite LQR rollout")
    return TrackingResult(state, acceleration, gains, disturbance.copy())
