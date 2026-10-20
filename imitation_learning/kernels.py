"""Analytic RBF position/velocity kernel, ordered by time then state dimension."""
import numpy as np


def state_kernel(a: np.ndarray, b: np.ndarray, length_scale: float,
                 amplitude: float = 1.0) -> np.ndarray:
    if length_scale <= 0 or amplitude <= 0:
        raise ValueError("kernel length and amplitude must be positive")
    difference = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    variance = length_scale ** 2
    kernel = amplitude * np.exp(-0.5 * difference ** 2 / variance)
    d_b = difference / variance * kernel
    d_a = -d_b
    d_ab = (1.0 / variance - difference ** 2 / variance ** 2) * kernel
    result = np.zeros((len(a), 4, len(b), 4))
    for d in range(2):
        result[:, d, :, d] = kernel
        result[:, d, :, d + 2] = d_b
        result[:, d + 2, :, d] = d_a
        result[:, d + 2, :, d + 2] = d_ab
    return result.reshape(4 * len(a), 4 * len(b))
