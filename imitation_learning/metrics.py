"""Metrics on physical-time, dataset-coordinate trajectories; no outlier removal."""
import numpy as np


def trajectory_metrics(truth: np.ndarray, predicted: np.ndarray, t: np.ndarray,
                       gap: np.ndarray | None = None, covariance: np.ndarray | None = None) -> dict:
    if truth.shape != predicted.shape or truth.shape != (len(t), 2):
        raise ValueError("truth and prediction must both be (T, 2) on the same time grid")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("metrics require finite positions")
    error = predicted - truth
    derivative = predicted.copy()
    for _ in range(3):
        derivative = np.gradient(derivative, t, axis=0, edge_order=2)
    result = {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "end_err": float(np.linalg.norm(error[-1])),
        "jerk": float(np.mean(np.linalg.norm(derivative, axis=1))),
        "gap_rmse": float(np.sqrt(np.mean(error[gap] ** 2))) if gap is not None and gap.any() else np.nan,
        "coverage95": np.nan, "gap_coverage95": np.nan, "mean_position_variance": np.nan,
    }
    if covariance is not None:
        variance = np.maximum(np.diagonal(covariance[:, :2, :2], axis1=1, axis2=2), 0.0)
        covered = np.abs(error) <= 1.96 * np.sqrt(variance)
        result.update(coverage95=float(covered.mean()), mean_position_variance=float(variance.mean()))
        if gap is not None and gap.any():
            result["gap_coverage95"] = float(covered[gap].mean())
    return result
