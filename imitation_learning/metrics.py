"""Metrics on physical-time, dataset-coordinate trajectories; no outlier removal."""
import numpy as np

from .uncertainty import Z95, gaussian_scores, position_covariance


def trajectory_metrics(truth: np.ndarray, predicted: np.ndarray, t: np.ndarray,
                       gap: np.ndarray | None = None, covariance: np.ndarray | None = None,
                       position_noise=0.0, calibration_scale: float | None = None) -> dict:
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
        "nlpd": np.nan, "gap_nlpd": np.nan, "joint_coverage95": np.nan,
        "interval_width95": np.nan, "calibrated_coverage95": np.nan,
        "calibrated_gap_coverage95": np.nan, "calibrated_interval_width95": np.nan,
    }
    if covariance is not None:
        position_cov = position_covariance(covariance, position_noise)
        variance = np.maximum(np.diagonal(position_cov, axis1=1, axis2=2), 0.0)
        std = np.sqrt(variance)
        covered = np.abs(error) <= Z95 * std
        nlpd, joint = gaussian_scores(error, position_cov)
        result.update(coverage95=float(covered.mean()), mean_position_variance=float(variance.mean()),
                      nlpd=float(nlpd.mean()), joint_coverage95=float(joint.mean()),
                      interval_width95=float((2 * Z95 * std).mean()))
        if gap is not None and gap.any():
            result["gap_coverage95"] = float(covered[gap].mean())
            result["gap_nlpd"] = float(nlpd[gap].mean())
        if calibration_scale is not None:
            if not np.isfinite(calibration_scale) or calibration_scale < 1:
                raise ValueError("empirical interval inflation must be finite and >= 1")
            calibrated = np.abs(error) <= Z95 * std * calibration_scale
            result.update(calibrated_coverage95=float(calibrated.mean()),
                          calibrated_interval_width95=float((2 * Z95 * std * calibration_scale).mean()))
            if gap is not None and gap.any():
                result["calibrated_gap_coverage95"] = float(calibrated[gap].mean())
    return result
