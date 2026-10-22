"""Marginal predictive scores and training-only empirical interval inflation.

Calibration examples are whole demonstrations excluded from model/scaler fit.
Pooled time/coordinate residuals are dependent; the empirical quantile is NOT
a distribution-free, trajectory-level conformal coverage guarantee.
"""
import numpy as np

Z95 = 1.959963984540054


def position_covariance(covariance, position_noise=0.0):
    if covariance is None:
        return None
    result = np.asarray(covariance, dtype=float)[:, :2, :2].copy()
    noise = np.broadcast_to(np.asarray(position_noise, dtype=float), (2,))
    if not np.isfinite(result).all() or not np.isfinite(noise).all() or np.any(noise < 0):
        raise ValueError("predictive covariance/noise must be finite and noise nonnegative")
    result[:, np.arange(2), np.arange(2)] += noise
    return (result + result.transpose(0, 2, 1)) * 0.5


def standardized_residuals(truth, prediction, mask=None):
    covariance = position_covariance(prediction.covariance, prediction.position_noise_variance)
    if covariance is None:
        return np.empty(0)
    variance = np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 1e-12)
    scores = np.abs(np.asarray(truth) - prediction.position) / np.sqrt(variance)
    return scores.ravel() if mask is None else scores[mask].ravel()


def interval_inflation(scores, level=0.95):
    scores = np.asarray(scores, dtype=float)
    if not 0 < level < 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("interval calibration requires finite held-out residuals and a valid level")
    from scipy.stats import norm
    z = float(norm.ppf((1 + level) / 2))
    quantile = float(np.quantile(scores, level, method="higher"))
    return {"scale": max(1.0, quantile / z), "level": level, "scores": len(scores),
            "quantile": quantile, "raw_calibration_coverage": float((scores <= z).mean()),
            "scope": "empirical marginal position bands; no trajectory-level coverage guarantee"}


def gaussian_scores(error, covariance):
    """Per-time bivariate position NLPD; joint coverage is a 2-D ellipse."""
    if np.linalg.eigvalsh(covariance).min() < -1e-6:
        raise ValueError("position covariance is not positive semidefinite")
    covariance = covariance + 1e-12 * np.maximum(
        np.max(np.abs(np.diagonal(covariance, axis1=1, axis2=2)), axis=1), 1.0)[:, None, None] * np.eye(2)
    sign, logdet = np.linalg.slogdet(covariance)
    if np.any(sign <= 0):
        raise ValueError("position covariance must be positive definite for scoring")
    mahalanobis = np.einsum("ti,ti->t", error, np.linalg.solve(covariance, error[..., None])[..., 0])
    nlpd = 0.5 * (2 * np.log(2 * np.pi) + logdet + mahalanobis)
    return nlpd, mahalanobis <= 5.991464547107979
