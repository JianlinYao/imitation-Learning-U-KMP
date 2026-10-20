"""Training-only GMM/GMR reference construction; no precomputed ref*.mat input."""
import warnings

import numpy as np
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture

from .data import state_from_positions


def gmr_reference(training: list[np.ndarray], t: np.ndarray, query_t: np.ndarray,
                  components: int = 8, seed: int = 42, regularization: float = 1e-5):
    duration = t[-1] - t[0]
    joint = np.concatenate([
        np.column_stack([(t - t[0]) / duration, state_from_positions(pos, t)])
        for pos in training
    ])
    center = joint.mean(axis=0)
    scale = np.maximum(joint.std(axis=0), 1e-8)
    standardized = (joint - center) / scale
    mixture = GaussianMixture(
        n_components=components, covariance_type="full", reg_covar=regularization,
        max_iter=400, n_init=1, random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        mixture.fit(standardized)
    if not mixture.converged_ or any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("GMM did not converge; adjust training parameters rather than hide the failure")
    query = (((query_t - t[0]) / duration) - center[0]) / scale[0]
    log_weights, conditional_means, conditional_covariances = [], [], []
    for prior, mu, covariance in zip(mixture.weights_, mixture.means_, mixture.covariances_):
        variance = covariance[0, 0]
        delta = query - mu[0]
        log_weights.append(np.log(prior) - 0.5 * (np.log(2 * np.pi * variance) + delta ** 2 / variance))
        gain = covariance[1:, 0] / variance
        conditional_means.append(mu[1:][None, :] + delta[:, None] * gain)
        conditional_covariances.append(covariance[1:, 1:] - np.outer(gain, covariance[0, 1:]))
    log_weights = np.array(log_weights).T
    weights = np.exp(log_weights - logsumexp(log_weights, axis=1, keepdims=True))
    means = np.stack(conditional_means, axis=1)
    mean = np.einsum("tk,tkd->td", weights, means)
    covariance = np.zeros((len(query_t), 4, 4))
    for k in range(components):
        delta = means[:, k] - mean
        covariance += weights[:, k, None, None] * (
            conditional_covariances[k] + delta[:, :, None] * delta[:, None, :]
        )
    mean = mean * scale[1:] + center[1:]
    covariance *= scale[None, 1:, None] * scale[None, None, 1:]
    covariance += regularization * np.eye(4)[None, :, :]
    metadata = {
        "source": "training-fold GMM/GMR", "components": components,
        "gmm_converged": bool(mixture.converged_), "gmm_iterations": int(mixture.n_iter_),
        "gmm_seed": seed, "standardization_mean": center.tolist(),
        "standardization_scale": scale.tolist(), "reference_points": len(query_t),
    }
    return mean, covariance, metadata
