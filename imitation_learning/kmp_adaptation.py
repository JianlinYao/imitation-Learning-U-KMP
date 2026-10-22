"""Reference-point replace/append KMP, with partial-state observations.

For full-state constraints this is the authors' reference database update.
Partial constraints replace a site's observations with only the specified
components, rather than inventing velocity targets. Mean and covariance use
their own regularizers even AFTER adaptation.
"""
import numpy as np
from scipy.linalg import block_diag, cho_solve

from .distributions import Prediction, factor_spd, observation_entries
from .kernels import state_kernel


def reference_prediction(ref_t, ref_mean, ref_covariance, query_t, observations, config):
    entries = observation_entries(observations, len(query_t))
    if not entries:
        raise ValueError("reference adaptation requires observations")
    grouped = {}
    for flat, value in sorted(entries.items()):
        grouped.setdefault(flat // 4, {})[flat % 4] = value
    sites = [(float(t), tuple(range(4)), mu.copy(), cov.copy())
             for t, mu, cov in zip(ref_t, ref_mean, ref_covariance)]
    for index, values in grouped.items():
        time = float(query_t[index])
        dims = tuple(sorted(values))
        replacement = (time, dims, np.array([values[d][0] for d in dims]),
                       np.diag([values[d][1] for d in dims]))
        nearest = int(np.argmin(np.abs(np.asarray([s[0] for s in sites]) - time)))
        if abs(sites[nearest][0] - time) < config.kmp_reference_tolerance:
            sites[nearest] = replacement
        else:
            sites.append(replacement)
    sites.sort(key=lambda s: s[0])
    times = np.array([s[0] for s in sites])
    selected = np.array([4 * i + d for i, s in enumerate(sites) for d in s[1]], dtype=int)
    target = np.concatenate([s[2] for s in sites])
    sigma = block_diag(*(s[3] for s in sites))
    kernel = state_kernel(times, times, config.length_scale)[np.ix_(selected, selected)]
    cross = state_kernel(query_t, times, config.length_scale)[:, selected]
    mean_factor = factor_spd(kernel + config.kmp_mean_regularization * sigma)
    cov_factor = factor_spd(kernel + config.kmp_covariance_regularization * sigma)
    mean = cross @ cho_solve(mean_factor, target)
    solved = cho_solve(cov_factor, cross.T).T.reshape(len(query_t), 4, -1)
    blocks = cross.reshape(len(query_t), 4, -1)
    prior = state_kernel(np.array([0.0]), np.array([0.0]), config.length_scale)
    covariance = len(sites) / config.kmp_covariance_regularization * (
        prior[None, :, :] - np.einsum("tik,tjk->tij", blocks, solved))
    covariance = (covariance + covariance.transpose(0, 2, 1)) * 0.5
    covariance += 1e-9 * np.eye(4)
    if np.linalg.eigvalsh(covariance).min() < -1e-6:
        raise FloatingPointError("reference adaptation produced an invalid covariance")
    return Prediction(mean.reshape(-1, 4), covariance)
