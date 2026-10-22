"""Movement primitives with a shared, training-only fit/predict interface.

KMP mean/covariance follow Huang et al. (2019) and Silvério et al. (2019).
U-KMP shares KMP's planned mean; its additional behavior is implemented and
evaluated as covariance-aware LQR in control.py, not as an invented new mean.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag, cho_solve

from .data import state_from_positions
from .distributions import GaussianTrajectory, Observation, Prediction, factor_spd
from .kernels import state_kernel
from .reference import gmr_reference

METHODS = ("DMP", "ProMP", "GP", "KMP", "U-KMP")


@dataclass(frozen=True)
class ModelConfig:
    basis_functions: int = 40
    ridge: float = 1e-5
    length_scale: float = 0.10
    gp_noise_variance: float = 0.02
    gmm_components: int = 8
    reference_points: int = 200
    kmp_mean_regularization: float = 0.1
    kmp_covariance_regularization: float = 200.0
    kmp_adaptation: str = "reference"
    kmp_reference_tolerance: float = 0.0005


class GaussianPrimitive:
    supports_via_points = True

    def __init__(self, config: ModelConfig | None = None, seed: int = 42):
        self.config = config or ModelConfig()
        self.seed = seed
        self.metadata = {}

    def predict(self, observations: list[Observation] | None = None) -> Prediction:
        return self.trajectory.predict(observations)


class DMP:
    """Discrete DMP, joint least-squares forcing fit to all training demos."""
    supports_via_points = False

    def __init__(self, config: ModelConfig | None = None, seed: int = 42):
        self.config = config or ModelConfig()
        self.seed = seed
        self.kp, self.kv, self.alpha = 150.0, 25.0, 5.0
        self.metadata = {"training": "all-demo pooled least squares", "integrator": "RK45"}

    def _basis(self, phase):
        phase = np.atleast_1d(phase)
        basis = np.exp(-self.widths * (phase[:, None] - self.centers) ** 2)
        return basis / np.maximum(basis.sum(axis=1, keepdims=True), 1e-300) * phase[:, None]

    def fit(self, training: list[np.ndarray], t: np.ndarray):
        self.t = t.copy()
        self.duration = float(t[-1] - t[0])
        count = self.config.basis_functions
        self.centers = np.exp(-self.alpha * np.linspace(0.0, 1.0, count))
        self.widths = np.r_[0.5 / (0.65 * np.diff(self.centers)) ** 2, 0.0]
        self.widths[-1] = self.widths[-2]
        phase = np.exp(-self.alpha * (t - t[0]) / self.duration)
        design = self._basis(phase)
        forces = []
        for pos in training:
            velocity = np.gradient(pos, t, axis=0, edge_order=2)
            acceleration = np.gradient(velocity, t, axis=0, edge_order=2)
            # tau^2*y_ddot = kp*(g-y) - tau*kv*y_dot + f.
            forces.append(self.duration ** 2 * acceleration - self.kp * (pos[-1] - pos)
                          + self.duration * self.kv * velocity)
        target = np.mean(forces, axis=0)
        normal = design.T @ design + self.config.ridge * np.eye(count)
        self.weights = np.linalg.solve(normal, design.T @ target)
        self.start = np.mean([pos[0] for pos in training], axis=0)
        self.goal = np.mean([pos[-1] for pos in training], axis=0)
        return self

    def predict(self, observations: list[Observation] | None = None) -> Prediction:
        start, goal = self.start.copy(), self.goal.copy()
        velocity = np.zeros(2)
        for obs in observations or []:
            for d, value in zip(obs.dimensions, obs.value):
                if obs.index == 0 and d < 2:
                    start[d] = value
                elif obs.index == len(self.t) - 1 and d < 2:
                    goal[d] = value
                elif obs.index == 0 and d >= 2:
                    velocity[d - 2] = value

        def ode(time, state):
            phase = np.exp(-self.alpha * (time - self.t[0]) / self.duration)
            force = (self._basis(phase) @ self.weights)[0]
            acceleration = (self.kp * (goal - state[:2])
                            - self.duration * self.kv * state[2:] + force) / self.duration ** 2
            return np.r_[state[2:], acceleration]

        solution = solve_ivp(ode, (self.t[0], self.t[-1]), np.r_[start, velocity],
                             t_eval=self.t, rtol=1e-7, atol=1e-9)
        if not solution.success:
            raise RuntimeError(f"DMP integration failed: {solution.message}")
        return Prediction(solution.y.T, None)


class ProMP(GaussianPrimitive):
    """Gaussian weight distribution with analytic position/velocity basis."""

    def fit(self, training: list[np.ndarray], t: np.ndarray):
        self.t = t.copy()
        count = self.config.basis_functions
        duration = t[-1] - t[0]
        phase = (t - t[0]) / duration
        centers = np.linspace(0, 1, count)
        width = 1.5 / (count - 1)
        raw = np.exp(-0.5 * ((phase[:, None] - centers) / width) ** 2)
        derivative = -(phase[:, None] - centers) / (width ** 2 * duration) * raw
        total = raw.sum(axis=1, keepdims=True)
        basis = raw / total
        d_basis = (derivative * total - raw * derivative.sum(axis=1, keepdims=True)) / total ** 2
        design = np.zeros((len(t), 4, 2 * count))
        for d in range(2):
            design[:, d, d * count:(d + 1) * count] = basis
            design[:, d + 2, d * count:(d + 1) * count] = d_basis
        design = design.reshape(4 * len(t), 2 * count)
        # Balance velocity rows (units 1/s) with position rows explicitly.
        row_weights = np.tile([1.0, 1.0, 0.1 * duration, 0.1 * duration], len(t))
        weighted = design * row_weights[:, None]
        factor = factor_spd(weighted.T @ weighted + self.config.ridge * np.eye(2 * count))
        targets = np.column_stack([state_from_positions(pos, t).ravel() for pos in training])
        weights = cho_solve(factor, weighted.T @ (targets * row_weights[:, None])).T
        mean_weights = weights.mean(axis=0)
        centered = weights - mean_weights
        cov_weights = centered.T @ centered / max(len(training) - 1, 1)
        cov_weights += self.config.ridge * np.eye(2 * count)
        covariance = design @ cov_weights @ design.T
        covariance += 1e-9 * np.eye(len(covariance))
        self.trajectory = GaussianTrajectory((design @ mean_weights).reshape(-1, 4), covariance)
        self.metadata = {"training": "Gaussian over all-demo basis weights", "basis_functions": count,
                         "velocity_row_weight": 0.1 * float(duration)}
        return self


class GP(GaussianPrimitive):
    """Independent-coordinate RBF GP with exact compression of time replicates.

    All demos share a time grid and IID noise. Their sample mean with noise/H is
    mathematically equivalent to conditioning on all H*T training observations.
    Hyperparameters are fixed by configuration, not optimized on held-out data.
    """

    def fit(self, training: list[np.ndarray], t: np.ndarray):
        self.t = t.copy()
        kernel = state_kernel(t, t, self.config.length_scale)
        indices = (4 * np.arange(len(t))[:, None] + np.arange(2)).ravel()
        noise = self.config.gp_noise_variance / len(training)
        factor = factor_spd(kernel[np.ix_(indices, indices)] + noise * np.eye(len(indices)))
        cross = kernel[:, indices]
        target = np.mean(training, axis=0).ravel()
        mean = cross @ cho_solve(factor, target)
        covariance = kernel - cross @ cho_solve(factor, cross.T)
        covariance += 1e-9 * np.eye(len(covariance))
        self.trajectory = GaussianTrajectory(mean.reshape(-1, 4), covariance)
        self.metadata = {"training": "exact replicate compression of all training demos",
                         "noise_variance_per_demo": self.config.gp_noise_variance,
                         "effective_noise_variance": noise, "hyperparameter_optimization": False}
        return self

    def predict(self, observations=None):
        latent = super().predict(observations)
        return Prediction(latent.state, latent.covariance, self.config.gp_noise_variance)


class KMP(GaussianPrimitive):
    def fit(self, training: list[np.ndarray], t: np.ndarray):
        self.t = t.copy()
        config = self.config
        ref_t = np.linspace(t[0], t[-1], min(config.reference_points, len(t)))
        mean, covariance, metadata = gmr_reference(
            training, t, ref_t, config.gmm_components, self.seed, config.ridge,
        )
        kernel = state_kernel(ref_t, ref_t, config.length_scale)
        sigma = block_diag(*covariance)
        mean_factor = factor_spd(kernel + config.kmp_mean_regularization * sigma)
        cov_factor = factor_spd(kernel + config.kmp_covariance_regularization * sigma)
        cross = state_kernel(t, ref_t, config.length_scale)
        predicted_mean = cross @ cho_solve(mean_factor, mean.ravel())
        scale = len(ref_t) / config.kmp_covariance_regularization
        predicted_cov = scale * (state_kernel(t, t, config.length_scale)
                                 - cross @ cho_solve(cov_factor, cross.T))
        predicted_cov += 1e-9 * np.eye(len(predicted_cov))
        self.trajectory = GaussianTrajectory(predicted_mean.reshape(-1, 4), predicted_cov)
        self.ref_t, self.ref_mean, self.ref_covariance = ref_t, mean, covariance
        self._cov_factor = cov_factor
        self.metadata = metadata | {"covariance_scale": scale, "state_order": ["px", "py", "vx", "vy"],
                                    "conditioning": config.kmp_adaptation}
        return self

    def predict(self, observations=None):
        if observations and self.config.kmp_adaptation == "reference":
            from .kmp_adaptation import reference_prediction
            return reference_prediction(self.ref_t, self.ref_mean, self.ref_covariance,
                                        self.t, observations, self.config)
        return super().predict(observations)

    def positional_variance(self, query_t: np.ndarray) -> np.ndarray:
        cross = state_kernel(query_t, self.ref_t, self.config.length_scale)
        prior = state_kernel(query_t, query_t, self.config.length_scale)
        covariance = len(self.ref_t) / self.config.kmp_covariance_regularization * (
            prior - cross @ cho_solve(self._cov_factor, cross.T)
        )
        return np.diagonal(covariance).reshape(-1, 4)[:, :2]


class UncertaintyAwareKMP(KMP):
    """Same trajectory distribution as KMP; adds covariance-aware tracking."""

    def track(self, prediction: Prediction, **kwargs):
        from .control import track_lqr
        return track_lqr(prediction, self.t, adaptive=True, **kwargs)


def make_model(method: str, config: ModelConfig | None = None, seed: int = 42):
    classes = {"DMP": DMP, "ProMP": ProMP, "GP": GP, "KMP": KMP, "U-KMP": UncertaintyAwareKMP}
    if method not in classes:
        raise ValueError(f"unknown method {method!r}; choose {', '.join(METHODS)}")
    return classes[method](config, seed)
