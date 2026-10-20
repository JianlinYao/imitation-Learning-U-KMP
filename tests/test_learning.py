"""Numerical and protocol tests; run with python -m unittest discover -s tests -v."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve
from threadpoolctl import threadpool_limits

from imitation_learning.control import track_lqr
from imitation_learning.data import SpatialScaler, folds, load_letter, observed_segments, resample, smooth, time_grid
from imitation_learning.distributions import Observation, Prediction, factor_spd
from imitation_learning.experiments import RunConfig, aggregate_results, case_observations, run_benchmark
from imitation_learning.geometry import Circle, clearance_metrics, detour_points
from imitation_learning.kernels import state_kernel
from imitation_learning.metrics import trajectory_metrics
from imitation_learning.models import GP, METHODS, ModelConfig, make_model

ROOT = Path(__file__).resolve().parents[1]


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_limit = threadpool_limits(limits=1)
        cls.t = time_grid(50)
        phase = cls.t / cls.t[-1]
        cls.training = [np.column_stack([np.sin(np.pi * phase) + shift * phase,
                                         phase + shift * np.sin(np.pi * phase)])
                        for shift in (-0.04, -0.02, 0.0, 0.02, 0.04)]
        cls.config = ModelConfig(basis_functions=25, reference_points=50, gmm_components=4,
                                kmp_covariance_regularization=50)

    @classmethod
    def tearDownClass(cls):
        cls.thread_limit.restore_original_limits()

    def test_kernel_matches_derivatives_and_is_positive_semidefinite(self):
        a, b, length, epsilon = 0.3, 0.19, 0.13, 1e-5
        def kernel(x, y):
            return np.exp(-0.5 * (x - y) ** 2 / length ** 2)
        result = state_kernel(np.array([a]), np.array([b]), length)
        derivative = (kernel(a, b + epsilon) - kernel(a, b - epsilon)) / (2 * epsilon)
        mixed = (kernel(a + epsilon, b + epsilon) - kernel(a + epsilon, b - epsilon)
                 - kernel(a - epsilon, b + epsilon) + kernel(a - epsilon, b - epsilon)) / (4 * epsilon ** 2)
        self.assertAlmostEqual(result[0, 2], derivative, places=5)
        self.assertAlmostEqual(result[2, 2], mixed, places=4)
        matrix = state_kernel(self.t[::3], self.t[::3], length)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(matrix).min(), -1e-8)

    def test_dmp_learns_correct_forcing_sign(self):
        model = make_model("DMP", self.config).fit(self.training, self.t)
        prediction = model.predict()
        expected = np.mean(self.training, axis=0)
        self.assertLess(np.sqrt(np.mean((prediction.position - expected) ** 2)), 0.04)
        self.assertIsNone(prediction.covariance)

    def test_gaussian_primitives_condition_positions_and_have_valid_covariance(self):
        for method in METHODS[1:]:
            with self.subTest(method=method):
                model = make_model(method, self.config).fit(self.training, self.t)
                prior = model.predict()
                desired = prior.position[-1] + [0.2, -0.15]
                conditioned = model.predict([Observation(len(self.t) - 1, tuple(desired), variance=1e-8)])
                self.assertLess(np.linalg.norm(conditioned.position[-1] - desired), 0.003)
                self.assertTrue(np.isfinite(conditioned.state).all())
                self.assertGreater(np.linalg.eigvalsh(conditioned.covariance).min(), -1e-7)
                # Predictions never mutate the fitted prior.
                np.testing.assert_array_equal(model.predict().state, prior.state)

    def test_gp_replication_compression_is_exact(self):
        t = self.t[::4]
        training = [p[::4] for p in self.training[:3]]
        model = GP(self.config).fit(training, t)
        kernel = state_kernel(t, t, self.config.length_scale)
        indices = (4 * np.arange(len(t))[:, None] + np.arange(2)).ravel()
        pp = kernel[np.ix_(indices, indices)]
        h = len(training)
        full = np.tile(pp, (h, h)) + self.config.gp_noise_variance * np.eye(h * len(indices))
        cross = np.tile(kernel[:, indices], (1, h))
        factor = factor_spd(full)
        target = np.concatenate([p.ravel() for p in training])
        expected = cross @ cho_solve(factor, target)
        expected_covariance = kernel - cross @ cho_solve(factor, cross.T)
        np.testing.assert_allclose(model.predict().state.ravel(), expected, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(model.trajectory.covariance, expected_covariance, rtol=1e-7, atol=2e-8)

    def test_kmp_and_ukmp_share_planned_distribution(self):
        ordinary = make_model("KMP", self.config).fit(self.training, self.t)
        aware = make_model("U-KMP", self.config).fit(self.training, self.t)
        np.testing.assert_allclose(ordinary.predict().state, aware.predict().state, atol=1e-12)
        np.testing.assert_allclose(ordinary.predict().covariance, aware.predict().covariance, atol=1e-12)

    def test_kmp_uncertainty_increases_far_outside_training_support(self):
        model = make_model("KMP", self.config).fit(self.training, self.t)
        query = np.array([self.t[len(self.t) // 2], 10.0])
        variance = model.positional_variance(query)
        self.assertGreater(variance[1].mean(), variance[0].mean() * 2)
        np.testing.assert_allclose(variance[1], len(model.ref_t) / self.config.kmp_covariance_regularization,
                                   atol=1e-8)

    def test_uncertain_lqr_has_lower_control_gains(self):
        mean = np.zeros((len(self.t), 4))
        low = Prediction(mean, np.broadcast_to(0.001 * np.eye(4), (len(self.t), 4, 4)).copy())
        high = Prediction(mean, np.broadcast_to(10.0 * np.eye(4), (len(self.t), 4, 4)).copy())
        certain = track_lqr(low, self.t)
        uncertain = track_lqr(high, self.t)
        self.assertGreater(np.linalg.norm(certain.gains[0]), np.linalg.norm(uncertain.gains[0]))
        np.testing.assert_allclose(track_lqr(low, self.t, adaptive=False).gains,
                                   track_lqr(high, self.t, adaptive=False).gains)

    def test_jerk_is_the_third_time_derivative(self):
        t = np.linspace(0, 2, 501)
        path = np.column_stack([t ** 3, np.zeros_like(t)])
        metrics = trajectory_metrics(path, path, t)
        self.assertEqual(metrics["rmse"], 0)
        self.assertEqual(metrics["end_err"], 0)
        self.assertAlmostEqual(metrics["jerk"], 6.0, delta=0.08)


class ProtocolTests(unittest.TestCase):
    def test_actual_dataset_and_complete_fold_partition(self):
        counts = [len(load_letter(ROOT / "2Dletters", letter)) for letter in "ABCDEFGH"]
        self.assertEqual(counts, [10, 12, 14, 14, 15, 14, 13, 13])
        partitions = folds(13, 5, 42)
        np.testing.assert_array_equal(np.sort(np.concatenate([test for _, test in partitions])), np.arange(13))
        for train, test in partitions:
            self.assertFalse(set(train) & set(test))

    def test_masked_values_cannot_affect_observations_or_predictions(self):
        t = time_grid(50)
        raw = [resample(p, t) for p in load_letter(ROOT / "2Dletters", "G")]
        training_raw = [smooth(p) for p in raw[1:]]
        scaler = SpatialScaler.fit(training_raw)
        config = RunConfig(points=50, observed_points=16,
                           model=ModelConfig(reference_points=50, gmm_components=3))
        with threadpool_limits(limits=1):
            model = make_model("GP", config.model).fit([scaler.transform(p) for p in training_raw], t)
            truth = raw[0]
            altered = truth.copy()
            hidden = (t >= 0.4 * t[-1]) & (t < 0.6 * t[-1])
            altered[hidden] = 1e12
            reference = np.mean(training_raw, axis=0)
            first, gap, _, _ = case_observations("missing", truth, t, scaler, reference, model, config)
            second, _, _, _ = case_observations("missing", altered, t, scaler, reference, model, config)
            self.assertEqual(first, second)
            self.assertTrue(all(not gap[o.index] for o in first))
            np.testing.assert_array_equal(model.predict(first).state, model.predict(second).state)
            masked = truth.copy()
            masked[hidden] = np.nan
            filtered = observed_segments(masked, ~hidden)
            self.assertTrue(np.isnan(filtered[hidden]).all())
            self.assertTrue(np.isfinite(filtered[~hidden]).all())

    def test_scaler_is_fitted_only_to_training_data(self):
        samples = [np.array([[0., 1.], [2., 3.], [4., 5.]])]
        scaler = SpatialScaler.fit(samples)
        np.testing.assert_array_equal(scaler.mean, [2., 3.])
        extreme_test = np.full((11, 2), 1e12)
        np.testing.assert_allclose(scaler.inverse(scaler.transform(extreme_test)), extreme_test)
        np.testing.assert_array_equal(scaler.mean, [2., 3.])

    def test_continuous_collision_including_degenerate_segments(self):
        circle = Circle(np.array([0., 0.]), 1., 0.1)
        # Endpoints outside, but segment crosses the obstacle.
        self.assertTrue(clearance_metrics(np.array([[-2., 0.], [2., 0.]]), circle)["collision"])
        # Entire segment inside; an edge-intersection-only check misses this.
        self.assertTrue(clearance_metrics(np.array([[0., 0.], [0., 0.]]), circle)["collision"])
        self.assertFalse(clearance_metrics(np.array([[2., 2.], [2., 2.]]), circle)["collision"])
        self.assertTrue(clearance_metrics(np.array([[-2., 1.], [2., 1.]]), circle)["collision"])

    def test_detour_keeps_distant_parts_of_revisited_loop(self):
        t = np.linspace(0, 1, 201)
        path = np.column_stack([3 * np.sin(2 * np.pi * t), np.zeros_like(t)])
        circle = Circle(np.array([1., 0.]), 0.2, 0.05)
        via = detour_points(path, circle)
        self.assertTrue(via)
        # The path visits x=1 twice. It must not create constraints across the
        # distant loop joining those visits, or project safe anchors elsewhere.
        self.assertTrue(all(np.linalg.norm(path[i] - circle.center) < 2 for i, _ in via))
        for index, point in via:
            if np.linalg.norm(path[index] - circle.center) >= 1.35 * (circle.radius + circle.margin):
                np.testing.assert_array_equal(point, path[index])

    def test_aggregation_keeps_outliers_and_failed_trial_counts(self):
        rows = pd.DataFrame([
            {"scenario": "missing", "method": "GP", "status": "ok", "rmse": 1.},
            {"scenario": "missing", "method": "GP", "status": "ok", "rmse": 1e8},
            {"scenario": "missing", "method": "GP", "status": "error", "rmse": np.nan},
        ])
        summary = aggregate_results(rows).iloc[0]
        self.assertEqual(summary["total_trials"], 3)
        self.assertEqual(summary["valid_trials"], 2)
        self.assertEqual(summary["failed_trials"], 1)
        self.assertEqual(summary["rmse_mean"], (1 + 1e8) / 2)

    def test_invalid_configuration_fails_before_any_run(self):
        for config in (RunConfig(missing_start=.8, missing_end=.2), RunConfig(methods=("FAKE",)),
                       RunConfig(timing_repeats=0), RunConfig(letters=("G", "G"))):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()


class EndToEndTests(unittest.TestCase):
    def test_all_methods_tasks_and_reproducible_rerun_without_precomputed_reference(self):
        config = RunConfig(letters=("G",), points=40, folds=5, fold_limit=1,
                           observed_points=12, timing_repeats=1, obstacle_repair_steps=2,
                           model=ModelConfig(basis_functions=18, reference_points=40,
                                             gmm_components=3, kmp_covariance_regularization=40))
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            data = base / "data"
            data.mkdir()
            shutil.copyfile(ROOT / "2Dletters" / "G.mat", data / "G.mat")
            first = run_benchmark(config, data, base / "first")
            second = run_benchmark(replace(config, save_predictions=False), data, base / "second")
            self.assertEqual(first["failed_trials"], 0)
            self.assertEqual(second["failed_trials"], 0)
            self.assertEqual(first["recorded_trials"], 45)
            a = pd.read_csv(base / "first" / "metrics.csv")
            b = pd.read_csv(base / "second" / "metrics.csv")
            keys = ["letter", "fold", "demo", "scenario", "method"]
            np.testing.assert_array_equal(a[keys], b[keys])
            np.testing.assert_allclose(a[["rmse", "end_err", "gap_rmse"]], b[["rmse", "end_err", "gap_rmse"]],
                                       atol=1e-10, equal_nan=True)
            self.assertTrue((a[["train_ms", "infer_ms"]] > 0).all().all())
            split = json.loads((base / "first" / "splits.json").read_text())[0]
            self.assertFalse(set(split["train_indices"]) & set(split["test_indices"]))
            self.assertEqual(set(a.groupby(["scenario", "method"]).size()), {3})
            self.assertEqual(len(pd.read_csv(base / "first" / "summary.csv")), 15)
            example = next((base / "first" / "predictions").rglob("reproduction-GP.npz"))
            with np.load(example) as saved:
                calculated = trajectory_metrics(saved["truth"], saved["state"][:, :2], saved["t"])
            demo_index = int(example.parent.name.removeprefix("demo-"))
            row = a[(a["method"] == "GP") & (a["scenario"] == "reproduction")
                    & (a["demo"] == demo_index)].iloc[0]
            self.assertAlmostEqual(calculated["rmse"], row["rmse"], places=10)
            dashboard = (base / "first" / "dashboard.html").read_text()
            self.assertNotIn("__RUN_DATA__", dashboard)
            self.assertIn('application/json', dashboard)
            self.assertTrue((base / "first" / "figures" / "summary_rmse.png").is_file())
            self.assertEqual(len(pd.read_csv(base / "first" / "control_ablation.csv")), 6)
            with self.assertRaises(ValueError):
                run_benchmark(config, data, base / "first")


if __name__ == "__main__":
    unittest.main()
