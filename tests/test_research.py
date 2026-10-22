"""Reference equations, honest uncertainty, split calibration and restart checks."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits

from imitation_learning.distributions import Observation, Prediction
from imitation_learning.experiments import RunConfig, calibration_split, run_benchmark, task_outcome
from imitation_learning.kernels import state_kernel
from imitation_learning.models import GP, KMP, ModelConfig, make_model
from imitation_learning.uncertainty import gaussian_scores, interval_inflation, position_covariance

ROOT = Path(__file__).resolve().parents[1]


class ResearchNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(limits=1)
        cls.t = np.linspace(0, 1, 32)
        cls.training = [np.column_stack([np.sin(2 * cls.t) + s * cls.t, cls.t ** 2 + s])
                        for s in (-.04, -.02, 0, .02, .04)]
        cls.config = ModelConfig(gmm_components=3, reference_points=32,
                                kmp_covariance_regularization=32)

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def test_full_state_reference_replacement_matches_independent_matrix_equations(self):
        model = KMP(self.config).fit(self.training, self.t)
        i = 15
        value = np.array([.9, .4, .2, -.1])
        noise = 1e-5
        actual = model.predict([Observation(i, tuple(value), (0, 1, 2, 3), noise)])
        mean = model.ref_mean.copy()
        covariance = model.ref_covariance.copy()
        mean[i] = value
        covariance[i] = noise * np.eye(4)
        kernel = state_kernel(self.t, self.t, self.config.length_scale)
        sigma = block_diag(*covariance)
        expected_mean = kernel @ np.linalg.solve(kernel + self.config.kmp_mean_regularization * sigma, mean.ravel())
        expected_joint = len(self.t) / self.config.kmp_covariance_regularization * (
            kernel - kernel @ np.linalg.solve(kernel + self.config.kmp_covariance_regularization * sigma, kernel))
        expected_blocks = expected_joint.reshape(len(self.t), 4, len(self.t), 4)[np.arange(len(self.t)), :, np.arange(len(self.t)), :]
        np.testing.assert_allclose(actual.state.ravel(), expected_mean, atol=2e-7)
        np.testing.assert_allclose(actual.covariance, expected_blocks, atol=2e-7)

    def test_covariance_tuning_does_not_change_reference_adapted_mean(self):
        obs = [Observation(12, (.8, .2), variance=1e-8)]
        low = KMP(replace(self.config, kmp_covariance_regularization=10)).fit(self.training, self.t)
        high = KMP(replace(self.config, kmp_covariance_regularization=100)).fit(self.training, self.t)
        np.testing.assert_allclose(low.predict(obs).state, high.predict(obs).state, atol=1e-12)
        self.assertGreater(np.max(np.abs(low.predict(obs).covariance - high.predict(obs).covariance)), 1e-5)
        np.testing.assert_allclose(low.predict(obs).position[12], [.8, .2], atol=1e-3)

    def test_gp_new_observation_noise_is_not_divided_by_demo_count(self):
        model = GP(self.config).fit(self.training, self.t)
        pred = model.predict()
        cov = position_covariance(pred.covariance, pred.position_noise_variance)
        difference = cov - pred.covariance[:, :2, :2]
        np.testing.assert_allclose(difference, np.broadcast_to(self.config.gp_noise_variance * np.eye(2), difference.shape))

    def test_gaussian_probability_score_and_empirical_interval_inflation(self):
        error = np.zeros((5, 2))
        nlpd, covered = gaussian_scores(error, np.broadcast_to(np.eye(2), (5, 2, 2)).copy())
        np.testing.assert_allclose(nlpd, np.log(2 * np.pi), atol=1e-9)
        self.assertTrue(covered.all())
        result = interval_inflation(np.linspace(0, 10, 100))
        self.assertGreater(result['scale'], 4)
        self.assertEqual(result['scores'], 100)
        self.assertIn('no trajectory-level', result['scope'])

    def test_missing_success_rejects_bad_interior_despite_correct_endpoint(self):
        result = task_outcome({'rmse': .2, 'gap_rmse': 3., 'end_err': 0.}, 10., 'missing', RunConfig())
        self.assertTrue(result['goal_reached'])
        self.assertFalse(result['task_success'])

    def test_calibration_demos_are_a_disjoint_reproducible_partition(self):
        fit, cal = calibration_split(np.arange(10), .2, 42)
        self.assertFalse(set(fit) & set(cal))
        self.assertEqual(set(fit) | set(cal), set(range(10)))
        np.testing.assert_array_equal(cal, calibration_split(np.arange(10), .2, 42)[1])


class ResearchProtocol(unittest.TestCase):
    def test_calibration_and_completed_run_resume_without_refitting(self):
        cfg = RunConfig(letters=('A',), methods=('GP', 'KMP'), scenarios=('reproduction', 'missing'),
                        fold_limit=1, points=32, observed_points=10, calibration_fraction=.2,
                        timing_repeats=1, control_ablation=False,
                        model=ModelConfig(reference_points=32, gmm_components=3, kmp_covariance_regularization=32))
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            run_benchmark(cfg, ROOT / '2Dletters', out)
            splits = json.loads((out/'splits.json').read_text())
            split = splits[0]
            self.assertFalse(set(split['train_indices']) & set(split['calibration_indices']))
            self.assertFalse((set(split['train_indices']) | set(split['calibration_indices'])) & set(split['test_indices']))
            records = json.loads((out/'calibration.json').read_text())
            self.assertEqual(len(records), 4)
            self.assertTrue(all(r['calibration_demo_indices'] == split['calibration_indices'] for r in records))
            before = pd.read_csv(out/'metrics.csv')
            self.assertTrue(before['calibrated_coverage95'].notna().all())
            with patch('imitation_learning.experiments.make_model', side_effect=AssertionError('completed fits must be reused')):
                manifest = run_benchmark(cfg, ROOT / '2Dletters', out, resume=True)
            self.assertEqual(manifest['recorded_trials'], len(before))
            np.testing.assert_allclose(pd.read_csv(out/'metrics.csv')['rmse'], before['rmse'])
            with self.assertRaisesRegex(ValueError, 'config changed'):
                run_benchmark(replace(cfg, seed=43), ROOT/'2Dletters', out, resume=True)

    def test_interrupted_run_reuses_only_completed_folds(self):
        cfg = RunConfig(letters=('A','G'), methods=('GP',), scenarios=('reproduction',),
                        fold_limit=1, points=24, timing_repeats=1, control_ablation=False)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            calls = []
            def interrupt_second(*args, **kwargs):
                calls.append(args)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return make_model(*args, **kwargs)
            with patch('imitation_learning.experiments.make_model', side_effect=interrupt_second):
                with self.assertRaises(KeyboardInterrupt):
                    run_benchmark(cfg, ROOT/'2Dletters', out)
            self.assertEqual(json.loads((out/'checkpoint.json').read_text())['completed_folds'], [['A',0]])
            with patch('imitation_learning.experiments.make_model', wraps=make_model) as fits:
                manifest = run_benchmark(cfg, ROOT/'2Dletters', out, resume=True)
            self.assertEqual(fits.call_count, 1)
            self.assertEqual(manifest['recorded_trials'], 5)
            rows = pd.read_csv(out/'metrics.csv')
            self.assertFalse(rows.duplicated(['letter','fold','demo','scenario','method']).any())
