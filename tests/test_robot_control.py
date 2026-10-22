"""Actual torque-loop checks for covariance-aware control and force response."""
from dataclasses import replace
import importlib.util
import unittest

import numpy as np

from imitation_learning.robot import (RobotConfig, forward_kinematics, model_xml,
    prepare_reference, rollout, tip_jacobian, tracking_metrics)


def stationary_case(variance):
    phase = np.linspace(0, 1, 40)
    case = {'time': phase.tolist(),
            'training_mean': np.column_stack([phase, .2 * np.sin(np.pi * phase)]).tolist()}
    entry = {'position': np.tile([.5, .1], (40, 1)).tolist(),
             'state_covariance': np.broadcast_to(variance * np.eye(4), (40, 4, 4)).tolist()}
    return case, entry


class RobotControlPreparation(unittest.TestCase):
    def test_jacobian_matches_forward_kinematics(self):
        q = np.array([-.5, 1.5])
        eps = 1e-6
        numeric = np.column_stack([(forward_kinematics((q + eps * np.eye(2)[i])[None, :])[0]
                    - forward_kinematics((q - eps * np.eye(2)[i])[None, :])[0]) / (2 * eps)
                   for i in range(2)])
        np.testing.assert_allclose(tip_jacobian(q), numeric, atol=1e-9)

    def test_covariance_mapping_preserves_units_time_warp_and_psd(self):
        case, entry = stationary_case(2.)
        cfg = RobotConfig(controller='covariance_lqr', duration=4.)
        ref, mapping = prepare_reference(case, entry, cfg)
        mid = len(ref['t']) // 2
        scale = mapping['scale_m_per_dataset_unit']
        self.assertAlmostEqual(ref['state_covariance'][mid, 0, 0], 2 * scale ** 2)
        self.assertAlmostEqual(ref['state_covariance'][mid, 2, 2], 2 * (scale * 1.875 / 4) ** 2)
        np.testing.assert_allclose(ref['state_covariance'][0, 2:, :], 0, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(ref['state_covariance']).min(), -1e-12)
        del entry['state_covariance']
        with self.assertRaisesRegex(ValueError, 'full state covariance'):
            prepare_reference(case, entry, cfg)


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'install the simulation extra')
class RobotControlDynamics(unittest.TestCase):
    def test_higher_covariance_changes_actual_force_response_and_reduces_gains(self):
        cfg = RobotConfig(controller='covariance_lqr', duration=4., disturbance_force=(.02, 0.))
        traces, metrics = [], []
        for variance in (.0001, 10.):
            ref, _ = prepare_reference(*stationary_case(variance), cfg)
            trace = rollout(ref, cfg, model_xml(cfg))
            traces.append(trace)
            metrics.append(tracking_metrics(ref, trace, cfg))
        self.assertLess(metrics[1]['mean_position_gain'], metrics[0]['mean_position_gain'] / 10)
        self.assertGreater(metrics[1]['disturbance_peak_tracking_error_m'],
                           metrics[0]['disturbance_peak_tracking_error_m'] * 2)
        self.assertLess(metrics[1]['torque_rms_nm'], metrics[0]['torque_rms_nm'])
        self.assertTrue(np.any(traces[1]['external_force_n']))
        self.assertTrue(np.all(np.abs(traces[1]['torque']) <= cfg.torque_limit))
        self.assertGreater(np.max(np.abs(traces[1]['q'] - traces[0]['q'])), 1e-4)

    def test_fixed_lqr_does_not_depend_on_saved_covariance(self):
        cfg = RobotConfig(controller='fixed_lqr', duration=2., disturbance_force=(.01, 0.))
        results = []
        for variance in (.0001, 10.):
            ref, _ = prepare_reference(*stationary_case(variance), cfg)
            results.append(rollout(ref, cfg, model_xml(cfg)))
        np.testing.assert_array_equal(results[0]['q'], results[1]['q'])
        np.testing.assert_array_equal(results[0]['torque'], results[1]['torque'])
