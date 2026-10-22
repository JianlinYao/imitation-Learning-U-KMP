"""One real CLI workflow and a second invocation that must reuse all completed work."""
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from imitation_learning.cli import main
from imitation_learning.experiments import RunConfig
from imitation_learning.models import ModelConfig


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'install the simulation extra')
class WorkflowTests(unittest.TestCase):
    def test_complete_cli_demo_controller_grid_and_verified_resume(self):
        cfg = RunConfig(letters=('A',), methods=('KMP','U-KMP'), scenarios=('reproduction',),
                        points=32, fold_limit=1, calibration_fraction=.2, timing_repeats=1,
                        control_ablation=False,
                        model=ModelConfig(reference_points=32, gmm_components=3, kmp_covariance_regularization=32))
        with (tempfile.TemporaryDirectory() as folder,
              io.TextIOWrapper(io.BytesIO(), encoding='cp1252', errors='strict') as console,
              patch('sys.stdout', console)):
            base = Path(folder)
            config = base/'demo.json'
            config.write_text(json.dumps(asdict(cfg)), encoding='utf-8')
            out = base/'演示输出'
            argv = ['demo','--config',str(config),'--output',str(out)]
            self.assertEqual(main(argv), 0)
            rows = pd.read_csv(out/'robot'/'summary.csv')
            self.assertEqual(set(rows['controller']), {'fixed_lqr','covariance_lqr'})
            self.assertEqual(rows['duration_s'].nunique(), 1)
            mtimes = {p: p.stat().st_mtime_ns for p in (out/'robot').rglob('robot_trace.npz')}
            self.assertEqual(len(mtimes), 2)
            with (patch('imitation_learning.experiments.make_model', side_effect=AssertionError('must reuse fits')),
                  patch('imitation_learning.robot.rollout', side_effect=AssertionError('must reuse physics'))):
                self.assertEqual(main(argv), 0)
            self.assertEqual(mtimes, {p: p.stat().st_mtime_ns for p in mtimes})
            self.assertTrue((out/'robot'/'dashboard.html').is_file())
