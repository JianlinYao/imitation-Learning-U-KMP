"""Synthetic log fixtures for viewer tests only; these are not physics results."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd

from imitation_learning.robot import RobotConfig


def write_robot_fixture(folder, letter="A", method="KMP", error_m=.001, controller=None):
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(501) * .002
    target = np.column_stack([.4 + .03 * np.sin(2 * np.pi * t), .02 * np.cos(2 * np.pi * t)])
    tip = target + [error_m, 0]
    elbow = np.column_stack([.2 + .01 * t, -.2 + .01 * t])
    error = np.linalg.norm(tip - target, axis=1)
    metrics = {"tracking_rmse_m": float(np.sqrt(np.mean(error ** 2))),
               "max_tracking_error_m": float(error.max()), "endpoint_error_m": float(error[-1]),
               "peak_torque_nm": .1, "saturation_fraction": 0.0, "contact_samples": 0,
               "actual_limits_satisfied": True, "tracking_passed": bool(error.max() <= .005), "samples": len(t)}
    manifest = {"robot": "educational_planar_2r", "stage": "synthetic UI test fixture, not a physics result",
                "status": "completed", "source_case": {"letter": letter, "method": method,
                "scenario": "reproduction", "fold": 0, "demo": 0},
                "config": asdict(RobotConfig(controller=controller or 'joint_pd')), "metrics": metrics}
    if controller is not None:
        manifest['controller_id'] = controller
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(folder / "robot_trace.npz", t=t, target=target, tip=tip, elbow=elbow, error_m=error)
    pd.DataFrame({"time_s": t, "error_m": error}).to_csv(folder / "robot_trace.csv", index=False)
    (folder / "REPORT.md").write_text(f"测试夹具：{letter}/{method}\n", encoding="utf-8")
    (folder / "dashboard.html").write_text("original single-case page", encoding="utf-8")
    return manifest
