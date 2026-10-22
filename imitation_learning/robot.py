"""Torque-driven MuJoCo simulation of a teaching model, with no hardware IO.

Each run tracks one saved reproduction example. Fixed PD, fixed-cost task-space
LQR and covariance-aware task-space LQR drive actual bounded actuator torques.
This validates a teaching-model simulation, not physical hardware.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .experiments import json_safe, write_json
from .models import METHODS
from .control import lqr_policy
from .distributions import Prediction

LINK_LENGTHS = np.array([0.32, 0.28])
JOINT_LIMITS = np.array([[-2.7, 2.7], [0.1, 3.0]])
TOOL_HEIGHT = 0.30
CONTROLLERS = ("joint_pd", "fixed_lqr", "covariance_lqr")


@dataclass(frozen=True)
class RobotConfig:
    width: float = 0.16
    center: tuple[float, float] = (0.40, 0.0)
    duration: float = 8.0
    timestep: float = 0.002
    max_velocity: float = 1.0
    max_acceleration: float = 2.0
    reference_limit_fraction: float = 0.8
    max_duration: float = 60.0
    torque_limit: float = 5.0
    kp: tuple[float, float] = (120.0, 80.0)
    kd: tuple[float, float] = (12.0, 8.0)
    tracking_tolerance: float = 0.005
    controller: str = "joint_pd"
    control_penalty: float = 1.0
    lqr_variance_floor: tuple[float, ...] = (0.0001, 0.0001, 0.0025, 0.0025)
    disturbance_force: tuple[float, float] = (0.0, 0.0)
    disturbance_start: float = 0.45
    disturbance_end: float = 0.55

    def validate(self):
        values = (self.width, self.duration, self.timestep, self.max_velocity,
                  self.max_acceleration, self.max_duration, self.torque_limit,
                  self.tracking_tolerance, self.control_penalty, *self.lqr_variance_floor, *self.kp, *self.kd)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("robot sizes, gains, limits and times must be positive and finite")
        if len(self.center) != 2 or not np.isfinite(self.center).all():
            raise ValueError("robot center must have two finite coordinates in meters")
        if len(self.kp) != 2 or len(self.kd) != 2:
            raise ValueError("robot controller requires two position and velocity gains")
        if not 0.0005 <= self.timestep <= 0.005:
            raise ValueError("robot timestep must be between 0.0005 and 0.005 seconds")
        if self.duration < 1 or self.duration > self.max_duration or self.max_duration > 60:
            raise ValueError("robot duration must be between 1 and max_duration (at most 60 seconds)")
        if not np.isfinite(self.reference_limit_fraction) or not 0 < self.reference_limit_fraction <= 1:
            raise ValueError("reference_limit_fraction must be in (0, 1]")
        if self.controller not in (*CONTROLLERS, "auto") or len(self.lqr_variance_floor) != 4:
            raise ValueError("choose a supported controller and four LQR variance floors")
        if len(self.disturbance_force) != 2 or not np.isfinite(self.disturbance_force).all():
            raise ValueError("disturbance force must contain two finite Newton values")
        if not 0 <= self.disturbance_start < self.disturbance_end <= 1:
            raise ValueError("disturbance interval must lie within [0,1]")


def resolved_config(config, method):
    controller = config.controller
    if controller == "auto":
        controller = "covariance_lqr" if method == "U-KMP" else "fixed_lqr"
    return replace(config, controller=controller)


def inverse_kinematics(position: np.ndarray) -> np.ndarray:
    """Continuous elbow-positive branch. Unreachable points are rejected."""
    position = np.asarray(position, dtype=float)
    if position.ndim != 2 or position.shape[1] != 2 or not np.isfinite(position).all():
        raise ValueError("robot positions must be a finite (T, 2) array in meters")
    l1, l2 = LINK_LENGTHS
    cosine = (np.sum(position ** 2, axis=1) - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    if np.any(np.abs(cosine) > 1 + 1e-12):
        raise ValueError("trajectory is outside the simulated arm reach; reduce width or change center")
    elbow = np.arccos(np.clip(cosine, -1, 1))  # roundoff only; no workspace clipping
    shoulder = np.arctan2(position[:, 1], position[:, 0]) - np.arctan2(l2 * np.sin(elbow), l1 + l2 * np.cos(elbow))
    shoulder = np.unwrap(shoulder)
    # Choose the equivalent continuous branch nearest the symmetric joint
    # range; atan2 alone would jump by 2*pi across the negative x axis.
    shoulder -= 2 * np.pi * np.round((shoulder.min() + shoulder.max()) / (4 * np.pi))
    joints = np.column_stack([shoulder, elbow])
    if np.any(joints < JOINT_LIMITS[:, 0]) or np.any(joints > JOINT_LIMITS[:, 1]):
        raise ValueError("trajectory violates simulated joint limits; reduce width or change center")
    return joints


def forward_kinematics(joints: np.ndarray) -> np.ndarray:
    l1, l2 = LINK_LENGTHS
    return np.column_stack([l1 * np.cos(joints[:, 0]) + l2 * np.cos(joints.sum(axis=1)),
                            l1 * np.sin(joints[:, 0]) + l2 * np.sin(joints.sum(axis=1))])


def tip_jacobian(q):
    l1, l2 = LINK_LENGTHS
    a, b = q[0], q.sum()
    return np.array([[-l1 * np.sin(a) - l2 * np.sin(b), -l2 * np.sin(b)],
                     [l1 * np.cos(a) + l2 * np.cos(b), l2 * np.cos(b)]])


def tip_convective_acceleration(q, qd):
    l1, l2 = LINK_LENGTHS
    return np.array([-l1 * np.cos(q[0]) * qd[0] ** 2 - l2 * np.cos(q.sum()) * qd.sum() ** 2,
                     -l1 * np.sin(q[0]) * qd[0] ** 2 - l2 * np.sin(q.sum()) * qd.sum() ** 2])


def load_robot_case(run: Path, letter: str, method: str):
    if len(letter) != 1 or letter not in "ABCDEFGH" or method not in METHODS:
        raise ValueError("choose letter A-H and a supported learning method")
    source = run / "examples.json"
    if not source.is_file():
        raise ValueError(f"no learning examples at {source}; run 'python -m imitation_learning demo' "
                         "to generate all letters/methods, or choose the directory containing examples.json")
    examples = json.loads(source.read_text(encoding="utf-8"))
    case = examples.get(f"{letter}/reproduction")
    if case is None or method not in case.get("methods", {}):
        raise ValueError(f"{letter}/{method} has no saved reproduction example in {run}")
    entry = case["methods"][method]
    if entry.get("metrics", {}).get("status") != "ok":
        raise ValueError("source learning prediction must have status ok")
    return case, entry, source


def prepare_reference(case, entry, config: RobotConfig):
    config.validate()
    raw = np.asarray(entry["position"], dtype=float)
    training = np.asarray(case["training_mean"], dtype=float)
    source_t = np.asarray(case["time"], dtype=float)
    if (raw.ndim != 2 or raw.shape[1] != 2 or len(raw) < 4 or training.shape != raw.shape
            or source_t.shape != (len(raw),) or not np.isfinite(raw).all()
            or not np.isfinite(training).all() or not np.isfinite(source_t).all()
            or np.any(np.diff(source_t) <= 0)):
        raise ValueError("source example must have matching finite positions, training mean and increasing times")
    low, high = training.min(axis=0), training.max(axis=0)
    span = float(np.max(high - low))
    if span < 1e-8:
        raise ValueError("training reference has no usable spatial extent")
    scale = config.width / span
    origin = (low + high) / 2
    mapped = (raw - origin) * scale + config.center
    inverse_kinematics(mapped)
    phase = (source_t - source_t[0]) / (source_t[-1] - source_t[0])
    interpolate = PchipInterpolator(phase, mapped, axis=0)
    source_cov = entry.get("state_covariance")
    if config.controller == "covariance_lqr" and source_cov is None:
        raise ValueError("covariance_lqr requires saved full state covariance; generate learning results with v0.5 or newer")
    if source_cov is not None:
        source_cov = np.asarray(source_cov, dtype=float)
        if source_cov.shape != (len(raw), 4, 4) or not np.isfinite(source_cov).all() or np.linalg.eigvalsh(source_cov).min() < -1e-6:
            raise ValueError("saved state covariance must be finite positive semidefinite (T,4,4)")
    duration = config.duration
    for _ in range(12):
        intervals = int(np.ceil(duration / config.timestep - 1e-10))
        duration = intervals * config.timestep
        if duration > config.max_duration + 1e-10:
            raise ValueError("trajectory cannot satisfy velocity/acceleration limits within max_duration")
        t = np.arange(intervals + 1) * config.timestep
        u = t / duration
        warped = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        position = interpolate(np.clip(warped, 0, 1))
        joints = inverse_kinematics(position)
        velocity = np.gradient(joints, config.timestep, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, config.timestep, axis=0, edge_order=2)
        ratio = max(float(np.max(np.abs(velocity))) / (config.max_velocity * config.reference_limit_fraction),
                    np.sqrt(float(np.max(np.abs(acceleration))) / (config.max_acceleration * config.reference_limit_fraction)))
        if ratio <= 1 + 1e-10:
            warp_rate = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / duration
            target_velocity = interpolate.derivative()(np.clip(warped, 0, 1)) * warp_rate[:, None]
            result = {"t": t, "target": position, "q_ref": joints, "qd_ref": velocity,
                      "qdd_ref": acceleration, "target_velocity": target_velocity}
            if source_cov is not None:
                interpolated = np.stack([np.interp(warped, phase, source_cov[:, a, b])
                                         for a in range(4) for b in range(4)], axis=-1).reshape(-1, 4, 4)
                source_rate = (source_t[-1] - source_t[0]) * warp_rate
                transform = scale * np.column_stack([np.ones_like(u), np.ones_like(u), source_rate, source_rate])
                result["state_covariance"] = interpolated * transform[:, :, None] * transform[:, None, :]
            return result, {"scale_m_per_dataset_unit": scale,
                    "training_bbox_min": low.tolist(), "training_bbox_max": high.tolist(),
                    "dataset_origin": origin.tolist(), "robot_center_m": list(config.center),
                    "plane_z_m": TOOL_HEIGHT, "motion_duration_s": duration,
                    "time_warp": "10*u^3-15*u^4+6*u^5; PCHIP of source positions",
                    "covariance_mapping": "linear PSD-preserving interpolation; diag(scale,scale,scale*source_rate,scale*source_rate)"}
        duration *= ratio * 1.02
    raise ValueError("robot time parameterization did not converge")


def model_xml(config: RobotConfig) -> str:
    root = ET.fromstring(Path(__file__).with_name("robot_arm.xml").read_text(encoding="utf-8"))
    root.find("option").set("timestep", str(config.timestep))
    root.find("default/motor").set("ctrlrange", f"{-config.torque_limit} {config.torque_limit}")
    return ET.tostring(root, encoding="unicode") + "\n"


def rollout(reference, config: RobotConfig, xml: str):
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError('robot simulation needs the optional dependency: python -m pip install -e ".[simulation]"') from error
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    if model.nq != 2 or model.nv != 2 or model.nu != 2:
        raise ValueError("expected the bundled two-joint teaching arm")
    count = len(reference["t"])
    arrays = {name: np.empty((count, 2)) for name in ("q", "qd", "tip", "elbow", "torque", "tip_velocity")}
    arrays["external_force_n"] = np.zeros((count, 2))
    arrays["position_gain_norm"] = np.full(count, float(np.linalg.norm(config.kp)))
    arrays["velocity_gain_norm"] = np.full(count, float(np.linalg.norm(config.kd)))
    contacts, saturated = np.zeros(count, dtype=int), np.zeros(count, dtype=bool)
    # Initial configuration only. Every later state comes from mj_step.
    data.qpos[:] = reference["q_ref"][0]
    mujoco.mj_forward(model, data)
    policy = None
    if config.controller != "joint_pd":
        floor = np.asarray(config.lqr_variance_floor)
        prediction = Prediction(np.column_stack([reference["target"], reference["target_velocity"]]),
                                reference.get("state_covariance"))
        policy = lqr_policy(prediction, reference["t"], adaptive=config.controller == "covariance_lqr",
                            control_penalty=config.control_penalty, variance_floor=floor,
                            fixed_cost=np.diag(1 / floor))
        arrays["position_gain_norm"] = np.r_[np.linalg.norm(policy.gains[:, :, :2], axis=(1, 2)),
                                             np.linalg.norm(policy.gains[-1, :, :2])]
        arrays["velocity_gain_norm"] = np.r_[np.linalg.norm(policy.gains[:, :, 2:], axis=(1, 2)),
                                             np.linalg.norm(policy.gains[-1, :, 2:])]
    def record(index, torque):
        arrays["q"][index] = data.qpos
        arrays["qd"][index] = data.qvel
        arrays["tip"][index] = data.site("tip").xpos[:2]
        arrays["tip_velocity"][index] = tip_jacobian(data.qpos) @ data.qvel
        arrays["elbow"][index] = data.site("elbow_site").xpos[:2]
        arrays["torque"][index] = torque
        contacts[index] = data.ncon
    record(0, np.zeros(2))
    for index in range(count - 1):
        jacobian = tip_jacobian(data.qpos)
        if policy is None:
            requested = (np.asarray(config.kp) * (reference["q_ref"][index] - data.qpos)
                         + np.asarray(config.kd) * (reference["qd_ref"][index] - data.qvel))
        else:
            if abs(np.linalg.det(jacobian)) < 1e-8:
                raise RuntimeError("task-space controller reached a singular robot configuration")
            measured = np.r_[data.site("tip").xpos[:2], jacobian @ data.qvel]
            command = -policy.gains[index] @ measured - policy.offsets[index]
            qdd = np.linalg.solve(jacobian, command - tip_convective_acceleration(data.qpos, data.qvel))
            requested = np.empty(2)
            mujoco.mj_mulM(model, data, requested, qdd)
            requested += data.qfrc_bias - data.qfrc_passive
        fraction = reference["t"][index] / reference["t"][-1]
        force = np.asarray(config.disturbance_force) if config.disturbance_start <= fraction < config.disturbance_end else np.zeros(2)
        data.qfrc_applied[:] = jacobian.T @ force
        arrays["external_force_n"][index + 1] = force
        torque = np.clip(requested, -config.torque_limit, config.torque_limit)
        saturated[index + 1] = np.any(np.abs(requested) > config.torque_limit)
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        # Recompute sites at the integrated state before logging (mj_step's
        # cached site positions otherwise refer to the preceding state).
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("non-finite simulated robot state")
        if not np.isclose(data.time, reference["t"][index + 1], atol=1e-8):
            raise RuntimeError("simulation clock diverged from reference time")
        record(index + 1, torque)
    arrays["contacts"] = contacts
    arrays["saturated"] = saturated
    arrays["error_m"] = np.linalg.norm(arrays["tip"] - reference["target"], axis=1)
    return arrays


def tracking_metrics(reference, arrays, config):
    error = arrays["error_m"]
    speed = float(np.max(np.abs(arrays["qd"])))
    acceleration = float(np.max(np.abs(np.gradient(arrays["qd"], config.timestep, axis=0, edge_order=2))))
    joints_valid = bool(np.all(arrays["q"] >= JOINT_LIMITS[:, 0] - 1e-6)
                        and np.all(arrays["q"] <= JOINT_LIMITS[:, 1] + 1e-6))
    within_limits = joints_valid and speed <= config.max_velocity and acceleration <= config.max_acceleration
    maximum = float(error.max())
    passed = bool(maximum <= config.tracking_tolerance and not arrays["contacts"].any() and within_limits)
    force_active = np.linalg.norm(arrays.get("external_force_n", np.zeros((len(error), 2))), axis=1) > 0
    recovery = None
    if force_active.any():
        last_force = int(np.flatnonzero(force_active)[-1])
        late = np.flatnonzero(error[last_force + 1:] > config.tracking_tolerance)
        settled = last_force + 1 if not len(late) else last_force + 2 + int(late[-1])
        if settled < len(error):
            recovery = float(reference["t"][settled] - reference["t"][last_force])
    work = np.abs(arrays["torque"][1:] * (arrays["qd"][1:] + arrays["qd"][:-1]) / 2).sum() * config.timestep
    return {"tracking_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "max_tracking_error_m": maximum, "endpoint_error_m": float(error[-1]),
            "peak_torque_nm": float(np.max(np.abs(arrays["torque"]))),
            "saturation_fraction": float(arrays["saturated"][1:].mean()),
            "contact_samples": int(np.count_nonzero(arrays["contacts"])),
            "joint_limits_satisfied": joints_valid, "actual_limits_satisfied": bool(within_limits),
            "actual_peak_velocity_rad_s": speed, "actual_peak_acceleration_rad_s2": acceleration,
            "reference_peak_velocity_rad_s": float(np.max(np.abs(reference["qd_ref"]))),
            "reference_peak_acceleration_rad_s2": float(np.max(np.abs(reference["qdd_ref"]))),
            "tracking_passed": passed, "samples": len(error),
            "torque_rms_nm": float(np.sqrt(np.mean(arrays["torque"][1:] ** 2))),
            "absolute_mechanical_work_j": float(work), "recovery_time_s": recovery,
            "disturbance_peak_tracking_error_m": float(error[force_active].max()) if force_active.any() else None,
            "mean_position_gain": float(arrays["position_gain_norm"].mean())}


def simulate_robot(run, output, letter="G", method="KMP", config=None):
    config = resolved_config(config or RobotConfig(), method)
    config.validate()
    print(f"Starting robot simulation: {letter}/{method}", flush=True)
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError('install simulation support: python -m pip install -e ".[simulation]"') from error
    run, output = Path(run).resolve(), Path(output).resolve()
    case, entry, source = load_robot_case(run, letter, method)
    reference, mapping = prepare_reference(case, entry, config)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output is not empty: {output}; choose a new simulation directory")
    xml = model_xml(config)
    arrays = rollout(reference, config, xml)
    metrics = tracking_metrics(reference, arrays, config)
    manifest = {"stage": "MuJoCo dynamics simulation; not physical robot validation",
                "status": "completed", "robot": "educational_planar_2r",
                "source_run": str(run), "source_examples_sha256": sha256(source.read_bytes()).hexdigest(),
                "source_case": {"letter": letter, "method": method, "scenario": "reproduction",
                                "fold": case["fold"], "demo": case["demo"]},
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "scipy")},
                "model_sha256": sha256(xml.encode()).hexdigest(),
                "source_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in
                                  (Path(__file__), Path(__file__).with_name("robot_dashboard.html"))},
                "config": asdict(config), "mapping": mapping, "metrics": metrics,
                "controller": "same fixed joint PD for every learning method; actuator torque is clipped",
                "initialization": "first reference joint position, zero velocity; approach motion not simulated",
                "logging": "post-step state; logged torque acts over the preceding timestep",
                "metrics_scope": "all physics samples, Euclidean TCP errors in meters; one example, not full CV",
                "limitations": ["generic 2R model, not an identified commercial robot",
                                "no hardware, sensor noise, network delay, contact-writing or moving obstacles",
                                "reproduction example only; no obstacle scene or physical avoidance claim",
                                "KMP/U-KMP share planning and fixed PD here; no U-KMP control advantage claim"]}
    manifest["controller_id"] = config.controller
    manifest["controller"] = ("fixed joint PD" if config.controller == "joint_pd" else
                               "task-space " + config.controller + "; affine LQR, Jacobian and inverse dynamics; clipped torque")
    manifest["limitations"] = ["generic 2R model, not an identified commercial robot",
        "simulation only; ideal known model; no hardware, sensing delay or contact-writing",
        "reproduction example; external-force pulse optional; no obstacle geometry",
        "empirically calibrated bands are evaluation outputs; LQR uses the saved latent state covariance"]
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "robot_trace.npz", **reference, **arrays)
    (output / "model.xml").write_text(xml, encoding="utf-8")
    write_json(output / "manifest.json", manifest)
    columns = {"time_s": reference["t"], "error_m": arrays["error_m"],
               "contacts": arrays["contacts"], "torque_saturated": arrays["saturated"]}
    for name in ("target", "tip", "elbow", "q_ref", "qd_ref", "qdd_ref", "q", "qd", "torque", "tip_velocity", "external_force_n"):
        for i in range(2):
            columns[f"{name}_{i}"] = (reference if name in reference else arrays)[name][:, i]
    columns["position_gain_norm"] = arrays["position_gain_norm"]
    columns["velocity_gain_norm"] = arrays["velocity_gain_norm"]
    pd.DataFrame(columns).to_csv(output / "robot_trace.csv", index=False)
    _write_robot_page(output / "dashboard.html", _robot_payload(manifest, reference | arrays))
    _plot_tracking(output, reference, arrays)
    report = (f"# Robot simulation: {letter} / {method}\n\n"
              "MuJoCo teaching-arm dynamics simulation. This is not physical robot validation.\n\n"
              f"- Tracking passed: **{metrics['tracking_passed']}**\n"
              f"- TCP Euclidean RMSE: {metrics['tracking_rmse_m'] * 1000:.4f} mm\n"
              f"- Maximum TCP error: {metrics['max_tracking_error_m'] * 1000:.4f} mm\n"
              f"- Endpoint error: {metrics['endpoint_error_m'] * 1000:.4f} mm\n"
              f"- Motion duration: {mapping['motion_duration_s']:.3f} s\n"
              f"- Contacts: {metrics['contact_samples']} physics samples\n"
              f"- Saturated commands: {metrics['saturation_fraction']:.2%}\n\n"
              "A 5 mm maximum-error threshold, contact count, joint bounds and actual velocity/acceleration "
              "limits determine the default pass result (see manifest config for the effective threshold). "
              "This threshold is a declared simulation criterion, not a hardware safety guarantee. "
              "The first point is the initialized robot state. All later states are integrated by MuJoCo. "
              "No per-step state reset or prescribed animation is used to generate the metrics.\n\n"
              "Mapping uses the training-reference bounding box, not each method's predicted extent. "
              f"Controller: {manifest['controller']}. One representative fold-0 example is used.\n\n"
              f"- RMS torque: {metrics['torque_rms_nm']:.6g} N m\n"
              f"- Absolute mechanical work estimate: {metrics['absolute_mechanical_work_j']:.6g} J\n"
              f"- Applied TCP force pulse: {config.disturbance_force} N, "
              f"fractions {config.disturbance_start} to {config.disturbance_end}\n"
              f"- Recovery time below the tracking threshold: {metrics['recovery_time_s']} s\n\n"
              "Task-space LQR uses actual simulated TCP position/velocity, analytic Jacobian, "
              "and the known model's inverse dynamics. Fixed and covariance-aware LQR use the "
              "same input cost and torque limit. Calibrated marginal bands are not substituted "
              "for the full state covariance. Read error together with control effort and deflection; "
              "more compliant behavior need not minimize tracking error.\n")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Robot simulation: tracking_passed={metrics['tracking_passed']}; "
          f"RMSE={metrics['tracking_rmse_m'] * 1000:.3f} mm; max={metrics['max_tracking_error_m'] * 1000:.3f} mm", flush=True)
    print(f"Open {output / 'dashboard.html'}", flush=True)
    return manifest


def _robot_payload(manifest, trace, artifacts=None):
    """Read the original full-rate logs; decimate only the browser display."""
    t = np.asarray(trace["t"], dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("robot trace must contain finite, increasing times")
    data = {name: np.asarray(trace[name], dtype=float) for name in ("target", "tip", "elbow", "error_m")}
    for name, array in data.items():
        expected = (len(t),) if name == "error_m" else (len(t), 2)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"invalid robot trace field: {name}")
    error = np.linalg.norm(data["tip"] - data["target"], axis=1)
    if not np.allclose(error, data["error_m"], rtol=1e-8, atol=1e-12):
        raise ValueError("saved robot error does not match full-rate positions")
    metrics = manifest["metrics"]
    if (not np.isclose(metrics["tracking_rmse_m"], np.sqrt(np.mean(error ** 2)), rtol=1e-8, atol=1e-12)
            or not np.isclose(metrics["max_tracking_error_m"], error.max(), rtol=1e-8, atol=1e-12)):
        raise ValueError("saved robot metrics do not match full-rate positions")
    indices = np.unique(np.linspace(0, len(t) - 1, min(1201, len(t)), dtype=int))
    payload = {"manifest": manifest, "t": t[indices].tolist(),
               "artifacts": artifacts or {"csv": "robot_trace.csv", "report": "REPORT.md"}}
    payload.update({name: array[indices].tolist() for name, array in data.items()})
    return payload


def _write_robot_page(path, payload):
    template = Path(__file__).with_name("robot_dashboard.html").read_text(encoding="utf-8")
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace("__ROBOT_DATA__", embedded), encoding="utf-8")


def build_robot_gallery(source, output=None):
    """Combine existing runs in one offline page, without running any physics."""
    source = Path(source).resolve()
    output = Path(output).resolve() if output is not None else source / "dashboard.html"
    if output.suffix.lower() != ".html":
        raise ValueError("gallery output must be an HTML filename")
    cases, seen, records = [], set(), []
    for path in sorted(source.rglob("manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("robot") != "educational_planar_2r":
            continue
        identity = manifest["source_case"]
        key = (identity["letter"], identity["method"], manifest.get("controller_id", "joint_pd"))
        if key in seen:
            raise ValueError(f"duplicate robot case {'/'.join(key)}; choose a directory with one run per case")
        seen.add(key)
        folder = path.parent
        def artifact(name):
            target = folder / name
            if not target.is_file():
                raise ValueError(f"missing robot artifact: {target}")
            try:
                return Path(os.path.relpath(target, output.parent)).as_posix()
            except ValueError:  # explicit output on another Windows drive
                return target.as_uri()
        artifacts = {"csv": artifact("robot_trace.csv"), "report": artifact("REPORT.md")}
        trace_path = folder / "robot_trace.npz"
        with np.load(trace_path, allow_pickle=False) as trace:
            cases.append(_robot_payload(manifest, trace, artifacts))
        records.append({"letter": key[0], "method": key[1], "controller": key[2],
                        "manifest_sha256": sha256(path.read_bytes()).hexdigest(),
                        "trace_sha256": sha256(trace_path.read_bytes()).hexdigest()})
    if not cases:
        raise ValueError(f"no saved robot simulation traces found in {source}")
    cases.sort(key=lambda case: (case["manifest"]["source_case"]["letter"],
                                METHODS.index(case["manifest"]["source_case"]["method"]),
                                case["manifest"].get("controller_id", "joint_pd")))
    _write_robot_page(output, {"cases": cases})
    print(f"Combined {len(cases)} robot cases into {output}", flush=True)
    return {"cases": records, "letters": sorted({key[0] for key in seen}),
            "all_tracking_passed": all(case["manifest"]["metrics"]["tracking_passed"] for case in cases)}


def simulate_robot_grid(run, output, letters=tuple("ABCDEFGH"), methods=METHODS,
                        controllers=("auto",), config=None, resume=False, common_duration=True):
    """Preflight all cases; optionally share a clock per letter and reuse intact logs."""
    config = config or RobotConfig()
    config.validate()
    run, output = Path(run).resolve(), Path(output).resolve()
    if (not letters or len(set(letters)) != len(letters) or not methods
            or len(set(methods)) != len(methods) or not controllers or len(set(controllers)) != len(controllers)):
        raise ValueError("letters, methods and controllers must be nonempty and unique")
    specs, seen = [], set()
    for letter in letters:
        selected = []
        for method in methods:
            case, entry, source = load_robot_case(run, letter, method)
            for controller in controllers:
                cfg = resolved_config(replace(config, controller=controller), method)
                cfg.validate()
                key = (letter, method, cfg.controller)
                if key in seen:
                    raise ValueError(f"duplicate grid case {'/'.join(key)}")
                seen.add(key)
                _, mapping = prepare_reference(case, entry, cfg)
                selected.append((letter, method, cfg, case, entry, mapping['motion_duration_s']))
        duration = max(item[-1] for item in selected)
        for letter, method, cfg, case, entry, _ in selected:
            if common_duration:
                cfg = replace(cfg, duration=duration)
                _, mapping = prepare_reference(case, entry, cfg)
                if abs(mapping['motion_duration_s'] - duration) > config.timestep * .1:
                    raise ValueError("shared duration needs a larger --duration; all methods must use the same clock")
            specs.append((letter, method, cfg))
    source_hash = sha256((run / 'examples.json').read_bytes()).hexdigest()
    calculation_sources = [Path(__file__), Path(__file__).with_name('control.py'),
                           Path(__file__).with_name('robot_arm.xml'), Path(__file__).with_name('robot_dashboard.html')]
    request = json_safe({'source_examples_sha256': source_hash,
        'source_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in calculation_sources},
        'common_duration': common_duration,
        'cases': [{'letter': l, 'method': m, 'config': asdict(c)} for l, m, c in specs]})
    if output.exists() and any(output.iterdir()):
        previous = output / 'grid.json'
        if not resume or not previous.is_file():
            raise ValueError(f"output is not an empty or resumable robot grid: {output}")
        if json.loads(previous.read_text(encoding='utf-8')).get('request') != request:
            raise ValueError("cannot resume robot grid: inputs/configuration/source changed; choose a new output")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'grid.json', {'status': 'running', 'request': request})
    manifests = []
    for number, (letter, method, cfg) in enumerate(specs, 1):
        destination = output / letter / method / cfg.controller
        print(f"Robot grid {number}/{len(specs)}: {letter}/{method}/{cfg.controller}", flush=True)
        if destination.exists():
            manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
            if (manifest.get('config') != json_safe(asdict(cfg)) or manifest.get('source_examples_sha256') != source_hash
                    or manifest.get('source_case', {}).get('letter') != letter
                    or manifest.get('source_case', {}).get('method') != method
                    or manifest.get('controller_id') != cfg.controller
                    or manifest.get('status') != 'completed'):
                raise ValueError(f"saved case does not match this grid: {destination}")
            for artifact in ('REPORT.md', 'robot_trace.csv'):
                if not (destination / artifact).is_file():
                    raise ValueError(f"saved case is incomplete: {destination / artifact}")
            with np.load(destination / 'robot_trace.npz', allow_pickle=False) as trace:
                _robot_payload(manifest, trace)
            print('  Reusing verified completed simulation', flush=True)
        else:
            pending_root = output.parent / (output.name + '-work') / letter / method
            attempt = 1
            temporary = pending_root / f'{cfg.controller}-{attempt}'
            while temporary.exists():
                attempt += 1
                temporary = pending_root / f'{cfg.controller}-{attempt}'
            manifest = simulate_robot(run, temporary, letter, method, cfg)
            manifest['comparison'] = {'duration_policy': 'shared within each letter' if common_duration else 'per trajectory'}
            write_json(temporary / 'manifest.json', manifest)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temporary), str(destination))
        manifests.append(manifest)
    result = build_robot_gallery(output)
    table = [{'letter': m['source_case']['letter'], 'method': m['source_case']['method'],
              'controller': m['controller_id'], 'duration_s': m['mapping']['motion_duration_s'], **m['metrics']}
             for m in manifests]
    pd.DataFrame(table).to_csv(output / 'summary.csv', index=False)
    write_json(output / 'grid.json', {'status': 'completed', 'request': request,
                                     'all_tracking_passed': result['all_tracking_passed']})
    print(f"Open {output / 'dashboard.html'}", flush=True)
    return manifests


def simulate_robot_letters(run, output, letters=tuple("ABCDEFGH"), method="KMP", config=None):
    config = config or RobotConfig()
    config.validate()
    run, output = Path(run).resolve(), Path(output).resolve()
    if not letters or len(set(letters)) != len(letters):
        raise ValueError("choose at least one letter, without duplicates")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output is not empty: {output}; choose a new simulation directory")
    for letter in letters:
        load_robot_case(run, letter, method)  # reject missing inputs before starting
    manifests = []
    for index, letter in enumerate(letters, 1):
        print(f"Robot case {index}/{len(letters)}: {letter}/{method}", flush=True)
        manifests.append(simulate_robot(run, output / letter, letter, method, config))
    build_robot_gallery(output)
    print(f"Open {output / 'dashboard.html'}", flush=True)
    return manifests


def _plot_tracking(output, reference, arrays):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(*reference["target"].T, "--", label="Reference")
    axes[0].plot(*arrays["tip"].T, label="MuJoCo TCP")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Simulated end-effector tracking", aspect="equal")
    axes[0].legend()
    axes[1].plot(reference["t"], arrays["error_m"] * 1000)
    axes[1].set(xlabel="Simulation time (s)", ylabel="Euclidean error (mm)", title="All physics samples")
    fig.savefig(output / "tracking.png", dpi=160)
    plt.close(fig)
