"""Torque-driven MuJoCo simulation of a teaching model, with no hardware IO.

Each run tracks one saved reproduction example. Learned planning methods share
the same training-reference mapping and fixed joint PD controller. This does
not test the separate uncertainty-aware LQR on a physical robot.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .experiments import write_json
from .models import METHODS

LINK_LENGTHS = np.array([0.32, 0.28])
JOINT_LIMITS = np.array([[-2.7, 2.7], [0.1, 3.0]])
TOOL_HEIGHT = 0.30


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

    def validate(self):
        values = (self.width, self.duration, self.timestep, self.max_velocity,
                  self.max_acceleration, self.max_duration, self.torque_limit,
                  self.tracking_tolerance, *self.kp, *self.kd)
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


def load_robot_case(run: Path, letter: str, method: str):
    if len(letter) != 1 or letter not in "ABCDEFGH" or method not in METHODS:
        raise ValueError("choose letter A-H and a supported learning method")
    source = run / "examples.json"
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
    duration = config.duration
    for _ in range(12):
        intervals = int(np.ceil(duration / config.timestep))
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
            return {"t": t, "target": position, "q_ref": joints, "qd_ref": velocity,
                    "qdd_ref": acceleration}, {"scale_m_per_dataset_unit": scale,
                    "training_bbox_min": low.tolist(), "training_bbox_max": high.tolist(),
                    "dataset_origin": origin.tolist(), "robot_center_m": list(config.center),
                    "plane_z_m": TOOL_HEIGHT, "motion_duration_s": duration,
                    "time_warp": "10*u^3-15*u^4+6*u^5; PCHIP of source positions"}
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
    arrays = {name: np.empty((count, 2)) for name in ("q", "qd", "tip", "elbow", "torque")}
    contacts, saturated = np.zeros(count, dtype=int), np.zeros(count, dtype=bool)
    # Initial configuration only. Every later state comes from mj_step.
    data.qpos[:] = reference["q_ref"][0]
    mujoco.mj_forward(model, data)
    def record(index, torque):
        arrays["q"][index] = data.qpos
        arrays["qd"][index] = data.qvel
        arrays["tip"][index] = data.site("tip").xpos[:2]
        arrays["elbow"][index] = data.site("elbow_site").xpos[:2]
        arrays["torque"][index] = torque
        contacts[index] = data.ncon
    record(0, np.zeros(2))
    for index in range(count - 1):
        requested = (np.asarray(config.kp) * (reference["q_ref"][index] - data.qpos)
                     + np.asarray(config.kd) * (reference["qd_ref"][index] - data.qvel))
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
    return {"tracking_rmse_m": float(np.sqrt(np.mean(error ** 2))),
            "max_tracking_error_m": maximum, "endpoint_error_m": float(error[-1]),
            "peak_torque_nm": float(np.max(np.abs(arrays["torque"]))),
            "saturation_fraction": float(arrays["saturated"][1:].mean()),
            "contact_samples": int(np.count_nonzero(arrays["contacts"])),
            "joint_limits_satisfied": joints_valid, "actual_limits_satisfied": bool(within_limits),
            "actual_peak_velocity_rad_s": speed, "actual_peak_acceleration_rad_s2": acceleration,
            "reference_peak_velocity_rad_s": float(np.max(np.abs(reference["qd_ref"]))),
            "reference_peak_acceleration_rad_s2": float(np.max(np.abs(reference["qdd_ref"]))),
            "tracking_passed": passed, "samples": len(error)}


def simulate_robot(run, output, letter="G", method="KMP", config=None):
    config = config or RobotConfig()
    config.validate()
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
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "robot_trace.npz", **reference, **arrays)
    (output / "model.xml").write_text(xml, encoding="utf-8")
    write_json(output / "manifest.json", manifest)
    columns = {"time_s": reference["t"], "error_m": arrays["error_m"],
               "contacts": arrays["contacts"], "torque_saturated": arrays["saturated"]}
    for name in ("target", "tip", "elbow", "q_ref", "qd_ref", "qdd_ref", "q", "qd", "torque"):
        for i in range(2):
            columns[f"{name}_{i}"] = (reference if name in reference else arrays)[name][:, i]
    pd.DataFrame(columns).to_csv(output / "robot_trace.csv", index=False)
    indices = np.unique(np.linspace(0, len(reference["t"]) - 1, min(1201, len(reference["t"])), dtype=int))
    payload = {"manifest": manifest, "t": reference["t"][indices].tolist()}
    payload.update({name: arrays[name][indices].tolist() for name in ("tip", "elbow", "error_m")})
    payload["target"] = reference["target"][indices].tolist()
    template = Path(__file__).with_name("robot_dashboard.html").read_text(encoding="utf-8")
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    (output / "dashboard.html").write_text(template.replace("__ROBOT_DATA__", embedded), encoding="utf-8")
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
              "All methods use the same fixed joint PD, so this validates tracking of a learned path, "
              "not uncertainty-aware control. One representative fold-0 example is used.\n")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Robot simulation: tracking_passed={metrics['tracking_passed']}; "
          f"RMSE={metrics['tracking_rmse_m'] * 1000:.3f} mm; max={metrics['max_tracking_error_m'] * 1000:.3f} mm", flush=True)
    print(f"Open {output / 'dashboard.html'}", flush=True)
    return manifest


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
