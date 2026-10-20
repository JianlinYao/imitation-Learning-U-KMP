"""Paired whole-demonstration cross-validation and auditable experiment outputs."""
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .control import track_lqr
from .data import SpatialScaler, folds, load_letter, observed_segments, resample, smooth, time_grid
from .distributions import Observation
from .geometry import clearance_metrics, detour_points, make_obstacle, repair_point
from .metrics import trajectory_metrics
from .models import METHODS, ModelConfig, make_model

SCENARIOS = ("reproduction", "missing", "obstacle")
NUMERIC_METRICS = ("rmse", "end_err", "jerk", "gap_rmse", "coverage95", "gap_coverage95",
                   "mean_position_variance", "min_clearance", "collision", "margin_satisfied",
                   "goal_reached", "task_success", "train_ms", "adapt_ms", "infer_ms")


@dataclass(frozen=True)
class RunConfig:
    letters: tuple[str, ...] = tuple("ABCDEFGH")
    methods: tuple[str, ...] = METHODS
    scenarios: tuple[str, ...] = SCENARIOS
    folds: int = 5
    seed: int = 42
    points: int = 200
    duration: float = 0.995
    smoothing_window: int = 11
    smoothing_order: int = 3
    missing_start: float = 0.4
    missing_end: float = 0.6
    observed_points: int = 20
    observation_variance: float = 1e-5
    obstacle_radius_fraction: float = 0.07
    obstacle_margin_fraction: float = 0.02
    obstacle_repair_steps: int = 8
    goal_tolerance_fraction: float = 0.03
    timing_repeats: int = 3
    fold_limit: int | None = None
    save_predictions: bool = True
    control_ablation: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self):
        if not self.letters or any(len(s) != 1 or s not in "ABCDEFGH" for s in self.letters):
            raise ValueError("letters must be selected from A through H")
        if len(set(self.letters)) != len(self.letters):
            raise ValueError("letters must be unique")
        if not self.methods or any(m not in METHODS for m in self.methods) or len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must be unique, supported method names")
        if not self.scenarios or any(s not in SCENARIOS for s in self.scenarios) or len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios must be unique, supported scenario names")
        time_grid(self.points, self.duration)
        smooth(np.zeros((self.points, 2)), self.smoothing_window, self.smoothing_order)
        if self.folds < 2 or not 0 < self.missing_start < self.missing_end < 1:
            raise ValueError("invalid fold count or missing interval")
        if self.observed_points < 4 or self.observed_points > self.points:
            raise ValueError("observed_points must be between 4 and points")
        if self.timing_repeats < 1 or self.obstacle_repair_steps < 0:
            raise ValueError("timing repeats must be positive; repair steps must be nonnegative")
        if self.fold_limit is not None and not 1 <= self.fold_limit <= self.folds:
            raise ValueError("fold_limit must be between 1 and folds")
        for value in (self.observation_variance, self.obstacle_radius_fraction,
                      self.goal_tolerance_fraction, self.model.ridge, self.model.length_scale,
                      self.model.gp_noise_variance, self.model.kmp_mean_regularization,
                      self.model.kmp_covariance_regularization):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("variances, lengths and regularizers must be positive and finite")
        if self.obstacle_margin_fraction < 0 or self.model.basis_functions < 3 or self.model.gmm_components < 1:
            raise ValueError("invalid geometry or model size")
        if self.model.reference_points < 3:
            raise ValueError("reference_points must be >= 3")


def read_config(path: Path | str) -> RunConfig:
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    values["model"] = ModelConfig(**values.get("model", {}))
    for key in ("letters", "methods", "scenarios"):
        if key in values:
            values[key] = tuple(values[key])
    config = RunConfig(**values)
    config.validate()
    return config


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def endpoints(truth, scaler, config):
    return [Observation(index, tuple(scaler.transform(truth[[index]])[0]), variance=config.observation_variance)
            for index in (0, len(truth) - 1)]


def case_observations(scenario, truth, t, scaler, reference, model, config):
    boundary = endpoints(truth, scaler, config)
    gap = (t >= config.missing_start * t[-1]) & (t < config.missing_end * t[-1])
    circle, attempts = None, 0
    observations = boundary
    if scenario == "missing":
        visible = ~gap
        observed = observed_segments(truth, visible, config.smoothing_window, config.smoothing_order)
        indices = np.flatnonzero(visible)
        selected = indices[np.unique(np.linspace(0, len(indices) - 1, config.observed_points, dtype=int))]
        observations = [Observation(int(i), tuple(scaler.transform(observed[[i]])[0]),
                                    variance=config.observation_variance) for i in selected] + boundary
    elif scenario == "obstacle":
        circle = make_obstacle(reference, config.obstacle_radius_fraction, config.obstacle_margin_fraction)
        if model.supports_via_points:
            observations = [Observation(index, tuple(scaler.transform(point[None, :])[0]), variance=1e-6)
                            for index, point in detour_points(reference, circle)] + boundary
            for _ in range(config.obstacle_repair_steps):
                candidate = scaler.inverse(model.predict(observations).position)
                if clearance_metrics(candidate, circle)["margin_satisfied"]:
                    break
                index, point = repair_point(candidate, circle)
                observations = [o for o in observations if o.index != index]
                observations.append(Observation(index, tuple(scaler.transform(point[None, :])[0]), variance=1e-7))
                attempts += 1
    return observations, gap if scenario == "missing" else None, circle, attempts


def measured_predict(model, observations, repeats):
    model.predict(observations)  # one explicit, untimed warmup
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        prediction = model.predict(observations)
        elapsed.append((perf_counter() - start) * 1000)
    return prediction, float(np.median(elapsed))


def _source_metadata(root, data_root, config):
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "threadpoolctl"):
        versions[package] = importlib.metadata.version(package)
    return {"started_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "packages": versions, "blas_threads": 1,
            "source_revision_at_run": git("rev-parse", "HEAD"),
            "source_dirty_at_run": bool(git("status", "--porcelain")),
            "source_sha256": {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                              for p in sorted((root / "imitation_learning").glob("*.py"))},
            "input_sha256": {f"{letter}.mat": sha256((data_root / f"{letter}.mat").read_bytes()).hexdigest()
                             for letter in config.letters},
            "config": asdict(config), "evaluation_stage": "planned trajectory",
            "coordinate_units": "original dataset units, not verified robot SI units",
            "time_assumption": f"all demonstrations resampled to {config.duration} seconds",
            "target_preprocessing": "resampled unsmoothed held-out positions",
            "intervals": "marginal Gaussian predictive bands; nominal 95%, not guaranteed calibrated",
            "kmp_ukmp": "identical planned distribution; covariance-aware control evaluated separately",
            "outlier_filter": None, "partial_cross_validation": config.fold_limit is not None and config.fold_limit < config.folds}


def aggregate_results(rows: pd.DataFrame):
    keys = ["scenario", "method"]
    successful = rows[rows["status"] == "ok"].copy()
    counts = rows.groupby(keys, sort=False).size().rename("total_trials")
    successes = successful.groupby(keys, sort=False).size().rename("valid_trials")
    parts = [counts, successes]
    for metric in NUMERIC_METRICS:
        if metric not in successful:
            continue
        group = successful.groupby(keys, sort=False)[metric]
        parts.extend([group.mean().rename(f"{metric}_mean"), group.std(ddof=1).rename(f"{metric}_std"),
                      group.count().rename(f"{metric}_count")])
    summary = pd.concat(parts, axis=1).reset_index()
    summary["valid_trials"] = summary["valid_trials"].fillna(0).astype(int)
    summary["failed_trials"] = summary["total_trials"] - summary["valid_trials"]
    return summary


def control_comparison(prediction, t, scaler, common, circle=None):
    """Same reference, system, R and disturbance; only state-cost policy changes."""
    disturbance = np.zeros((len(t) - 1, 2))
    active = (t[:-1] >= 0.45 * t[-1]) & (t[:-1] < 0.55 * t[-1])
    disturbance[active] = [4.0, -2.0]
    rows, examples = [], {}
    for adaptive in (False, True):
        start = perf_counter()
        result = track_lqr(prediction, t, adaptive=adaptive, disturbance=disturbance)
        control_ms = (perf_counter() - start) * 1000
        raw_state = scaler.inverse_state(result.state)
        label = "covariance-aware LQR" if adaptive else "fixed-Q LQR"
        row = common | {"controller": label, "status": "ok", "control_ms": control_ms,
                        "tracking_rmse": float(np.sqrt(np.mean((result.state[:, :2] - prediction.position) ** 2))),
                        "acceleration_rms": float(np.sqrt(np.mean(result.acceleration ** 2))),
                        "mean_position_gain": float(np.mean(np.linalg.norm(result.gains[:, :, :2], axis=(1, 2)))),
                        "peak_position_gain": float(np.max(np.linalg.norm(result.gains[:, :, :2], axis=(1, 2))))}
        if circle is not None:
            row.update(clearance_metrics(raw_state[:, :2], circle))
        rows.append(row)
        examples[label] = {"position": raw_state[:, :2], "position_gain": np.linalg.norm(result.gains[:, :, :2], axis=(1, 2)),
                           "metrics": row}
    return rows, examples


def run_benchmark(config: RunConfig, data_root: Path | str, output: Path | str) -> dict:
    config.validate()
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    # Validate all inputs before creating output or starting expensive work.
    dataset = {letter: load_letter(data_root, letter) for letter in config.letters}
    split_map = {letter: folds(len(items), config.folds, config.seed) for letter, items in dataset.items()}
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output is not empty: {output}; choose a new run directory")
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest = _source_metadata(root, data_root, config)
    manifest["status"] = "running"
    write_json(output / "manifest.json", manifest)
    write_json(output / "effective_config.json", asdict(config))
    t = time_grid(config.points, config.duration)
    rows, controls, saved_cases, split_records, errors = [], [], {}, [], []
    expected_trials = 0
    control_method = "KMP" if "KMP" in config.methods else "U-KMP"
    with threadpool_limits(limits=1):
        for letter, demos in dataset.items():
            selected_folds = split_map[letter][:config.fold_limit]
            for fold_index, (train_indices, test_indices) in enumerate(selected_folds):
                print(f"[{letter}] fold {fold_index + 1}/{len(selected_folds)}: "
                      f"{len(train_indices)} train, {len(test_indices)} test", flush=True)
                expected_trials += len(test_indices) * len(config.methods) * len(config.scenarios)
                preprocessing_start = perf_counter()
                training_raw = [smooth(resample(demos[i], t), config.smoothing_window, config.smoothing_order)
                                for i in train_indices]
                scaler = SpatialScaler.fit(training_raw)
                training = [scaler.transform(pos) for pos in training_raw]
                reference = np.mean(training_raw, axis=0)
                preprocess_ms = (perf_counter() - preprocessing_start) * 1000
                split_record = {"letter": letter, "fold": fold_index, "train_indices": train_indices.tolist(),
                                "test_indices": test_indices.tolist(), "scaler": scaler.as_dict(), "models": {}}
                split_records.append(split_record)
                for method in config.methods:
                    model = make_model(method, config.model, config.seed + fold_index)
                    training_start = perf_counter()
                    try:
                        model.fit(training, t)
                        train_ms = (perf_counter() - training_start) * 1000
                        split_record["models"][method] = model.metadata
                    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                        for test_index in test_indices:
                            for scenario in config.scenarios:
                                row = {"letter": letter, "fold": fold_index, "demo": int(test_index),
                                       "scenario": scenario, "method": method, "status": "fit_error", "error": str(error)}
                                rows.append(row)
                                errors.append(row)
                        continue
                    for test_index in test_indices:
                        truth = resample(demos[test_index], t)
                        for scenario in config.scenarios:
                            common = {"letter": letter, "fold": fold_index, "demo": int(test_index),
                                      "scenario": scenario, "method": method}
                            try:
                                adaptation_start = perf_counter()
                                observations, gap, circle, attempts = case_observations(
                                    scenario, truth, t, scaler, reference, model, config,
                                )
                                adapt_ms = (perf_counter() - adaptation_start) * 1000
                                prediction, infer_ms = measured_predict(model, observations, config.timing_repeats)
                                raw_state = scaler.inverse_state(prediction.state)
                                raw_covariance = None if prediction.covariance is None else scaler.inverse_covariance(prediction.covariance)
                                metrics = trajectory_metrics(truth, raw_state[:, :2], t, gap, raw_covariance)
                                span = max(float(np.linalg.norm(np.ptp(reference, axis=0))), 1e-8)
                                goal_reached = metrics["end_err"] <= config.goal_tolerance_fraction * span
                                row = common | metrics | {"status": "ok", "train_ms": train_ms, "adapt_ms": adapt_ms,
                                                          "infer_ms": infer_ms, "preprocess_ms": preprocess_ms,
                                                          "obstacle_repair_steps": attempts, "goal_reached": goal_reached,
                                                          "adaptation_supported": model.supports_via_points,
                                                          "task_success": goal_reached,
                                                          "observation_count": len(observations)}
                                if circle is not None:
                                    row.update(clearance_metrics(raw_state[:, :2], circle))
                                    row["task_success"] = bool(goal_reached and row["margin_satisfied"])
                                rows.append(row)
                                if config.save_predictions:
                                    folder = output / "predictions" / letter / f"fold-{fold_index}" / f"demo-{test_index}"
                                    folder.mkdir(parents=True, exist_ok=True)
                                    np.savez_compressed(folder / f"{scenario}-{method}.npz", t=t, truth=truth, state=raw_state,
                                                        covariance=np.empty((0, 4, 4)) if raw_covariance is None else raw_covariance,
                                                        gap=np.zeros(len(t), dtype=bool) if gap is None else gap,
                                                        observation_indices=np.array([o.index for o in observations]))
                                is_example = fold_index == 0 and test_index == test_indices[0]
                                if is_example:
                                    key = f"{letter}/{scenario}"
                                    case = saved_cases.setdefault(key, {"letter": letter, "scenario": scenario,
                                        "fold": fold_index, "demo": int(test_index), "time": t,
                                        "truth": truth, "training_mean": reference,
                                        "gap": np.zeros(len(t), dtype=bool) if gap is None else gap,
                                        "obstacle": None if circle is None else circle.as_dict(), "methods": {}})
                                    case["methods"][method] = {"position": raw_state[:, :2], "metrics": row,
                                        "position_std": None if raw_covariance is None else np.sqrt(np.maximum(
                                            np.diagonal(raw_covariance[:, :2, :2], axis1=1, axis2=2), 0.0))}
                                if config.control_ablation and method == control_method and test_index == test_indices[0]:
                                    control_rows, control_examples = control_comparison(prediction, t, scaler, common, circle)
                                    controls.extend(control_rows)
                                    if is_example:
                                        saved_cases[f"{letter}/{scenario}"]["controllers"] = control_examples
                            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                                # A failure is recorded and makes the CLI exit nonzero; no failed trial is discarded.
                                row = common | {"status": "error", "error": str(error)}
                                if rows and all(rows[-1].get(k) == common[k] for k in common):
                                    row = rows.pop() | {"status": "error", "error": str(error)}
                                rows.append(row)
                                errors.append(row)
                # Checkpoint at each completed fold.
                pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
                write_json(output / "splits.json", split_records)
                write_json(output / "errors.json", errors)
    frame = pd.DataFrame(rows)
    summary = aggregate_results(frame)
    summary.to_csv(output / "summary.csv", index=False)
    fold_metrics = frame[frame["status"] == "ok"].groupby(["scenario", "method", "letter", "fold"])[
        [m for m in NUMERIC_METRICS if m in frame]
    ].mean().reset_index()
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(controls).to_csv(output / "control_ablation.csv", index=False)
    write_json(output / "examples.json", saved_cases)
    manifest.update(status="completed" if not errors else "completed_with_errors", expected_trials=expected_trials,
                    recorded_trials=len(rows), valid_trials=int((frame["status"] == "ok").sum()), failed_trials=len(errors),
                    finished_at_utc=datetime.now(timezone.utc).isoformat())
    if len(rows) != expected_trials:
        raise RuntimeError(f"trial accounting mismatch: expected {expected_trials}, recorded {len(rows)}")
    write_json(output / "manifest.json", manifest)
    from .reporting import make_report
    make_report(output)
    print(f"Saved {len(rows)} trials ({len(errors)} errors) to {output}", flush=True)
    return manifest
