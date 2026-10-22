"""Generate static figures, a Markdown result report and an offline HTML viewer."""
import html
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as CirclePatch
import numpy as np
import pandas as pd

from .models import METHODS

COLORS = {"DMP": "#d47720", "ProMP": "#7659a8", "GP": "#238865", "KMP": "#3165bd", "U-KMP": "#dd4962"}


def _pm(row, metric):
    mean, deviation = row.get(f"{metric}_mean"), row.get(f"{metric}_std")
    if mean is None or pd.isna(mean):
        return "—"
    return f"{mean:.4g}" if pd.isna(deviation) else f"{mean:.4g} ± {deviation:.3g}"


def make_report(output: Path | str):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["report_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    for source in (Path(__file__), Path(__file__).with_name("dashboard.html")):
        manifest["source_sha256"][f"imitation_learning/{source.name}"] = sha256(source.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cases = json.loads((output / "examples.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(output / "summary.csv")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "figure.dpi": 120})
    scenarios = [s for s in ("reproduction", "missing", "obstacle") if s in set(summary["scenario"])]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4), squeeze=False)
    for axis, scenario in zip(axes[0], scenarios):
        selected = summary[(summary["scenario"] == scenario) & (summary["valid_trials"] > 0)]
        if "rmse_mean" in selected:
            axis.bar(selected["method"], selected["rmse_mean"],
                     yerr=selected["rmse_std"].fillna(0), capsize=3,
                     color=[COLORS[m] for m in selected["method"]])
        axis.set(title=scenario.capitalize(), ylabel="RMSE (dataset units)")
    fig.suptitle("Held-out demonstration error · bars show trial standard deviation")
    fig.tight_layout()
    fig.savefig(figures / "summary_rmse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    for scenario in scenarios:
        key = f"G/{scenario}" if f"G/{scenario}" in cases else next((k for k in cases if k.endswith("/" + scenario)), None)
        if key is None:
            continue
        case = cases[key]
        truth = np.asarray(case["truth"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(truth[:, 0], truth[:, 1], "k--", linewidth=1.5, label="Held-out truth")
        if case.get("obstacle"):
            obstacle = case["obstacle"]
            axes[0].add_patch(CirclePatch(obstacle["center"], obstacle["radius"], color="#dd4962", alpha=0.22))
            axes[0].add_patch(CirclePatch(obstacle["center"], obstacle["radius"] + obstacle["margin"],
                                          fill=False, linestyle=":", color="#dd4962"))
        for method, data in case["methods"].items():
            path = np.asarray(data["position"])
            style = "--" if method == "U-KMP" else "-"
            axes[0].plot(path[:, 0], path[:, 1], style, color=COLORS[method], label=method, linewidth=1.6)
            axes[1].plot(case["time"], path[:, 0], style, color=COLORS[method], label=method)
        for method in ("U-KMP", "KMP", "GP", "ProMP"):
            data = case["methods"].get(method)
            if data and data["position_std"] is not None:
                mean, std = np.asarray(data["position"]), np.asarray(data["position_std"])
                axes[1].fill_between(case["time"], mean[:, 0] - 1.96 * std[:, 0], mean[:, 0] + 1.96 * std[:, 0],
                                     color=COLORS[method], alpha=0.13, label=f"{method} nominal 95%")
                break
        axes[1].plot(case["time"], truth[:, 0], "k--", linewidth=1.2, label="Held-out truth")
        gap = np.asarray(case["gap"], dtype=bool)
        if gap.any():
            times = np.asarray(case["time"])
            axes[1].axvspan(times[gap][0], times[gap][-1], color="#d47720", alpha=0.12, label="Hidden interval")
        axes[0].set(aspect="equal", xlabel="x (dataset units)", ylabel="y (dataset units)")
        axes[1].set(xlabel="Assumed time (s)", ylabel="x (dataset units)")
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{case['letter']} · {scenario} · held-out demo {case['demo']} · fold {case['fold']}")
        fig.tight_layout()
        fig.savefig(figures / f"trajectory_{scenario}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    lines = ["# Recomputed experiment report", "", "These are new measurements from the recorded configuration, not recovered dissertation tables.", "",
             f"- Status: **{manifest['status']}**", f"- Trials: {manifest['valid_trials']} valid / {manifest['expected_trials']} expected; {manifest['failed_trials']} errors.",
             f"- Partial cross-validation: {manifest['partial_cross_validation']}", "- Planned trajectories and closed-loop control are evaluated separately.",
             "- KMP and U-KMP have the same planned mean/covariance; their planning errors may be identical.",
             "- GP position bands include per-demo observation noise; controllers use the latent state covariance.",
             "- Calibration, when enabled, reserves whole outer-training demos from both model and scaler fit. It is empirical marginal interval inflation, not a trajectory-level coverage guarantee.",
             f"- KMP adaptation mode: {manifest['config'].get('model', {}).get('kmp_adaptation', 'gaussian')}",
             f"- Reproduction RMSE / training span threshold: {manifest['config'].get('reproduction_rmse_tolerance_fraction', 'legacy endpoint-only')}",
             f"- Missing-gap RMSE / training span threshold: {manifest['config'].get('missing_gap_tolerance_fraction', 'legacy endpoint-only')}",
             "- No outlier filtering or preferred-method selection is applied.", ""]
    for scenario in scenarios:
        lines += [f"## {scenario.capitalize()}", "", "| Method | Valid / total | RMSE | End error | Jerk | Gap RMSE | Collision rate | Inference ms |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for _, row in summary[summary["scenario"] == scenario].iterrows():
            lines.append(f"| {row['method']} | {int(row['valid_trials'])} / {int(row['total_trials'])} | "
                         + " | ".join(_pm(row, m) for m in ("rmse", "end_err", "jerk", "gap_rmse", "collision", "infer_ms")) + " |")
        lines.append("")
    if 'calibrated_coverage95_mean' in summary:
        lines += ['## Predictive uncertainty', '',
                  '| Scenario | Method | Raw coordinate coverage | Gap coverage | Calibrated coverage | Calibrated gap coverage | Marginal position NLPD | Task success |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for _, row in summary.iterrows():
            lines.append(f"| {row['scenario']} | {row['method']} | " + ' | '.join(_pm(row, key) for key in
                ('coverage95','gap_coverage95','calibrated_coverage95','calibrated_gap_coverage95','nlpd','task_success')) + ' |')
        lines += ['', 'Raw and calibrated coverage are recorded separately; a nominal 95% label is not proof of calibration. '
                  'No held-out test residual is used to fit inflation. Obstacle detours have no calibrated band because '
                  'the original demonstration is not a target distribution for a deliberate detour. See calibration.json and splits.json.', '']
    lines += ["## Interpretation", "", "Obstacle RMSE measures deviation from the original demonstration, which can increase for a valid detour. Read it alongside continuous clearance, goal reach and collision rate. DMP is an endpoint-only baseline with no obstacle coupling or gap conditioning.", "",
              "Training is performed once per method/fold and its measured cost is repeated on the corresponding trial rows; do not sum these repeated costs. Inference uses one warmup followed by the median of the configured repeats. Adaptation includes constraint construction and obstacle repair. Timing depends on the recorded runtime and hardware.", "",
              "Control ablation uses the same predicted reference, unit-mass system, input cost and acceleration disturbance for fixed-Q and covariance-aware LQR. It is a separate numerical experiment, not physical robot validation.", "",
              "See `manifest.json`, `splits.json`, `metrics.csv`, `fold_metrics.csv`, `control_ablation.csv` and the offline `dashboard.html` for provenance and details.", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {"manifest": manifest, "cases": cases,
               "summary": json.loads(summary.to_json(orient="records"))}
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    template = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
    title = html.escape("Imitation Learning · Experiment Viewer")
    rendered = template.replace("__PAGE_TITLE__", title).replace("__RUN_DATA__", encoded)
    (output / "dashboard.html").write_text(rendered, encoding="utf-8")
