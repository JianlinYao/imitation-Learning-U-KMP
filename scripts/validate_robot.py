"""Run every saved A-H reproduction example; retain traces and all failures."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from imitation_learning.experiments import write_json
from imitation_learning.models import METHODS
from imitation_learning.robot import (RobotConfig, load_robot_case, model_xml,
                                     prepare_reference, rollout, tracking_metrics)


def main():
    import mujoco
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty")
    source = args.run / "examples.json"
    source_hash = sha256(source.read_bytes()).hexdigest()
    config = RobotConfig()
    xml = model_xml(config)
    args.output.mkdir(parents=True, exist_ok=True)
    traces = args.output / "traces"
    traces.mkdir()
    rows = []
    for letter in "ABCDEFGH":
        for method in METHODS:
            try:
                case, entry, _ = load_robot_case(args.run, letter, method)
                reference, mapping = prepare_reference(case, entry, config)
                arrays = rollout(reference, config, xml)
                np.savez_compressed(traces / f"{letter}-{method}.npz", **reference, **arrays)
                row = {"letter": letter, "method": method, "status": "ok", "fold": case["fold"],
                       "demo": case["demo"], "duration_s": mapping["motion_duration_s"],
                       **tracking_metrics(reference, arrays, config)}
            except (ValueError, RuntimeError, FloatingPointError) as error:
                row = {"letter": letter, "method": method, "status": "error", "error": str(error), "tracking_passed": False}
            rows.append(row)
        print(letter, f"{sum(r['tracking_passed'] for r in rows[-len(METHODS):])}/{len(METHODS)} passed", flush=True)
        pd.DataFrame(rows).to_csv(args.output / "summary.csv", index=False)
    manifest = {"stage": "40 representative MuJoCo reproduction cases, not full cross-validation or hardware",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "mujoco": mujoco.__version__,
                "source_examples_sha256": source_hash, "config": asdict(config),
                "model_sha256": sha256(xml.encode()).hexdigest(), "expected_cases": 8 * len(METHODS),
                "recorded_cases": len(rows), "tracking_passed": sum(bool(row["tracking_passed"]) for row in rows),
                "execution_errors": sum(row["status"] != "ok" for row in rows)}
    write_json(args.output / "manifest.json", manifest)
    (args.output / "model.xml").write_text(xml, encoding="utf-8")
    return 0 if manifest["tracking_passed"] == manifest["expected_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
