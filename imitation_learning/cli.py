"""CLI: inspect data, run experiments, regenerate reports or serve a local viewer."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .data import load_letter
from .experiments import read_config, run_benchmark


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Reproducible movement-primitive experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="validate and list original .mat demonstrations")
    inspect.add_argument("--data", type=Path, default=root / "2Dletters")
    run = sub.add_parser("run", help="run paired whole-demonstration cross-validation")
    run.add_argument("--config", type=Path, default=root / "configs" / "benchmark.json")
    run.add_argument("--data", type=Path, default=root / "2Dletters")
    run.add_argument("--output", type=Path)
    run.add_argument("--letters", nargs="+")
    run.add_argument("--methods", nargs="+")
    run.add_argument("--fold-limit", type=int)
    report = sub.add_parser("report", help="regenerate figures and offline viewer from recorded results")
    report.add_argument("output", type=Path)
    serve = sub.add_parser("serve", help="serve an existing run on localhost only")
    serve.add_argument("output", type=Path)
    serve.add_argument("--port", type=int, default=8000)
    simulate = sub.add_parser("simulate", help="track a saved reproduction example with a torque-driven MuJoCo teaching arm")
    simulate.add_argument("run", type=Path, help="learning run containing examples.json")
    selection = simulate.add_mutually_exclusive_group()
    selection.add_argument("--letter", default="G", choices=list("ABCDEFGH"))
    selection.add_argument("--letters", nargs="+", choices=list("ABCDEFGH"))
    selection.add_argument("--all-letters", action="store_true", help="simulate A-H and create one page with a letter selector")
    simulate.add_argument("--method", default="KMP", choices=["DMP", "ProMP", "GP", "KMP", "U-KMP"])
    simulate.add_argument("--output", type=Path)
    simulate.add_argument("--duration", type=float, default=8.0, help="minimum simulation duration in seconds; extended to respect reference limits")
    simulate.add_argument("--width", type=float, default=0.16, help="training-reference workspace width in meters")
    robot_view = sub.add_parser("robot-view", help="combine existing robot runs into one offline page with letter/method selectors")
    robot_view.add_argument("source", type=Path, help="directory containing saved robot simulations")
    robot_view.add_argument("--output", type=Path, help="HTML filename; defaults to SOURCE/dashboard.html")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            for letter in "ABCDEFGH":
                demos = load_letter(args.data, letter)
                print(f"{letter}: {len(demos)} demonstrations; lengths {sorted({len(p) for p in demos})}; coordinates=2")
            return 0
        if args.command == "run":
            config = read_config(args.config)
            overrides = {key: tuple(getattr(args, key)) for key in ("letters", "methods") if getattr(args, key) is not None}
            if args.fold_limit is not None:
                overrides["fold_limit"] = args.fold_limit
            config = replace(config, **overrides)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = args.output or root / "outputs" / stamp
            manifest = run_benchmark(config, args.data, output)
            print(f"Open {output.resolve() / 'dashboard.html'}")
            return 0 if manifest["failed_trials"] == 0 else 1
        if args.command == "simulate":
            from .robot import RobotConfig, simulate_robot, simulate_robot_letters
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = args.output or root / "outputs" / f"robot-{stamp}"
            config = RobotConfig(duration=args.duration, width=args.width)
            if args.all_letters or args.letters:
                letters = tuple("ABCDEFGH") if args.all_letters else tuple(args.letters)
                manifests = simulate_robot_letters(args.run, output, letters, args.method, config)
                return 0 if all(m["metrics"]["tracking_passed"] for m in manifests) else 1
            manifest = simulate_robot(args.run, output, args.letter, args.method, config)
            return 0 if manifest["metrics"]["tracking_passed"] else 1
        if args.command == "robot-view":
            from .robot import build_robot_gallery
            build_robot_gallery(args.source, args.output)
            return 0
        if args.command == "report":
            from .reporting import make_report
            make_report(args.output)
            print(f"Report regenerated in {args.output.resolve()}")
            return 0
        if args.command == "serve":
            directory = args.output.resolve()
            if not (directory / "dashboard.html").is_file():
                raise ValueError("output must contain a generated dashboard.html")
            handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
            with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
                print(f"Viewer: http://127.0.0.1:{args.port}/dashboard.html (Ctrl+C to stop)", flush=True)
                server.serve_forever()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
