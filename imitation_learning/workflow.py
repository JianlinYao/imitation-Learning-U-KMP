"""One restartable learning -> physics -> offline viewer workflow."""
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

from .experiments import json_safe, run_benchmark, write_json
from .robot import RobotConfig, simulate_robot_grid


def run_demo(config, data_root, output):
    config.validate()
    if importlib.util.find_spec('mujoco') is None:
        raise RuntimeError('install simulation support first: python -m pip install -e ".[simulation]"')
    output = Path(output).resolve()
    request = json_safe({'config': asdict(config), 'data_root': str(Path(data_root).resolve()),
                         'controller': 'auto', 'common_duration': True})
    state_path = output / 'workflow.json'
    if output.exists() and any(output.iterdir()):
        if not state_path.is_file() or json.loads(state_path.read_text(encoding='utf-8')).get('request') != request:
            raise ValueError('demo output belongs to another configuration; choose a new --output directory')
    output.mkdir(parents=True, exist_ok=True)
    state = {'schema_version': 1, 'status': 'running', 'request': request,
             'started_at_utc': datetime.now(timezone.utc).isoformat(), 'stage': 'learning'}
    write_json(state_path, state)
    print('Stage 1/2: learning and held-out evaluation (completed folds are reused)', flush=True)
    learning = output / 'learning'
    manifest = run_benchmark(config, data_root, learning, resume=learning.exists())
    if manifest['failed_trials']:
        state.update(status='learning_errors', failed_trials=manifest['failed_trials'])
        write_json(state_path, state)
        raise RuntimeError(f"learning reported {manifest['failed_trials']} execution errors; inspect {learning / 'errors.json'}")
    state.update(stage='robot', learning_status=manifest['status'])
    write_json(state_path, state)
    print('Stage 2/2: physics simulation; shared duration per letter; U-KMP uses covariance-aware LQR', flush=True)
    robots = simulate_robot_grid(learning, output / 'robot', config.letters, config.methods,
                                 config=RobotConfig(controller='auto'), resume=True)
    passed = all(item['metrics']['tracking_passed'] for item in robots)
    state.update(status='completed' if passed else 'completed_with_tracking_failures',
                 stage='completed', robot_cases=len(robots), all_tracking_passed=passed,
                 page=str(output / 'robot' / 'dashboard.html'),
                 finished_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(state_path, state)
    print(f"Demo page: {state['page']}", flush=True)
    return state
