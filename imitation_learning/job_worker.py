"""One local queue supervisor; scientific computation runs in child processes."""
import csv
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback


class WorkspaceLock:
    def __init__(self, root):
        self.path = Path(root) / '.server.lock'
        self.file = None

    def acquire(self):
        self.file = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.file.write(b'0')
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.file.close()
            self.file = None
            raise RuntimeError('another server is using this workspace') from error

    def release(self):
        if self.file:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
            self.file = None


def metric_value(value):
    if value == '':
        return None
    if value in ('True', 'False'):
        return value == 'True'
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return value


def collect_results(folder, kind):
    output = folder / 'output'
    learning = output / 'learning' if kind == 'demo' else output
    metrics, summary = [], {}
    manifest = learning / 'manifest.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        summary['learning'] = {key: data.get(key) for key in ('status', 'recorded_trials', 'failed_trials', 'partial_cross_validation')}
    for source, path in (('learning', learning / 'metrics.csv'), ('simulation', output / 'robot' / 'summary.csv')):
        if path.is_file():
            with path.open(encoding='utf-8', newline='') as stream:
                records = [{k: metric_value(v) for k, v in row.items()} for row in csv.DictReader(stream)]
            metrics.extend((source, record) for record in records)
            summary[source + '_rows'] = len(records)
    pages = {}
    for name, path in (('learning', learning / 'dashboard.html'), ('simulation', output / 'robot' / 'dashboard.html')):
        if path.is_file():
            pages[name] = path.relative_to(folder).as_posix()
    summary['pages'] = pages
    extensions = {'.html', '.csv', '.json', '.png', '.npz', '.xml', '.md'}
    artifacts = [(p.relative_to(folder).as_posix(), p.stat().st_size) for p in output.rglob('*')
                 if p.is_file() and not p.is_symlink() and p.suffix.lower() in extensions
                 and p.resolve().is_relative_to(folder.resolve())]
    return summary, metrics, artifacts


class QueueWorker:
    def __init__(self, store, command_factory=None):
        self.store = store
        self.stop_event = threading.Event()
        self.thread = None
        self.command_factory = command_factory or self.command

    @staticmethod
    def command(job, folder):
        action = 'demo' if job['kind'] == 'demo' else 'run'
        return [sys.executable, '-u', '-m', 'imitation_learning', action,
                '--config', str(folder / 'config.json'), '--output', str(folder / 'output')]

    def start(self):
        self.store.recover()
        self.thread = threading.Thread(target=self.loop, name='experiment-queue', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(15)
            if self.thread.is_alive():
                raise RuntimeError('experiment worker did not stop')

    def loop(self):
        while not self.stop_event.is_set():
            try:
                job = self.store.claim()
                if job:
                    self.execute(job)
                else:
                    self.stop_event.wait(.2)
            except Exception:
                logging.exception('experiment queue error')
                self.stop_event.wait(.5)

    @staticmethod
    def terminate(process):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def execute(self, job):
        folder = self.store.root / 'jobs' / job['id']
        process = None
        try:
            folder.mkdir(exist_ok=True)
            (folder / 'config.json').write_text(json.dumps(job['config'], indent=2), encoding='utf-8')
            env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:backslashreplace',
                       MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            with (folder / 'task.log').open('ab', buffering=0) as log:
                process = subprocess.Popen(self.command_factory(job, folder), stdout=log, stderr=subprocess.STDOUT,
                                           cwd=Path(__file__).resolve().parents[1], env=env, shell=False)
                interrupted = False
                while process.poll() is None:
                    if self.stop_event.wait(.1):
                        interrupted = True
                        self.terminate(process)
                        break
                    if self.store.get(job['id'])['state'] == 'cancelling':
                        self.terminate(process)
                        break
                if interrupted:
                    self.store.complete(job['id'], 'interrupted', process.returncode, 'Server stopped; retry creates a new run.')
                elif self.store.get(job['id'])['state'] == 'cancelling':
                    self.store.complete(job['id'], 'cancelled', process.returncode)
                else:
                    summary, metrics, artifacts = collect_results(folder, job['kind'])
                    state = 'succeeded' if process.returncode == 0 else 'failed'
                    error = None if state == 'succeeded' else f'Experiment exited with code {process.returncode}; inspect logs and recorded results.'
                    self.store.complete(job['id'], state, process.returncode, error, summary, metrics, artifacts)
        except Exception as error:
            if process:
                self.terminate(process)
            try:
                with (folder / 'task.log').open('a', encoding='utf-8') as log:
                    log.write(traceback.format_exc())
            except OSError:
                pass
            self.store.complete(job['id'], 'failed', error=str(error))
