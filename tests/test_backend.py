"""Persistence, process lifecycle, HTTP boundaries and real scientific-job integration."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from imitation_learning.job_store import JobStore, TERMINAL
from imitation_learning.job_worker import QueueWorker, WorkspaceLock

SERVER = all(importlib.util.find_spec(name) for name in ('fastapi', 'httpx'))
MUJOCO = importlib.util.find_spec('mujoco') is not None


def fixture_command(job, folder):
    """Explicit process/UI fixture, not an algorithm result or benchmark."""
    seed = job['params'].get('seed', 42)
    code = '''from pathlib import Path
import json,time,sys
root=Path(sys.argv[1]); seed=int(sys.argv[2])
print('fixture process started',flush=True)
if seed==14: time.sleep(60)
if seed==13: print('deliberate fixture failure',flush=True); sys.exit(17)
out=root/'output'; out.mkdir()
(out/'metrics.csv').write_text('letter,method,scenario,rmse\\nG,KMP,reproduction,0.125\\n',encoding='utf-8')
(out/'dashboard.html').write_text('<html><body>Explicit UI test fixture</body></html>',encoding='utf-8')
(out/'manifest.json').write_text(json.dumps({'status':'completed','failed_trials':0}),encoding='utf-8')
print('fixture process done',flush=True)
'''
    return [sys.executable, '-u', '-c', code, str(folder), str(seed)]


def wait_job(client, job_id, predicate=lambda job: job['state'] in TERMINAL, timeout=30):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        job = client.get('/api/jobs/' + job_id).json()
        if predicate(job):
            return job
        time.sleep(.08)
    raise AssertionError('task did not reach expected state: ' + str(job))


class QueuePersistenceTests(unittest.TestCase):
    def test_durable_fifo_claims_are_atomic_across_connections(self):
        with tempfile.TemporaryDirectory() as folder:
            store = JobStore(folder)
            ids = [store.create('learning', {'seed': index}, {'seed': index})['id'] for index in range(8)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                jobs = list(pool.map(lambda _: store.claim(), range(12)))
            claimed = [job['id'] for job in jobs if job]
            self.assertEqual(set(claimed), set(ids))
            self.assertEqual(len(claimed), len(set(claimed)))
            pending = store.create('learning', {}, {})
            reopened = JobStore(folder)
            reopened.recover()
            self.assertEqual(reopened.list('interrupted')['total'], 8)
            self.assertEqual(reopened.get(pending['id'])['state'], 'queued')
            self.assertEqual(reopened.claim()['id'], pending['id'])

    def test_exclusive_workspace_ownership_and_release(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = WorkspaceLock(folder), WorkspaceLock(folder)
            a.acquire()
            try:
                with self.assertRaises(RuntimeError):
                    b.acquire()
            finally:
                a.release()
            b.acquire()
            b.release()


@unittest.skipUnless(SERVER, 'install server extra and httpx')
class BackendApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from imitation_learning.server import create_app
        self.temp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp.name) / '实验记录'
        self.app = create_app(self.workspace, start_worker=False)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def submit(self, **values):
        response = self.client.post('/api/jobs', json=values)
        self.assertEqual(response.status_code, 202, response.text)
        return response.json()

    def test_validation_listing_cancellation_and_retry_persist(self):
        for payload in ({'letters': ['G', 'G']}, {'methods': ['unknown']}, {'points': 1}, {'output': '../outside'},
                        {'kind': 'demo', 'scenarios': ['missing']}, {'seed': -1}):
            self.assertEqual(self.client.post('/api/jobs', json=payload).status_code, 422)
        job = self.submit(letters=['A'], methods=['KMP'])
        self.assertEqual(self.client.post('/api/jobs/' + job['id'] + '/retry').status_code, 409)
        cancelled = self.client.post('/api/jobs/' + job['id'] + '/cancel').json()
        self.assertEqual(cancelled['state'], 'cancelled')
        retried = self.client.post('/api/jobs/' + job['id'] + '/retry').json()
        self.assertNotEqual(retried['id'], job['id'])
        self.assertEqual(retried['parent_id'], job['id'])
        self.assertEqual(retried['config'], job['config'])
        self.assertEqual(JobStore(self.workspace).list()['total'], 2)
        self.assertEqual(self.client.get('/api/jobs?state=cancelled').json()['total'], 1)
        self.assertEqual(self.client.get('/api/jobs?limit=1&offset=1').json()['items'][0]['id'], job['id'])
        self.assertEqual(self.client.get('/api/jobs?state=invalid').status_code, 422)
        self.assertEqual(self.client.get('/api/jobs/missing').status_code, 404)

    def test_metrics_isolation_registered_files_and_path_boundaries(self):
        job = self.submit()
        other = self.submit()
        folder = self.workspace / 'jobs' / job['id']
        (folder / 'output').mkdir(parents=True)
        (folder / 'output' / 'report.html').write_text('<html>report</html>', encoding='utf-8')
        (folder / 'secret.txt').write_text('not registered')
        self.app.state.store.claim()
        self.app.state.store.complete(job['id'], 'succeeded', 0, metrics=[('learning', {'letter': 'G', 'method': 'KMP', 'rmse': .125})],
                                      artifacts=[('output/report.html', 19)])
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/metrics?method=KMP").json()['items'][0]['rmse'], .125)
        self.assertEqual(self.client.get(f"/api/jobs/{other['id']}/metrics").json()['total'], 0)
        prefix = f"/api/jobs/{job['id']}/files/"
        self.assertEqual(self.client.get(prefix + 'output/report.html').status_code, 200)
        self.assertIn('attachment', self.client.get(prefix + 'output/report.html?download=true').headers['content-disposition'])
        for path in ('secret.txt', '%2e%2e/experiments.sqlite3', '..%5Cexperiments.sqlite3', 'C:private'):
            self.assertEqual(self.client.get(prefix + path).status_code, 404)
        outside = self.workspace / 'outside.html'
        outside.write_text('private')
        linked = folder / 'output' / 'linked.html'
        try:
            linked.symlink_to(outside)
        except OSError:
            pass  # Windows runners may lack symlink privileges.
        else:
            with self.app.state.store.connection(True) as db:
                db.execute('INSERT INTO artifacts VALUES(?,?,?)', (job['id'], 'output/linked.html', 7))
            self.assertEqual(self.client.get(prefix + 'output/linked.html').status_code, 404)

    def test_origin_host_and_log_limits(self):
        self.assertEqual(self.client.post('/api/jobs', json={}, headers={'origin': 'https://elsewhere.example'}).status_code, 403)
        self.assertEqual(self.client.get('/api/health', headers={'host': 'elsewhere.example'}).status_code, 400)
        job = self.submit()
        folder = self.workspace / 'jobs' / job['id']
        folder.mkdir()
        (folder / 'task.log').write_text('中文路径\n' + 'x' * 2000, encoding='utf-8')
        data = self.client.get(f"/api/jobs/{job['id']}/logs?tail_bytes=100").json()
        self.assertTrue(data['truncated'])
        self.assertEqual(len(data['text']), 100)
        self.assertEqual(self.client.get(f"/api/jobs/{job['id']}/logs?tail_bytes=0").status_code, 422)


@unittest.skipUnless(SERVER, 'install server extra and httpx')
class BackendProcessTests(unittest.TestCase):
    def test_process_results_failure_cancel_and_shutdown(self):
        from fastapi.testclient import TestClient
        from imitation_learning.server import create_app
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(folder, command_factory=fixture_command)) as client:
                job = client.post('/api/jobs', json={}).json()
                done = wait_job(client, job['id'])
                self.assertEqual(done['state'], 'succeeded', done)
                self.assertEqual(done['summary']['learning_rows'], 1)
                self.assertIn('fixture process done', client.get(f"/api/jobs/{job['id']}/logs").json()['text'])
                failed = client.post('/api/jobs', json={'seed': 13}).json()
                failed = wait_job(client, failed['id'])
                self.assertEqual((failed['state'], failed['exit_code']), ('failed', 17))
                slow = client.post('/api/jobs', json={'seed': 14}).json()
                wait_job(client, slow['id'], lambda j: j['state'] == 'running')
                start = time.monotonic()
                self.assertEqual(client.get('/api/health').status_code, 200)
                self.assertLess(time.monotonic() - start, 1)
                client.post(f"/api/jobs/{slow['id']}/cancel")
                self.assertEqual(wait_job(client, slow['id'])['state'], 'cancelled')
                interrupted = client.post('/api/jobs', json={'seed': 14}).json()
                wait_job(client, interrupted['id'], lambda j: j['state'] == 'running')
                queued = client.post('/api/jobs', json={}).json()
            self.assertEqual(JobStore(folder).get(interrupted['id'])['state'], 'interrupted')
            with TestClient(create_app(folder, command_factory=fixture_command)) as client:
                self.assertEqual(wait_job(client, queued['id'])['state'], 'succeeded')
                self.assertEqual(client.get(f"/api/jobs/{interrupted['id']}").json()['state'], 'interrupted')

    def test_real_learning_cli_stores_queryable_metrics_and_reports(self):
        from fastapi.testclient import TestClient
        from imitation_learning.server import create_app
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(Path(folder) / '真实评测')) as client:
                job = client.post('/api/jobs', json={'letters': ['G'], 'methods': ['KMP'],
                                  'scenarios': ['reproduction', 'missing'], 'points': 32}).json()
                done = wait_job(client, job['id'], timeout=120)
                self.assertEqual(done['state'], 'succeeded', client.get(f"/api/jobs/{job['id']}/logs").json())
                rows = client.get(f"/api/jobs/{job['id']}/metrics?source=learning&method=KMP").json()
                self.assertGreater(rows['total'], 0)
                page = done['summary']['pages']['learning']
                self.assertEqual(client.get(f"/api/jobs/{job['id']}/files/{page}").status_code, 200)
                self.assertEqual(rows['total'], done['summary']['learning_rows'])

    @unittest.skipUnless(MUJOCO, 'install simulation extra')
    def test_real_demo_runs_both_physical_controllers(self):
        from fastapi.testclient import TestClient
        from imitation_learning.server import create_app
        with tempfile.TemporaryDirectory() as folder:
            with TestClient(create_app(folder)) as client:
                job = client.post('/api/jobs', json={'kind': 'demo', 'letters': ['A'], 'methods': ['KMP', 'U-KMP'],
                                                   'scenarios': ['reproduction'], 'points': 32}).json()
                done = wait_job(client, job['id'], timeout=120)
                self.assertEqual(done['state'], 'succeeded', client.get(f"/api/jobs/{job['id']}/logs").json())
                rows = client.get(f"/api/jobs/{job['id']}/metrics?source=simulation").json()
                self.assertEqual(rows['total'], 2)
                self.assertEqual({r['controller'] for r in rows['items']}, {'fixed_lqr', 'covariance_lqr'})
                page = done['summary']['pages']['simulation']
                self.assertEqual(client.get(f"/api/jobs/{job['id']}/files/{page}").status_code, 200)


if __name__ == '__main__':
    unittest.main()
