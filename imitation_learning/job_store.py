"""Durable SQLite experiment queue, metrics and artifact catalog."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import uuid

TERMINAL = ('succeeded', 'failed', 'cancelled', 'interrupted')
STATES = ('queued', 'running', 'cancelling') + TERMINAL


def now():
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, workspace):
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'jobs').mkdir(exist_ok=True)
        self.database = self.root / 'experiments.sqlite3'
        with self.connection() as db:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise RuntimeError('unsupported experiment database version')
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
                  params TEXT NOT NULL, config TEXT NOT NULL, parent_id TEXT REFERENCES jobs(id),
                  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                  exit_code INTEGER, error TEXT, summary TEXT NOT NULL DEFAULT '{}');
                CREATE INDEX IF NOT EXISTS queue_order ON jobs(state, created_at);
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                  created_at TEXT NOT NULL, state TEXT NOT NULL, message TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS metrics (
                  id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                  source TEXT NOT NULL, letter TEXT, method TEXT, scenario TEXT, controller TEXT,
                  record TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS metrics_job ON metrics(job_id, source, method);
                CREATE TABLE IF NOT EXISTS artifacts (
                  job_id TEXT NOT NULL REFERENCES jobs(id), path TEXT NOT NULL, bytes INTEGER NOT NULL,
                  PRIMARY KEY(job_id, path));
                PRAGMA user_version=1;
            ''')

    @contextmanager
    def connection(self, write=False):
        db = sqlite3.connect(self.database, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            if write:
                db.execute('BEGIN IMMEDIATE')
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            raise KeyError('job not found')
        value = dict(row)
        for name in ('params', 'config', 'summary'):
            value[name] = json.loads(value[name])
        return value

    @staticmethod
    def event(db, job_id, state, message):
        db.execute('INSERT INTO events(job_id,created_at,state,message) VALUES(?,?,?,?)',
                   (job_id, now(), state, message))

    def create(self, kind, params, config, parent_id=None):
        job_id = uuid.uuid4().hex
        with self.connection(True) as db:
            db.execute('INSERT INTO jobs(id,kind,state,params,config,parent_id,created_at) VALUES(?,?,?,?,?,?,?)',
                       (job_id, kind, 'queued', json.dumps(params), json.dumps(config), parent_id, now()))
            self.event(db, job_id, 'queued', 'Task accepted')
        return self.get(job_id)

    def get(self, job_id):
        with self.connection() as db:
            return self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())

    def list(self, state=None, limit=50, offset=0):
        clause, args = (' WHERE state=?', [state]) if state else ('', [])
        with self.connection() as db:
            total = db.execute('SELECT COUNT(*) FROM jobs' + clause, args).fetchone()[0]
            rows = db.execute('SELECT * FROM jobs' + clause + ' ORDER BY created_at DESC,id LIMIT ? OFFSET ?',
                              args + [limit, offset]).fetchall()
        return {'total': total, 'items': [self.decode(row) for row in rows]}

    def claim(self):
        with self.connection(True) as db:
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created_at,id LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET state='running',started_at=? WHERE id=?", (now(), row['id']))
            self.event(db, row['id'], 'running', 'Worker claimed task')
            return self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (row['id'],)).fetchone())

    def cancel(self, job_id):
        with self.connection(True) as db:
            job = self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())
            if job['state'] in TERMINAL:
                raise ValueError('task is already finished')
            state = 'cancelled' if job['state'] == 'queued' else 'cancelling'
            db.execute('UPDATE jobs SET state=?,finished_at=? WHERE id=?',
                       (state, now() if state == 'cancelled' else None, job_id))
            self.event(db, job_id, state, 'Cancellation requested')
        return self.get(job_id)

    def recover(self):
        with self.connection(True) as db:
            rows = db.execute("SELECT id FROM jobs WHERE state IN ('running','cancelling')").fetchall()
            for row in rows:
                db.execute("UPDATE jobs SET state='interrupted',finished_at=?,error=? WHERE id=?",
                           (now(), 'Previous server stopped before task completion; retry creates a new run.', row['id']))
                self.event(db, row['id'], 'interrupted', 'Previous worker ended without completion')

    def complete(self, job_id, state, exit_code=None, error=None, summary=None, metrics=(), artifacts=()):
        if state not in TERMINAL:
            raise ValueError('invalid terminal state')
        with self.connection(True) as db:
            job = self.decode(db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())
            if job['state'] in TERMINAL:
                return
            if job['state'] == 'cancelling':
                state = 'cancelled'
                summary, metrics, artifacts = {}, (), ()
            for source, record in metrics:
                db.execute('INSERT INTO metrics(job_id,source,letter,method,scenario,controller,record) VALUES(?,?,?,?,?,?,?)',
                           (job_id, source, record.get('letter'), record.get('method'), record.get('scenario'),
                            record.get('controller'), json.dumps(record, allow_nan=False)))
            for path, size in artifacts:
                db.execute('INSERT INTO artifacts(job_id,path,bytes) VALUES(?,?,?)', (job_id, path, size))
            db.execute('UPDATE jobs SET state=?,finished_at=?,exit_code=?,error=?,summary=? WHERE id=?',
                       (state, now(), exit_code, error, json.dumps(summary or {}, allow_nan=False), job_id))
            self.event(db, job_id, state, error or 'Task completed')

    def events(self, job_id):
        self.get(job_id)
        with self.connection() as db:
            return [dict(r) for r in db.execute('SELECT * FROM events WHERE job_id=? ORDER BY id', (job_id,))]

    def metrics(self, job_id, source=None, method=None, limit=100, offset=0):
        self.get(job_id)
        clause, args = ' WHERE job_id=?', [job_id]
        for column, value in (('source', source), ('method', method)):
            if value:
                clause += f' AND {column}=?'
                args.append(value)
        with self.connection() as db:
            total = db.execute('SELECT COUNT(*) FROM metrics' + clause, args).fetchone()[0]
            rows = db.execute('SELECT source,record FROM metrics' + clause + ' ORDER BY id LIMIT ? OFFSET ?',
                              args + [limit, offset]).fetchall()
        return {'total': total, 'items': [{'source': r['source'], **json.loads(r['record'])} for r in rows]}

    def artifacts(self, job_id):
        self.get(job_id)
        with self.connection() as db:
            return [dict(r) for r in db.execute('SELECT path,bytes FROM artifacts WHERE job_id=? ORDER BY path', (job_id,))]
