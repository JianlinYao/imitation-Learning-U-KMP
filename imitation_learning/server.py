"""Local FastAPI experiment service. Install the optional server extra."""
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .experiments import read_config
from .job_store import JobStore, STATES, TERMINAL
from .job_worker import QueueWorker, WorkspaceLock

Method = Literal['DMP', 'ProMP', 'GP', 'KMP', 'U-KMP']
Letter = Literal['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
Scenario = Literal['reproduction', 'missing', 'obstacle']


class JobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    kind: Literal['learning', 'demo'] = 'learning'
    letters: list[Letter] = Field(default_factory=lambda: ['G'], min_length=1, max_length=8)
    methods: list[Method] = Field(default_factory=lambda: ['KMP', 'U-KMP'], min_length=1, max_length=5)
    scenarios: list[Scenario] = Field(default_factory=lambda: ['reproduction', 'missing', 'obstacle'], min_length=1, max_length=3)
    points: int = Field(default=80, ge=32, le=200, strict=True)
    fold_limit: int = Field(default=1, ge=1, le=5, strict=True)
    seed: int = Field(default=42, ge=0, le=2147483647, strict=True)
    calibration_fraction: float = Field(default=.2, ge=0, le=.4, allow_inf_nan=False)

    @field_validator('letters', 'methods', 'scenarios')
    @classmethod
    def unique(cls, value):
        if len(value) != len(set(value)):
            raise ValueError('selections must be unique')
        return value

    @model_validator(mode='after')
    def demo_needs_reproduction(self):
        if self.kind == 'demo' and 'reproduction' not in self.scenarios:
            raise ValueError('demo requires reproduction examples for simulation')
        return self


def effective_config(body):
    base = read_config(Path(__file__).resolve().parents[1] / 'configs' / 'demo.json')
    config = replace(base, letters=tuple(body.letters), methods=tuple(body.methods), scenarios=tuple(body.scenarios),
                     points=body.points, fold_limit=body.fold_limit, seed=body.seed,
                     calibration_fraction=body.calibration_fraction, control_ablation=False,
                     model=replace(base.model, reference_points=min(base.model.reference_points, body.points)))
    config.validate()
    return asdict(config)


def create_app(workspace='outputs/server', start_worker=True, command_factory=None):
    store = JobStore(workspace)
    lock = WorkspaceLock(store.root)
    worker = QueueWorker(store, command_factory)

    @asynccontextmanager
    async def lifespan(app):
        if start_worker:
            lock.acquire()
            try:
                worker.start()
                yield
            finally:
                worker.stop()
                lock.release()
        else:
            yield

    app = FastAPI(title='Robot Experiment Manager', version=__version__, lifespan=lifespan,
                  description='Local experiment queue, persistent metrics, logs and generated reports.')
    app.state.store, app.state.worker = store, worker
    app.state.instance_id = uuid.uuid4().hex
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', '[::1]', 'testserver'])

    @app.middleware('http')
    async def same_origin(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in ('http', 'https') or parsed.netloc != request.headers.get('host'):
                return JSONResponse({'detail': 'cross-origin mutations are not allowed'}, status_code=403)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def not_found(request, error):
        return JSONResponse({'detail': 'job not found'}, status_code=404)

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    def home():
        return HTMLResponse(Path(__file__).with_name('job_manager.html').read_text(encoding='utf-8'))

    @app.get('/api/health')
    def health():
        with store.connection() as db:
            counts = {r['state']: r['count'] for r in db.execute('SELECT state,COUNT(*) AS count FROM jobs GROUP BY state')}
        return {'status': 'ok', 'version': __version__, 'database': 'sqlite', 'counts': counts,
                'worker_active': bool(worker.thread and worker.thread.is_alive()), 'instance_id': app.state.instance_id}

    @app.get('/api/catalog')
    def catalog():
        return {'letters': list('ABCDEFGH'), 'methods': ['DMP', 'ProMP', 'GP', 'KMP', 'U-KMP'],
                'scenarios': ['reproduction', 'missing', 'obstacle'],
                'simulation_available': importlib.util.find_spec('mujoco') is not None,
                'defaults': JobRequest().model_dump()}

    @app.post('/api/jobs', status_code=202)
    def submit(body: JobRequest):
        if body.kind == 'demo' and importlib.util.find_spec('mujoco') is None:
            raise HTTPException(409, 'Install simulation support: python -m pip install -e ".[server,simulation]"')
        return store.create(body.kind, body.model_dump(), effective_config(body))

    @app.get('/api/jobs')
    def jobs(state: str | None = None, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        if state is not None and state not in STATES:
            raise HTTPException(422, 'unknown task state')
        return store.list(state, limit, offset)

    @app.get('/api/jobs/{job_id}')
    def detail(job_id: str):
        return store.get(job_id)

    @app.post('/api/jobs/{job_id}/cancel')
    def cancel(job_id: str):
        try:
            return store.cancel(job_id)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.post('/api/jobs/{job_id}/retry', status_code=202)
    def retry(job_id: str):
        previous = store.get(job_id)
        if previous['state'] not in TERMINAL:
            raise HTTPException(409, 'finish or cancel the active task before retrying')
        body = JobRequest.model_validate(previous['params'])
        if body.kind == 'demo' and importlib.util.find_spec('mujoco') is None:
            raise HTTPException(409, 'simulation support is not installed')
        return store.create(body.kind, body.model_dump(), effective_config(body), parent_id=job_id)

    @app.get('/api/jobs/{job_id}/events')
    def events(job_id: str):
        return {'items': store.events(job_id)}

    @app.get('/api/jobs/{job_id}/logs')
    def logs(job_id: str, tail_bytes: int = Query(16000, ge=1, le=200000)):
        store.get(job_id)
        path = store.root / 'jobs' / job_id / 'task.log'
        if not path.is_file():
            return {'text': '', 'bytes': 0, 'truncated': False}
        with path.open('rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - tail_bytes))
            text = stream.read(tail_bytes).decode('utf-8', errors='replace')
        return {'text': text, 'bytes': size, 'truncated': size > tail_bytes}

    @app.get('/api/jobs/{job_id}/metrics')
    def metrics(job_id: str, source: Literal['learning', 'simulation'] | None = None, method: Method | None = None,
                limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
        return store.metrics(job_id, source, method, limit, offset)

    @app.get('/api/jobs/{job_id}/artifacts')
    def artifacts(job_id: str):
        return {'items': store.artifacts(job_id)}

    @app.get('/api/jobs/{job_id}/files/{relative:path}', include_in_schema=False)
    def file(job_id: str, relative: str, download: bool = False):
        store.get(job_id)
        parts = PurePosixPath(relative).parts
        if not parts or '\\' in relative or ':' in relative or relative.startswith('/') or '..' in parts:
            raise HTTPException(404, 'artifact not found')
        root = (store.root / 'jobs' / job_id).resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise HTTPException(404, 'artifact not found')
        with store.connection() as db:
            exists = db.execute('SELECT 1 FROM artifacts WHERE job_id=? AND path=?', (job_id, relative)).fetchone()
        if not exists:
            raise HTTPException(404, 'artifact not found')
        return FileResponse(path, filename=path.name if download else None)

    return app


def serve(workspace, port=8000, open_browser=False):
    import uvicorn
    app = create_app(workspace)
    if open_browser:
        import json
        import threading
        import urllib.request
        import webbrowser
        def open_when_ready():
            for _ in range(40):
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=.3) as response:
                        if json.load(response).get('instance_id') == app.state.instance_id:
                            webbrowser.open(f'http://127.0.0.1:{port}/')
                            return
                except OSError:
                    pass
                threading.Event().wait(.25)
        threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run(app, host='127.0.0.1', port=port)
