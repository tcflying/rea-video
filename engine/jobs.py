"""Durable one-GPU job queue: submit films, watch them, resume only what is safe.

The queue survives a service restart. A job that was running when the process died is
marked interrupted, never silently resubmitted: its ComfyUI prompt ids are reconciled
against /queue and /history, and only an operator-confirmed resume continues it.
"""
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / 'engine' / 'state'
QUEUE = Path(os.environ.get('REA_QUEUE') or (STATE / 'queue.json'))
LOGS = ROOT / 'engine' / 'logs'
H3 = os.environ.get('REA_ENGINE_URL') or 'http://127.0.0.1:1022'
STATES = ('queued', 'running', 'interrupted', 'done', 'failed', 'cancelled')

_lock = threading.Lock()


def _now():
    return round(time.time(), 3)


def _read():
    if not QUEUE.is_file():
        return {'version': 1, 'jobs': []}
    return json.loads(QUEUE.read_text(encoding='utf-8'))


def _write(doc):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE.with_suffix('.tmp')
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, QUEUE)


def _h3(path):
    try:
        req = build_opener(ProxyHandler({})).open(H3 + path, timeout=5)
        return json.load(req)
    except Exception:
        return None


def pending_prompts(out):
    """Prompt ids this film already submitted, with their local record paths."""
    found = {}
    for record in sorted((Path(out) / 'remake' / 'history').glob('G*.json')):
        state = json.loads(record.read_text(encoding='utf-8'))
        if state.get('prompt_id'):
            found[record.stem] = {'prompt_id': state['prompt_id'],
                                  'done': bool((state.get('history') or {}).get('status'))}
    return found


def reconcile(job):
    """Decide what the GPU actually did with this job, without resubmitting anything."""
    out = Path(job['out'])
    prompts = pending_prompts(out)
    if not prompts:
        return {'verdict': 'not_started', 'detail': 'no ComfyUI prompt was submitted'}
    queue = _h3('/queue')
    history = _h3('/history?max_items=200')
    if queue is None or history is None:
        return {'verdict': 'unknown', 'detail': 'ComfyUI is unreachable; state cannot be proven'}
    live = {row[1] for key in ('queue_running', 'queue_pending') for row in queue.get(key, [])}
    done = set(history)
    stuck = [name for name, row in prompts.items()
             if not row['done'] and row['prompt_id'] not in live and row['prompt_id'] not in done]
    if any(row['prompt_id'] in live for row in prompts.values()) and not stuck:
        return {'verdict': 'still_running', 'detail': 'ComfyUI still owns every unfinished prompt'}
    if stuck and not any(row['prompt_id'] in done for row in prompts.values()):
        return {'verdict': 'unknown', 'detail': 'prompt ids are no longer queued or in history'}
    return {'verdict': 'finished' if not stuck else 'partial',
            'detail': '%d prompt(s) have no result' % len(stuck) if stuck else 'every submitted prompt finished'}


def submit(source, photos, out, narration=True, skip_gate=False):
    source, photos, out = Path(source).resolve(), [Path(p).resolve() for p in photos], Path(out).resolve()
    if not source.is_file():
        raise ValueError('source video is missing: %s' % source)
    if not 1 <= len(photos) <= 3 or not all(p.is_file() for p in photos):
        raise ValueError('give 1-3 existing product photos')
    with _lock:
        doc = _read()
        job = {'id': uuid.uuid4().hex[:12], 'state': 'queued', 'created': _now(),
               'source': str(source), 'photos': [str(p) for p in photos], 'out': str(out),
               'narration': bool(narration), 'skip_gate': bool(skip_gate),
               'attempts': 0}
        job['log'] = str(LOGS / ('job-%s-%s.log' % (job['id'], out.name)))
        doc['jobs'].append(job)
        _write(doc)
        return job


def listing():
    with _lock:
        return _read()['jobs']


def get(job_id):
    with _lock:
        return next((j for j in _read()['jobs'] if j['id'] == job_id), None)


def cancel(job_id, proc=None):
    with _lock:
        doc = _read()
        job = next((j for j in doc['jobs'] if j['id'] == job_id), None)
        if not job:
            raise KeyError(job_id)
        if job['state'] in ('done', 'failed', 'cancelled'):
            raise ValueError('job is already %s' % job['state'])
        job['state'] = 'cancelled'
        job['cancelled_at'] = _now()
        _write(doc)
    if proc is not None and proc.poll() is None:
        proc.terminate()
    return job


def reorder(job_id, before=None):
    """Move a waiting job inside the waiting positions only; running work never moves."""
    with _lock:
        doc = _read()
        waiting = [j for j in doc['jobs'] if j['state'] == 'queued']
        job = next((j for j in waiting if j['id'] == job_id), None)
        if not job:
            raise ValueError('only a queued job can be reordered')
        waiting.remove(job)
        at = len(waiting)
        if before:
            at = next((i for i, j in enumerate(waiting) if j['id'] == before), len(waiting))
        waiting.insert(at, job)
        others = [j for j in doc['jobs'] if j['state'] != 'queued']
        doc['jobs'] = others + waiting
        _write(doc)
        return job


def resume(job_id, confirm=False):
    """Continue an interrupted film. Never resubmits unless the operator confirms the reconciliation."""
    job = get(job_id)
    if not job:
        raise KeyError(job_id)
    if job['state'] != 'interrupted':
        raise ValueError('only an interrupted job needs resume; this one is %s' % job['state'])
    verdict = reconcile(job)
    job['reconciliation'] = verdict
    if not confirm:
        return {'job': job, 'reconciliation': verdict, 'resumed': False,
                'reason': 'the operator must confirm after reading the reconciliation'}
    if verdict['verdict'] == 'unknown':
        raise ValueError('cannot resume safely: ' + verdict['detail'])
    with _lock:
        doc = _read()
        row = next(j for j in doc['jobs'] if j['id'] == job_id)
        row['state'] = 'queued'
        row['reconciliation'] = verdict
        _write(doc)
    return {'job': job, 'reconciliation': verdict, 'resumed': True}


def command(job):
    args = [sys.executable, '-u', '-B', str(ROOT / 'engine' / 'rea_remake.py'), 'render', job['out']]
    if not job.get('narration', True):
        args.append('--no-narration')
    if job.get('skip_gate'):
        args.append('--skip-gate')
    return args


def mark_interrupted():
    """Startup: nothing may still claim to be running."""
    with _lock:
        doc = _read()
        changed = 0
        for job in doc['jobs']:
            if job['state'] == 'running':
                job['state'] = 'interrupted'
                job['reconciliation'] = reconcile(job)
                changed += 1
        if changed:
            _write(doc)
        return changed


def worker(stop=None, poll=2.0):
    """One job at a time; the whole queue is the caller's order."""
    stop = stop or threading.Event()
    while not stop.is_set():
        job = next((j for j in listing() if j['state'] == 'queued'), None)
        if not job:
            stop.wait(poll)
            continue
        with _lock:
            doc = _read()
            row = next(j for j in doc['jobs'] if j['id'] == job['id'])
            row['state'] = 'running'
            row['started'] = _now()
            row['attempts'] = int(row.get('attempts', 0)) + 1
            _write(doc)
        Path(job['log']).parent.mkdir(parents=True, exist_ok=True)
        with open(job['log'], 'a', encoding='utf-8') as log:
            log.write('\n=== %s %s ===\n' % (job['id'], time.strftime('%Y-%m-%d %H:%M:%S')))
            log.flush()
            proc = subprocess.Popen(command(job), stdout=log, stderr=subprocess.STDOUT,
                                    cwd=str(ROOT), env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            while proc.poll() is None and not stop.is_set():
                time.sleep(1)
            if stop.is_set() and proc.poll() is None:
                proc.terminate()
        with _lock:
            doc = _read()
            row = next(j for j in doc['jobs'] if j['id'] == job['id'])
            row['finished'] = _now()
            if row['state'] != 'cancelled':
                row['state'] = 'done' if proc.returncode == 0 else 'failed'
                row['returncode'] = proc.returncode
            _write(doc)
