"""Durable single-host research scheduler: serial execution, no implicit test access.

Local-only; a single scheduler owner and a single training writer are guarded
with POSIX advisory locks. One invocation performs at most one pass over jobs.
It never guesses that a failed external command is safe to retry.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from .records import canonical_hash, write_json
from .runtime_state import run_writer_lock

STATES = ('PENDING', 'RUNNING', 'PAUSED', 'COMPLETED', 'FAILED', 'CANCELLED')
ALLOWED = {
    'PENDING': {'RUNNING', 'CANCELLED'},
    'RUNNING': {'PAUSED', 'COMPLETED', 'FAILED'},
    'PAUSED': {'RUNNING', 'CANCELLED'},
    'COMPLETED': set(), 'FAILED': set(), 'CANCELLED': set(),
}


def _utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('schema_version') != 1:
        raise ValueError('Plan requires schema_version=1')
    if not isinstance(plan.get('jobs'), list) or not plan['jobs']:
        raise ValueError('Plan needs a nonempty jobs list')
    max_cpu = plan.get('max_cpu_threads', 1)
    if type(max_cpu) is not int or max_cpu < 1:
        raise ValueError('max_cpu_threads must be positive integer')
    seen = set()
    for job in plan['jobs']:
        if not isinstance(job, dict):
            raise ValueError('Each job must be an object')
        ident = job.get('id')
        if (not isinstance(ident, str) or not ident or len(ident) > 96 or
                not ident.replace('_', '').replace('-', '').isalnum() or ident in seen):
            raise ValueError('Job IDs must be unique and safe')
        seen.add(ident)
        if job.get('kind') not in ('esc50_train', 'command'):
            raise ValueError('Unsupported job kind')
        cap = job.get('timeout_seconds')
        if not isinstance(cap, (int, float)) or isinstance(cap, bool) or not 1 <= cap <= 86400:
            raise ValueError('timeout_seconds must be between 1 and 86400')
        if job['kind'] == 'esc50_train':
            args = job.get('train_args')
            if not isinstance(args, list) or not all(isinstance(x, str) and x for x in args):
                raise ValueError('train_args must be an argv array')
            forbidden = ('--output', '--max-seconds', '--threads', '--device',
                         '--pause-after-epochs', '--allow-test', '--test-fold')
            if any(any(v == key or v.startswith(key+'=') for key in forbidden) for v in args):
                raise ValueError('Scheduler controls output, execution and device; test fold override is not allowed')
            from .cli import build_parser
            parsed = build_parser().parse_args(['train', *args, '--output', '__check__'])
            if parsed.command != 'train' or parsed.epochs <= 0:
                raise ValueError('Invalid train job')
            threads = job.get('cpu_threads', 1)
            if type(threads) is not int or not 1 <= threads <= max_cpu:
                raise ValueError('cpu_threads exceeds scheduler budget')
            if job.get('device', 'cpu') not in ('cpu', 'cuda'):
                raise ValueError('device must be cpu or cuda')
        else:
            command = job.get('command')
            if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
                raise ValueError('command must be an argv list, not a shell string')
    return json.loads(json.dumps(plan, allow_nan=False))


@contextlib.contextmanager
def controller_lock(workspace):
    """Prevent concurrent dispatch on a single POSIX host; lockfile is never unlinked."""
    path = Path(workspace) / '.controller.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Scheduler workspace already has an active controller') from exc
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _save(ws, state, job_id, old, new, reason):
    if (old not in ALLOWED or new not in ALLOWED[old] or
            state['jobs'][job_id]['state'] != old):
        raise ValueError(f'Invalid scheduler transition: {old} -> {new}')
    # The snapshot is authoritative. Events are supplementary audit; each event
    # carries a revision so gaps after a hard crash are visible, not hidden.
    state['revision'] += 1
    state['jobs'][job_id]['state'] = new
    state['jobs'][job_id]['updated_utc'] = _utc()
    write_json(ws / 'state.json', state)
    entry = dict(revision=state['revision'], job_id=job_id, previous=old,
                 state=new, reason=reason, at=_utc())
    fd = os.open(ws / 'events.jsonl', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(entry, allow_nan=False) + '\n').encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def initialize(workspace, plan):
    plan = _validate_plan(plan)
    ws = Path(workspace).resolve()
    ws.mkdir(parents=True, exist_ok=False)
    (ws / 'jobs').mkdir()
    write_json(ws / 'plan.json', plan)
    state = dict(schema_version=1, plan_sha256=canonical_hash(plan), revision=0,
                 jobs={j['id']: dict(state='PENDING', attempts=0, updated_utc=_utc()) for j in plan['jobs']})
    write_json(ws / 'state.json', state)
    return state


def _read(ws):
    plan = _validate_plan(json.loads((ws / 'plan.json').read_text()))
    state = json.loads((ws / 'state.json').read_text())
    if state.get('plan_sha256') != canonical_hash(plan):
        raise ValueError('Scheduler plan integrity mismatch')
    if (set(state.get('jobs', {})) != {j['id'] for j in plan['jobs']} or
            type(state.get('revision')) is not int or state['revision'] < 0 or
            any(j.get('state') not in STATES or type(j.get('attempts')) is not int or
                j['attempts'] < 0 for j in state['jobs'].values())):
        raise ValueError('Scheduler state is inconsistent with plan')
    return plan, state


def status(workspace):
    ws = Path(workspace).resolve()
    plan, state = _read(ws)
    return dict(workspace=str(ws), revision=state['revision'],
                jobs=[dict(id=j['id'], **state['jobs'][j['id']]) for j in plan['jobs']])


def cancel(workspace, ident):
    ws = Path(workspace).resolve()
    with controller_lock(ws):
        _, state = _read(ws)
        if ident not in state['jobs'] or state['jobs'][ident]['state'] not in ('PENDING', 'PAUSED'):
            raise ValueError('Can only cancel a pending or paused job')
        before = state['jobs'][ident]['state']
        _save(ws, state, ident, before, 'CANCELLED', 'explicit operator cancellation')
    return status(ws)


def _safe_recover(ws, plan, state):
    """If an orchestrator died, never relaunch until the training writer is gone."""
    for job in plan['jobs']:
        ident = job['id']
        if state['jobs'][ident]['state'] != 'RUNNING':
            continue
        experiment = ws / 'jobs' / ident / 'experiment'
        if job['kind'] == 'esc50_train':
            with run_writer_lock(experiment):
                # A valid snapshot can be resumed; resume-train performs the
                # definitive manifest, cache, environment and test-access audit.
                resumable = all((experiment / filename).is_file()
                                for filename in ('last.pt', 'manifest.json', 'run.json'))
            if resumable:
                try:
                    training_record = json.loads((experiment / 'run.json').read_text())
                    target = 'COMPLETED' if training_record.get('status') == 'completed' else 'PAUSED'
                except (OSError, ValueError):
                    target = 'FAILED'
            else:
                target = 'FAILED'
        else:
            target = 'FAILED'   # Cannot assume external command idempotence
        _save(ws, state, ident, 'RUNNING', target, 'orchestrator restart reconciliation')


def dispatch(workspace, *, max_jobs=None, base_dir=None, echo=False):
    """Run up to max_jobs, sequentially, then return; never a background daemon."""
    from .runner import run_logged
    ws = Path(workspace).resolve()
    if max_jobs is not None and (type(max_jobs) is not int or max_jobs < 1):
        raise ValueError('max_jobs must be positive')
    with controller_lock(ws):
        plan, state = _read(ws)
        _safe_recover(ws, plan, state)
        executed = 0
        for job in plan['jobs']:
            ident = job['id']
            initial = state['jobs'][ident]['state']
            if initial not in ('PENDING', 'PAUSED'):
                continue
            if max_jobs is not None and executed >= max_jobs:
                break
            root = ws / 'jobs' / ident
            root.mkdir(exist_ok=True)
            output = root / 'experiment'
            attempts = state['jobs'][ident]['attempts']
            record_log = root / f'attempt_{attempts + 1:03d}.log'
            if job['kind'] == 'esc50_train':
                if initial == 'PENDING':
                    cmd = [sys.executable, '-m', 'esc50_sound', 'train', *job['train_args'],
                           '--output', str(output), '--max-seconds', str(job['timeout_seconds']),
                           '--threads', str(job.get('cpu_threads', 1)), '--device', job.get('device', 'cpu')]
                else:
                    cmd = [sys.executable, '-m', 'esc50_sound', 'resume-train', '--output', str(output),
                           '--max-seconds', str(job['timeout_seconds'])]
            else:
                cmd = job['command']
            state['jobs'][ident]['attempts'] += 1
            _save(ws, state, ident, initial, 'RUNNING', f'attempt {attempts+1} launched')
            try:
                result = run_logged(cmd, record_log, timeout=job['timeout_seconds'] + 3,
                                    grace_seconds=.25, cwd=base_dir, check=False, echo=echo)
                state['jobs'][ident]['last_process_status'] = result['status']
                state['jobs'][ident]['last_exit_code'] = result['exit_code']
                if job['kind'] == 'command':
                    target = 'COMPLETED' if result['status'] == 'completed' else 'FAILED'
                elif result['status'] == 'completed':
                    record = json.loads((output / 'run.json').read_text())
                    if record.get('status') == 'completed':
                        target = 'COMPLETED'
                    elif record.get('status') == 'time_limit' and (output / 'last.pt').is_file():
                        target = 'PAUSED'
                    else:
                        target = 'FAILED'
                elif result['status'] == 'timeout':
                    target = ('PAUSED' if all((output / f).is_file() for f in ('last.pt', 'manifest.json', 'run.json'))
                              else 'FAILED')
                else:
                    target = 'FAILED'  # Do not silently retry code errors as time slices
                _save(ws, state, ident, 'RUNNING', target, f'process outcome: {result["status"]}')
            except BaseException as exc:
                _save(ws, state, ident, 'RUNNING', 'FAILED', f'dispatch exception: {type(exc).__name__}')
                raise
            executed += 1
            if target == 'FAILED':
                break  # Fail closed: do not run later experiments after failure.
        return status(ws)
