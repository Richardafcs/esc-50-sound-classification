"""Durable local scheduling and writer exclusion; all data is synthetic."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from esc50_sound import scheduler
from esc50_sound.runtime_state import run_writer_lock


def py(code):
    return [sys.executable, '-u', '-c', code]


def plan(*jobs):
    return dict(schema_version=1, max_cpu_threads=3, jobs=list(jobs))


def command_job(ident, code='print("done")'):
    return dict(id=ident, kind='command', command=py(code), timeout_seconds=5)


def escjob(ident='train1'):
    return dict(id=ident, kind='esc50_train', train_args=['--model', 'cnn_freq', '--epochs', '2',
                      '--cache', 'missing_for_synthetic_test'], timeout_seconds=5, cpu_threads=2)


def test_plan_rejects_invalid_jobs_before_creating_any_workspace(tmp_path):
    invalid = [plan(command_job('reused'), command_job('reused')),
               plan(dict(id='../bad', kind='command', command=py('pass'), timeout_seconds=3)),
               plan(dict(id='bad', kind='command', command='echo dangerous', timeout_seconds=3)),
               plan({**escjob(), 'train_args': ['--allow-test']}),
               plan({**escjob(), 'train_args': ['--test-fold', '1']}),
               plan({**escjob(), 'cpu_threads': 5}),
               plan({**escjob(), 'train_args': ['--output', '/tmp/unsafe']})]
    for i, cfg in enumerate(invalid):
        with pytest.raises((ValueError, SystemExit)):
            scheduler.initialize(tmp_path / f'invalid_{i}', cfg)
        assert not (tmp_path / f'invalid_{i}').exists()


def test_controller_lock_blocks_second_scheduler_owner(tmp_path):
    ws = tmp_path / 'queue'
    scheduler.initialize(ws, plan(command_job('a')))
    with scheduler.controller_lock(ws):
        with pytest.raises(RuntimeError, match='active controller'):
            scheduler.dispatch(ws)


def test_writer_lock_blocks_second_training_writer(tmp_path):
    run = tmp_path / 'one_run'
    with run_writer_lock(run):
        with pytest.raises(RuntimeError, match='active writer'):
            with run_writer_lock(run):
                pass
    with run_writer_lock(run):
        pass


def test_successful_job_is_not_restarted(tmp_path):
    ws = tmp_path / 'queue'
    counter = tmp_path / 'counter'
    source = f'from pathlib import Path; p=Path({str(counter)!r}); p.write_text(p.read_text()+"x" if p.exists() else "x")'
    scheduler.initialize(ws, plan(command_job('a', source)))
    first = scheduler.dispatch(ws)
    assert first['jobs'][0]['state'] == 'COMPLETED'
    assert scheduler.dispatch(ws)['jobs'][0]['state'] == 'COMPLETED'
    assert counter.read_text() == 'x'
    assert first['jobs'][0]['attempts'] == 1
    assert len((ws / 'events.jsonl').read_text().splitlines()) == 2


def test_failed_command_prevents_later_jobs_and_preserves_log(tmp_path):
    marker = tmp_path / 'not_launched'
    ws = tmp_path / 'queue'
    scheduler.initialize(ws, plan(command_job('bad', 'raise SystemExit(7)'),
                                  command_job('next', f'from pathlib import Path; Path({str(marker)!r}).touch()')))
    result = scheduler.dispatch(ws)
    assert [j['state'] for j in result['jobs']] == ['FAILED', 'PENDING']
    assert not marker.exists()
    assert (ws / 'jobs/bad/attempt_001.log.json').exists()
    assert result['jobs'][0]['last_exit_code'] == 7


def test_operator_cancel_queued_or_paused_only(tmp_path):
    ws = tmp_path / 'queue'
    scheduler.initialize(ws, plan(command_job('a')))
    assert scheduler.cancel(ws, 'a')['jobs'][0]['state'] == 'CANCELLED'
    assert scheduler.dispatch(ws)['jobs'][0]['state'] == 'CANCELLED'
    with pytest.raises(ValueError):
        scheduler.cancel(ws, 'a')


def test_orphan_running_command_is_failed_not_reexecuted(tmp_path):
    ws = tmp_path / 'queue'
    scheduler.initialize(ws, plan(command_job('a')))
    state = json.loads((ws / 'state.json').read_text())
    state['jobs']['a']['state'] = 'RUNNING'
    from esc50_sound.records import write_json
    write_json(ws / 'state.json', state)
    assert scheduler.dispatch(ws)['jobs'][0]['state'] == 'FAILED'


def test_orphan_train_with_checkpoint_pauses_but_refuses_live_writer(tmp_path):
    ws = tmp_path / 'queue'
    cfg = plan(escjob())
    scheduler.initialize(ws, cfg)
    state = json.loads((ws / 'state.json').read_text())
    state['jobs']['train1']['state'] = 'RUNNING'
    from esc50_sound.records import write_json
    write_json(ws / 'state.json', state)
    out = ws / 'jobs/train1/experiment'
    out.mkdir(parents=True)
    for name in ('last.pt', 'manifest.json', 'run.json'):
        (out / name).write_text('{"status":"running"}' if name == 'run.json' else 'fixture')
    with run_writer_lock(out):
        with pytest.raises(RuntimeError, match='active writer'):
            scheduler.dispatch(ws, max_jobs=1)
    # Set max_jobs to 1: recovery itself makes run PAUSED and then tries resume,
    # which fails deliberately because the synthetic checkpoint is not real.
    # Inspect reconciliation in isolation instead of invoking torch loading.
    state = json.loads((ws / 'state.json').read_text())
    scheduler._safe_recover(ws, cfg, state)
    assert json.loads((ws / 'state.json').read_text())['jobs']['train1']['state'] == 'PAUSED'


def test_train_time_slice_and_resume_dispatch_paths(tmp_path, monkeypatch):
    ws = tmp_path / 'queue'
    scheduler.initialize(ws, plan(escjob()))
    from esc50_sound import runner
    commands = []
    def fake_run(cmd, log_path, **kwargs):
        commands.append(cmd)
        out = ws / 'jobs/train1/experiment'
        out.mkdir(parents=True, exist_ok=True)
        if len(commands) == 1:
            (out / 'last.pt').write_bytes(b'fake-checkpoint')
            (out / 'manifest.json').write_text('{}')
            (out / 'run.json').write_text('{"status":"time_limit"}')
        else:
            (out / 'run.json').write_text('{"status":"completed"}')
        return dict(status='completed', exit_code=0)
    monkeypatch.setattr(runner, 'run_logged', fake_run)
    a = scheduler.dispatch(ws)
    assert a['jobs'][0]['state'] == 'PAUSED'
    b = scheduler.dispatch(ws)
    assert b['jobs'][0]['state'] == 'COMPLETED'
    assert a['jobs'][0]['attempts'] == 1 and b['jobs'][0]['attempts'] == 2
    assert 'train' in commands[0] and 'resume-train' in commands[1]
    assert '--allow-test' not in ' '.join(commands[0] + commands[1])


def test_terminal_transitions_are_rejected(tmp_path):
    ws = tmp_path / 'queue'
    scheduler.initialize(ws, plan(command_job('a')))
    scheduler.cancel(ws, 'a')
    state = json.loads((ws / 'state.json').read_text())
    with pytest.raises(ValueError, match='Invalid scheduler transition'):
        scheduler._save(ws, state, 'a', 'CANCELLED', 'RUNNING', 'forbidden replay')
    assert scheduler.status(ws)['jobs'][0]['state'] == 'CANCELLED'
