"""Regression checks of the actual executable notebook controller cells."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / 'notebooks/templates/ESC50_Colab_Research_Console.ipynb').read_text())

def cell(index, env):
    exec(''.join(NOTEBOOK['cells'][index]['source']), env)

def controller():
    env = dict(Path=Path,json=json,EXECUTE_SETUP=False)
    cell(10,env)
    return env

def matrix_controller():
    env=dict(pathlib=__import__('pathlib'),Counter=Counter,json=json,
             RUN_MODE='plan',GROUPS_TO_RUN=['A_baselines'],TASK_IDS=[],
             MAX_TASKS_PER_SESSION=1,FULL_EPOCHS=60,BATCH_SIZE=32,
             LEARNING_RATE=.001,STUDY_MATRIX_PATH='')
    cell(4,env)
    return env

def test_matrix_checks_semantics_and_consumes_exact_task_fields():
    env=matrix_controller()
    matrix=json.loads((ROOT/'configs/colab_study_matrix_v1.json').read_text())
    assert env['checked_matrix'](matrix)==matrix['tasks']
    assert 'requires' in env['checked_matrix'](matrix)[0]
    matrix['tasks'][2]['model']='crnn_freq_tca'
    with pytest.raises(ValueError,match='semantics'):
        env['checked_matrix'](matrix)

def test_matrix_rejects_changed_protocol_and_training_budget():
    env=matrix_controller()
    matrix=json.loads((ROOT/'configs/colab_study_matrix_v1.json').read_text())
    matrix['train_protocol']['epochs']=2
    with pytest.raises(ValueError,match='settings'):
        env['checked_matrix'](matrix)
    matrix['train_protocol']['epochs']=60
    matrix['train_protocol']['validation_fold']=5
    with pytest.raises(ValueError,match='protocol'):
        env['checked_matrix'](matrix)

def test_mismatched_science_and_invalid_weights_are_not_completed(tmp_path):
    env=controller()
    env.update(BATCH_SIZE=32,LEARNING_RATE=.001)
    (tmp_path/'run.json').write_text(json.dumps(dict(status='completed',epochs_recorded=60,
                                                   config=dict(model='cnn',seed=999,lr=9))))
    (tmp_path/'best.pt').write_text('not a checkpoint')
    assert not env['completed'](dict(kind='supervised',id='B_crnn_freq_tca_s101',
                                     model='crnn_freq_tca',seed=101),tmp_path,60)

def test_cached_learning_rate_change_is_not_resumed(tmp_path):
    env=controller()
    env.update(BATCH_SIZE=32,LEARNING_RATE=.001)
    task=dict(model='cnn_freq',seed=101,augmentation='clean')
    config=env['expected_supervised'](task,60)
    config['lr']=.01
    (tmp_path/'run.json').write_text(json.dumps(dict(config=config)))
    assert not env['compatible_supervised'](task,tmp_path,60)

def execution_env(tmp_path):
    env=dict(Path=Path,json=json,os=os,sys=sys,subprocess=subprocess,time=time,
             EXECUTE_SETUP=True,ROOT=ROOT,CPU_THREADS=2,LOGS=tmp_path,
             MAX_SECONDS_PER_TRAIN=5)
    cell(8,env)
    return env

def test_silent_subprocess_is_really_terminated_at_timeout(tmp_path):
    env=execution_env(tmp_path)
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        env['execute_command']([sys.executable,'-c','import time;time.sleep(20)'],
                               'silent',timeout=.3)
    assert time.monotonic()-started<3

def test_remaining_session_deadline_bounds_feature_like_process(tmp_path):
    env=execution_env(tmp_path)
    env['_session_deadline']=time.monotonic()+.3
    started=time.monotonic()
    with pytest.raises(TimeoutError):
        env['execute_command']([sys.executable,'-c','import time;time.sleep(20)'],'feature')
    assert time.monotonic()-started<3

def test_subprocess_success_preserves_output(tmp_path):
    env=execution_env(tmp_path)
    env['execute_command']([sys.executable,'-c','print("retained output")'],'success')
    assert 'retained output' in (tmp_path/'success.log').read_text()
