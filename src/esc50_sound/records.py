"""Small experiment-record helpers: identities, environment, JSON and append-only audit."""
from __future__ import annotations
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import numpy as np
import torch

def write_json(path, value):
    """Atomically replace a JSON record (same-filesystem temporary + fsync).

    Existing training code intentionally updates run/history records in place;
    atomic replacement prevents readers seeing a partially written JSON file.
    This is not append-only or an immutable storage guarantee.
    """
    path = Path(path)
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def environment(device='cpu'):
    return dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda, device=str(device), available_cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def append_evaluation_audit(path, record):
    """One write with append semantics; prior records are never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 384)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)

def write_json_new(path, value):
    """Publish a new immutable-by-convention JSON report without replace races.

    Same-filesystem hard linking is atomic and fails if destination exists.
    Existing evidence is never overwritten, even with concurrent writers.
    """
    path = Path(path)
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
