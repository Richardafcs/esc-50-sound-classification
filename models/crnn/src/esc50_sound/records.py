"""Small experiment-record helpers: identities, environment, JSON and append-only audit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def package_source_identity():
    """Return an inspectable digest of every Python module used by this package."""
    package_root = Path(__file__).resolve().parent
    files = {path.name: sha256(path) for path in sorted(package_root.glob("*.py"))}
    return dict(code_sha256=canonical_hash(files), code_files_sha256=files,
                code_identity_kind="esc50_sound package Python sources")


def environment(device="cpu"):
    return dict(python=sys.version, platform=platform.platform(), torch=str(torch.__version__),
                numpy=np.__version__, cuda=torch.version.cuda,
                device=str(device),
                available_cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                **package_source_identity())


def append_evaluation_audit(path, record):
    """One write with append semantics; prior records are never rewritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


