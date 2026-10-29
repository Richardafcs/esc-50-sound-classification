"""Epoch-boundary runtime snapshots, restricted to safe tensor-only payloads.

No attempt is made to replay a partially processed epoch.  This deliberately
avoids the unsound assumption that a sampler or optimizer can resume mid-batch.
"""
from __future__ import annotations
import contextlib

import random

import numpy as np
import torch


def capture_rng(generator: torch.Generator) -> dict:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {"kind": numpy_state[0], "keys": torch.from_numpy(numpy_state[1].copy()),
                  "pos": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
                  "cached_gauss": float(numpy_state[4])},
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "sampler": generator.get_state().clone(),
    }


def restore_rng(snapshot: dict, generator: torch.Generator) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda", "sampler"}
    if not isinstance(snapshot, dict) or not required.issubset(snapshot):
        raise ValueError("Epoch resume requires full Python/NumPy/Torch/sampler RNG state")
    if len(snapshot["torch_cuda"]) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("CUDA device count differs from saved RNG state")
    n = snapshot["numpy"]
    random.setstate(snapshot["python"])
    np.random.set_state((n["kind"], n["keys"].cpu().numpy().copy(), n["pos"], n["has_gauss"], n["cached_gauss"]))
    torch.set_rng_state(snapshot["torch_cpu"].cpu())
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])
    generator.set_state(snapshot["sampler"].cpu())


def clone_weights(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@contextlib.contextmanager
def run_writer_lock(output):
    """Non-blocking, same-host POSIX single writer guard for one experiment.

    The sibling lock file is permanent: unlinking a flock lockfile while it is
    in use could let another process create a second inode and acquire it.
    """
    import fcntl
    import os
    from pathlib import Path
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / ('.' + output.name + '.writer.lock')
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Experiment has an active writer: {output}') from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
