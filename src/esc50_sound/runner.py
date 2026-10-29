"""Bounded, evidence-preserving process runner for local and Colab studies.

No shell execution, retries, model selection, or implicit test evaluation occurs.
Grid schema: global_wall_seconds, optional grace_seconds, and runs containing
id, timeout_seconds, and either command:[executable,...] or esc50_args:[...].
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import signal
import subprocess
import sys
import threading
import time


class RunFailure(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(f"Process {record['status']}: {record['command']}; log: {record['log_path']}")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write(path, data, exclusive=False):
    path = Path(path)
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name + ".updating")
        with temporary.open("w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _command(command):
    if not isinstance(command, (list, tuple)) or not command or not all(isinstance(x, str) and x for x in command):
        raise ValueError("command must be a nonempty list of nonempty strings; shell strings are forbidden")
    return list(command)


def _terminate_tree(process, grace_seconds):
    """Clean up our isolated POSIX group, including descendants retaining stdout."""
    def send(sig):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif process.poll() is None:
                process.terminate() if sig == signal.SIGTERM else process.kill()
        except ProcessLookupError:
            pass
    send(signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        send(signal.SIGKILL)
        # The grace interval controls when escalation occurs, not how long an
        # overloaded host is allowed to take to reap an already-killed child.
        # A second .1s wait can itself time out and incorrectly replace a
        # correct `timeout` outcome with a generic `failed` result.
        process.wait(timeout=max(2.0, grace_seconds))
    finally:
        # The direct child may exit but a descendant can still hold the group.
        # This final KILL is intentional and does not change the exit status.
        if os.name == "posix":
            send(signal.SIGKILL)


def run_logged(command, log_path, timeout, *, cwd=None, env=None, grace_seconds=1.0,
               check=True, echo=False):
    """Run one argv command with a silence-independent watchdog.

    timeout caps execution; bounded cleanup can add at most two grace intervals.
    New log and status paths are reserved exclusively. Status is updated atomically;
    stdout is appended, flushed continuously, and retained on every failure.
    Environment overrides are used but their values are never serialized.
    """
    command = _command(command)
    timeout = _positive(timeout, "timeout")
    grace_seconds = _positive(grace_seconds, "grace_seconds")
    log_path = Path(log_path).resolve()
    status_path = Path(str(log_path) + ".json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists() or status_path.exists():
        raise FileExistsError(f"Refusing to replace run evidence: {log_path}")
    # Exclusive reservation protects against another runner racing this preflight.
    with log_path.open("x"):
        pass
    started = time.monotonic()
    record = dict(command=command, command_sha256=_digest(command), log_path=str(log_path),
                  cwd=str(Path(cwd or os.getcwd()).resolve()), timeout_seconds=timeout,
                  grace_seconds=grace_seconds, status="starting", exit_code=None,
                  started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  environment=dict(python=sys.version, platform=platform.platform()),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  environment_override_keys=sorted((env or {}).keys()),
                  process_tree_cleanup="isolated POSIX process group" if os.name == "posix" else "direct process only")
    _write(status_path, record, exclusive=True)
    process = None
    reader = None
    messages = queue.Queue(maxsize=1024)
    stop_reader = threading.Event()
    eof = object()
    cleanup_error = None
    try:
        merged_env = None if env is None else dict(os.environ, **env)
        process = subprocess.Popen(command, cwd=cwd, env=merged_env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                                   start_new_session=os.name == "posix")
        record.update(status="running", pid=process.pid)
        _write(status_path, record)

        def read_output():
            try:
                for line in process.stdout:
                    while not stop_reader.is_set():
                        try:
                            messages.put(line, timeout=.05)
                            break
                        except queue.Full:
                            continue
                    if stop_reader.is_set():
                        break
            finally:
                try:
                    messages.put(eof, timeout=.05)
                except queue.Full:
                    pass

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        with log_path.open("a", buffering=1) as stream:
            stream.write(json.dumps({"runner": "started", "command": command}) + "\n")
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= timeout:
                    record["status"] = "timeout"
                    break
                try:
                    message = messages.get(timeout=min(.05, max(.001, timeout - elapsed)))
                    if message is not eof:
                        stream.write(message)
                        if echo:
                            print(message, end="", flush=True)
                except queue.Empty:
                    pass
                code = process.poll()
                if code is not None:
                    record.update(status="completed" if code == 0 else "failed", exit_code=code)
                    break
            _terminate_tree(process, grace_seconds)
            # Drain buffered output without waiting on a possibly blocked writer.
            drain_until = time.monotonic() + grace_seconds
            while time.monotonic() < drain_until:
                try:
                    message = messages.get(timeout=.02)
                except queue.Empty:
                    if reader is not None and not reader.is_alive():
                        break
                    continue
                if message is eof:
                    break
                stream.write(message)
                if echo:
                    print(message, end="", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            record["interruption_requested"] = True
    finally:
        if process is not None:
            try:
                _terminate_tree(process, grace_seconds)
            except BaseException as error:
                cleanup_error = str(error)
                record.update(status="cleanup_failed", cleanup_error=cleanup_error)
            record["exit_code"] = process.poll()
            if reader is not None and reader.is_alive():
                record["output_may_be_truncated"] = True
                stop_reader.set()
                reader.join(timeout=min(grace_seconds, .2))
            if process.stdout is not None and (reader is None or not reader.is_alive()):
                process.stdout.close()
        record.update(duration_seconds=time.monotonic() - started,
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        _write(status_path, record)
    if record.get("interruption_requested"):
        raise KeyboardInterrupt("Study execution interrupted; evidence retained")
    if check and record["status"] != "completed":
        raise RunFailure(record)
    return record


def run_grid(config, output, *, base_dir=None, echo=False):
    """Execute registered jobs serially; no retries and fail closed by default."""
    if not isinstance(config, dict) or not isinstance(config.get("runs"), list):
        raise ValueError("Grid requires a runs list")
    config = json.loads(json.dumps(config, allow_nan=False))
    global_seconds = _positive(config.get("global_wall_seconds"), "global_wall_seconds")
    grace = _positive(config.get("grace_seconds", 1.0), "grace_seconds")
    planned = []
    seen = set()
    for index, run in enumerate(config["runs"]):
        name = run.get("id", f"job_{index:03d}")
        if (not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") or name in seen):
            raise ValueError("Grid run IDs must be unique safe directory names")
        seen.add(name)
        if ("command" in run) == ("esc50_args" in run):
            raise ValueError("Each run requires exactly one of command or esc50_args")
        if "esc50_args" in run:
            _command(run["esc50_args"])
        command = _command(run["command"] if "command" in run else [sys.executable, 'scripts/legacy/esc50.py', *run["esc50_args"]])
        cap = _positive(run.get("timeout_seconds"), "timeout_seconds")
        planned.append((name, command, cap))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _write(output / "config.json", config, exclusive=True)
    summary = dict(config_sha256=_digest(config), status="running", runs=[],
                   global_wall_seconds=global_seconds,
                   environment=dict(python=sys.version, platform=platform.platform()),
                   cleanup_reserve_seconds=3 * grace + .3)
    _write(output / "grid.json", summary, exclusive=True)
    try:
        for name, command, cap in planned:
            remaining = global_seconds - (time.monotonic() - started)
            # Include termination, output drain, final writes and a scheduling margin.
            allowance = min(cap, remaining - summary["cleanup_reserve_seconds"])
            if allowance <= 0:
                summary.update(status="global_time_limit", unstarted_run=name)
                break
            job = output / name
            job.mkdir()
            _write(job / "registered.json", dict(command=command, command_sha256=_digest(command),
                                                 timeout_seconds=allowance, remaining_global_seconds=remaining), exclusive=True)
            record = run_logged(command, job / "stdout.log", allowance, cwd=base_dir,
                                grace_seconds=grace, check=False, echo=echo)
            summary["runs"].append(dict(id=name, **record))
            _write(output / "grid.json", summary)
            if record["status"] != "completed":
                summary["status"] = "failed"
                break
        else:
            summary["status"] = "completed"
    except BaseException as error:
        summary.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        summary["duration_seconds"] = time.monotonic() - started
        _write(output / "grid.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    grid = sub.add_parser("run-grid")
    grid.add_argument("--config", required=True)
    grid.add_argument("--output", required=True)
    grid.add_argument("--base-dir", default=os.getcwd())
    args = parser.parse_args()
    summary = run_grid(json.loads(Path(args.config).read_text()), args.output, base_dir=args.base_dir, echo=True)
    print(json.dumps({k: summary[k] for k in ("status", "duration_seconds")}, indent=2))
    if summary["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
