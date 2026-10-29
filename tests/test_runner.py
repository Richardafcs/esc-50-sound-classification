"""Real short subprocess checks; no model optimization or cloud resources."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runner_independent_tests", ROOT / 'scripts/legacy/runner.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def command(source):
    return [sys.executable, "-u", "-c", source]


def test_success_preserves_stdout_and_terminal_metadata(tmp_path):
    log = tmp_path / "success.log"
    record = runner.run_logged(command("print('visible result')"), log, timeout=2, grace_seconds=.1)
    assert record["status"] == "completed" and record["exit_code"] == 0
    assert "visible result" in log.read_text()
    assert json.loads(Path(str(log) + ".json").read_text()) == record
    assert record["duration_seconds"] < 2


def test_silent_process_times_out_without_waiting_for_output(tmp_path):
    start = time.monotonic()
    record = runner.run_logged(command("import time; time.sleep(30)"), tmp_path / "silent.log",
                               timeout=.2, grace_seconds=.1, check=False)
    assert record["status"] == "timeout"
    assert record["exit_code"] is not None
    assert time.monotonic() - start < 1.5


def test_partial_line_is_retained_after_timeout(tmp_path):
    log = tmp_path / "partial.log"
    record = runner.run_logged(command("import sys,time;sys.stdout.write('partial line');sys.stdout.flush();time.sleep(30)"),
                               log, timeout=.2, grace_seconds=.1, check=False)
    assert record["status"] == "timeout"
    assert "partial line" in log.read_text()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group termination check")
def test_timeout_cleans_descendant_that_ignores_termination(tmp_path):
    script = ("import subprocess,sys,time; "
              "child=subprocess.Popen([sys.executable,'-u','-c',"
              "'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']); "
              "print(child.pid,flush=True); time.sleep(30)")
    log = tmp_path / "tree.log"
    record = runner.run_logged(command(script), log, timeout=6.0, grace_seconds=.2, check=False)
    assert record["status"] == "timeout"
    child_pid = int(log.read_text().splitlines()[1])
    # A killed descendant may briefly remain a zombie until init reaps it.
    import subprocess
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True).stdout.strip()
    assert not state or state.startswith("Z")


def test_failed_child_is_recorded_and_raises_after_evidence_write(tmp_path):
    log = tmp_path / "failure.log"
    with pytest.raises(runner.RunFailure) as caught:
        runner.run_logged(command("print('before failure'); raise SystemExit(7)"), log, timeout=6, grace_seconds=.2)
    assert caught.value.record["exit_code"] == 7
    assert caught.value.record["status"] == "failed"
    assert "before failure" in log.read_text()
    assert json.loads(Path(str(log) + ".json").read_text())["exit_code"] == 7


def test_launch_failure_has_terminal_evidence(tmp_path):
    log = tmp_path / "launch.log"
    with pytest.raises(runner.RunFailure):
        runner.run_logged([str(tmp_path / "nonexistent_executable")], log, timeout=2, grace_seconds=.1)
    assert json.loads(Path(str(log) + ".json").read_text())["status"] == "failed"


def test_logs_are_never_overwritten(tmp_path):
    log = tmp_path / "existing.log"
    log.write_text("prior evidence")
    with pytest.raises(FileExistsError):
        runner.run_logged(command("print('replacement')"), log, timeout=2)
    assert log.read_text() == "prior evidence"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_deadline_rejected_before_launch(tmp_path, timeout):
    with pytest.raises(ValueError):
        runner.run_logged(command("pass"), tmp_path / "invalid.log", timeout=timeout)
    assert not list(tmp_path.iterdir())


def test_grid_writes_registration_before_first_child_and_preserves_success(tmp_path):
    out = tmp_path / "grid"
    registered = out / "config.json"
    cfg = {"global_wall_seconds": 10, "grace_seconds": .05, "runs": [
        {"id": "one", "command": command(f"from pathlib import Path; assert Path({str(registered)!r}).is_file(); print('registered')"), "timeout_seconds": 6}]}
    result = runner.run_grid(cfg, out)
    assert result["status"] == "completed" and len(result["runs"]) == 1
    assert json.loads(registered.read_text()) == cfg
    assert (out / "one/registered.json").exists()
    with pytest.raises(FileExistsError):
        runner.run_grid(cfg, out)


def test_expired_global_allowance_does_not_launch_first_job(tmp_path):
    marker = tmp_path / "must_not_exist"
    cfg = {"global_wall_seconds": .01, "grace_seconds": .05, "runs": [
        {"id": "never", "command": command(f"from pathlib import Path; Path({str(marker)!r}).touch()"), "timeout_seconds": 1}]}
    result = runner.run_grid(cfg, tmp_path / "expired")
    assert result["status"] == "global_time_limit" and not result["runs"]
    assert not marker.exists()


def test_grid_fails_closed_and_does_not_launch_later_jobs(tmp_path):
    marker = tmp_path / "must_not_exist"
    cfg = {"global_wall_seconds": 3, "grace_seconds": .05, "runs": [
        {"id": "failure", "command": command("raise SystemExit(2)"), "timeout_seconds": 1},
        {"id": "never", "command": command(f"from pathlib import Path; Path({str(marker)!r}).touch()"), "timeout_seconds": 1}]}
    result = runner.run_grid(cfg, tmp_path / "failure_grid")
    assert result["status"] == "failed" and len(result["runs"]) == 1
    assert not marker.exists()


def test_grid_clips_child_deadline_to_remaining_global_budget(tmp_path):
    cfg = {"global_wall_seconds": .9, "grace_seconds": .05, "runs": [
        {"id": "silent", "command": command("import time;time.sleep(30)"), "timeout_seconds": 10}]}
    start = time.monotonic()
    result = runner.run_grid(cfg, tmp_path / "bounded_grid")
    assert result["status"] == "failed" and result["runs"][0]["status"] == "timeout"
    assert 0 < result["runs"][0]["timeout_seconds"] < .9
    assert time.monotonic() - start < 1.5


def test_grid_rejects_shell_strings_and_malformed_cli_args_before_artifacts(tmp_path):
    for field, value in (("command", "python script.py"), ("esc50_args", "train --model cnn")):
        with pytest.raises(ValueError):
            runner.run_grid({"global_wall_seconds": 3, "runs": [{"id": "invalid", field: value, "timeout_seconds": 1}]},
                            tmp_path / field)
        assert not (tmp_path / field).exists()
