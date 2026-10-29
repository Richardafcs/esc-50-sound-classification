# Phase 2: Durable single-host experiment lifecycle

## Scope and evidence

This phase provides a local, **serial** research dispatcher. It is not Kubernetes, a
cloud service, a GPU cluster scheduler, or a daemon. Every `run` invocation executes
at most one pass through the pending/paused queue and then exits.

A `plan.json` is created once. `state.json` records the authoritative current
job status; `events.jsonl` is a supplementary, fsynced audit log with sequential
revision numbers. `state.json` is replaced atomically but event/snapshot writes
are not a single filesystem transaction: gaps can occur after a hard crash.
Do not infer cryptographic tamper-proofing from this logging scheme.

### State semantics

- `PENDING`: no execution attempted.
- `RUNNING`: controller recorded an attempt launch; controller lock prevents a second dispatcher.
- `PAUSED`: the train command ended with `time_limit` or the runner timed out with
  a committed `last.pt`, `manifest.json`, and `run.json`. All identities are **revalidated**
  by `resume-train` when the next invocation starts. File existence is not proof
  that those artifacts are valid.
- `COMPLETED`: train process finished successfully and `run.json` reported completed,
  or a generic command exited with zero status.
- `FAILED`: no safe snapshot for resume, nonzero non-timeout failure, or a non-idempotent
  generic command was orphaned by the scheduler dying.
- `CANCELLED`: an explicit operator cancellation for a `PENDING` or `PAUSED` job only.

There is intentionally **no** automatic retry of a failed generic command.
The scheduler is fail-closed and stops dispatching later jobs after a failure.

### Single writer and crash recovery

`train` and `resume-train` now acquire a nonblocking POSIX `flock` at the sibling
`.<runname>.writer.lock` file. This is an advisory same-host lock: it does not protect
against code that ignores the convention, deleted lockfiles, distributed filesystems
with unreliable locks, or external workers on other hosts. Never unlink a live lockfile.
The controller similarly locks `.controller.lock` for the complete dispatch cycle.

If the controller is killed and restarted, any `RUNNING` training job is reconciled:
only when its training writer lock is free is its status inspected. A completed run
stays complete; a run with the minimum recovery artifacts becomes paused; no
checkpoint becomes failed. A live orphan training writer causes a refusal instead
of a duplicate launch. The training runtime alone checks checkpoint provenance,
environment, cache SHA, and the no-held-out-test-access constraint.

## Using the CLI

Install the local package (`pip install -e .`), then from the repository root:

```bash
# Plan includes two controlled CPU experiments; not run until you dispatch.
python scripts/research_scheduler.py init \
  --plan examples/scheduler_attention_cpu.json \
  --workspace local_scheduler/study_01

# Run at most one job synchronously. Re-run the same command to resume a PAUSED job.
python scripts/research_scheduler.py run \
  --workspace local_scheduler/study_01 --base-dir . --max-jobs 1

python scripts/research_scheduler.py status --workspace local_scheduler/study_01

# Can cancel only pending/paused jobs, not running ones.
python scripts/research_scheduler.py cancel \
  --workspace local_scheduler/study_01 --job attention_ta_seed101
```

Training data must already exist. Run `esc50 preflight` first. Scheduling is not a
replacement for metadata/source leakage checks. Example includes no test evaluation.

`max_cpu_threads` validates each job's declared thread allotment, and jobs execute
serially; it is **not** a system-wide CPU lease or multi-workspace quota manager.
`device="cuda"` passes through to the trainer, but there is currently no GPU
memory reservation, device allocation or GPU concurrency. Run on local, trusted
job plans only: `kind="command"` intentionally executes supplied argv locally,
without a shell, and should never be used with untrusted plans.

Time limits are soft for the trainer and bounded by the external process watchdog.
An interrupt during the first epoch leaves no resumable state and therefore fails
closed. A timeout after at least one committed epoch can resume at that boundary.
No mid-epoch replay, distributed consensus, or durable asynchronous cloud worker
is provided. Controller-initiated live cancellation is not yet implemented.

## Verification

- Unit checks: plan safety, controller lock, direct train writer lock, failed-job
  gating, cancel policy, restart reconciliation, and train/resume command selection.
- One subprocess test creates a **synthetic** log-Mel cache, runs `esc50 train` via
  the dispatcher and verifies epoch 1 checkpoint files, a completed lifecycle,
  and no held-out predictions. This is not an ESC-50 accuracy result.
- Existing runner and resumable-training tests are preserved.
- Historical `.pt` checkpoints and recorded experiments remain unchanged.

## Known gaps / next stage

1. Stronger transactional store (SQLite with committed state transitions) and
   immutable artifact registry for multi-process query workloads.
2. Cooperative stop / drain and signal reconciliation with the active worker.
3. Concurrent CPU/GPU workers with global host-level resource accounting and
   resource-lease recovery, if workload requires concurrency.
4. Multi-host execution with explicit distributed lock semantics, only after local
   invariants and crash tests are established.
5. True audio cache and complete ESC-50 training for scientific conclusions.
