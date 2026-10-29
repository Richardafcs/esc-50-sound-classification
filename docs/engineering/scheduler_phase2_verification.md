# Phase 2 verification record — 2026-10-28

## What passed in this container

- New scheduler + subprocess integration + pre-existing runner: **27 passed**.
- Data/Foundation, infrastructure and attention unit tests: **28 passed**.
- Epoch-boundary recovery comparison and crash-sidecar repair: **3 passed**.
- Remaining resume tests: **5 passed**.
- Total distinct selected tests: **63 passed** across four invocations.
- An earlier full-suite command exceeded the container execution budget after
  printing >90% progress. **Do not report that invocation as a passing complete suite.**
- A separate combined focused invocation also timed out after partial progress;
  its subsets were subsequently run successfully as listed above.
- One real `esc50 train` subprocess was executed on a small **synthetic** log-Mel
  feature cache. The job reached `COMPLETED`, wrote `last.pt`, `best.pt`, and
  `run.json`, and did not write test predictions. No authentic ESC-50 accuracy
  or long-duration recovery result was produced.
- All **49** historical `.pt` files were byte-for-byte SHA-256 identical to
  the previous Phase 1 full project directory.

## Limitations not yet addressed

- No complete ESC-50 WAV dataset / feature cache was available. No authentic
  research comparison or full-GPU load testing was executed.
- No tests for real host power loss, disk full, unreliable shared network locking,
  multi-machine scheduler ownership, or mid-epoch replay.
- Concurrency is intentionally **one job at a time per workspace**. An independent
  second workspace can still overload the host: there is no host-wide token broker.
- Active cancellation while a train subprocess is running is not implemented.
- State snapshot and event audit file are independently fsynced, but there is
  no atomic transaction spanning both files; sequence gaps are detectable.
- Failed jobs are not automatically retried, and an interrupted generic command
  is treated as failed to avoid duplicate external side effects.
- Locking only guards cooperating POSIX processes on a single host.
