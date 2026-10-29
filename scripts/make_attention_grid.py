"""Generate (never execute) a pre-registered Phase 1 training grid.

Needs full ESC-50 feature cache and a GPU when the resulting grid is executed.
This tool does not evaluate a test fold and does not choose a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("crnn_freq", "crnn_freq_ta", "crnn_freq_ca", "crnn_freq_tca")
SEEDS = (101, 202, 303)


def make_grid(*, cache="data/esc50_features.npz", run_root='results/runs/attention_v1',
              device="cuda"):
    if device not in ("cuda", "cpu", "auto"):
        raise ValueError("device must be cuda, cpu, or auto")
    jobs = []
    for model in MODELS:
        for seed in SEEDS:
            run_name = f"{model}_seed{seed}"
            argv = [
                "train", "--cache", cache, "--model", model,
                "--output", f"{run_root}/{run_name}",
                "--epochs", "60", "--max-seconds", "3500",
                "--batch-size", "32", "--lr", "0.001", "--scheduler", "cosine",
                "--seed", str(seed), "--threads", "2", "--device", device,
                "--train-folds", "1,2,3", "--val-fold", "4", "--test-fold", "5",
            ]
            jobs.append({"id": run_name, "timeout_seconds": 3600, "esc50_args": argv})
    return {
        "global_wall_seconds": 12 * 3600 + 60,
        "grace_seconds": 1,
        "runs": jobs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--run-root", default='results/runs/attention_v1')
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    args = p.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Existing registrations and experiment evidence must not be overwritten.
    with path.open("x") as stream:
        json.dump(make_grid(cache=args.cache, run_root=args.run_root, device=args.device),
                  stream, indent=2)
        stream.write("\n")
    print(f"Created 12-run, TRAIN-ONLY grid at {path} (nothing executed).")


if __name__ == "__main__":
    main()
