#!/usr/bin/env python3
"""Reproducible, lightweight feature x sampling-rate ablations (no test access)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
# Prefer this checkout over any unrelated installed copy.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / 'src'))
from esc50_sound.feature_ablation import GRID, by_name, prepare_case, run_case, summary, paired_contrasts

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("plan", help="Print the 12 prespecified cases")
    prepare = sub.add_parser("prepare", help="Generate caches from official WAV; no file hashes")
    prepare.add_argument("--raw-root", default="data/ESC-50")
    prepare.add_argument("--cache-dir", default="data/feature_ablation")
    prepare.add_argument("--case", default="all", help="Specific case from plan, or all")
    train = sub.add_parser("train", help="Train one case/seed; validation only; new directory required")
    train.add_argument("--raw-root", default="data/ESC-50")
    train.add_argument("--cache-dir", default="data/feature_ablation")
    train.add_argument("--results-dir", default="results/feature_ablation")
    train.add_argument("--case", required=True)
    train.add_argument("--seed", type=int, default=101)
    train.add_argument("--model", choices=("cnn", "crnn"), default="cnn")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--threads", type=int, default=2)
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    summarize = sub.add_parser("summary", help="Aggregate validation results into CSV")
    summarize.add_argument("--results-dir", default="results/feature_ablation")
    summarize.add_argument("--output", default="results/feature_ablation_summary.csv")
    contrasts = sub.add_parser("contrasts", help="Compute seed-paired validation accuracy/F1 differences")
    contrasts.add_argument("--results-dir", default="results/feature_ablation")
    contrasts.add_argument("--output", default="results/feature_ablation_contrasts.csv")
    a = parser.parse_args(argv)
    if a.action == "plan":
        print(json.dumps([dict(name=s.name, sample_rate=s.sample_rate,
                               max_frequency=s.max_frequency, feature=s.feature,
                               feature_bands=s.bands, fft=s.n_fft, hop=s.hop)
                          for s in GRID], indent=2))
    elif a.action == "prepare":
        for s in GRID if a.case == "all" else (by_name(a.case),):
            print(prepare_case(a.raw_root, a.cache_dir, s), flush=True)
    elif a.action == "train":
        spec = by_name(a.case)
        out = Path(a.results_dir) / a.model / spec.name / f"seed{a.seed}"
        run_case(a.raw_root, a.cache_dir, out, spec,
                 model_name=a.model, seed=a.seed, epochs=a.epochs,
                 batch_size=a.batch_size, learning_rate=a.lr,
                 threads=a.threads, device=a.device)
    elif a.action == "summary":
        rows = summary(a.results_dir, a.output)
        print(f"Wrote {len(rows)} development-validation rows to {a.output}")
    elif a.action == "contrasts":
        rows = paired_contrasts(a.results_dir, a.output)
        print(f"Wrote {len(rows)} seed-paired development contrasts to {a.output}")

if __name__ == "__main__":
    main()
