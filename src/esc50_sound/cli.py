"""One command line for dataset preparation, experiments and short-clip prediction."""
from __future__ import annotations

import argparse
import json

from .data import audit_metadata, prepare, read_metadata
from .evaluation import evaluate
from .models import MODEL_NAMES
from .training import smoke, train, resume_train
from .infra import preflight_report, inspect_run, compatibility, run_index, write_index_csv
from .records import write_json

def build_parser():
    from .baseline import baseline, evaluate_test
    from .inference import package, predict

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download pinned dataset, audit, cache deterministic features")
    p.add_argument("--root", default="data/ESC-50")
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.set_defaults(func=prepare)
    p = sub.add_parser("audit")
    p.add_argument("--root", default="data/ESC-50")
    p.set_defaults(func=lambda a: print(json.dumps(audit_metadata(read_metadata(a.root)), indent=2)))
    p = sub.add_parser("smoke")
    p.add_argument("--output", default='results/runs/smoke.json')
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=smoke)
    p = sub.add_parser("train")
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--model", choices=MODEL_NAMES, default="cnn")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-seconds", type=float, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--scheduler", choices=("none", "cosine"), default="none",
                   help="Cosine decay over the declared maximum epochs; no restarts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--train-folds", default="1,2,3")
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--test-fold", type=int, default=5)
    p.add_argument("--specaugment", action="store_true")
    p.add_argument("--raw-audio-root", default="data/ESC-50", help="Official WAV root, required when using waveform augmentation")
    p.add_argument("--noise-probability", type=float, default=0.0, help="Per-training-example Gaussian noise probability; 0 disables")
    p.add_argument("--noise-snr-min-db", type=float, default=10.0, help="Minimum added Gaussian noise SNR")
    p.add_argument("--noise-snr-max-db", type=float, default=30.0, help="Maximum added Gaussian noise SNR")
    p.add_argument("--mixup-alpha", type=float, default=0.0, help="Waveform Mixup Beta alpha; 0 disables")
    p.add_argument("--official-folds", action="store_true", help="Retain known official source overlaps; disclose this protocol")
    p.add_argument("--pause-after-epochs", type=int, help="Stop at a committed epoch boundary (operational only)")
    p.set_defaults(func=train)
    p = sub.add_parser("resume-train", help="Resume a v1 epoch snapshot in an existing run directory")
    p.add_argument("--output", required=True, help="Existing run directory with last.pt")
    p.add_argument("--max-seconds", type=float, default=300, help="Time budget for this invocation")
    p.add_argument("--pause-after-epochs", type=int, help="Cumulative committed epoch limit")
    p.set_defaults(func=resume_train)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--allow-legacy-checkpoint", action="store_true")
    p.add_argument("--output", required=True)
    p.set_defaults(func=evaluate)
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--json-output")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--allow-legacy-checkpoint", action="store_true")
    p.set_defaults(func=predict)
    p = sub.add_parser("package", help="Package one selected checkpoint and its model card")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-card", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-legacy-checkpoint", action="store_true")
    p.set_defaults(func=package)
    p = sub.add_parser("baseline", help="Fit the cheap temporal-statistics classifier")
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--output", default='results/runs/statistics_baseline')
    p.add_argument("--train-folds", default="1,2,3")
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--test-fold", type=int, default=5)
    p.add_argument("--official-folds", action="store_true")
    p.add_argument("--maxiter", type=int, default=1000)
    p.set_defaults(func=baseline)
    p = sub.add_parser("baseline-evaluate", help="Explicit held-out evaluation of the statistical baseline")
    p.add_argument("--artifact", required=True)
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--allow-test", action="store_true")
    p.set_defaults(func=evaluate_test)
    p = sub.add_parser("preflight", help="Read-only data/cache/split training readiness check")
    p.add_argument("--root", default="data/ESC-50")
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--train-folds", default="1,2,3")
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--test-fold", type=int, default=5)
    p.add_argument("--official-folds", action="store_true")
    p.add_argument("--quick", action="store_true", help="Skip full cache digest; blocks final approval")
    p.add_argument("--verify-audio", action="store_true", help="Require full WAV headers and SHA256 provenance (needs current cache manifest)")
    p.add_argument("--output", help="Optional NEW JSON output; existing evidence is never overwritten")
    def _preflight(a):
        report = preflight_report(a.root, a.cache, train_folds=tuple(int(n) for n in a.train_folds.split(",")),
                                  val_fold=a.val_fold, test_fold=a.test_fold,
                                  official_folds=a.official_folds, verify_cache=not a.quick,
                                  verify_audio=a.verify_audio)
        if a.output:
            from pathlib import Path
            if Path(a.output).exists():
                raise FileExistsError("Preflight output already exists")
            write_json(a.output, report)
        print(json.dumps(report, indent=2))
        if not report["ready"]:
            raise SystemExit(2)
    p.set_defaults(func=_preflight)
    p = sub.add_parser("index-runs", help="Read-only run registry; NO test metrics")
    p.add_argument("--root", default="runs")
    p.add_argument("--output", required=True, help="Path for a NEW JSON index")
    p.add_argument("--csv", help="Optional NEW CSV summary")
    def _index(a):
        from pathlib import Path
        if Path(a.output).exists() or (a.csv and Path(a.csv).exists()):
            raise FileExistsError("Index destination already exists")
        index = run_index(a.root)
        write_json(a.output, index)
        if a.csv:
            write_index_csv(a.csv, index)
        print(json.dumps({k: index[k] for k in ("total", "completed", "incomplete", "test_metrics_read")}, indent=2))
    p.set_defaults(func=_index)
    p = sub.add_parser("compare-runs", help="Require comparable provenance for validation results")
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--output", help="Optional NEW JSON report")
    def _compare(a):
        from pathlib import Path
        left, right = inspect_run(a.run_a), inspect_run(a.run_b)
        gate = compatibility(left, right)
        result = dict(comparison=gate, left=left, right=right)
        if a.output:
            if Path(a.output).exists():
                raise FileExistsError("Comparison report already exists")
            write_json(a.output, result)
        print(json.dumps(result, indent=2))
        if not gate["eligible"]:
            raise SystemExit(2)
    p.set_defaults(func=_compare)
    p = sub.add_parser("cepstral-baseline", help="DCT logmel cepstral-statistics + linear SVM baseline")
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--train-folds", default="1,2,3")
    p.add_argument("--val-fold", type=int, default=4)
    p.add_argument("--test-fold", type=int, default=5)
    p.add_argument("--official-folds", action="store_true")
    p.add_argument("--maxiter", type=int, default=5000)
    from .cepstral import fit_cepstral_svm, evaluate_cepstral_svm
    p.set_defaults(func=fit_cepstral_svm)
    p = sub.add_parser("cepstral-evaluate", help="Explicit test evaluation of frozen cepstral SVM")
    p.add_argument("--artifact", required=True)
    p.add_argument("--cache", default="data/esc50_features.npz")
    p.add_argument("--output", required=True)
    p.add_argument("--allow-test", action="store_true")
    p.set_defaults(func=evaluate_cepstral_svm)
    p = sub.add_parser("audit-baseline", help="Read-only validation-only baseline artifact audit")
    p.add_argument("--artifact", required=True)
    p.add_argument("--output", help="Optional NEW JSON audit destination")
    def _audit_baseline(a):
        from pathlib import Path
        from .quality import baseline_artifact_contract, cepstral_artifact_contract
        from pathlib import Path
        metadata = json.loads((Path(a.artifact) / "model.json").read_text())
        if "LinearSVC" in metadata.get("model", ""):
            report = cepstral_artifact_contract(a.artifact)
        else:
            report = baseline_artifact_contract(a.artifact)
        if a.output:
            if Path(a.output).exists():
                raise FileExistsError("Baseline audit output already exists")
            write_json(a.output, report)
        print(json.dumps(report, indent=2))
        if not report["valid"]:
            raise SystemExit(2)
    p.set_defaults(func=_audit_baseline)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "max_seconds") and args.max_seconds <= 0:
        parser.error("max-seconds must be positive")
    if hasattr(args, "epochs") and args.epochs <= 0:
        parser.error("epochs must be positive")
    if getattr(args, "pause_after_epochs", None) is not None and args.pause_after_epochs <= 0:
        parser.error("pause-after-epochs must be positive")
    # argparse's function object must not enter the JSON run configuration.
    function = args.func
    del args.func
    try:
        function(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


