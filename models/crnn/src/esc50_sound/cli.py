"""One command line for dataset preparation, experiments and short-clip prediction."""
from __future__ import annotations

import argparse
import json

from .data import audit_metadata, prepare, read_metadata
from .evaluation import evaluate
from .models import MODEL_NAMES
from .training import smoke, train

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
    p.add_argument("--output", default="results/smoke.json")
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
    p.add_argument("--official-folds", action="store_true", help="Retain known official source overlaps; disclose this protocol")
    p.set_defaults(func=train)
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
    p.add_argument("--output", default="results/statistics_baseline")
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
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "max_seconds") and (args.max_seconds <= 0 or args.epochs <= 0):
        parser.error("epochs and max-seconds must be positive")
    # argparse's function object must not enter the JSON run configuration.
    function = args.func
    del args.func
    try:
        function(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


