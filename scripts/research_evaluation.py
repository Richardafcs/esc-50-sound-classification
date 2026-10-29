"""Audit existing prediction evidence; holdout access requires an explicit flag.

Example:
  python scripts/research_evaluation.py audit --protocol configs/eval_dev_v1.json \
      --predictions .../validation_predictions.csv --metadata-root data/ESC-50
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from esc50_sound.research_eval import Protocol, audit_predictions, compare_predictions, audit_run_link, summarize_cv, _public
from esc50_sound.records import write_json_new


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for action in ("audit", "compare"):
        p = sub.add_parser(action)
        p.add_argument("--protocol", required=True)
        p.add_argument("--metadata-root", default="data/ESC-50")
        p.add_argument("--allow-holdout", action="store_true", help="Explicitly authorize reading previously accessed test predictions")
        p.add_argument("--output", required=True, help="New output JSON; refuses overwrite")
        if action == "audit":
            p.add_argument("--predictions", required=True)
            p.add_argument("--metrics")
            p.add_argument("--run-dir")
            p.add_argument("--checkpoint")
        else:
            p.add_argument("--left", required=True)
            p.add_argument("--right", required=True)
            p.add_argument("--left-metrics")
            p.add_argument("--left-run")
            p.add_argument("--right-run")
            p.add_argument("--left-checkpoint")
            p.add_argument("--right-checkpoint")
            p.add_argument("--right-metrics")
            p.add_argument("--bootstrap", type=int, default=2000)
            p.add_argument("--seed", type=int, default=2026)
    p = sub.add_parser("summarize-cv", help="Retrospective, 5-fold prediction and checkpoint audit")
    p.add_argument("--manifest", required=True)
    p.add_argument("--metadata-root", default="data/ESC-50")
    p.add_argument("--allow-holdout", action="store_true")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; audit evidence is append-only by file path")
    try:
        protocol = None if args.command == "summarize-cv" else Protocol.from_file(args.protocol)
        if args.command == "summarize-cv":
            result = summarize_cv(args.manifest, args.metadata_root, allow_heldout=args.allow_holdout)
        elif args.command == "audit":
            result = _public(audit_predictions(args.predictions, args.metadata_root, protocol,
                                              metrics_file=args.metrics, allow_heldout=args.allow_holdout))
            if args.run_dir:
                result["run_link"] = audit_run_link(args.run_dir, result, metrics_file=args.metrics, checkpoint=args.checkpoint)
                if not result["run_link"]["linked"]:
                    raise ValueError("Run/evaluation evidence could not be linked: " + str(result["run_link"]["reasons"]))
        else:
            result = compare_predictions(args.left, args.right, args.metadata_root, protocol,
                                         allow_heldout=args.allow_holdout, repetitions=args.bootstrap,
                                         seed=args.seed, left_metrics=args.left_metrics,
                                         right_metrics=args.right_metrics, left_run=args.left_run,
                                         right_run=args.right_run, left_checkpoint=args.left_checkpoint,
                                         right_checkpoint=args.right_checkpoint)
        write_json_new(output, result)
        print(json.dumps({k: result.get(k) for k in ("valid", "delta_accuracy", "n") if k in result}))
    except (ValueError, OSError, PermissionError) as exc:
        parser.exit(2, f"Audit refused: {exc}\n")


if __name__ == "__main__":
    main()
