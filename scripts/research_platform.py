#!/usr/bin/env python3
"""Phase 5 contract-first interface. Never evaluates the held-out test split."""
import argparse
import json
from pathlib import Path
import sys

# Work from an editable install or PYTHONPATH=src, as with the existing tools.
# Prefer this checkout over any unrelated installed copy.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / 'src'))
from esc50_sound.platform import builtin_models, TrainingPlan, ContractError


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("catalog", help="List currently implemented, not aspirational, adapters")
    c = sub.add_parser("check", help="Validate a frozen plan and actual data provenance")
    c.add_argument("--plan", required=True)
    r = sub.add_parser("run", help="Explicitly run a checked supervised experiment")
    r.add_argument("--plan", required=True)
    r.add_argument("--execute", action="store_true", help="Mandatory opt-in to run training")
    v = sub.add_parser("verify-run", help="Read-only provenance check against a frozen plan; no test access")
    v.add_argument("--plan", required=True)
    v.add_argument("--run-dir", required=True)
    args = p.parse_args(argv)
    try:
        if args.action == "catalog":
            report = {"models": builtin_models().describe(),
                      "dataset_backends": [
                          {"name": "feature_cache", "capabilities": ["supervised_train", "inspect"]},
                          {"name": "raw_wav", "capabilities": ["inspect"]}],
                      "not_implemented": ["PANNs fine-tune", "AST fine-tune", "CLAP zero-shot"]}
        else:
            plan = TrainingPlan.from_file(args.plan)
            if args.action == "verify-run":
                from esc50_sound.infra import inspect_run
                run_dir = Path(args.run_dir)
                if run_dir.resolve() != Path(plan.output).resolve():
                    raise ContractError("Run directory does not match frozen experiment output")
                identity_file = run_dir / "manifest.json"
                identity = json.loads(identity_file.read_text())
                if identity.get("config", {}).get("platform_plan_sha256") != plan.identity():
                    raise ContractError("Run manifest is not bound to the supplied frozen plan")
                checked = inspect_run(run_dir)
                report = {"plan_sha256": plan.identity(), "run": checked,
                          "plan_bound": True, "test_predictions_read": False,
                          "scope": "read-only record inspection, not independent inference replay"}
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0
            report = plan.check()
            if args.action == "run":
                if not args.execute:
                    raise ContractError("Execution requires --execute; no silent training")
                if not report["ready"]:
                    raise ContractError("Preflight blocked: " + "; ".join(report["blockers"]))
                plan.execute()
                report["execution"] = "trainer returned; inspect run.json to distinguish completed from paused"
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.action != "catalog" and not report["ready"]:
            return 2
        return 0
    except (ContractError, ValueError, OSError, KeyError) as exc:
        print(f"Contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
