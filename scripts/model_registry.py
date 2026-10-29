"""Local research model registry CLI. No training, metrics calculation or remote writes."""
from __future__ import annotations
import argparse
import json
# Prefer this checkout over any unrelated installed copy.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / 'src'))
from esc50_sound import model_registry as r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='Local registry directory')
    sub = p.add_subparsers(dest='command', required=True)
    for cmd in ('register', 'verify', 'validate', 'approve', 'release', 'archive', 'promote', 'rollback'):
        c = sub.add_parser(cmd)
        c.add_argument('model_id')
        c.add_argument('version')
        if cmd == 'register': c.add_argument('--bundle', required=True)
        if cmd == 'approve':
            c.add_argument('--reviewer', required=True)
            c.add_argument('--review-document', required=True)
            c.add_argument('--evaluation-report', required=True)
        if cmd not in ('verify', 'approve'):
            c.add_argument('--actor', required=True)
        if cmd != 'verify': c.add_argument('--reason', required=True)
    sub.add_parser('resolve')
    sub.add_parser('list')
    a = p.parse_args(argv)
    common = dict(root=a.root)
    if a.command == 'register': result = r.register(**common, model_id=a.model_id, version=a.version, bundle=a.bundle, actor=a.actor, reason=a.reason)
    elif a.command == 'verify': result = r.verify(**common, model_id=a.model_id, version=a.version)
    elif a.command == 'validate': result = r.validate(**common, model_id=a.model_id, version=a.version, actor=a.actor, reason=a.reason)
    elif a.command == 'approve': result = r.approve(**common, model_id=a.model_id, version=a.version, reviewer=a.reviewer, review_document=a.review_document, evaluation_report=a.evaluation_report, reason=a.reason)
    elif a.command == 'release': result = r.release(**common, model_id=a.model_id, version=a.version, actor=a.actor, reason=a.reason)
    elif a.command == 'archive': result = r.archive(**common, model_id=a.model_id, version=a.version, actor=a.actor, reason=a.reason)
    elif a.command in ('promote', 'rollback'):
        result = r.set_research_channel(**common, model_id=a.model_id, version=a.version, actor=a.actor, reason=a.reason, rollback=a.command=='rollback')
    elif a.command == 'resolve': result = r.resolve_research_channel(**common)
    elif a.command == 'list': result = r.list_models(**common)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
