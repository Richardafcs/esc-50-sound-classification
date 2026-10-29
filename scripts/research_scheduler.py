"""Local persistent research-job controller (serial, POSIX-only)."""
import argparse
import json
from pathlib import Path
# Prefer this checkout over any unrelated installed copy.
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1] / 'src'))
from esc50_sound.scheduler import initialize, status, dispatch, cancel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    p = subs.add_parser('init'); p.add_argument('--plan', required=True); p.add_argument('--workspace', required=True)
    p = subs.add_parser('run'); p.add_argument('--workspace', required=True); p.add_argument('--max-jobs', type=int)
    p.add_argument('--base-dir', default='.'); p.add_argument('--echo', action='store_true')
    p = subs.add_parser('status'); p.add_argument('--workspace', required=True)
    p = subs.add_parser('cancel'); p.add_argument('--workspace', required=True); p.add_argument('--job', required=True)
    a = parser.parse_args()
    if a.action == 'init':
        result = initialize(a.workspace, json.loads(Path(a.plan).read_text()))
    elif a.action == 'run':
        result = dispatch(a.workspace, max_jobs=a.max_jobs, base_dir=a.base_dir, echo=a.echo)
    elif a.action == 'status':
        result = status(a.workspace)
    else:
        result = cancel(a.workspace, a.job)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
