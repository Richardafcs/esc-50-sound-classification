"""Preview or finish six validation-only robustness tasks from existing weights."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from esc50_sound.robustness import execute, tasks, validation_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-tasks', type=int, default=1)
    parser.add_argument('--run-root', type=Path, default=ROOT / 'results/matrix_20261028/execution/runs')
    parser.add_argument('--metadata-root', type=Path, default=ROOT / 'data/ESC-50')
    parser.add_argument('--audio-root', type=Path, default=ROOT / 'data/ESC-50/audio')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/followup_20261104/robustness')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args(argv)
    if args.max_tasks < 1:
        parser.error('--max-tasks must be positive')
    rows = validation_rows(args.metadata_root)
    planned = tasks(args.run_root, args.output, rows)
    pending = [t for t in planned if t['status'] != 'completed']
    print(json.dumps(dict(mode='execute' if args.execute else 'preview',
                          pending=len(pending), tasks=planned), indent=2), flush=True)
    if args.execute and pending:
        execute(pending[:args.max_tasks], rows, args.audio_root,
                batch_size=args.batch_size, threads=args.threads, device=args.device)


if __name__ == '__main__':
    main()
