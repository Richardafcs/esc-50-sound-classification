"""Check saved dataset predictions through the packaged WAV inference path.

This deliberately decodes the original audio instead of reading the training
feature cache. It checks portable delivery consistency, not model accuracy on
new data. Supply validation predictions while developing; use test predictions
only after the study's confirmation boundary has been frozen.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from esc50_sound.inference import AudioClassifier


def check(checkpoint, predictions, audio_root):
    classifier = AudioClassifier(checkpoint)
    with Path(predictions).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    mismatches, abstentions = [], 0
    for row in rows:
        result = classifier.classify(audio_root / row['filename'])
        actual = result['top5'][0]['class_id'] if result['status'] == 'classified' else None
        abstentions += actual is None
        if actual != int(row['prediction']):
            mismatches.append(dict(filename=row['filename'], saved=int(row['prediction']), delivered=actual))
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=classifier.checkpoint_sha256,
                predictions=str(predictions), clips=len(rows), abstentions=abstentions,
                matches=len(rows)-len(mismatches), mismatches=mismatches,
                execution_device='cpu', torch_version=str(torch.__version__), threads=torch.get_num_threads(),
                scope='Original WAV through packaged decoding/frontend/model/postprocessing; no cached features')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--audio-root', type=Path, default=Path('data/ESC-50/audio'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = check(args.checkpoint, args.predictions, args.audio_root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key:result[key] for key in ('clips', 'matches', 'abstentions')}))
    if result['mismatches']:
        raise SystemExit(1)
