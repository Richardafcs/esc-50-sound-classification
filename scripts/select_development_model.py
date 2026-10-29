"""Apply the registered practical preference to saved development results.

Run after all three seeds and the registered augmentation pilot are recovered.
The augmentation candidate can enter practical selection only after its pilot
passes the declared gate and its repeats finish. An augmented-versus-unaugmented
comparison concerns the whole recipe, not the causal benefit of recurrence.
"""
import argparse
import json
from pathlib import Path

SEEDS = (101, 202, 303)
NONRECURRENT = ('cnn_freq', 'cnn_freq_matched', 'cnn_freq_token')


def read(path):
    return json.loads(Path(path).read_text())


def summarize(root, model, augment=False):
    suffix = '_specaugment' if augment else ''
    records = []
    for seed in SEEDS:
        tag = f'{model}_seed{seed}_60ep{suffix}'
        run = read(root / 'runs' / tag / 'run.json')
        if run['status'] != 'completed' or run['epochs_recorded'] != 60:
            raise ValueError(f'Incomplete registered run: {tag}')
        metric = read(root / 'runs' / tag / 'validation.json')
        records.append(dict(seed=seed, accuracy=metric['accuracy'], macro_f1=metric['macro_f1']))
    return dict(model=model, specaugment=augment, seeds=records,
                accuracy=sum(r['accuracy'] for r in records) / len(records),
                macro_f1=sum(r['macro_f1'] for r in records) / len(records),
                parameters=run['parameters'])


def select(root, latency_root):
    nonrecurrent = [summarize(root, model) for model in NONRECURRENT]
    for candidate in nonrecurrent:
        latency = read(latency_root / f"{candidate['model']}_seed101" / 'latency.json')
        candidate['cpu_p95_ms'] = latency['waveform_file_to_logits']['p95_ms']
        candidate['checkpoint_bytes'] = latency['checkpoint_bytes']
    eligible = [c for c in nonrecurrent if c['cpu_p95_ms'] <= 100 and c['checkpoint_bytes'] <= 5 * 1024**2]
    if not eligible:
        raise ValueError('No nonrecurrent candidate meets the declared prototype limits')
    strongest = min(eligible, key=lambda c: (-c['accuracy'], c['cpu_p95_ms'], c['parameters']))
    recurrent = summarize(root, 'crnn_freq')
    pilot = read(root / 'runs/crnn_freq_seed101_60ep_specaugment/validation.json')
    base101 = recurrent['seeds'][0]
    augmentation_passed = (pilot['accuracy'] - base101['accuracy'] >= .02
                           and pilot['macro_f1'] >= base101['macro_f1'])
    if augmentation_passed:
        augmented = summarize(root, 'crnn_freq', augment=True)
        if augmented['accuracy'] > recurrent['accuracy']:
            recurrent = augmented
    gain = recurrent['accuracy'] - strongest['accuracy']
    preference = gain >= .03 and recurrent['macro_f1'] >= strongest['macro_f1']
    recurrent_tag = 'crnn_freq_seed101' + ('_specaugment' if recurrent['specaugment'] else '')
    latency = read(latency_root / recurrent_tag / 'latency.json')
    recurrent_eligible = (latency['waveform_file_to_logits']['p95_ms'] <= 100
                          and latency['checkpoint_bytes'] <= 5 * 1024**2)
    chosen = recurrent if preference and recurrent_eligible else strongest
    return dict(nonrecurrent_candidates=nonrecurrent, strongest_nonrecurrent=strongest,
                recurrent_candidate=recurrent, recurrent_mean_gain_pp=100 * gain,
                augmentation_pilot_passed=augmentation_passed, chosen=chosen,
                confirmation_seed=101, test_used_for_selection=False,
                interpretation='Development recipe selection; augmentation is not a recurrence-only intervention')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'results/historical/repeated_development')
    parser.add_argument('--latency-root', type=Path, default=Path(__file__).resolve().parents[1] / 'results/historical/latency')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = select(args.root, args.latency_root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
