"""Analyze saved ESC-50 validation predictions; no model/WAV/hash access."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from esc50_sound.data import read_metadata, split_rows
from esc50_sound.prediction_errors import SourceBootstrap, loss_reduction_summary, paired_correctness, read_predictions

SEEDS = (101, 202, 303)
CONDITIONS = ('clean', 'snr_20', 'snr_10', 'snr_0', 'gain_-6', 'gain_6')
VARIANTS = {
    'cnn': ('B_architecture', 'B_cnn_freq_s{seed}'),
    'crnn': ('B_architecture', 'B_crnn_freq_s{seed}'),
    'temporal_attention': ('B_architecture', 'B_crnn_freq_ta_s{seed}'),
    'channel_attention': ('B_architecture', 'B_crnn_freq_ca_s{seed}'),
    'temporal_channel_attention': ('B_architecture', 'B_crnn_freq_tca_s{seed}'),
    'specaugment': ('C_augmentation', 'C_specaugment_s{seed}'),
    'noise_cnn': ('C_augmentation', 'C_noise_s{seed}'),
    'mixup': ('C_augmentation', 'C_mixup_s{seed}'),
}


def write_csv(path, records):
    if not records:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for record in records for k in record))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT/'results/matrix_20261028/execution/runs')
    parser.add_argument('--metadata-root', type=Path, default=ROOT/'data/ESC-50')
    parser.add_argument('--robustness-root', type=Path, default=ROOT/'reports/followup_20261104/robustness')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/followup_20261104')
    parser.add_argument('--resamples', type=int, default=4000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261104)
    args = parser.parse_args(argv)
    rows_all = read_metadata(args.metadata_root)
    indices = split_rows(rows_all, strict=True)['validation']
    rows = sorted([rows_all[i] for i in indices], key=lambda row: row['filename'])
    targets = np.asarray([int(row['target']) for row in rows])
    sources = [row['src_file'] for row in rows]
    bootstrap = SourceBootstrap(sources, args.resamples, args.bootstrap_seed)
    clean, clean_predictions, clean_runs, class_rows, paired_clean, confusion_rows = {}, {}, [], [], [], []
    for variant, (group, task) in VARIANTS.items():
        for seed in SEEDS:
            prediction = read_predictions(args.runs/group/task.format(seed=seed)/'validation_predictions.csv', rows)
            correct = prediction == targets
            clean[variant, seed] = correct
            clean_predictions[variant, seed] = prediction
            clean_runs.append(dict(variant=variant, seed=seed, n=len(rows), correct=int(correct.sum()), accuracy=float(correct.mean()), n_sources=bootstrap.group_count))
        for target in sorted(set(targets)):
            selected = targets == target
            per_seed = [float(clean[variant, seed][selected].mean()) for seed in SEEDS]
            class_rows.append(dict(variant=variant, target=int(target), category=rows[np.flatnonzero(selected)[0]]['category'],
                                   unique_clips=int(selected.sum()), mean_accuracy=float(np.mean(per_seed)),
                                   seed101_accuracy=per_seed[0], seed202_accuracy=per_seed[1], seed303_accuracy=per_seed[2]))
    for variant in VARIANTS:
        events = {}
        for seed in SEEDS:
            for i in np.flatnonzero(clean_predictions[variant, seed] != targets):
                key = int(targets[i]), int(clean_predictions[variant, seed][i])
                events.setdefault(key, []).append((seed, i))
        category_names = {int(row['target']): row['category'] for row in rows}
        for (target, prediction), observed in sorted(events.items(), key=lambda item: (-len(item[1]), item[0])):
            confusion_rows.append(dict(variant=variant, target=target, target_category=category_names[target],
                                       prediction=prediction, predicted_category=category_names[prediction],
                                       seed_clip_error_events=len(observed), unique_clips=len({i for seed, i in observed}),
                                       unique_sources=len({sources[i] for seed, i in observed})))
    for variant in VARIANTS:
        if variant == 'cnn':
            continue
        differences = []
        for seed in SEEDS:
            a, b = clean['cnn', seed], clean[variant, seed]
            difference = b.astype(float)-a.astype(float)
            differences.append(difference)
            paired_clean.append(dict(comparison=f'{variant}-cnn', seed=seed,
                                     **{**paired_correctness(a, b), **bootstrap.summarize(difference)}))
        paired_clean.append(dict(comparison=f'{variant}-cnn', seed='fixed_three_seed_mean',
                                 **bootstrap.summarize(np.stack(differences, axis=1))))
    persistent, persistent_summary = [], {}
    for label, variants in [('cnn', ['cnn']), ('crnn', ['crnn']),
                             ('shared_four_families', ['cnn', 'crnn', 'channel_attention', 'noise_cnn'])]:
        wrong = np.logical_and.reduce([~clean[variant, seed] for variant in variants for seed in SEEDS])
        persistent_summary[label] = dict(clips=int(wrong.sum()), sources=len(set(np.asarray(sources)[wrong])))
        for i in np.flatnonzero(wrong):
            persistent.append(dict(set=label, filename=rows[i]['filename'], target=int(targets[i]),
                                   category=rows[i]['category'], src_file=sources[i]))
    robust, missing, robust_provenance = {}, [], []
    legacy = {'cnn': 'E_clean_cnn', 'crnn': 'E_clean_crnn', 'noise_cnn': 'E_noise_cnn'}
    for variant in legacy:
        task = VARIANTS[variant][1]
        for seed in SEEDS:
            folder = args.robustness_root/task.format(seed=seed)
            if not folder.is_dir() and seed == 101:
                folder = args.runs/'E_robustness'/legacy[variant]
            for condition in CONDITIONS:
                path = folder/f'{condition}_predictions.csv'
                if not path.is_file():
                    missing.append(dict(variant=variant, seed=seed, condition=condition, path=str(path)))
                    continue
                prediction = read_predictions(path, rows)
                correct = prediction == targets
                if condition == 'clean' and not np.array_equal(prediction, clean_predictions[variant, seed]):
                    raise ValueError(f'Clean robustness predictions differ: {variant} seed {seed}')
                robust[variant, seed, condition] = correct
                robust_provenance.append(dict(variant=variant, seed=seed, condition=condition, path=str(path),
                                              correct=int(correct.sum()), accuracy=float(correct.mean())))
    paired_robustness = []
    for comparator in ('noise_cnn', 'crnn'):
        for condition in CONDITIONS:
            differences = []
            available_seeds = []
            for seed in SEEDS:
                if ('cnn', seed, condition) not in robust or (comparator, seed, condition) not in robust:
                    continue
                a, b = robust['cnn', seed, condition], robust[comparator, seed, condition]
                difference = b.astype(float)-a.astype(float)
                differences.append(difference)
                available_seeds.append(seed)
                paired_robustness.append(dict(comparison=f'{comparator}-cnn', condition=condition, seed=seed,
                                              **{**paired_correctness(a, b), **bootstrap.summarize(difference)}))
            if len(available_seeds) == 3:
                paired_robustness.append(dict(comparison=f'{comparator}-cnn', condition=condition,
                                              seed='fixed_three_seed_mean', **bootstrap.summarize(np.stack(differences, axis=1))))
    loss_rows = []
    for condition in CONDITIONS:
        if condition == 'clean':
            continue
        clean_deltas, perturbed_deltas = [], []
        for seed in SEEDS:
            if ('cnn', seed, condition) not in robust or ('noise_cnn', seed, condition) not in robust:
                continue
            clean_delta = clean['noise_cnn', seed].astype(float)-clean['cnn', seed].astype(float)
            perturbed_delta = robust['noise_cnn', seed, condition].astype(float)-robust['cnn', seed, condition].astype(float)
            clean_deltas.append(clean_delta)
            perturbed_deltas.append(perturbed_delta)
            loss_rows.append(dict(comparison='noise_cnn-cnn', condition=condition, seed=seed,
                                  **loss_reduction_summary(clean_delta, perturbed_delta, bootstrap)))
        if len(clean_deltas) == 3:
            loss_rows.append(dict(comparison='noise_cnn-cnn', condition=condition, seed='fixed_three_seed_mean',
                                  **loss_reduction_summary(np.stack(clean_deltas, axis=1), np.stack(perturbed_deltas, axis=1), bootstrap)))
    summary = dict(scope='Source-strict Fold4 development analysis; historical folds were exposed; no independent confirmation, training, test-fold inference, hash or file-integrity checks',
                   n_clips=len(rows), n_sources=bootstrap.group_count, clean_prediction_sets=len(clean),
                   seeds=list(SEEDS), bootstrap_resamples=args.resamples, bootstrap_seed=args.bootstrap_seed,
                   estimand='Clip-weighted paired accuracy difference. Resample G original sources with replacement and keep every clip of each selected source; each draw uses its actual clip count denominator. Use identical source draws for all paired models and fixed training seeds.',
                   uncertainty='Intervals condition on the fitted models and one fixed corruption realization per clip/condition. Three seeds reuse the same audio and are not 3N independent observations. Fixed-three-seed mean intervals do not quantify the distribution of future training runs.',
                   persistent_errors=persistent_summary, robustness_prediction_sets=len(robust),
                   robustness_complete=len(robust)==3*3*len(CONDITIONS), missing_robustness=missing,
                   source_size_min=int(bootstrap.counts.min()), source_size_max=int(bootstrap.counts.max()),
                   bootstrap_clip_denominator_min=int(bootstrap.denominators.min()), bootstrap_clip_denominator_max=int(bootstrap.denominators.max()))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, records in [('clean_runs', clean_runs), ('paired_clean', paired_clean), ('paired_robustness', paired_robustness),
                           ('class_accuracy', class_rows), ('persistent_errors', persistent), ('confusions', confusion_rows), ('robustness_runs', robust_provenance), ('robustness_loss_reduction', loss_rows)]:
        write_csv(args.output/f'{name}.csv', records)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# ESC-50 Prediction Follow-up — 4 November 2026', '',
             f"Analyzed **{len(clean)} clean prediction sets**, {len(rows)} validation clips and {bootstrap.group_count} original sources. Available robustness prediction sets: {len(robust)}/54.", '',
             '## Paired clean CNN vs CRNN', '',
             '| Seed | CRNN repairs | New errors | Net clips | Delta accuracy (pp) | 95% source bootstrap interval (pp) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for record in paired_clean:
        if record['comparison']=='crnn-cnn':
            lines.append(f"| {record['seed']} | {record.get('b_only_correct', '—')} | {record.get('a_only_correct', '—')} | {record.get('net_correct', '—')} | {record['delta_accuracy_pp']:+.3f} | [{record['source_bootstrap_low_pp']:+.3f}, {record['source_bootstrap_high_pp']:+.3f}] |")
    lines += ['', '## Noise-trained CNN vs clean-trained CNN', '',
              '| Condition | Seed | Noise-only correct | Clean-only correct | Net clips | Delta accuracy (pp) | 95% source bootstrap interval (pp) |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for record in paired_robustness:
        if record['comparison']=='noise_cnn-cnn':
            lines.append(f"| {record['condition']} | {record['seed']} | {record.get('b_only_correct', '—')} | {record.get('a_only_correct', '—')} | {record.get('net_correct', '—')} | {record['delta_accuracy_pp']:+.3f} | [{record['source_bootstrap_low_pp']:+.3f}, {record['source_bootstrap_high_pp']:+.3f}] |")
    lines += ['', '## Reduction of corruption-induced accuracy loss', '',
              'For each fixed fit pair, loss reduction = (noise-CNN corrupted accuracy minus clean-CNN corrupted accuracy) minus their clean-input accuracy difference. Positive values mean noise training reduces the corruption-induced accuracy loss. The table averages the three fixed seeds with the same source draws; per-seed values are in `robustness_loss_reduction.csv`.', '',
              '| Condition | Mean loss reduction (pp) | 95% conditional source interval (pp) |',
              '| --- | ---: | ---: |']
    for record in loss_rows:
        if record['seed']=='fixed_three_seed_mean':
            lines.append(f"| {record['condition']} | {record['reduced_accuracy_loss_pp']:+.3f} | [{record['source_bootstrap_low_pp']:+.3f}, {record['source_bootstrap_high_pp']:+.3f}] |")
    lines += ['', '## Persistent errors', '']
    for key, value in persistent_summary.items():
        lines.append(f"- {key}: {value['clips']} clips from {value['sources']} original sources, incorrect in every included seed/model fit.")
    lines += ['', '## Statistical meaning and limits', '', 'Tables: `class_accuracy.csv` averages the same unique clips across three fixed fits. `confusions.csv` distinguishes seed-clip error events from unique clips and sources; repeated-fit errors are not independent observations. `persistent_errors.csv` lists concrete samples for subsequent listening and spectrogram review.', '', summary['estimand'], '', summary['uncertainty'], '',
              f"Sources contain {summary['source_size_min']}–{summary['source_size_max']} clips. Resampled denominators range from {summary['bootstrap_clip_denominator_min']} to {summary['bootstrap_clip_denominator_max']} clips. Replacing those denominators with 398 changes the resampling statistic; averaging source accuracies equally changes the estimand.", '',
              'The bootstrap uses 4,000 resamples by default and an explicit RNG seed (20261104); percentile endpoints can differ from earlier runs that used another seed or a different weighting convention. They are approximate conditional intervals, not a proof of real-world robustness. No family-wise correction is applied to these exploratory contrasts.', '',
              'Persistent error lists and class differences prioritize listening/spectrogram review. They do not identify annotation errors, causal acoustic mechanisms or information inaccessible to a model. Roughly eight unique clips per class remain roughly eight clips after three training seeds.', '',
              'This work uses already-exposed development validation audio. Existing reports and the 65 historical training tasks are preserved. No retraining or test-fold inference is performed.', '']
    if missing:
        lines += [f'**Pending:** {len(missing)} condition-level prediction files remain absent. No missing results are imputed; run the robustness inference command to fill the missing model/seed tasks.', '']
    (args.output/'report_en.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    main()
