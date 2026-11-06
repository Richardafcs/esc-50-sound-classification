"""Matched recorded-background adaptation of a frozen PANNs linear readout."""
from pathlib import Path
import argparse
import csv
import json
import platform
import signal
import sys
import time
import warnings

import numpy as np
import sklearn
import soundfile as sf
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from esc50_sound.audio import read_audio
from esc50_sound.data import read_metadata, split_rows
from esc50_sound.prediction_errors import SourceBootstrap
from esc50_sound.robustness import perturb
from run_noise_transfer import aligned_cache, mix, write_csv
from run_transfer_followup import embeddings, load_panns


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_head(path):
    with np.load(path, allow_pickle=False) as archive:
        head = {key: archive[key].copy() for key in archive.files}
    if not np.array_equal(head['classes'], np.arange(50)):
        raise ValueError('Expected the fixed 50-class mapping')
    return head


def background(path):
    values, rate = sf.read(path, dtype='float32')
    if rate != 16000 or values.ndim != 1 or len(values) < 80000 or not np.isfinite(values).all():
        raise ValueError(f'Unexpected background format: {path.name}')
    return torch.from_numpy(values)


def signal_description(name, start, segment, role):
    values = segment.double().numpy()
    values = values - values.mean()
    power = np.abs(np.fft.rfft(values)) ** 2
    frequencies = np.fft.rfftfreq(len(values), 1 / 16000)
    total = power.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Evaluation background segment has zero/nonfinite power')
    return dict(environment=name, role=role, channel='ch01', start_seconds=start,
                end_seconds=start + 5, rms=float(np.sqrt(np.mean(values ** 2))),
                peak=float(np.abs(values).max()),
                below_500hz_power_fraction=float(power[frequencies < 500].sum() / total),
                above_4000hz_power_fraction=float(power[frequencies >= 4000].sum() / total),
                spectral_centroid_hz=float((power * frequencies).sum() / total))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--assets', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = args.output
    if (out / 'summary.json').exists():
        print('Completed comparison exists; retain its saved results.')
        return
    protocol = json.loads(args.protocol.read_text())
    training = protocol['training']
    evaluation = protocol['evaluation']
    train_backgrounds = training['backgrounds']
    eval_backgrounds = evaluation['backgrounds']
    starts = evaluation['start_seconds']
    snr_grid = evaluation['snr_db_grid']
    if not train_backgrounds or not eval_backgrounds or set(train_backgrounds) & set(eval_backgrounds):
        raise ValueError('Training/evaluation background recording IDs must be nonempty and disjoint')
    if len(set(train_backgrounds)) != len(train_backgrounds) or len(set(eval_backgrounds)) != len(eval_backgrounds):
        raise ValueError('Duplicate background IDs')
    if not starts or not snr_grid or not np.isfinite(snr_grid).all():
        raise ValueError('Need fixed evaluation offsets and finite SNR grid')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    feature_cache = ROOT / 'results' / ('noise_distribution_' + out.name)
    feature_cache.mkdir(parents=True, exist_ok=True)
    def timeout(*_):
        raise TimeoutError('Fixed 30-minute compute budget reached')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    started = time.perf_counter()
    torch.set_num_threads(2)
    metadata = read_metadata(ROOT / 'data/ESC-50')
    split = split_rows(metadata)
    train = sorted([metadata[i] for i in split['train']], key=lambda row: row['filename'])
    val = sorted([metadata[i] for i in split['validation']], key=lambda row: row['filename'])
    if (len(train), len(val), len(set(row['src_file'] for row in val))) != (1200, 398, 295):
        raise ValueError('Unexpected source-strict target population')
    names = [row['filename'] for row in val]
    targets = np.array([int(row['target']) for row in val])
    train_targets = np.array([int(row['target']) for row in train])
    clean_train = aligned_cache(ROOT / 'results/embedding_cache/train_embeddings.npz', train)
    clean_val = aligned_cache(ROOT / 'results/embedding_cache/val_embeddings.npz', val)
    heads = {
        'clean_refit': load_head(ROOT / 'models/panns_noise_transfer/clean_refit_head.npz'),
        'gaussian_adapted': load_head(ROOT / 'models/panns_noise_transfer/gaussian_adapted_head.npz')}
    reference = heads['clean_refit']
    if not all(np.array_equal(head[key], reference[key]) for head in heads.values() for key in ('mean', 'scale')):
        raise ValueError('Existing readouts do not share the fixed clean scaler')
    scaler = StandardScaler()
    scaler.mean_ = reference['mean'].copy()
    scaler.scale_ = reference['scale'].copy()
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = 1200
    with project_path(training.get('exposure_csv', 'reports/noise_transfer_20261106/training_exposure.csv')).open(newline='') as stream:
        exposure_rows = list(csv.DictReader(stream))
    exposure_lookup = {(int(row['variant']), row['filename']): row for row in exposure_rows}
    if len(exposure_lookup) != 2400 or len(exposure_rows) != 2400:
        raise ValueError('Expected exactly two saved Gaussian SNR assignments per training target')
    train_waves = torch.stack([read_audio(args.assets / 'ESC-50/audio' / row['filename']) for row in train])
    val_waves = torch.stack([read_audio(args.assets / 'ESC-50/audio' / row['filename']) for row in val])
    if not torch.isfinite(train_waves).all() or not torch.isfinite(val_waves).all():
        raise ValueError('Nonfinite foreground waveform')
    device = torch.device(args.device)
    model = load_panns(ROOT / 'official_panns', args.assets / 'Cnn14_16k_mAP=0.438.pth', device)
    backgrounds = {name: background(args.assets / (name + '_ch01.wav')) for name in train_backgrounds}
    # One label-independent permutation, balanced distinct source pairs.
    count = len(train_backgrounds)
    if count < 2 or len(train) % count:
        raise ValueError('The fixed distinct-source design needs balanced background counts')
    permutation = np.random.default_rng(training['seed']).permutation(len(train))
    position = np.arange(len(train))
    assignments = np.empty((2, len(train)), dtype=int)
    assignments[0, permutation] = position % count
    assignments[1, permutation] = (position + 1 + (position // count) % (count - 1)) % count
    rng = np.random.default_rng(training['crop_seed'])
    augmented = []
    exposure = []
    snr_records = []
    for variant in range(2):
        assignment = assignments[variant]
        noise_rows = []
        snrs = []
        for index, row in enumerate(train):
            prior = exposure_lookup[(variant, row['filename'])]
            if int(prior['target']) != int(row['target']) or float(prior['weight']) != .25:
                raise ValueError('Saved exposure target/weight mismatch')
            name = train_backgrounds[int(assignment[index])]
            recording = backgrounds[name]
            offset = int(rng.integers(0, len(recording) - 80000 + 1))
            noise_rows.append(recording[offset:offset + 80000])
            snr = float(prior['snr_db'])
            snrs.append(snr)
            exposure.append(dict(filename=row['filename'], target=int(row['target']), variant=variant,
                                 environment=name, channel='ch01', start_sample=offset,
                                 start_seconds=offset / 16000, snr_db=snr, weight=.25))
        changed, realized = mix(train_waves, torch.stack(noise_rows), torch.tensor(snrs, dtype=torch.float32))
        for row, value in zip(exposure[-len(train):], realized):
            row['realized_snr_db'] = float(value)
        augmented.append(embeddings(model, changed, device))
        snr_records.append(dict(condition='training_variant_' + str(variant), requested_min_db=min(snrs),
                                requested_max_db=max(snrs), realized_min_db=float(realized.min()),
                                realized_max_db=float(realized.max()), max_abs_error_db=float(np.max(abs(realized - snrs))),
                                mixture_peak=float(changed.abs().max())))
        print('Extracted recorded training variant', variant + 1, 'of 2.', flush=True)
    write_csv(out / 'training_exposure.csv', exposure)
    del train_waves, backgrounds, noise_rows, changed
    features = np.concatenate([clean_train, *augmented])
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite training embedding')
    transformed_train = scaler.transform(features)
    labels = np.tile(train_targets, 3)
    weights = np.concatenate([np.full(1200, .5), np.full(2400, .25)])
    fit_start = time.perf_counter()
    head = LogisticRegression(C=1., solver='lbfgs', max_iter=3000, tol=1e-6)
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        head.fit(transformed_train, labels, sample_weight=weights)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    fit = dict(model='recorded_adapted', rows=len(features), total_weight=float(weights.sum()),
               iterations=int(head.n_iter_.max()), converged=converged, seconds=time.perf_counter() - fit_start,
               feature_dtype=str(features.dtype), transformed_dtype=str(transformed_train.dtype))
    write_csv(out / 'head_fits.csv', [fit])
    recorded_head = dict(mean=reference['mean'], scale=reference['scale'], coef=head.coef_,
                         intercept=head.intercept_, classes=head.classes_)
    np.savez_compressed(out / 'recorded_adapted_head.npz', **recorded_head)
    if not converged:
        raise RuntimeError('New head did not converge; results are incomplete, not a scientific negative')
    heads['recorded_adapted'] = recorded_head
    del features, transformed_train, augmented, clean_train
    metrics = []
    predictions = []
    correctness = {}
    conditions = []
    background_rows = []

    def evaluate(condition, values, environment='', start_seconds='', snr_db=''):
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite evaluation embedding')
        info = dict(condition=condition, environment=environment, start_seconds=start_seconds, snr_db=snr_db)
        conditions.append(info)
        for model_name, fitted in heads.items():
            logits = ((values - fitted['mean']) / fitted['scale']) @ fitted['coef'].T + fitted['intercept']
            stable = logits - logits.max(1, keepdims=True)
            probability = np.exp(stable)
            probability /= probability.sum(1, keepdims=True)
            pred = fitted['classes'][logits.argmax(1)]
            correctness[model_name, condition] = pred == targets
            other = logits.copy()
            other[np.arange(len(targets)), targets] = -np.inf
            margin = logits[np.arange(len(targets)), targets] - other.max(1)
            metrics.append(dict(model=model_name, **info, accuracy=accuracy_score(targets, pred),
                                macro_f1=f1_score(targets, pred, labels=np.arange(50), average='macro', zero_division=0),
                                nll=log_loss(targets, probability, labels=np.arange(50)),
                                correct=int((pred == targets).sum()), clips=len(targets)))
            predictions.extend(dict(model=model_name, **info, filename=row['filename'], src_file=row['src_file'],
                                    target=int(target), prediction=int(prediction), true_probability=float(p), true_margin=float(m))
                               for row, target, prediction, p, m in zip(val, targets, pred, probability[np.arange(len(targets)), targets], margin))
        print('Evaluated', condition, flush=True)

    evaluate('clean', clean_val)
    changed = torch.stack([perturb(wave, 'snr_10', name, 2026) for wave, name in zip(val_waves, names)])
    evaluate('historical_shared_gaussian10', embeddings(model, changed, device), environment='Gaussian', snr_db=10)
    del changed
    for name in eval_backgrounds:
        recording = background(args.assets / (name + '_ch01.wav'))
        for start in starts:
            offset = int(round(float(start) * 16000))
            if offset < 0 or offset + 80000 > len(recording):
                raise ValueError('Fixed evaluation offset exceeds recording')
            segment = recording[offset:offset + 80000]
            background_rows.append(signal_description(name, float(start), segment, 'evaluation'))
            for snr in snr_grid:
                condition = f'{name}_{start}s_snr{snr}'
                changed, realized = mix(val_waves, segment[None].expand(len(val), -1), torch.full((len(val),), float(snr)))
                values = embeddings(model, changed, device)
                evaluate(condition, values, name, start, snr)
                np.savez_compressed(feature_cache / (condition + '.npz'), embeddings=values, filenames=np.array(names))
                snr_records.append(dict(condition=condition, requested_min_db=float(snr), requested_max_db=float(snr),
                                        realized_min_db=float(realized.min()), realized_max_db=float(realized.max()),
                                        max_abs_error_db=float(np.max(abs(realized - float(snr)))),
                                        mixture_peak=float(changed.abs().max())))
    del model, val_waves
    torch.cuda.empty_cache()
    bootstrap = SourceBootstrap([row['src_file'] for row in val], resamples=4000, seed=20261107)
    recorded_conditions = [item for item in conditions if item['environment'] in eval_backgrounds]
    effects = []
    aggregates = []
    transitions = []

    def aggregate(group, selected):
        if not selected:
            raise ValueError('Empty aggregation group')
        condition_names = [item['condition'] for item in selected]
        for model_name in heads:
            subset = [item for item in metrics if item['model'] == model_name and item['condition'] in condition_names]
            aggregates.append(dict(model=model_name, group=group, conditions=len(selected),
                                   accuracy=float(np.mean([item['accuracy'] for item in subset])),
                                   macro_f1=float(np.mean([item['macro_f1'] for item in subset])),
                                   nll=float(np.mean([item['nll'] for item in subset]))))
        for comparator in ('gaussian_adapted', 'clean_refit'):
            differences = np.column_stack([correctness['recorded_adapted', name].astype(float) -
                                           correctness[comparator, name].astype(float) for name in condition_names])
            draws = bootstrap.distribution(differences)
            low, high = np.percentile(100 * draws, [2.5, 97.5])
            effects.append(dict(group=group, intervention='recorded_adapted', comparator=comparator,
                                delta_pp=100 * float(differences.mean()), low_pp=float(low), high_pp=float(high),
                                conditions=len(selected), clips=len(val), target_sources=bootstrap.group_count,
                                interval_scope='Target-source resampling conditional on fixed heads, segments and SNR grid'))

    aggregate('recorded_grid_primary', recorded_conditions)
    aggregate('clean', [item for item in conditions if item['condition'] == 'clean'])
    aggregate('historical_shared_gaussian10', [item for item in conditions if item['condition'] == 'historical_shared_gaussian10'])
    for snr in snr_grid:
        aggregate('snr_' + str(snr), [item for item in recorded_conditions if item['snr_db'] == snr])
    for name in eval_backgrounds:
        aggregate('environment_' + name, [item for item in recorded_conditions if item['environment'] == name])
        if len(eval_backgrounds) > 1:
            aggregate('leave_out_' + name, [item for item in recorded_conditions if item['environment'] != name])
    for item in conditions:
        condition = item['condition']
        for comparator in ('gaussian_adapted', 'clean_refit'):
            a = correctness[comparator, condition]
            b = correctness['recorded_adapted', condition]
            transitions.append(dict(condition=condition, comparator=comparator, repairs=int((~a & b).sum()),
                                    new_errors=int((a & ~b).sum()), both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum())))
    for filename, rows in [('metrics.csv', metrics), ('aggregate_metrics.csv', aggregates), ('paired_effects.csv', effects),
                           ('predictions.csv', predictions), ('transitions.csv', transitions),
                           ('background_segments.csv', background_rows), ('realized_snr.csv', snr_records), ('conditions.csv', conditions)]:
        write_csv(out / filename, rows)
    summary = dict(status='completed', protocol=protocol, new_head_fits=1, encoder_fits=0,
                   primary=next(item for item in effects if item['group'] == 'recorded_grid_primary' and item['comparator'] == 'gaussian_adapted'),
                   fit=fit, train_clips=1200, development_clips=398, development_sources=295,
                   training_background_recordings=len(train_backgrounds), evaluation_background_recordings=len(eval_backgrounds),
                   recorded_conditions=len(recorded_conditions), target_noise_pairs=len(recorded_conditions) * 398,
                   test_waveforms_read=0, compute_seconds=time.perf_counter() - started,
                   environment=dict(python=platform.python_version(), torch=torch.__version__,
                                    numpy=np.__version__, scikit_learn=sklearn.__version__,
                                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    signal.alarm(0)
    print('DISTRIBUTION_COMPARISON_COMPLETE', json.dumps(summary['primary']), flush=True)


if __name__ == '__main__':
    main()
