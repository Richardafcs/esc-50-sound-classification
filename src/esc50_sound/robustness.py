"""Validation-only inference with the original study's paired perturbations.

No training, cache loading, or file-integrity hashing. A small SHA-256 of the
noise-address string preserves the original experiment's random-noise mapping;
it does not read or verify file contents.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .audio import log_mel, read_audio
from .data import read_metadata, split_rows
from .evaluation import compute_metrics
from .models import build_model

CONDITIONS = ('clean', 'snr_20', 'snr_10', 'snr_0', 'gain_-6', 'gain_6')
PROTOCOL = 'esc50_legacy_stress_v1'
FAMILIES = {
    'cnn': ('B_architecture', 'B_cnn_freq', 'cnn_freq'),
    'crnn': ('B_architecture', 'B_crnn_freq', 'crnn_freq'),
    'noise_cnn': ('C_augmentation', 'C_noise', 'cnn_freq'),
}


def validation_rows(metadata_root: Path) -> list[dict]:
    rows = read_metadata(metadata_root)
    ids = split_rows(rows, (1, 2, 3), 4, 5, strict=True)['validation']
    selected = [rows[i] for i in ids]
    if len(selected) != 398 or len({r['src_file'] for r in selected}) != 295:
        raise ValueError('Expected Source-Strict Fold4: 398 clips from 295 sources')
    return selected


def noise_address(filename: str, condition: str, seed: int = 2026) -> int:
    """Legacy per-clip RNG address, independent of model, order, and batching."""
    return int(hashlib.sha256(f'{seed}:{condition}:{filename}'.encode()).hexdigest()[:8], 16)


def perturb(wave: torch.Tensor, condition: str, filename: str,
            noise_seed: int = 2026) -> torch.Tensor:
    if wave.ndim != 1 or wave.numel() != 80000 or wave.dtype != torch.float32 or not torch.isfinite(wave).all():
        raise ValueError('Expected a finite float32 waveform [80000] on CPU')
    if wave.device.type != 'cpu':
        raise ValueError('Construct the historical Gaussian perturbation on CPU')
    if condition not in CONDITIONS:
        raise ValueError('Unknown robustness condition')
    if condition == 'clean':
        return wave
    if condition.startswith('gain_'):
        return wave * 10 ** (float(condition[5:]) / 20)
    signal_rms = wave.square().mean().sqrt()
    if signal_rms <= 1e-8:
        raise ValueError('SNR is undefined for a silent reference waveform')
    rng = torch.Generator().manual_seed(noise_address(filename, condition, noise_seed))
    noise = torch.randn(wave.shape, generator=rng)
    noise -= noise.mean()
    noise /= noise.square().mean().sqrt().clamp_min(1e-10)
    return wave + noise * signal_rms / 10 ** (float(condition[4:]) / 20)


def read_predictions(path: Path, rows: list[dict]) -> list[int]:
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['filename', 'target', 'prediction']:
            raise ValueError(f'Unexpected prediction schema: {path}')
        values = list(reader)
    by_name = {r['filename']: r for r in values}
    if len(values) != len(rows) or len(by_name) != len(rows) or set(by_name) != {r['filename'] for r in rows}:
        raise ValueError(f'Prediction membership differs from validation: {path}')
    output = []
    for row in rows:
        record = by_name[row['filename']]
        if int(record['target']) != int(row['target']) or not 0 <= int(record['prediction']) < 50:
            raise ValueError(f'Prediction target/class mismatch: {path}')
        output.append(int(record['prediction']))
    return output


def tasks(run_root: Path, output_root: Path, rows: list[dict], seeds=(202, 303)) -> list[dict]:
    result = []
    for family, (group, prefix, model) in FAMILIES.items():
        for seed in seeds:
            task_id = f'{prefix}_s{seed}'
            folder = output_root / task_id
            record = folder / 'stress.json'
            complete = False
            if record.is_file():
                document = json.loads(record.read_text())
                if (document.get('status') == 'completed' and document.get('protocol') == PROTOCOL
                        and document.get('checkpoint_task') == task_id and document.get('noise_seed') == 2026
                        and tuple(document.get('conditions', {})) == CONDITIONS):
                    try:
                        for condition in CONDITIONS:
                            read_predictions(folder / f'{condition}_predictions.csv', rows)
                        complete = True
                    except (ValueError, OSError, KeyError):
                        pass
            result.append(dict(id=task_id, family=family, seed=seed, model=model,
                               checkpoint=str(run_root / group / task_id / 'best.pt'),
                               output=str(folder), status='completed' if complete else 'pending'))
    return result


def _write_json(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def _write_predictions(path: Path, rows: list[dict], predictions: list[int]) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('filename', 'target', 'prediction'))
        writer.writerows((r['filename'], r['target'], p) for r, p in zip(rows, predictions))
    temp.replace(path)


def execute(selected: list[dict], rows: list[dict], audio_root: Path, *,
            batch_size: int = 32, threads: int = 2, device: str = 'cpu') -> list[dict]:
    """One waveform decode and one frontend pass/condition, shared by all models."""
    if batch_size < 1 or threads < 1 or device not in ('cpu', 'cuda'):
        raise ValueError('Positive batch size/threads and cpu/cuda device required')
    torch.set_num_threads(threads)
    waves = []
    for row in rows:
        path = audio_root / row['filename']
        waves.append(read_audio(path))  # Only the 398 validation WAVs are opened.
    targets = [int(r['target']) for r in rows]
    loaded = []
    for task in selected:
        checkpoint = torch.load(task['checkpoint'], map_location='cpu', weights_only=True)
        expected_classes = {int(r['target']): r['category'] for r in rows}
        if (checkpoint['model_name'] != task['model'] or checkpoint['seed'] != task['seed']
                or tuple(checkpoint.get('train_folds', ())) != (1, 2, 3)
                or checkpoint.get('val_fold') != 4 or checkpoint.get('test_fold') != 5
                or checkpoint['class_names'] != [expected_classes[i] for i in range(50)]):
            raise ValueError(f'Checkpoint model/seed/split/class contract differs: {task["id"]}')
        mean, std = checkpoint['mean'], checkpoint['std']
        if mean.shape != (1, 1, 64, 1) or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError('Invalid checkpoint training normalization')
        model = build_model(task['model']).to(device).eval()
        model.load_state_dict(checkpoint['model'])
        folder = Path(task['output'])
        folder.mkdir(parents=True, exist_ok=True)
        record = dict(schema_version=1, status='running', protocol=PROTOCOL,
                      checkpoint_task=task['id'], training_seed=task['seed'], noise_seed=2026,
                      split='validation', clips=len(rows), sources=len({r['src_file'] for r in rows}),
                      checkpoint=task['checkpoint'], audio_root=str(audio_root),
                      torch_version=str(torch.__version__), device=device, threads=threads,
                      conditions={}, scope='Fixed checkpoints and one Gaussian realization per clip/condition; no training or test inference')
        _write_json(folder / 'stress.json', record)
        loaded.append((task, model, mean, std, record))
    started = time.monotonic()
    for condition in CONDITIONS:
        features, snrs = [], []
        for row, wave in zip(rows, waves):
            changed = perturb(wave, condition, row['filename'])
            if condition.startswith('snr_'):
                achieved = float(10 * torch.log10(wave.double().square().mean() / (changed.double() - wave.double()).square().mean()))
                if not math.isfinite(achieved) or abs(achieved - float(condition[4:])) > .01:
                    raise ValueError(f'Realized added-noise SNR differs: {row["filename"]}')
                snrs.append(achieved)
            features.append(log_mel(changed))
        x = torch.stack(features)
        with torch.inference_mode():
            for task, model, mean, std, record in loaded:
                predictions = []
                for begin in range(0, len(x), batch_size):
                    batch = ((x[begin:begin + batch_size] - mean) / std).to(device)
                    predictions.extend(model(batch).argmax(1).cpu().tolist())
                if condition == 'clean':
                    reference = read_predictions(Path(task['checkpoint']).parent / 'validation_predictions.csv', rows)
                    disagreements = sum(p != q for p, q in zip(predictions, reference))
                    if disagreements:
                        raise ValueError(f'{task["id"]}: clean WAV replay changed {disagreements} predictions; investigate before pairing')
                folder = Path(task['output'])
                _write_predictions(folder / f'{condition}_predictions.csv', rows, predictions)
                metrics = compute_metrics(targets, predictions)
                record['conditions'][condition] = {k: metrics[k] for k in ('n', 'accuracy', 'macro_f1')}
                if snrs:
                    record['conditions'][condition]['realized_snr_db_range'] = [min(snrs), max(snrs)]
                _write_json(folder / 'stress.json', record)
                print(json.dumps(dict(task=task['id'], condition=condition,
                                      accuracy=metrics['accuracy'], clips=len(predictions))), flush=True)
    completed = []
    for task, _, _, _, record in loaded:
        record.update(status='completed', elapsed_seconds=time.monotonic() - started,
                      clean_prediction_matches=len(rows))
        _write_json(Path(task['output']) / 'stress.json', record)
        completed.append(record)
    return completed
