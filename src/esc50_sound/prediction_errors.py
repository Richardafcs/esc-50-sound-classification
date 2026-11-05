"""Paired prediction analysis with source-cluster, clip-weighted resampling.

The resampling unit is an original recording source. The estimand remains
accuracy per clip, conditional on the fitted models and corruption realizations.
Training seeds are repeated fits of the same clips, never extra audio samples.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np

def read_predictions(path: str | Path, rows: Sequence[dict]) -> np.ndarray:
    """Align by filename; reject duplicate/missing clips and inconsistent labels."""
    expected = {row['filename']: int(row['target']) for row in rows}
    if len(expected) != len(rows):
        raise ValueError('Duplicate metadata filenames')
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['filename', 'target', 'prediction']:
            raise ValueError('Expected filename,target,prediction columns')
        records = {}
        for record in reader:
            name = record['filename']
            if name in records:
                raise ValueError(f'Duplicate prediction filename: {name}')
            try:
                target, prediction = (int(record['target']), int(record['prediction']))
            except (ValueError, TypeError) as error:
                raise ValueError(f'Invalid class ID: {name}') from error
            if name not in expected or expected[name] != target or (not 0 <= prediction < 50):
                raise ValueError(f'Unexpected clip, target mismatch, or invalid prediction: {name}')
            records[name] = prediction
    if records.keys() != expected.keys():
        raise ValueError('Prediction membership differs from source-strict validation')
    return np.asarray([records[row['filename']] for row in rows], dtype=np.int64)

def paired_correctness(a: np.ndarray, b: np.ndarray) -> dict:
    """B-only correct is a repair, A-only correct is a newly introduced error."""
    a, b = (np.asarray(a, dtype=bool), np.asarray(b, dtype=bool))
    if a.ndim != 1 or b.shape != a.shape or (not len(a)):
        raise ValueError('Expected equally sized nonempty correctness vectors')
    repaired = int((~a & b).sum())
    introduced = int((a & ~b).sum())
    return dict(n=len(a), a_correct=int(a.sum()), b_correct=int(b.sum()), both_correct=int((a & b).sum()), both_wrong=int((~a & ~b).sum()), b_only_correct=repaired, a_only_correct=introduced, net_correct=repaired - introduced, delta_accuracy_pp=100.0 * (repaired - introduced) / len(a))

@dataclass
class SourceBootstrap:
    """One reusable source resampling plan, shared by every fixed seed/model.

    Draw G source IDs with replacement, retain all clips of each sampled source,
    and divide the sum of clip differences by the resulting number of clips.
    Unequal source sizes mean this denominator varies between resamples.
    """
    sources: Sequence[str]
    resamples: int = 4000
    seed: int = 20261104

    def __post_init__(self):
        if self.resamples < 1 or not len(self.sources):
            raise ValueError('Need positive resamples and nonempty sources')
        names, self.inverse = np.unique(np.asarray(self.sources, dtype=str), return_inverse=True)
        self.group_count = len(names)
        self.counts = np.bincount(self.inverse, minlength=self.group_count)
        rng = np.random.default_rng(self.seed)
        self.multiplicities = rng.multinomial(self.group_count, np.full(self.group_count, 1 / self.group_count), size=self.resamples)
        self.denominators = self.multiplicities @ self.counts

    def distribution(self, differences: np.ndarray) -> np.ndarray:
        """Return accuracy differences; input [clips] or [clips,fixed_seeds]."""
        differences = np.asarray(differences, dtype=np.float64)
        if differences.ndim not in (1, 2) or differences.shape[0] != len(self.inverse):
            raise ValueError('Expected [clips] or [clips,fixed_seeds] differences')
        if not np.isfinite(differences).all() or (np.abs(differences) > 1).any():
            raise ValueError('Accuracy differences must be finite and in [-1,1]')
        if differences.ndim == 2:
            if differences.shape[1] == 0:
                raise ValueError('At least one fixed seed is required')
            differences = differences.mean(axis=1)
        group_sums = np.bincount(self.inverse, weights=differences, minlength=self.group_count)
        return self.multiplicities @ group_sums / self.denominators

    def summarize(self, differences: np.ndarray) -> dict:
        differences = np.asarray(differences, dtype=np.float64)
        draws = self.distribution(differences)
        low, high = np.percentile(draws * 100, [2.5, 97.5])
        return dict(delta_accuracy_pp=float(differences.mean() * 100), source_bootstrap_low_pp=float(low), source_bootstrap_high_pp=float(high), n_clips=len(self.inverse), n_sources=self.group_count, bootstrap_resamples=self.resamples, bootstrap_seed=self.seed, minimum_resample_clips=int(self.denominators.min()), maximum_resample_clips=int(self.denominators.max()), uncertainty_scope='source sampling conditional on the fixed models and corruption realization; not training-seed or noise-draw population uncertainty')

def loss_reduction_summary(clean_difference: np.ndarray, perturbed_difference: np.ndarray, bootstrap: SourceBootstrap) -> dict:
    """Difference-in-loss: (B_pert-A_pert) - (B_clean-A_clean).

    This separates a model's clean-accuracy advantage from its reduction of
    corruption-induced loss. The same source draws preserve all four pairings.
    Per-clip values may reach +/-2, so subtract the two paired distributions
    rather than incorrectly treating this as a single [-1,1] accuracy contrast.
    """
    clean_difference = np.asarray(clean_difference, dtype=float)
    perturbed_difference = np.asarray(perturbed_difference, dtype=float)
    if clean_difference.shape != perturbed_difference.shape:
        raise ValueError('Clean and perturbed differences must have equal shapes')
    draws = bootstrap.distribution(perturbed_difference) - bootstrap.distribution(clean_difference)
    low, high = np.percentile(draws * 100, [2.5, 97.5])
    return dict(reduced_accuracy_loss_pp=float((perturbed_difference.mean() - clean_difference.mean()) * 100), source_bootstrap_low_pp=float(low), source_bootstrap_high_pp=float(high), n_clips=len(bootstrap.inverse), n_sources=bootstrap.group_count, bootstrap_resamples=bootstrap.resamples, bootstrap_seed=bootstrap.seed)
