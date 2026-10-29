"""Load tensor-only checkpoints and validate their portable experimental provenance."""
from __future__ import annotations

import warnings

import torch

from .audio import FRONTEND
from .data import split_rows
from .records import canonical_hash

def load_checkpoint(path, device="cpu"):
    # Load only locally produced checkpoints; weights_only protects against arbitrary pickle objects.
    return torch.load(path, map_location=device, weights_only=True)


def validate_checkpoint(checkpoint, allow_legacy=False):
    if checkpoint.get("frontend") != FRONTEND:
        raise ValueError("Checkpoint frontend is incompatible with this implementation")
    mean, std = checkpoint.get("mean"), checkpoint.get("std")
    if (not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor) or
            tuple(mean.shape) != (1, 1, 64, 1) or tuple(std.shape) != (1, 1, 64, 1) or
            not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all() or
            len(checkpoint.get("class_names", [])) != 50 or
            len(set(checkpoint.get("class_names", []))) != 50):
        raise ValueError("Checkpoint normalization/class mapping is invalid")
    if checkpoint.get("schema_version") != 2:
        if not allow_legacy:
            raise ValueError("Legacy checkpoint lacks portable provenance; explicitly allow legacy compatibility")
        warnings.warn("Legacy checkpoint: source policy and manifest identity are unverified", RuntimeWarning)
        return "legacy: source policy/manifest unverified"
    manifest = checkpoint.get("manifest")
    if (not isinstance(manifest, dict) or canonical_hash(manifest) != checkpoint.get("manifest_id") or
            manifest.get("source_policy") != checkpoint.get("source_policy") or
            manifest.get("cache_sha256") != checkpoint.get("cache_sha256") or
            manifest.get("frontend") != checkpoint.get("frontend") or
            manifest.get("class_names") != checkpoint.get("class_names") or
            manifest.get("config", {}).get("model") != checkpoint.get("model_name")):
        raise ValueError("Checkpoint portable manifest is inconsistent")
    try:
        config = manifest["config"]
        train_folds = tuple(int(f) for f in config["train_folds"].split(","))
        val_fold, test_fold = config["val_fold"], config["test_fold"]
        split_rows([], train_folds, val_fold, test_fold, strict=False)
        if (tuple(checkpoint["train_folds"]) != train_folds or
                checkpoint["val_fold"] != val_fold or checkpoint["test_fold"] != test_fold):
            raise ValueError("Checkpoint fold roles differ from its portable manifest")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Checkpoint fold-role provenance is incomplete") from error
    return "verified embedded manifest"


