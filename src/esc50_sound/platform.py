"""Phase 5: explicit data/model/experiment contracts over the stable legacy runtime.

No dynamic imports from experiment files, no implicit pretrained weights, no
silent test access, and no change to historical checkpoint or CLI semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .audio import FRONTEND, log_mel, read_audio
from .data import audit_metadata, load_cache, metadata_records, read_metadata, split_rows
from .models import MODEL_NAMES, build_model
from .quality import cache_metadata_contract, metadata_contract
from .records import canonical_hash, sha256


class ContractError(ValueError):
    """A declared dataset, model or execution mode cannot be used safely."""


@dataclass(frozen=True)
class InputContract:
    modality: str
    shape: tuple[int, ...]
    dtype: str
    frontend_id: str

    def validate(self, batch: torch.Tensor) -> None:
        if not isinstance(batch, torch.Tensor) or batch.ndim != len(self.shape) + 1:
            raise ContractError(f"Expected batched {self.modality} tensor [B,{self.shape}]")
        if tuple(batch.shape[1:]) != self.shape or batch.shape[0] <= 0:
            raise ContractError(f"Invalid {self.modality} shape: expected [B,{self.shape}]")
        if str(batch.dtype).replace("torch.", "") != self.dtype or not torch.isfinite(batch).all():
            raise ContractError(f"Expected finite {self.dtype} {self.modality} tensor")


LOGMEL = InputContract("log_mel", (1, 64, 313), "float32", canonical_hash(FRONTEND))
WAVEFORM = InputContract("waveform", (80000,), "float32", canonical_hash(FRONTEND))


@dataclass(frozen=True)
class ModelAdapter:
    name: str
    input_contract: InputContract
    execution_modes: tuple[str, ...]
    pretrained: bool
    builder: Callable[[int], torch.nn.Module]
    num_classes: int = 50

    def build(self, num_classes: int = 50) -> torch.nn.Module:
        if num_classes != self.num_classes:
            raise ContractError(f"{self.name}: this study requires {self.num_classes} classes")
        model = self.builder(num_classes)
        if not isinstance(model, torch.nn.Module):
            raise ContractError("Model adapter did not return a torch.nn.Module")
        return model

    def infer_logits(self, model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        """Check shape, dtype and logits of a built supervised model."""
        if "supervised_train" not in self.execution_modes:
            raise ContractError("This adapter has no supervised logits contract")
        self.input_contract.validate(batch)
        result = model(batch)
        if (not isinstance(result, torch.Tensor) or
                result.shape != (batch.shape[0], self.num_classes) or
                not torch.isfinite(result).all()):
            raise ContractError("Model must return finite [B,num_classes] logits")
        return result

    def describe(self) -> dict:
        return dict(name=self.name, input=asdict(self.input_contract),
                    modes=list(self.execution_modes), pretrained=self.pretrained,
                    num_classes=self.num_classes)


class ModelCatalog:
    """Explicit registration, avoiding arbitrary Python execution from JSON."""

    def __init__(self):
        self._models: dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        if not adapter.name or adapter.name in self._models:
            raise ContractError(f"Duplicate or empty model adapter: {adapter.name!r}")
        if not adapter.execution_modes or any(m not in ("supervised_train", "zero_shot_inference")
                                              for m in adapter.execution_modes):
            raise ContractError("Model adapter declares unsupported execution mode")
        self._models[adapter.name] = adapter

    def resolve(self, name: str, mode: str) -> ModelAdapter:
        if name not in self._models:
            raise ContractError(f"Model {name!r} is not registered; no implicit download or fallback")
        result = self._models[name]
        if mode not in result.execution_modes:
            raise ContractError(f"{name}: execution mode {mode!r} is not supported")
        return result

    def describe(self) -> list[dict]:
        return [a.describe() for a in self._models.values()]


def builtin_models() -> ModelCatalog:
    catalog = ModelCatalog()
    for name in MODEL_NAMES:
        catalog.register(ModelAdapter(name, LOGMEL, ("supervised_train",), False,
                                      builder=lambda classes, model_name=name: build_model(model_name, classes)))
    return catalog


@dataclass(frozen=True)
class SplitPolicy:
    train_folds: tuple[int, ...] = (1, 2, 3)
    val_fold: int = 4
    test_fold: int = 5
    source_strict: bool = True

    def indices(self, rows: list[dict]) -> dict[str, list[int]]:
        try:
            return split_rows(rows, self.train_folds, self.val_fold, self.test_fold,
                              strict=self.source_strict)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Invalid split protocol: {exc}") from exc


@dataclass(frozen=True)
class AudioExample:
    name: str
    target: int
    source: str
    fold: int
    features: torch.Tensor
    raw_sha256: str | None = None


class FeatureCacheBackend:
    """Provenance-checked cached log-mel features with explicit split read guards."""

    kind = "feature_cache"
    output_contract = LOGMEL
    supported_modes = ("supervised_train",)

    def __init__(self, path: str | Path, metadata_root: str | Path, split: SplitPolicy):
        self.path, self.metadata_root, self.split = Path(path), Path(metadata_root), split
        self.cache = load_cache(self.path)  # Existing hash, frontend and schema checks.
        self.rows = read_metadata(self.metadata_root)
        audit_metadata(self.rows)
        contract = metadata_contract(self.rows)
        if not contract["valid"]:
            raise ContractError("Metadata filename/source contract is invalid")
        check = cache_metadata_contract(self.cache, self.rows)
        if not check["valid"]:
            raise ContractError("Cache rows are not aligned to authoritative metadata")
        self.indices = split.indices(self.rows)
        if not self.indices["train"] or not self.indices["validation"]:
            raise ContractError("Train and validation split must not be empty")
        self.cache_sha256 = sha256(self.path)
        self.metadata_sha256 = sha256(self.metadata_root / "meta" / "esc50.csv")

    def get(self, role: str, index: int, *, allow_test: bool = False) -> AudioExample:
        if role not in self.indices:
            raise ContractError("Unknown split role")
        if role == "test" and not allow_test:
            raise PermissionError("Held-out feature access requires explicit allow_test=True")
        try:
            idx = self.indices[role][index]
        except (IndexError, TypeError) as exc:
            raise ContractError("Invalid index within split") from exc
        row = self.rows[idx]
        return AudioExample(row["filename"], int(row["target"]), str(row["src_file"]),
                            int(row["fold"]), torch.from_numpy(self.cache["x"][idx].copy()))

    def describe(self) -> dict:
        return dict(kind=self.kind, output_contract=asdict(self.output_contract),
                    rows=len(self.rows), split_sizes={k: len(v) for k,v in self.indices.items()},
                    cache_sha256=self.cache_sha256, metadata_sha256=self.metadata_sha256,
                    source_strict=self.split.source_strict,
                    supported_modes=list(self.supported_modes))


class RawAudioBackend:
    """Lazy WAV -> logmel adapter. No implicit cache or training support.

    Raw audio may be used to audit individual non-test samples. Training does not
    silently switch to this backend: a future streaming runtime must declare and
    implement its own caching, shuffle, normalization and resume contracts.
    """

    kind = "raw_wav"
    output_contract = LOGMEL
    supported_modes = ("inspect",)

    def __init__(self, root: str | Path, split: SplitPolicy):
        self.root, self.split = Path(root), split
        self.rows = read_metadata(self.root)
        audit_metadata(self.rows)
        if not metadata_contract(self.rows)["valid"]:
            raise ContractError("Metadata filename/source contract is invalid")
        self.indices = split.indices(self.rows)
        self.metadata_sha256 = sha256(self.root / "meta" / "esc50.csv")

    def get(self, role: str, index: int, *, allow_test: bool = False) -> AudioExample:
        if role not in self.indices:
            raise ContractError("Unknown split role")
        if role == "test" and not allow_test:
            raise PermissionError("Held-out raw audio access requires explicit allow_test=True")
        try:
            idx = self.indices[role][index]
        except (IndexError, TypeError) as exc:
            raise ContractError("Invalid index within split") from exc
        row = self.rows[idx]
        path = self.root / "audio" / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Required WAV is unavailable: {path}")
        import soundfile as sf
        info = sf.info(path)
        if info.samplerate != 44100 or info.channels != 1 or info.frames != 5 * 44100:
            raise ContractError("WAV does not match the official ESC-50 44.1kHz mono/5s contract")
        features = log_mel(read_audio(path))
        return AudioExample(row["filename"], int(row["target"]), str(row["src_file"]),
                            int(row["fold"]), features, raw_sha256=sha256(path))

    def describe(self) -> dict:
        return dict(kind=self.kind, output_contract=asdict(self.output_contract),
                    rows=len(self.rows), split_sizes={k:len(v) for k,v in self.indices.items()},
                    metadata_sha256=self.metadata_sha256,
                    source_strict=self.split.source_strict,
                    supported_modes=list(self.supported_modes))


@dataclass(frozen=True)
class TrainingPlan:
    """Frozen, versioned contract; never imports a model by name or reads test."""

    schema_version: int
    model: str
    dataset_backend: str
    metadata_root: str
    cache: str
    output: str
    train_folds: tuple[int, ...]
    val_fold: int
    test_fold: int
    source_strict: bool
    epochs: int
    batch_size: int
    lr: float
    scheduler: str
    seed: int
    threads: int
    device: str
    specaugment: bool
    max_seconds: float
    # Schema v2 only: waveform augmentation; v1 plans retain original identity.
    raw_audio_root: str = "data/ESC-50"
    noise_probability: float = 0.0
    noise_snr_min_db: float = 10.0
    noise_snr_max_db: float = 30.0
    mixup_alpha: float = 0.0

    @classmethod
    def from_dict(cls, raw: dict) -> "TrainingPlan":
        if not isinstance(raw, dict):
            raise ContractError("Experiment plan must be a JSON object")
        v2_keys = {"raw_audio_root", "noise_probability", "noise_snr_min_db",
                   "noise_snr_max_db", "mixup_alpha"}
        if type(raw.get("schema_version")) is not int or raw["schema_version"] not in (1, 2):
            raise ContractError("Unsupported experiment plan schema_version")
        expected = set(cls.__dataclass_fields__) - (v2_keys if raw["schema_version"] == 1 else set())
        if set(raw) != expected:
            raise ContractError(f"Experiment keys mismatch: missing={sorted(expected-set(raw))}, extra={sorted(set(raw)-expected)}")
        if raw["schema_version"] == 2:
            from .augmentation import validate_augmentation
            if not isinstance(raw["raw_audio_root"], str) or not raw["raw_audio_root"].strip():
                raise ContractError("raw_audio_root must be a nonempty path")
            validate_augmentation(raw["noise_probability"], raw["noise_snr_min_db"],
                                  raw["noise_snr_max_db"], raw["mixup_alpha"])
        if not isinstance(raw["train_folds"], list) or not raw["train_folds"] or any(type(f) is not int for f in raw["train_folds"]):
            raise ContractError("train_folds must be nonempty list of integers")
        raw = dict(raw, train_folds=tuple(raw["train_folds"]))
        for key in ("model", "dataset_backend", "metadata_root", "cache", "output", "scheduler", "device"):
            if not isinstance(raw[key], str) or not raw[key]:
                raise ContractError(f"{key} must be a nonempty string")
        for key in ("source_strict", "specaugment"):
            if type(raw[key]) is not bool:
                raise ContractError(f"{key} must be a boolean")
        for key in ("epochs", "batch_size", "threads", "val_fold", "test_fold", "seed"):
            if type(raw[key]) is not int or (key == "seed" and not (0 <= raw[key] < 2**63)) or (key not in ("seed", "val_fold", "test_fold") and raw[key] <= 0):
                raise ContractError(f"{key} must be a valid integer")
        for key in ("lr", "max_seconds"):
            if type(raw[key]) not in (int, float) or not np.isfinite(raw[key]) or raw[key] <= 0:
                raise ContractError(f"{key} must be positive and finite")
        if raw["scheduler"] not in ("none", "cosine") or raw["device"] not in ("cpu", "cuda", "auto"):
            raise ContractError("Unsupported scheduler or device")
        instance = cls(**raw)
        instance.split().indices([])  # Validate disjoint fold roles without reading any test data.
        if instance.dataset_backend != "feature_cache":
            raise ContractError("Existing trainer only supports feature_cache; raw_wav is inspect-only")
        builtin_models().resolve(instance.model, "supervised_train")
        if not instance.output.strip() or Path(instance.output).resolve() in (Path.cwd(), Path.cwd().parent):
            raise ContractError("A distinct experiment output directory is required")
        return instance

    @classmethod
    def from_file(cls, path: str | Path) -> "TrainingPlan":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def split(self) -> SplitPolicy:
        return SplitPolicy(self.train_folds, self.val_fold, self.test_fold, self.source_strict)

    def identity(self) -> str:
        fields = asdict(self)
        if self.schema_version == 1:
            for key in ("raw_audio_root", "noise_probability", "noise_snr_min_db",
                        "noise_snr_max_db", "mixup_alpha"):
                fields.pop(key)
        return canonical_hash(fields)

    def to_legacy_args(self):
        """Invoke the proven existing train implementation without changing artifacts."""
        from argparse import Namespace
        return Namespace(cache=self.cache, model=self.model, output=self.output,
                         epochs=self.epochs, max_seconds=self.max_seconds,
                         batch_size=self.batch_size, lr=self.lr, scheduler=self.scheduler,
                         seed=self.seed, threads=self.threads, device=self.device,
                         train_folds=",".join(map(str, self.train_folds)),
                         val_fold=self.val_fold, test_fold=self.test_fold,
                         specaugment=self.specaugment, official_folds=not self.source_strict,
                         pause_after_epochs=None, platform_plan_sha256=self.identity(),
                         platform_plan_schema_version=self.schema_version,
                         platform_metadata_root=self.metadata_root,
                         platform_dataset_backend=self.dataset_backend,
                         raw_audio_root=self.raw_audio_root,
                         noise_probability=self.noise_probability,
                         noise_snr_min_db=self.noise_snr_min_db,
                         noise_snr_max_db=self.noise_snr_max_db,
                         mixup_alpha=self.mixup_alpha)

    def check(self, *, inspect_cache: bool = True) -> dict:
        model = builtin_models().resolve(self.model, "supervised_train")
        if model.input_contract != LOGMEL:
            raise ContractError("The existing trainer requires the pinned log-mel frontend")
        blockers = []
        if self.device == "cuda" and not torch.cuda.is_available():
            blockers.append("CUDA requested but unavailable; no silent CPU fallback")
        if Path(self.output).exists() and any(Path(self.output).iterdir()):
            blockers.append("Output contains artifacts; use resume-train or a new run")
        dataset = None
        if inspect_cache:
            try:
                dataset = FeatureCacheBackend(self.cache, self.metadata_root, self.split())
                if self.noise_probability > 0 or self.mixup_alpha > 0:
                    from .augmentation import verify_waveform_sources
                    verify_waveform_sources(dataset.cache, self.cache, dataset.indices["train"],
                                            self.raw_audio_root, load_waveforms=False)
            except (OSError, ValueError, KeyError) as exc:
                blockers.append(f"Dataset not ready: {exc}")
        else:
            blockers.append("Cache not checked; dry plan cannot be authorized for execution")
        return dict(schema_version=1, plan_sha256=self.identity(), ready=not blockers,
                    blockers=blockers, model=model.describe(),
                    dataset=dataset.describe() if dataset is not None else None,
                    research_scope="training/validation only; test not read or evaluated")

    def execute(self) -> dict:
        report = self.check(inspect_cache=True)
        if not report["ready"]:
            raise ContractError("Preflight blocked: " + "; ".join(report["blockers"]))
        from .training import train
        train(self.to_legacy_args())
        return report
