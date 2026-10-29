"""Pinned ESC-50 acquisition, metadata/source splits and validated feature caches.

No fitted normalization statistics are computed until training selects its rows.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

import numpy as np
import torch

from .audio import FRONTEND, log_mel, read_audio
from .records import canonical_hash, environment, sha256, write_json

DATA_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"


def metadata_records(rows):
    return [dict(filename=str(r["filename"]), fold=int(r["fold"]), target=int(r["target"]),
                 category=str(r["category"]), src_file=str(r["src_file"])) for r in rows]


def read_metadata(root):
    with open(Path(root) / "meta/esc50.csv", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_metadata(rows, require_complete=True):
    """Reject malformed labels/files; report official cross-fold source overlaps."""
    names, sources, labels, counts = set(), {}, {}, {}
    for row in rows:
        name, fold, target = row["filename"], int(row["fold"]), int(row["target"])
        if Path(name).name != name or not name.endswith(".wav"):
            raise ValueError(f"Unsafe audio filename: {name}")
        if name in names or fold not in range(1, 6) or target not in range(50):
            raise ValueError(f"Duplicate filename or invalid fold/target: {name}")
        names.add(name)
        source = row.get("src_file") or name.split("-")[1]
        sources.setdefault(source, set()).add(fold)
        category = row["category"]
        if target in labels and labels[target] != category:
            raise ValueError(f"Inconsistent target mapping: {target}")
        labels[target] = category
        counts[fold, target] = counts.get((fold, target), 0) + 1
    if len(set(labels.values())) != len(labels):
        raise ValueError("Category maps to multiple target IDs")
    if require_complete and (len(rows) != 2000 or len(labels) != 50 or
                             any(counts.get((f, c), 0) != 8 for f in range(1, 6)
                                 for c in range(50))):
        raise ValueError("Expected 2000 clips, 50 classes, eight per class per fold")
    return dict(clips=len(rows), classes=len(labels), sources=len(sources),
                folds={str(f): sum(int(r["fold"]) == f for r in rows) for f in range(1, 6)},
                source_fold_leakage=any(len(v) > 1 for v in sources.values()),
                source_fold_overlaps={k: sorted(v) for k, v in sources.items() if len(v) > 1},
                class_names=[labels[c] for c in sorted(labels)])


def split_rows(rows, train_folds=(1, 2, 3), val_fold=4, test_fold=5, strict=True):
    train_folds = tuple(train_folds)
    if (not train_folds or len(set(train_folds)) != len(train_folds) or
            val_fold == test_fold or val_fold in train_folds or test_fold in train_folds or
            any(f not in range(1, 6) for f in (*train_folds, val_fold, test_fold))):
        raise ValueError("Train, validation, test folds must be nonempty and disjoint in 1..5")
    indices = {"train": [i for i, r in enumerate(rows) if int(r["fold"]) in train_folds],
            "validation": [i for i, r in enumerate(rows) if int(r["fold"]) == val_fold],
            "test": [i for i, r in enumerate(rows) if int(r["fold"]) == test_fold]}
    if strict and rows and "src_file" in rows[0]:
        test_sources = {str(rows[i]["src_file"]) for i in indices["test"]}
        validation_sources = {str(rows[i]["src_file"]) for i in indices["validation"]}
        indices["validation"] = [i for i in indices["validation"] if str(rows[i]["src_file"]) not in test_sources]
        indices["train"] = [i for i in indices["train"] if str(rows[i]["src_file"]) not in test_sources | validation_sources]
    return indices


def prepare(args):
    torch.set_num_threads(2)
    root, cache = Path(args.root), Path(args.cache)
    metadata_exists = (root / "meta/esc50.csv").exists()
    dataset_complete = metadata_exists and all((root / "audio" / row["filename"]).is_file()
                                              for row in read_metadata(root))
    if not dataset_complete:
        root.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root.parent).free < 2 * 1024**3:
            raise RuntimeError("Need at least 2 GiB free for download, extraction, feature cache")
        archive = root.parent / f"esc50-{DATA_COMMIT}.zip"
        url = f"https://github.com/karoldvl/ESC-50/archive/{DATA_COMMIT}.zip"
        print(f"Downloading official pinned dataset: {url}", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "ESC50-research-baseline"})
        size = 0
        with urllib.request.urlopen(request, timeout=60) as response, open(archive, "wb") as out:
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > 800 * 1024**2:
                    raise RuntimeError("Download exceeded 800 MiB safety limit")
                out.write(block)
        archive_hash = sha256(archive)
        with zipfile.ZipFile(archive) as zipped:
            members = [m for m in zipped.infolist() if "/audio/" in m.filename or
                       m.filename.endswith(("/meta/esc50.csv", "/LICENSE", "/README.md"))]
            if sum(m.file_size for m in members) > 1200 * 1024**2:
                raise RuntimeError("Extraction exceeds 1200 MiB safety limit")
            for member in members:
                relative = Path(*Path(member.filename).parts[1:])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe archive path")
                target = root / relative
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        write_json(root / "provenance.json", dict(commit=DATA_COMMIT, archive_url=url,
                                                  archive_sha256=archive_hash))
        archive.unlink()  # Only the temporary archive created by this command.
    rows = read_metadata(root)
    audit = audit_metadata(rows)
    # Check every audio header/hash; deterministic cache includes test features but no fitted statistics.
    import soundfile as sf
    audio_hashes, features = {}, []
    for i, row in enumerate(rows):
        audio = root / "audio" / row["filename"]
        info = sf.info(audio)
        if info.samplerate != 44100 or info.channels != 1 or abs(info.duration - 5) > 0.001:
            raise ValueError(f"Unexpected official audio format: {audio}")
        digest = sha256(audio)
        if digest in audio_hashes and audio_hashes[digest][1] != int(row["fold"]):
            raise ValueError(f"Exact audio duplicate crosses folds: {row['filename']}")
        audio_hashes[digest] = (row["filename"], int(row["fold"]))
        features.append(log_mel(read_audio(audio)).numpy())
        if (i + 1) % 100 == 0:
            print(f"Cached {i + 1}/{len(rows)} clips", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=np.stack(features).astype(np.float32),
                        y=np.array([int(r["target"]) for r in rows]),
                        folds=np.array([int(r["fold"]) for r in rows]),
                        sources=np.array([r["src_file"] for r in rows]),
                        names=np.array([r["filename"] for r in rows]),
                        class_names=np.array(audit["class_names"]))
    provenance = json.loads((root / "provenance.json").read_text()) if (root / "provenance.json").exists() else {"commit": "unverified local copy"}
    write_json(str(cache) + ".json", dict(schema_version=2, frontend=FRONTEND,
                                         metadata_records_sha256=canonical_hash(metadata_records(rows)),
                                         metadata_sha256=sha256(root / "meta/esc50.csv"),
                                         cache_sha256=sha256(cache), audit=audit,
                                         provenance=provenance, environment=environment()))
    print(json.dumps(dict(cache=str(cache), shape=list(np.stack(features).shape), audit=audit), indent=2))


def load_cache(path):
    """Validate bytes, frontend, schema and row alignment before exposing arrays.

    Original v1 sidecars are accepted only with matching adjacent official raw
    metadata; v2 caches include a portable digest of canonical metadata records.
    """
    path = Path(path)
    sidecar_path = Path(str(path) + ".json")
    if not sidecar_path.is_file():
        raise ValueError("Cache provenance sidecar is missing; rerun prepare")
    sidecar = json.loads(sidecar_path.read_text())
    if sidecar.get("frontend") != FRONTEND:
        raise ValueError("Cache frontend is incompatible with this implementation")
    if sidecar.get("cache_sha256") != sha256(path):
        raise ValueError("Cache SHA256 does not match its sidecar")
    if sidecar.get("schema_version", 1) not in (1, 2):
        raise ValueError("Unsupported cache schema version")
    with np.load(path, allow_pickle=False) as archive:
        required = {"x", "y", "folds", "sources", "names", "class_names"}
        if not required.issubset(archive.files):
            raise ValueError("Cache is missing required arrays")
        cache = {k: archive[k].copy() for k in required}
    n = len(cache["y"])
    if (cache["x"].shape != (n, 1, 64, 313) or cache["x"].dtype != np.float32 or
            not np.isfinite(cache["x"]).all() or n == 0):
        raise ValueError("Invalid feature tensor: expected finite float32 [N,1,64,313]")
    for key in ("y", "folds", "sources", "names"):
        if cache[key].shape != (n,):
            raise ValueError(f"Cache row alignment is invalid: {key}")
    if (not np.issubdtype(cache["y"].dtype, np.integer) or
            not np.issubdtype(cache["folds"].dtype, np.integer) or
            cache["class_names"].shape != (50,) or
            len(set(cache["class_names"].tolist())) != 50 or
            cache["sources"].dtype.kind not in "US" or cache["names"].dtype.kind not in "US" or
            any(not str(x) for x in cache["sources"])):
        raise ValueError("Invalid label, fold, source or category schema")
    if np.any(cache["y"] < 0) or np.any(cache["y"] >= 50):
        raise ValueError("Class ID outside 0..49")
    rows = [dict(filename=str(name), fold=int(fold), target=int(target),
                 category=str(cache["class_names"][target]), src_file=str(source))
            for name, fold, target, source in zip(cache["names"], cache["folds"], cache["y"], cache["sources"])]
    audit = audit_metadata(rows, require_complete=False)
    expected_audit = sidecar.get("audit", {})
    for key in ("clips", "classes", "folds", "class_names"):
        if expected_audit.get(key) != audit[key]:
            raise ValueError(f"Cache metadata audit differs from sidecar: {key}")
    record_hash = canonical_hash(metadata_records(rows))
    if "metadata_records_sha256" in sidecar:
        if sidecar["metadata_records_sha256"] != record_hash:
            raise ValueError("Cache metadata records do not match their digest")
    else:
        raw = path.parent / "ESC-50" / "meta" / "esc50.csv"
        if not raw.is_file() or sha256(raw) != sidecar.get("metadata_sha256"):
            raise ValueError("Legacy cache needs matching adjacent ESC-50 metadata; rerun prepare for portable v2")
        if canonical_hash(metadata_records(read_metadata(path.parent / "ESC-50"))) != record_hash:
            raise ValueError("Cache rows disagree with raw metadata")
    return cache


def fit_normalization(x):
    """Per-mel train-only mean/std over clips and time, float64 accumulation."""
    mean = x.mean(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    std = x.std(axis=(0, 3), keepdims=True, dtype=np.float64).astype(np.float32).clip(1e-5)
    return mean, std


