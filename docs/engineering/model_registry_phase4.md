# Phase 4 — Model Registry, artifact lifecycle and research-only release governance

**Date:** 2026-10-28 · **Status:** Implemented and unit-tested. Previously saved ESC-50 CNN/CRNN bundles technically registered. No human research approval or production release was performed.

## Why the old packaging system was insufficient

`inference.package_bundle()` already produces a self-contained folder with a checkpoint, model card, source files and SHA-256 hashes. This was necessary but not sufficient: there was no version-level identity, explicit immutable lifecycle record, controlled approval, promotion channel or rollback history.

## Architecture and invariants

`src/esc50_sound/model_registry.py` and `scripts/model_registry.py` implement a **local research-only registry** with the following layout:

```
registry/
  .registry.lock                # process lock (Linux / POSIX flock)
  objects/<2-digit-prefix>/<sha256>  # content-addressed file bytes
  models/<model-id>/<version>/
    manifest.json               # immutable version identity and object hashes
    events/000001.json ...       # append-only lifecycle events with hash-chain links
  releases/<model-id>/<version>.zip  # no-overwrite portable snapshot
  channels/research/000001.json ...  # append-only deployment selection history
```

A model is registered only from an existing `bundle_schema_version=2` portable bundle. Registration verifies every declared source file's **size and SHA-256**, rejects unsafe file paths and symlinks, loads the checkpoint with `weights_only=True`, validates its embedded provenance and checks its weights against the actual model architecture. The registry snapshots the files, not mutable external source paths. New version manifests and their initial events are staged and published by directory rename.

All lifecycle changes are serialized by an OS file lock. A version is **never overwritten**. Content-addressed objects are published without replacing existing objects. Verification re-hashes objects, compares source manifests, validates event sequence and provenance, and checks release archive digests. The new ZIP is deterministic for a fixed set of source bytes and includes `REGISTRY_RELEASE.json` that **explicitly says research-only**.

Lifecycle: `candidate -> validated -> approved -> released -> archived`. `validated` means structural and cryptographic integrity of the stored artifacts; it does **not** mean scientific validation of the model. `approved` requires a **separate review document** and a **model-matched Phase 3 evaluation audit** with clear selection-history disclosure, together with a reviewer identity and rationale. Review artifacts are stored as SHA-256-addressed bytes. A researcher can approve only the noncommercial *research* scope. There is **no production approval** path. An audit marked valid is not independently reproduced inference.

The `research` channel can only point at released, unarchived versions. Rollbacks must target a previously activated, still-released version. A currently active version cannot be archived until the channel is switched away. The channel's full promotion/rollback history is append-only.

## Use

From project root (with `PYTHONPATH=src` or after editable install):

```sh
# Import existing saved CNN bundle: technically inspect, do not touch held-out predictions.
python scripts/model_registry.py --root local_registry register esc50-cnn-freq historical-1 \
  --bundle models/cnn --actor researcher --reason "Import historical model"
python scripts/model_registry.py --root local_registry validate esc50-cnn-freq historical-1 \
  --actor researcher --reason "Checks passed; no new performance claim"
python scripts/model_registry.py --root local_registry list
python scripts/model_registry.py --root local_registry verify esc50-cnn-freq historical-1

# HUMAN action: review documented selection history and intended use BEFORE approval.
python scripts/model_registry.py --root local_registry approve esc50-cnn-freq historical-1 \
  --reviewer YOUR_REVIEWER_ID \
  --review-document YOUR_SIGNOFF_NOTE.md \
  --evaluation-report results/historical/infra_v3/cnn_freq_cv_seed101_audit.json \
  --reason "Research-only release reviewed; not independently confirmed"
python scripts/model_registry.py --root local_registry release esc50-cnn-freq historical-1 \
  --actor YOUR_REVIEWER_ID --reason "Issue research-only portable version"
python scripts/model_registry.py --root local_registry promote esc50-cnn-freq historical-1 \
  --actor YOUR_REVIEWER_ID --reason "Select as current research reference"
python scripts/model_registry.py --root local_registry resolve
```

To revert the research channel to an earlier released and previously active model version, use `rollback` with the same positional model/version and a recorded `--reason`. There is deliberately no `--production`, `--force`, or overwrite option.

## Verified demonstration

The local `results/historical/infra_v4/registry/` contains two **technically validated** historical versions corresponding to existing CNN and BiGRU deliverables. Both stop at `validated`; neither is `approved`, `released`, or an active channel target. This preserves review responsibility and the disclosure that the ESC-50 historical five-fold results are retrospective, not independent confirmation.

## Security, validity and operational limitations

- SHA-256 and hash-linked files **detect accidental and many straightforward unauthorized modifications**, but they are not authenticated digital signatures. A privileged writer could rewrite the entire event chain and registry. For adversarial provenance, use signed attestations, external checkpoints of audit roots, immutable remote storage and actual access control.
- Actor and reviewer are plain recorded strings, **not verified identities**. Real organizational approval requires an identity provider and role-based authorization.
- The registry uses POSIX `fcntl.flock` for a single trusted local filesystem. It is not a distributed database and is not designed for flaky network filesystems or Kubernetes multi-writer deployments.
- Atomic per-file and per-directory publication are not a transaction across the whole registry, object store and release events. A crash can leave unreferenced objects or a release ZIP not yet referenced by a committed release event; such orphaned files require cautious maintenance, not blind deletion.
- The registry does not reproduce a WAV inference result, train models or decide whether benchmark claims are true. A valid Phase 3 audit links historical records but **does not provide independent field validation, calibration, robustness guarantees, safe autonomous decisions or commercial licensing rights**.
- No untrusted weights should be loaded outside the project's `weights_only=True` loader. Release bundle files are inspectable but still require normal software supply-chain review before execution on another machine.
- Earlier test folds were already accessed; no registry status can undo that. ESC-50 has a noncommercial license, which is disclosed by current model cards.
- End-to-end production requirements still pending: real-world external test data, near-duplicate checks on full audio, independent re-inference, signature/RBAC, runtime monitoring, SLOs, and drift detection.

## Next phase

Add a properly authenticated storage/backend, external validation and a controlled release-review workflow only after obtaining independent evaluation evidence. Pretrained PANNs/AST/CLAP should register through the same model-card/evidence contract, but need explicit upstream data-source and third-party license disclosures.
