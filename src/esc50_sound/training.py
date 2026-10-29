"""Explicit training loop, training-only augmentation and implementation smoke check.

This module owns optimization and checkpoint selection. Evaluation has its own
module so training-mode losses cannot be mistaken for evaluation-mode readouts.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from argparse import Namespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .audio import FRONTEND, log_mel
from .data import fit_normalization, load_cache, split_rows
from .checkpoints import save_checkpoint_atomic, load_checkpoint, validate_checkpoint
from .evaluation import run_eval, save_predictions
from .models import build_model
from .records import canonical_hash, environment, seed_all, sha256, write_json
from .runtime_state import capture_rng, restore_rng, clone_weights, run_writer_lock
from .augmentation import (augmentation_config, enabled, load_training_waveforms,
                           add_gaussian_noise, mix_waveforms, log_mel_batch)

def spec_augment(x):
    x = x.clone()
    for axis, limit in ((2, 8), (3, 24)):
        for sample in x:
            width = int(torch.randint(limit + 1, ()).item())
            start = int(torch.randint(x.shape[axis] - width + 1, ()).item())
            slices = [slice(None)] * 3
            slices[axis - 1] = slice(start, start + width)
            sample[tuple(slices)] = 0
    return x


def _stable_config(args):
    # Control-plane interruptions are not part of the science protocol identity.
    return {k: v for k, v in vars(args).items()
            if k not in ("pause_after_epochs",)}


def train(args):
    out = Path(args.output)
    with run_writer_lock(out):
        if out.exists() and any(out.iterdir()):
            raise ValueError("Output already contains evidence; use resume-train or a new directory")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "run.json", dict(config=_stable_config(args), status="initializing", test_evaluated=False))
        _run_guarded(args, out, resuming=False)


def resume_train(args):
    """Resume only a new-format committed epoch under an exclusive writer lock."""
    out = Path(args.output)
    with run_writer_lock(out):
        _resume_train_locked(args, out)


def _resume_train_locked(args, out):
    manifest_path, run_path = out / "manifest.json", out / "run.json"
    if not manifest_path.is_file() or not run_path.is_file():
        raise ValueError("Existing manifest and run record are required for resume")
    identity = json.loads(manifest_path.read_text())
    record = json.loads(run_path.read_text())
    if record.get("status") not in ("failed", "time_limit", "running"):
        raise ValueError("Only interrupted/incomplete runs may be resumed")
    if record.get("test_evaluated") or (out / "evaluation_history.jsonl").exists():
        raise ValueError("Cannot resume after any held-out test access")
    if (canonical_hash(identity) != record.get("manifest_id") or
            sha256(manifest_path) != record.get("manifest_sha256")):
        raise ValueError("Manifest integrity is inconsistent with the run record")
    original = Namespace(**identity["config"])
    original.output = str(out)
    original.max_seconds = args.max_seconds
    original.pause_after_epochs = args.pause_after_epochs
    _run_guarded(original, out, resuming=True)


def _run_guarded(args, out, *, resuming):
    start = time.monotonic()
    try:
        # Keep the historical two-argument _train call compatible with
        # existing exception-injection and integration tests.
        if resuming:
            _train(args, out, resuming=True)
        else:
            _train(args, out)
    except BaseException as error:
        manifest = json.loads((out / "run.json").read_text())
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error),
                        failure_elapsed_seconds=time.monotonic() - start)
        write_json(out / "run.json", manifest)
        raise


def _train(args, out, *, resuming=False):
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    train_folds = tuple(int(f) for f in args.train_folds.split(","))
    if getattr(args, "platform_plan_sha256", None):
        # New, opt-in research platform contract; legacy runs retain the exact
        # original loader path and checkpoint architecture/state names.
        from .platform import FeatureCacheBackend, SplitPolicy
        if getattr(args, "platform_dataset_backend", None) != "feature_cache":
            raise ValueError("Legacy trainer supports only validated feature_cache")
        dataset = FeatureCacheBackend(
            args.cache, args.platform_metadata_root,
            SplitPolicy(train_folds, args.val_fold, args.test_fold,
                        source_strict=not args.official_folds))
        cache = dataset.cache
    else:
        cache = load_cache(args.cache)
    split_metadata = [{"fold": f, "src_file": s} for f, s in zip(cache["folds"], cache["sources"])]
    indices = split_rows(split_metadata, train_folds, args.val_fold, args.test_fold, strict=not args.official_folds)
    train_ids, val_ids = indices["train"], indices["validation"]
    if not train_ids or not val_ids:
        raise ValueError("Empty training or validation split")
    x, y = cache["x"], cache["y"]
    mean, std = fit_normalization(x[train_ids])
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else
                          "cpu" if args.device == "auto" else args.device)
    if getattr(args, "platform_plan_sha256", None):
        from .platform import builtin_models
        model = builtin_models().resolve(args.model, "supervised_train").build().to(device)
    else:
        model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_name = getattr(args, "scheduler", "none")
    if scheduler_name not in ("none", "cosine"):
        raise ValueError("scheduler must be none or cosine")
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if scheduler_name == "cosine" else None)
    generator = torch.Generator().manual_seed(args.seed)
    augmented = augmentation_config(args)
    train_data = TensorDataset(torch.from_numpy((x[train_ids] - mean) / std),
                               torch.from_numpy(y[train_ids]).long())
    if enabled(augmented):
        # Explicit opt-in. Only the training split is read from raw WAV;
        # validation/test remain pinned clean cache features.
        loader_data = load_training_waveforms(cache, args.cache, train_ids, augmented["raw_audio_root"])
    else:
        loader_data = train_data
    train_loader = DataLoader(loader_data, batch_size=args.batch_size, shuffle=True, generator=generator)
    normalization_mean = torch.from_numpy(mean).to(device)
    normalization_std = torch.from_numpy(std).to(device)
    train_eval_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(TensorDataset(torch.from_numpy((x[val_ids] - mean) / std),
                                          torch.from_numpy(y[val_ids]).long()), batch_size=args.batch_size,
                            generator=torch.Generator().manual_seed(0))
    actual_environment = environment(device)
    if resuming:
        identity = json.loads((out / "manifest.json").read_text())
        run_manifest = json.loads((out / "run.json").read_text())
        last = load_checkpoint(out / "last.pt", device="cpu")
        validate_checkpoint(last)
        snapshot = last.get("resume_state")
        if not isinstance(snapshot, dict) or snapshot.get("version") != 1 or not snapshot.get("epoch_complete"):
            raise ValueError("Cannot resume legacy or partial-epoch checkpoint: missing resume v1 snapshot")
        if (last.get("manifest_id") != canonical_hash(identity) or
                last.get("manifest_sha256") != run_manifest.get("manifest_sha256") or
                last.get("cache_sha256") != sha256(args.cache)):
            raise ValueError("Resume checkpoint identity or cache digest mismatch")
        old_environment = identity.get("environment", {})
        for key in ("code_sha256", "torch", "numpy", "python", "platform", "device"):
            if old_environment.get(key) != actual_environment.get(key):
                raise ValueError(f"Resume requires identical environment: {key} differs")
        if (not torch.equal(last["mean"], torch.from_numpy(mean)) or
                not torch.equal(last["std"], torch.from_numpy(std))):
            raise ValueError("Train-fitted normalization differs from checkpoint")
        history = snapshot["history"]
        if (last.get("scheduler_name") != scheduler_name or
                last.get("best_epoch") is None or
                last["best_epoch"] > last["epoch"] or
                not isinstance(history, list) or
                len(history) != last["epoch"] or
                [r["epoch"] for r in history] != list(range(1, last["epoch"] + 1)) or
                any(r.get("partial_epoch", False) for r in history)):
            raise ValueError("Checkpoint history is not consecutive completed epochs")
        start_epoch = last["epoch"]
        if start_epoch >= args.epochs:
            raise ValueError("Checkpoint has no remaining training epochs")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        if scheduler is not None:
            if last["scheduler"] is None:
                raise ValueError("Checkpoint scheduler state missing")
            scheduler.load_state_dict(last["scheduler"])
        elif last["scheduler"] is not None:
            raise ValueError("Unexpected scheduler state in checkpoint")
        best, best_epoch, updates = last["best_validation_loss"], last["best_epoch"], snapshot["updates"]
        best_weights, best_val, best_train = snapshot["best_model"], snapshot["best_validation"], snapshot["best_train"]
        best_preds = snapshot["best_predictions"]
        # Restore random streams AFTER model/optimizer/loader construction.
        restore_rng(snapshot["rng"], generator)
        # last.pt is the authoritative epoch commit. Repair sidecar files after
        # crashes between checkpoint, history, best and run JSON writes.
        _repair_records(out, last, history, best_weights, best_val, best_train, best_preds,
                        cache, val_ids, last["manifest_id"])
        run_manifest.update(status="running", epochs_recorded=start_epoch,
                            recovered_from_epoch=start_epoch, optimizer_updates=updates)
        run_manifest.pop("error_type", None)
        run_manifest.pop("error", None)
        write_json(out / "run.json", run_manifest)
        manifest = run_manifest
        identity_hash = last["manifest_id"]
    else:
        identity = dict(schema_version=2, config=_stable_config(args), environment=actual_environment,
                        frontend=FRONTEND, seed=args.seed, cache_sha256=sha256(args.cache),
                        class_names=cache["class_names"].tolist(),
                        split={k: dict(n=len(v), names=cache["names"][v].tolist(),
                                       sources=sorted(set(cache["sources"][v].tolist())))
                               for k, v in indices.items()},
                        normalization_fit="train clips only",
                        parameters=sum(p.numel() for p in model.parameters()),
                        source_policy="official folds without source repair" if args.official_folds else
                        "keep test; remove test sources from validation and held-out sources from train",
                        selection="minimum validation cross-entropy; earliest tie", contribution="agent-produced")
        identity_hash = canonical_hash(identity)
        write_json(out / "manifest.json", identity)
        manifest = dict(identity, manifest_id=identity_hash, manifest_sha256=sha256(out / "manifest.json"),
                        status="running", test_evaluated=False)
        write_json(out / "run.json", manifest)
        start_epoch, best, best_epoch, updates, history = 0, float("inf"), None, 0, []
        best_weights = best_val = best_train = best_preds = None
    attempts = list(manifest.get("training_attempts", []))
    attempts.append(dict(kind="resume" if resuming else "initial",
                         start_epoch=start_epoch, max_seconds=args.max_seconds,
                         progress="running"))
    manifest["training_attempts"] = attempts
    write_json(out / "run.json", manifest)
    start = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timed_out = False
    paused = False
    for epoch in range(start_epoch, args.epochs):
        if time.monotonic() - start >= args.max_seconds:
            timed_out = True
            break
        model.train()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        total_loss, seen, batches = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            if time.monotonic() - start >= args.max_seconds:
                timed_out = True
                break
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            partner_y, mixing = batch_y, None
            if enabled(augmented):
                batch_x = add_gaussian_noise(batch_x, augmented["noise_probability"],
                                             augmented["noise_snr_min_db"], augmented["noise_snr_max_db"])
                batch_x, partner_y, mixing = mix_waveforms(batch_x, batch_y, augmented["mixup_alpha"])
                batch_x = (log_mel_batch(batch_x) - normalization_mean) / normalization_std
            if args.specaugment:
                batch_x = spec_augment(batch_x)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            if mixing is None:
                loss = F.cross_entropy(logits, batch_y)
            else:
                loss = (mixing * F.cross_entropy(logits, batch_y, reduction="none") +
                        (1 - mixing) * F.cross_entropy(logits, partner_y, reduction="none")).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total_loss += loss.item() * len(batch_y)
            seen += len(batch_y)
            batches += 1
        if timed_out or batches != len(train_loader):
            # The partial epoch is discarded, NEVER checkpointed/validated.
            # Resume replays it from the preceding committed epoch.
            timed_out = True
            break
        training_metrics, _ = run_eval(model, train_eval_loader, device)
        metrics, predictions = run_eval(model, val_loader, device)
        def compact(value):
            return {k: v for k, v in value.items() if k not in ("confusion_matrix", "per_class_f1")}
        improved = metrics["loss"] < best
        if improved:
            best, best_epoch = metrics["loss"], epoch + 1
            best_weights = clone_weights(model)
            best_val, best_train, best_preds = metrics, training_metrics, list(predictions)
        if scheduler is not None:
            scheduler.step()
        record = dict(epoch=epoch + 1, training_loss=total_loss / seen,
                      learning_rate=epoch_learning_rate,
                      next_learning_rate=optimizer.param_groups[0]["lr"],
                      scheduler=scheduler_name,
                      training_loss_definition=("online waveform-mixup weighted cross-entropy over changing weights"
                                                if augmented["mixup_alpha"] > 0 else
                                                "online training-mode cross-entropy over changing weights"),
                      train_evaluation=compact(training_metrics),
                      training_examples=seen, optimizer_updates=updates,
                      validation=compact(metrics), best_epoch=best_epoch,
                      seconds=time.monotonic() - start, partial_epoch=False)
        history.append(record)
        payload = dict(schema_version=2, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict() if scheduler is not None else None,
                       scheduler_name=scheduler_name, model_name=args.model,
                       epoch=epoch + 1, seed=args.seed,
                       best_epoch=best_epoch, best_validation_loss=best,
                       mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                       class_names=cache["class_names"].tolist(), frontend=FRONTEND,
                       cache_sha256=manifest["cache_sha256"], train_folds=train_folds,
                       val_fold=args.val_fold, test_fold=args.test_fold,
                       source_policy=manifest["source_policy"], manifest=identity,
                       manifest_id=identity_hash, manifest_sha256=manifest["manifest_sha256"],
                       torch_rng=torch.get_rng_state(), generator_rng=generator.get_state(),
                       resume_policy="epoch boundary v1; partial epochs discarded",
                       resume_state=dict(version=1, epoch_complete=True, updates=updates,
                                         history=history.copy(), rng=capture_rng(generator),
                                         best_model=best_weights,
                                         best_validation=best_val, best_train=best_train,
                                         best_predictions=best_preds))
        # One durable commit per COMPLETE epoch; companion files are repaired
        # from last.pt on resume when an interruption falls between writes.
        save_checkpoint_atomic(out / "last.pt", payload)
        _repair_records(out, payload, history, best_weights, best_val, best_train, best_preds,
                        cache, val_ids, identity_hash)
        manifest.update(status="running", optimizer_updates=updates,
                        epochs_recorded=len(history), best_epoch=best_epoch,
                        best_validation_loss=best)
        write_json(out / "run.json", manifest)
        print(json.dumps(record), flush=True)
        if getattr(args, "pause_after_epochs", None) is not None and len(history) >= args.pause_after_epochs:
            if len(history) < args.epochs:
                paused = True
            break
    attempts[-1].update(progress="completed" if len(history) == args.epochs else "stopped",
                        committed_epoch=len(history), elapsed_seconds=time.monotonic() - start)
    # A partially processed epoch does not count toward committed optimization.
    committed_updates = history[-1]["optimizer_updates"] if history else 0
    manifest.update(status="completed" if len(history) == args.epochs else "time_limit",
                    seconds=time.monotonic() - start, optimizer_updates=committed_updates,
                    epochs_recorded=len(history), best_epoch=best_epoch,
                    best_validation_loss=best if history else None,
                    pause_reason="operator_epoch_boundary" if paused else
                    "wall_clock" if timed_out else None,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    time_limit_scope="partial epochs discarded; max_seconds checked before each optimizer batch")
    write_json(out / "run.json", manifest)
    if not history:
        raise RuntimeError("No complete epoch; no resumable checkpoint. Run with a larger time budget")


def _repair_records(out, last, history, best_weights, best_val, best_train,
                    best_preds, cache, val_ids, manifest_id):
    """Rebuild non-authoritative run views from the committed epoch snapshot."""
    if not best_weights or best_val is None or best_train is None or best_preds is None:
        raise ValueError("Committed checkpoint has no complete best-model evidence")
    # Use the full committed checkpoint for inference packaging, replacing the
    # current model state with the selected best epoch's state. Public checkpoint
    # stays schema v2; no resume snapshot is included in best.pt.
    best_payload = {key: value for key, value in last.items() if key not in
                    ("resume_state", "torch_rng", "generator_rng", "optimizer", "scheduler")}
    # Optimizer/scheduler belong to the most recent completed epoch, which
    # may differ from the selected best epoch. Never advertise mismatched
    # training states in a prediction-only checkpoint.
    best_payload.update(model=best_weights, epoch=last["best_epoch"],
                        checkpoint_role="validation-selected inference only",
                        resume_policy="resume ONLY from last.pt")
    save_checkpoint_atomic(out / "best.pt", best_payload)
    write_json(out / "validation.json", best_val)
    write_json(out / "training_evaluation.json", best_train)
    save_predictions(out / "validation_predictions.csv", cache["names"][val_ids],
                     cache["y"][val_ids], best_preds)
    write_json(out / "history.json", history)


def smoke(args):
    """Synthetic correctness only: NOT an ESC-50 accuracy measurement."""
    seed_all(args.seed)
    torch.set_num_threads(2)
    report = dict(evidence="synthetic implementation smoke; no generalization claim", environment=environment(), models={})
    silence = log_mel(torch.zeros(80000))
    report["frontend"] = dict(shape=list(silence.shape), finite=bool(torch.isfinite(silence).all()))
    x = torch.randn(8, 1, 64, 64)
    # Easily separable frequency bands, fixed examples, four labels.
    y = torch.arange(8) % 4
    for i in range(8):
        x[i, :, int(y[i]) * 16:(int(y[i]) + 1) * 16] += 4
    for name in ("cnn", "crnn"):
        model = build_model(name, num_classes=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        first = None
        start = time.monotonic()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if first is None:
                first = loss.item()
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite gradients")
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(x)
            final = F.cross_entropy(logits, y).item()
            accuracy = (logits.argmax(1) == y).float().mean().item()
        if not final < first:
            raise AssertionError(f"{name} did not reduce synthetic training loss")
        report["models"][name] = dict(initial_loss=first, final_loss=final,
                                       training_accuracy=accuracy, steps=args.steps,
                                       output_shape=list(logits.shape), seconds=time.monotonic() - start,
                                       finite_gradients=True,
                                       parameters=sum(p.numel() for p in model.parameters()))
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


