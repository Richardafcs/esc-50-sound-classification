"""Explicit training loop, training-only augmentation and implementation smoke check.

This module owns optimization and checkpoint selection. Evaluation has its own
module so training-mode losses cannot be mistaken for evaluation-mode readouts.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .audio import FRONTEND, log_mel
from .data import fit_normalization, load_cache, split_rows
from .evaluation import run_eval, save_predictions
from .models import build_model
from .records import canonical_hash, environment, seed_all, sha256, write_json

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


def train(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise ValueError("Output already has a run; choose a new directory to preserve evidence")
    write_json(out / "run.json", dict(config=vars(args), status="initializing", test_evaluated=False))
    started = time.monotonic()
    try:
        _train(args, out)
    except BaseException as error:
        manifest = json.loads((out / "run.json").read_text())
        manifest.update(status="failed", error_type=type(error).__name__, error=str(error),
                        failure_elapsed_seconds=time.monotonic() - started)
        write_json(out / "run.json", manifest)
        raise


def _train(args, out):
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    cache = load_cache(args.cache)
    train_folds = tuple(int(f) for f in args.train_folds.split(","))
    split_metadata = [{"fold": f, "src_file": s} for f, s in zip(cache["folds"], cache["sources"])]
    indices = split_rows(split_metadata, train_folds, args.val_fold, args.test_fold, strict=not args.official_folds)
    train_ids, val_ids = indices["train"], indices["validation"]
    if not train_ids or not val_ids:
        raise ValueError("Empty training or validation split")
    x, y = cache["x"], cache["y"]
    mean, std = fit_normalization(x[train_ids])
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else
                          "cpu" if args.device == "auto" else args.device)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler_name = getattr(args, "scheduler", "none")
    if scheduler_name not in ("none", "cosine"):
        raise ValueError("scheduler must be none or cosine")
    # One step after each completed epoch, T_max fixed to the declared maximum.
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if scheduler_name == "cosine" else None)
    generator = torch.Generator().manual_seed(args.seed)
    train_data = TensorDataset(torch.from_numpy((x[train_ids] - mean) / std),
                               torch.from_numpy(y[train_ids]).long())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator)
    # Explicit generators prevent diagnostic DataLoader construction/iteration
    # from consuming the model's global random stream (dropout/augmentation).
    train_eval_loader = DataLoader(train_data, batch_size=args.batch_size,
                                  generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(TensorDataset(torch.from_numpy((x[val_ids] - mean) / std),
                                          torch.from_numpy(y[val_ids]).long()), batch_size=args.batch_size,
                            generator=torch.Generator().manual_seed(0))
    identity = dict(schema_version=2, config=vars(args), environment=environment(device), frontend=FRONTEND,
                    seed=args.seed, cache_sha256=sha256(args.cache),
                    class_names=cache["class_names"].tolist(),
                    split={k: dict(n=len(v), names=cache["names"][v].tolist(),
                                   sources=sorted(set(cache["sources"][v].tolist()))) for k, v in indices.items()},
                    normalization_fit="train clips only", parameters=sum(p.numel() for p in model.parameters()),
                    source_policy="official folds without source repair" if args.official_folds else
                    "keep test; remove test sources from validation and held-out sources from train",
                    selection="minimum validation cross-entropy; earliest tie", contribution="agent-produced")
    identity_hash = canonical_hash(identity)
    write_json(out / "manifest.json", identity)
    manifest = dict(identity, manifest_id=identity_hash, manifest_sha256=sha256(out / "manifest.json"),
                    status="running", test_evaluated=False)
    write_json(out / "run.json", manifest)
    start, best, best_epoch, updates, history = time.monotonic(), float("inf"), None, 0, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(args.epochs):
        model.train()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        total_loss, seen = 0.0, 0
        timed_out = False
        for batch_x, batch_y in train_loader:
            if time.monotonic() - start >= args.max_seconds:
                timed_out = True
                break
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if args.specaugment:
                batch_x = spec_augment(batch_x)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total_loss += loss.item() * len(batch_y)
            seen += len(batch_y)
        if not seen:
            break
        training_metrics, _ = run_eval(model, train_eval_loader, device)
        metrics, predictions = run_eval(model, val_loader, device)
        def compact(value):
            return {k: v for k, v in value.items() if k not in ("confusion_matrix", "per_class_f1")}
        improved = metrics["loss"] < best
        if improved:
            best, best_epoch = metrics["loss"], epoch + 1
        if scheduler is not None and not timed_out:
            scheduler.step()
        record = dict(epoch=epoch + 1, training_loss=total_loss / seen,
                      learning_rate=epoch_learning_rate,
                      next_learning_rate=optimizer.param_groups[0]["lr"],
                      scheduler=scheduler_name,
                      training_loss_definition="online training-mode cross-entropy over changing weights",
                      train_evaluation=compact(training_metrics),
                      training_examples=seen, optimizer_updates=updates,
                      validation=compact(metrics), best_epoch=best_epoch,
                      seconds=time.monotonic() - start, partial_epoch=timed_out)
        history.append(record)
        write_json(out / "history.json", history)
        print(json.dumps(record), flush=True)
        payload = dict(schema_version=2, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict() if scheduler is not None else None,
                       scheduler_name=scheduler_name,
                       model_name=args.model, epoch=epoch + 1, seed=args.seed,
                       best_epoch=best_epoch, best_validation_loss=best,
                       mean=torch.from_numpy(mean), std=torch.from_numpy(std),
                       class_names=cache["class_names"].tolist(), frontend=FRONTEND,
                       cache_sha256=manifest["cache_sha256"], train_folds=train_folds,
                       val_fold=args.val_fold, test_fold=args.test_fold,
                       source_policy=manifest["source_policy"], manifest=identity,
                       manifest_id=identity_hash, manifest_sha256=manifest["manifest_sha256"],
                       torch_rng=torch.get_rng_state(), generator_rng=generator.get_state(),
                       resume_policy="optimizer/random state stored; resume CLI and exact mid-epoch replay are not implemented")
        torch.save(payload, out / "last.pt")
        if improved:
            torch.save(payload, out / "best.pt")
            write_json(out / "validation.json", metrics)
            write_json(out / "training_evaluation.json", training_metrics)
            save_predictions(out / "validation_predictions.csv", cache["names"][val_ids], y[val_ids], predictions)
        if timed_out:
            break
    manifest.update(status="completed" if len(history) == args.epochs and not history[-1]["partial_epoch"] else "time_limit",
                    seconds=time.monotonic() - start, optimizer_updates=updates,
                    epochs_recorded=len(history), best_epoch=best_epoch,
                    best_validation_loss=best if history else None,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    time_limit_scope="checked before each optimizer batch; epoch-end train/validation/checkpoint overhead additional")
    write_json(out / "run.json", manifest)
    if not history:
        raise RuntimeError("Time budget expired before any training batch; no model selected")


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


