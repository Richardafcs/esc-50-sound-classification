"""Bounded diagnostics and stress measurements; holdout requires an explicit freeze.

These tools do not select models from test feedback. Development is the default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import esc50 as esc


def new_output(path):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=False)
    return out


def split_ids(cache, checkpoint=None):
    rows = [dict(fold=int(f), src_file=str(s)) for f, s in zip(cache["folds"], cache["sources"])]
    checkpoint = checkpoint or {}
    official = checkpoint.get("manifest", {}).get("config", {}).get("official_folds", False)
    return esc.split_rows(rows, checkpoint.get("train_folds", (1, 2, 3)),
                          checkpoint.get("val_fold", 4), checkpoint.get("test_fold", 5), strict=not official)


def requested_measurement_protocol(args):
    """Canonical request compared exactly with the predeclared confirmation file."""
    command = getattr(args, "command", None)
    if command == "order":
        seeds = list(args.permutation_seeds)
        if (not seeds or len(set(seeds)) != len(seeds) or
                any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63 for seed in seeds)):
            raise ValueError("Order measurements need distinct integer permutation seeds in 0..2^63-1")
        return dict(split=args.split, permutation_seeds=seeds)
    if command == "stress":
        conditions = list(args.conditions)
        validate_stress_conditions(conditions)
        if not isinstance(args.seed, int) or isinstance(args.seed, bool) or not 0 <= args.seed < 2**63:
            raise ValueError("Stress noise seed must be an integer in 0..2^63-1")
        return dict(split=args.split, conditions=conditions, seed=args.seed)
    raise ValueError("Confirmation measurements must declare command order or stress")


def validate_stress_conditions(conditions):
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Stress conditions must be a nonempty distinct list")
    for condition in conditions:
        if condition == "clean":
            continue
        if not isinstance(condition, str) or not condition.startswith(("snr_", "gain_")):
            raise ValueError(f"Unknown stress condition: {condition}")
        try:
            value = float(condition.split("_", 1)[1])
        except ValueError as error:
            raise ValueError(f"Invalid stress condition: {condition}") from error
        if not np.isfinite(value) or not -120 <= value <= 120:
            raise ValueError("Stress dB values must be finite and within [-120,120] for numeric safety")


def check_holdout(args, checkpoint_path):
    if args.split != "test":
        return None
    if not args.allow_test or not args.freeze:
        raise ValueError("Holdout measurements require --allow-test and a frozen study manifest")
    freeze_path = Path(args.freeze)
    raw = freeze_path.read_bytes()
    freeze = json.loads(raw)
    digest = esc.sha256(checkpoint_path)
    registered_checkpoints = freeze.get("checkpoint_sha256", {})
    if not isinstance(registered_checkpoints, dict) or digest not in registered_checkpoints.values():
        raise ValueError("Checkpoint is not registered in the frozen study")
    if freeze.get("confirmation_schema_version") != 1:
        raise ValueError("Frozen study needs confirmation_schema_version=1 and explicit measurement_protocol")
    requested = requested_measurement_protocol(args)
    protocols = freeze.get("measurement_protocol")
    if not isinstance(protocols, dict) or protocols.get(args.command) != requested:
        raise ValueError("Requested measurement settings are not registered exactly in the frozen study")
    return dict(path=str(freeze_path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                confirmation_schema_version=1, checkpoint_sha256=digest,
                measurement=args.command, protocol=requested, document=freeze)


def begin_measurement_audit(args, checkpoint, output, freeze):
    """Record holdout access before any forward pass, including failed measurements."""
    if args.split != "test":
        return None
    checkpoint_path = Path(args.checkpoint)
    audit_path = checkpoint_path.parent / "evaluation_history.jsonl"
    record = dict(evaluation_id=esc.canonical_hash(dict(output=str(output.resolve()),time_ns=time.time_ns())),
                  timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  checkpoint_sha256=checkpoint.get("_loaded_checkpoint_sha256") or esc.sha256(checkpoint_path),
                  manifest_id=checkpoint.get("manifest_id"),
                  source_policy=checkpoint.get("source_policy","legacy: unverified"),
                  measurement=args.command, output=str(output.resolve()),freeze=freeze,status="started")
    esc.append_evaluation_audit(audit_path,record)
    run_path=checkpoint_path.parent/"run.json"
    if run_path.exists():
        run=json.loads(run_path.read_text())
        if run.get("manifest_id")==checkpoint.get("manifest_id"):
            run.update(test_evaluated=True,test_evaluation_audit=str(audit_path))
            esc.write_json(run_path,run)
    return audit_path,record


def complete_measurement_audit(audit):
    if audit is not None:
        path,record=audit
        esc.append_evaluation_audit(path,dict(record,status="completed"))


def audited_measurement(args, measurement):
    freeze = check_holdout(args, args.checkpoint)
    # This preflight restores weights only, before accessing held-out cache/audio.
    checkpoint, model = restore(args.checkpoint, args.allow_legacy_checkpoint)
    loaded_digest = checkpoint.get("_loaded_checkpoint_sha256") or esc.sha256(args.checkpoint)
    if freeze is not None and loaded_digest != freeze["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed after confirmation registration was checked")
    out = new_output(args.output)
    audit = begin_measurement_audit(args,checkpoint,out,freeze)
    try:
        # Conservatively record access before reading a cache containing test rows.
        cache = esc.load_cache(args.cache)
        if esc.sha256(args.cache) != checkpoint["cache_sha256"]:
            raise ValueError("Checkpoint/cache identity mismatch")
        if freeze is not None:
            esc.write_json(out / "freeze_snapshot.json", freeze)
        result = measurement(args,out,checkpoint,model,cache,freeze)
    except BaseException as error:
        if audit is not None:
            path,record=audit
            esc.append_evaluation_audit(path,dict(record,status="failed",error_type=type(error).__name__,error=str(error)))
        raise
    else:
        complete_measurement_audit(audit)
        return result


def restore(path, allow_legacy=False):
    # Bind the instantiated weights to the exact immutable bytes whose hash is
    # checked, even if another process replaces the checkpoint path later.
    raw = Path(path).read_bytes()
    checkpoint = esc.load_checkpoint(io.BytesIO(raw))
    esc.validate_checkpoint(checkpoint, allow_legacy=allow_legacy)
    checkpoint["_loaded_checkpoint_sha256"] = hashlib.sha256(raw).hexdigest()
    model = esc.build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model"])
    return checkpoint, model.eval()


def normalized_loader(cache, ids, checkpoint, batch_size=32):
    x = (cache["x"][ids] - checkpoint["mean"].numpy()) / checkpoint["std"].numpy()
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(cache["y"][ids]).long()),
                      batch_size=batch_size, generator=torch.Generator().manual_seed(0))


def tiny(args):
    out = new_output(args.output)
    esc.seed_all(args.seed)
    cache = esc.load_cache(args.cache)
    train = split_ids(cache)["train"]
    ids = [i for label in (0, 10, 20, 30) for i in
           [i for i in train if cache["y"][i] == label][:4]]
    if len(ids) != 16:
        raise ValueError("Tiny real-audio check needs four training clips per designated class")
    mean, std = esc.fit_normalization(cache["x"][ids])
    x = torch.from_numpy((cache["x"][ids] - mean) / std)
    y = torch.from_numpy(cache["y"][ids]).long()
    report = dict(scope="training-only real-audio memorization, not generalization", seed=args.seed,
                  cache_sha256=esc.sha256(args.cache), names=cache["names"][ids].tolist(),
                  normalization="16 selected training examples only", models={})
    for name in args.models:
        esc.seed_all(args.seed)
        model = esc.build_model(name)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        started, history = time.monotonic(), []
        for step in range(args.steps):
            if time.monotonic() - started > args.max_seconds:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite real-audio loss")
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite real-audio gradient")
            optimizer.step()
            if (step + 1) % 20 == 0 or step == args.steps - 1:
                model.eval()
                with torch.inference_mode():
                    logits = model(x)
                    accuracy = (logits.argmax(1) == y).float().mean().item()
                    item = dict(step=step + 1, loss=F.cross_entropy(logits, y).item(), accuracy=accuracy)
                    history.append(item)
                if accuracy == 1 and item["loss"] < .05:
                    break
        passed = bool(history and history[-1]["accuracy"] == 1 and history[-1]["loss"] < .05)
        report["models"][name] = dict(passed=passed, history=history, seconds=time.monotonic()-started)
        esc.write_json(out / "real_audio_overfit.json", report)
        print(json.dumps(dict(model=name, **report["models"][name])), flush=True)
    if not all(x["passed"] for x in report["models"].values()):
        raise AssertionError("At least one model did not memorize the real training subset within the cap")


def order(args):
    return audited_measurement(args,_order)


def _order(args,out,checkpoint,model,cache,freeze):
    ids = split_ids(cache, checkpoint)[args.split]
    loader = normalized_loader(cache, ids, checkpoint)
    clean, predictions = esc.run_eval(model, loader, "cpu")
    esc.save_predictions(out / "clean_predictions.csv", cache["names"][ids], cache["y"][ids], predictions)
    trials = []
    for seed in args.permutation_seeds:
        metrics, permuted = esc.run_eval(model, loader, "cpu", time_shuffle_seed=seed)
        esc.save_predictions(out / f"shuffle_{seed}_predictions.csv", cache["names"][ids], cache["y"][ids], permuted)
        trials.append(dict(seed=seed, accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                           loss=metrics["loss"], changed_predictions=sum(a!=b for a,b in zip(predictions,permuted))))
    report = dict(model=checkpoint["model_name"], checkpoint_sha256=checkpoint.get("_loaded_checkpoint_sha256") or esc.sha256(args.checkpoint),
                  split=args.split, freeze=freeze, clean=clean, trials=trials,
                  interpretation="Encoder-token permutation is a distribution-shift dependence probe, not a causal proof of useful recurrence")
    esc.write_json(out / "order_ablation.json", report)
    print(json.dumps(dict(clean_accuracy=clean["accuracy"],trials=trials)), flush=True)


def perturb(wave, condition, seed):
    if condition == "clean":
        return wave
    if condition.startswith("gain_"):
        return wave * 10 ** (float(condition[5:]) / 20)
    if not condition.startswith("snr_"):
        raise ValueError("Unknown stress condition")
    signal_rms = wave.square().mean().sqrt()
    if signal_rms <= 1e-8:
        raise ValueError("SNR is undefined for a silent reference waveform")
    noise = torch.randn(wave.shape, generator=torch.Generator().manual_seed(seed))
    noise -= noise.mean()
    noise /= noise.square().mean().sqrt().clamp_min(1e-10)
    return wave + noise * signal_rms / 10 ** (float(condition[4:]) / 20)


def stress(args):
    return audited_measurement(args,_stress)


def verified_stress_audio(args, cache, ids):
    """Verify the raw source and retain immutable processed waveforms for the run.

    An absolute tolerance of 1e-5 in natural-log feature units (rtol=0) permits
    small backend rounding differences. No adaptive widening is performed.
    All selected clips are checked before any model forward pass.
    """
    root = Path(args.root)
    sidecar = json.loads(Path(str(args.cache) + ".json").read_text())
    metadata_path = root / "meta" / "esc50.csv"
    expected_metadata = sidecar.get("metadata_sha256")
    if (not isinstance(expected_metadata, str) or len(expected_metadata) != 64 or
            not metadata_path.is_file() or esc.sha256(metadata_path) != expected_metadata):
        raise ValueError("Raw stress root metadata does not match the verified feature cache")
    raw_rows = esc.read_metadata(root)
    cached_rows = [dict(filename=str(name), fold=int(fold), target=int(target),
                        category=str(cache["class_names"][target]), src_file=str(source))
                   for name,fold,target,source in zip(cache["names"],cache["folds"],cache["y"],cache["sources"])]
    if esc.canonical_hash(esc.metadata_records(raw_rows)) != esc.canonical_hash(esc.metadata_records(cached_rows)):
        raise ValueError("Raw stress root metadata rows are not aligned with the verified cache")
    waveforms, audio_hashes, maximum = {}, {}, 0.0
    for index in ids:
        name = str(cache["names"][index])
        path = root / "audio" / name
        # Hash and decode the same immutable bytes. This prevents a source change
        # between validation and conditions from silently changing the experiment.
        audio_bytes = path.read_bytes()
        audio_hashes[name] = hashlib.sha256(audio_bytes).hexdigest()
        wave = esc.read_audio(io.BytesIO(audio_bytes))
        actual = esc.log_mel(wave)
        expected = torch.from_numpy(cache["x"][index])
        difference = float((actual - expected).abs().max())
        if not torch.allclose(actual, expected, rtol=0, atol=1e-5):
            raise ValueError(f"Raw stress audio features differ from cache for {name}: max_abs={difference:.8g}, atol=1e-5")
        maximum = max(maximum, difference)
        waveforms[int(index)] = wave
    provenance = dict(metadata_sha256=expected_metadata,
                      verification_environment=esc.environment(),
                      cache_environment=sidecar.get("environment"),
                      cache_sha256=esc.sha256(args.cache),
                      cache_sidecar_sha256=esc.sha256(str(args.cache)+".json"),
                      audio_sha256=audio_hashes,
                      clean_feature_equivalence=dict(clips=len(ids),atol=1e-5,rtol=0,
                                                     max_absolute_difference=maximum),
                      reference_policy="Decode immutable verified bytes once; reuse processed waveform for every condition")
    return waveforms, provenance


def _stress(args,out,checkpoint,model,cache,freeze):
    validate_stress_conditions(args.conditions)
    ids = split_ids(cache, checkpoint)[args.split]
    waveforms, provenance = verified_stress_audio(args, cache, ids)
    esc.write_json(out / "raw_audio_provenance.json", provenance)
    report = dict(checkpoint_sha256=checkpoint.get("_loaded_checkpoint_sha256") or esc.sha256(args.checkpoint), split=args.split, freeze=freeze,
                  noise="white Gaussian, exact per-clip RMS SNR, no peak clipping; one deterministic realization/clip/condition",
                  interpretation="Synthetic corruption sensitivity on ESC-50; not real-world acoustic robustness",
                  raw_audio_provenance=provenance, conditions={}, seed=args.seed)
    for condition in args.conditions:
        started, preds, targets = time.monotonic(), [], []
        for begin in range(0, len(ids), 32):
            batch = ids[begin:begin+32]
            features = []
            for index in batch:
                name = str(cache["names"][index])
                seed = int(hashlib.sha256(f"{args.seed}:{condition}:{name}".encode()).hexdigest()[:8],16)
                features.append(esc.log_mel(perturb(waveforms[index],condition,seed)))
            x = (torch.stack(features)-checkpoint["mean"])/checkpoint["std"]
            with torch.inference_mode():
                preds.extend(model(x).argmax(1).tolist())
            targets.extend(cache["y"][batch].tolist())
        metrics = esc.compute_metrics(targets,preds)
        metrics["seconds"] = time.monotonic()-started
        report["conditions"][condition] = metrics
        esc.save_predictions(out/f"{condition}_predictions.csv",cache["names"][ids],targets,preds)
        esc.write_json(out/"stress.json",report)
        print(json.dumps(dict(condition=condition,accuracy=metrics["accuracy"],macro_f1=metrics["macro_f1"])),flush=True)


def latency(args):
    out = new_output(args.output)
    checkpoint, model = restore(args.checkpoint,args.allow_legacy_checkpoint)
    wave = esc.read_audio(args.audio)
    def frontend():
        return (esc.log_mel(esc.read_audio(args.audio)).unsqueeze(0)-checkpoint["mean"])/checkpoint["std"]
    feature = (esc.log_mel(wave).unsqueeze(0)-checkpoint["mean"])/checkpoint["std"]
    def measure(full):
        samples=[]
        with torch.inference_mode():
            for i in range(args.warmup+args.repetitions):
                start=time.perf_counter()
                model(frontend() if full else feature)
                elapsed=(time.perf_counter()-start)*1000
                if i>=args.warmup:samples.append(elapsed)
        return dict(p50_ms=float(np.percentile(samples,50)),p95_ms=float(np.percentile(samples,95)),
                    mean_ms=float(np.mean(samples)),samples_ms=samples)
    report=dict(environment=esc.environment(),execution_device="cpu",threads=torch.get_num_threads(),batch=1,
                checkpoint_sha256=checkpoint.get("_loaded_checkpoint_sha256") or esc.sha256(args.checkpoint),checkpoint_bytes=Path(args.checkpoint).stat().st_size,
                audio_sha256=esc.sha256(args.audio),warmup=args.warmup,repetitions=args.repetitions,
                parameters=sum(p.numel() for p in model.parameters()),model_only=measure(False),
                waveform_file_to_logits=measure(True),scope="Warmed CPU, repeated same WAV with warm OS file cache; excludes model loading; not CUDA or service latency")
    esc.write_json(out/"latency.json",report)
    print(json.dumps({k:report[k] for k in ('parameters','checkpoint_bytes','model_only','waveform_file_to_logits')}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser('tiny')
    p.add_argument('--cache',default='data/esc50_features.npz')
    p.add_argument('--output',required=True)
    p.add_argument('--models',nargs='+',default=['cnn','crnn','cnn_matched'])
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--max-seconds',type=float,default=120)
    p.set_defaults(func=tiny)
    for command,func in [('order',order),('stress',stress),('latency',latency)]:
        p=sub.add_parser(command)
        p.add_argument('--checkpoint',required=True)
        p.add_argument('--output',required=True)
        p.add_argument('--allow-legacy-checkpoint',action='store_true')
        if command!='latency':
            p.add_argument('--cache',default='data/esc50_features.npz')
            p.add_argument('--split',choices=['validation','test'],default='validation')
            p.add_argument('--allow-test',action='store_true')
            p.add_argument('--freeze')
        if command=='order':p.add_argument('--permutation-seeds',type=int,nargs='+',default=[101,202,303,404,505])
        if command=='stress':
            p.add_argument('--root',default='data/ESC-50')
            p.add_argument('--conditions',nargs='+',default=['clean','snr_20','snr_10','snr_0','gain_-6','gain_6'])
            p.add_argument('--seed',type=int,default=2026)
        if command=='latency':
            p.add_argument('--audio',required=True)
            p.add_argument('--warmup',type=int,default=10)
            p.add_argument('--repetitions',type=int,default=100)
        p.set_defaults(func=func)
    args=parser.parse_args()
    torch.set_num_threads(2)
    args.func(args)


if __name__=='__main__':main()
