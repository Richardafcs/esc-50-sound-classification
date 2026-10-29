"""Classify one short environmental-sound clip or build a portable local bundle.

Examples:
  python classify.py --checkpoint best.pt --audio clip.wav --json-output result.json
  python classify.py package --checkpoint best.pt --model-card MODEL_CARD.md --output bundle

Inputs must be at most five seconds. Short recordings are zero-padded on the
right, channels are averaged to mono, and audio is resampled to 16 kHz. Long
recordings must be deliberately trimmed by the caller before classification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import soundfile as sf
import torch

import esc50

SILENCE_RMS_THRESHOLD = 1e-5
MAX_DURATION_SECONDS = 5.0
SCORE_INTERPRETATION = (
    "Uncalibrated softmax scores over the 50 training categories; "
    "not calibrated confidence or probability of a correct prediction."
)


class AudioClassifier:
    """Portable CNN/CRNN inference with explicit preprocessing and provenance.

    Silence abstention is a fixed input-energy rule. It does not provide general
    out-of-distribution detection. The caller retains responsibility for using
    the model only in the conditions documented by its accompanying model card.
    """

    def __init__(self, checkpoint, device="cpu", allow_legacy=False):
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; no device fallback was used")
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_file():
            raise ValueError(f"Checkpoint does not exist: {self.checkpoint_path}")
        try:
            checkpoint_data = esc50.load_checkpoint(self.checkpoint_path, device="cpu")
        except Exception as error:
            raise ValueError("Checkpoint could not be loaded as a weights-only artifact") from error
        if not isinstance(checkpoint_data, dict):
            raise ValueError("Checkpoint must contain a mapping")
        self.verification = esc50.validate_checkpoint(checkpoint_data, allow_legacy=allow_legacy)
        self.class_names = checkpoint_data["class_names"]
        if not all(isinstance(name, str) and name for name in self.class_names):
            raise ValueError("Checkpoint class names must be nonempty strings")
        self.device = torch.device(device)
        self.frontend = dict(checkpoint_data["frontend"])
        self.mean = checkpoint_data["mean"].detach().cpu()
        self.std = checkpoint_data["std"].detach().cpu()
        self.model_name = checkpoint_data["model_name"]
        self.manifest_id = checkpoint_data.get("manifest_id")
        self.source_policy = checkpoint_data.get("source_policy", "legacy: unverified")
        self.checkpoint_sha256 = esc50.sha256(self.checkpoint_path)
        try:
            self.model = esc50.build_model(self.model_name)
            self.model.load_state_dict(checkpoint_data["model"], strict=True)
        except Exception as error:
            raise ValueError("Checkpoint weights do not match the declared architecture") from error
        if not all(torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("Checkpoint contains nonfinite model tensors")
        self.model.to(self.device).eval()

    def classify(self, audio_path):
        """Return JSON-compatible metadata and top five scores or silence abstention."""
        path = Path(audio_path)
        if not path.is_file():
            raise ValueError(f"Audio file does not exist: {path}")
        try:
            info = sf.info(path)
        except Exception as error:
            raise ValueError("Audio file is unreadable or uses an unsupported format") from error
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise ValueError("Audio recording must be nonempty")
        raw_duration = info.frames / info.samplerate
        # Permit a single-sample rounding discrepancy; never choose a window from
        # a genuinely longer recording without the caller's explicit trimming.
        if raw_duration > MAX_DURATION_SECONDS + 1 / info.samplerate:
            raise ValueError(
                f"Audio duration is {raw_duration:.6f} seconds; maximum is five seconds. "
                "Deliberately trim a five-second window before classification."
            )
        try:
            waveform = esc50.read_audio(path)
        except Exception as error:
            raise ValueError("Audio must decode to a nonempty finite waveform") from error
        if waveform.shape != (80000,) or not torch.isfinite(waveform).all():
            raise ValueError("Preprocessing did not produce a finite five-second mono window")
        rms = float(torch.sqrt(waveform.to(torch.float64).square().mean()))
        processed_length_before_padding = (
            info.frames * self.frontend["sample_rate"] + info.samplerate - 1
        ) // info.samplerate
        preprocessing = dict(
            source_sample_rate=info.samplerate, source_channels=info.channels,
            source_frames=info.frames, raw_duration_seconds=raw_duration,
            analyzed_duration_seconds=MAX_DURATION_SECONDS, analyzed_sample_rate=16000,
            channel_policy="arithmetic mean across input channels",
            resampling_policy="scipy resample_poly to 16000 Hz when required",
            duration_policy="right zero-padding for short input; rejects recordings longer than five seconds",
            right_padding_samples=max(0, 80000 - processed_length_before_padding),
            rounding_crop_samples=max(0, processed_length_before_padding - 80000),
            analyzed_waveform_rms=rms,
        )
        result = dict(
            status="abstained" if rms < SILENCE_RMS_THRESHOLD else "classified",
            input_file=path.name, model=self.model_name, device=str(self.device),
            checkpoint_sha256=self.checkpoint_sha256, manifest_id=self.manifest_id,
            source_policy=self.source_policy, provenance_verification=self.verification,
            preprocessing=preprocessing, score_interpretation=SCORE_INTERPRETATION,
            silence_rms_threshold=SILENCE_RMS_THRESHOLD,
        )
        if rms < SILENCE_RMS_THRESHOLD:
            result.update(predicted_category=None, top5=[],
                          abstention_reason="silence_or_near_silence",
                          abstention_scope="input-energy rule only; not general unknown-sound detection")
            return result
        features = esc50.log_mel(waveform).unsqueeze(0)
        if not torch.isfinite(features).all():
            raise ValueError("Feature computation produced nonfinite values")
        features = ((features - self.mean) / self.std).to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            logits = self.model(features)
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced nonfinite scores")
            values, indices = logits.softmax(dim=-1).topk(5)
        top5 = [dict(rank=rank + 1, class_id=int(index), category=self.class_names[index],
                     softmax_score=float(score))
                for rank, (score, index) in enumerate(zip(values[0].cpu().tolist(), indices[0].cpu().tolist()))]
        result.update(predicted_category=top5[0]["category"], top5=top5)
        return result


def package_bundle(checkpoint, model_card, output, allow_legacy=False):
    """Copy code, one selected checkpoint and its card; never include recordings."""
    classifier = AudioClassifier(checkpoint, device="cpu", allow_legacy=allow_legacy)
    model_card = Path(model_card)
    if not model_card.is_file() or not model_card.read_text().strip():
        raise ValueError("A nonempty model-card file is required for a portable bundle")
    source_root = Path(__file__).resolve().parent
    sources = {
        "classify.py": Path(__file__).resolve(),
        "esc50.py": source_root / "esc50.py",
        "requirements.txt": source_root / "requirements.txt",
        "checkpoint.pt": Path(checkpoint),
        "MODEL_CARD.md": model_card,
    }
    if any(not path.is_file() for path in sources.values()):
        raise ValueError("One or more required bundle files are missing")
    # Capture immutable bytes before reserving output so a later source edit
    # cannot produce a manifest for bytes different from those actually copied.
    contents = {name: path.read_bytes() for name, path in sources.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        (output / name).write_bytes(content)
    manifest = dict(
        bundle_schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        model=classifier.model_name, frontend=classifier.frontend,
        model_manifest_id=classifier.manifest_id, source_policy=classifier.source_policy,
        provenance_verification=classifier.verification,
        runtime_at_packaging=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__),
        command="python classify.py --checkpoint checkpoint.pt --audio YOUR_SHORT_CLIP.wav",
        files={name: dict(sha256=esc50.sha256(output / name), bytes=len(content))
               for name, content in contents.items()},
        contents_scope="code, one selected checkpoint, dependency specification and model card; no dataset audio",
    )
    esc50.write_json(output / "bundle_manifest.json", manifest)
    return manifest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("predict", "package"):
        argv.insert(0, "predict")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="Classify one recording at most five seconds long")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True)
    p.add_argument("--json-output")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--allow-legacy-checkpoint", action="store_true")
    p = sub.add_parser("package", help="Create a new portable local bundle directory")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-card", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-legacy-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "package":
            result = package_bundle(args.checkpoint, args.model_card, args.output, args.allow_legacy_checkpoint)
        else:
            if args.threads < 1:
                raise ValueError("threads must be positive")
            torch.set_num_threads(args.threads)
            result = AudioClassifier(args.checkpoint, args.device, args.allow_legacy_checkpoint).classify(args.audio)
            if args.json_output:
                destination = Path(args.json_output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with open(destination, "x") as stream:
                    stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileExistsError, OSError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
