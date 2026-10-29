"""Legacy launcher for recorded commands; new work uses ``python -m esc50_sound``.

These explicit reexports temporarily keep old research tools importable. They
are ordinary imported functions: patch or test the module that owns a function,
not this launcher. No implementation or mutable-global forwarding lives here.
"""
from pathlib import Path
import sys

# Keep direct execution usable before an editable install and inside old bundles
# that include src/. Installed commands do not depend on this compatibility file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from esc50_sound.audio import FRONTEND, log_mel, read_audio
from esc50_sound.checkpoints import load_checkpoint, validate_checkpoint
from esc50_sound.cli import main
from esc50_sound.data import (
    DATA_COMMIT, audit_metadata, fit_normalization, load_cache, metadata_records,
    prepare, read_metadata, split_rows,
)
from esc50_sound.evaluation import compute_metrics, evaluate, run_eval, save_predictions
from esc50_sound.inference import predict
from esc50_sound.models import build_model
from esc50_sound.records import (
    append_evaluation_audit, canonical_hash, environment, seed_all, sha256, write_json,
)
from esc50_sound.training import smoke, spec_augment, train

if __name__ == "__main__":
    main()
