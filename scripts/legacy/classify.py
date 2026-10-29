"""Compatibility launcher; use ``python -m esc50_sound predict`` for new work."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from esc50_sound.inference import AudioClassifier, package_bundle, main

if __name__ == "__main__":
    main()
