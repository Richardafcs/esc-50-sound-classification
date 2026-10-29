"""Project source and legacy adapters for existing migration/regression checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for folder in [ROOT,ROOT/'src',ROOT/'scripts/legacy']:
    sys.path.insert(0,str(folder))
