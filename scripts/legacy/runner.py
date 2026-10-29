"""Compatibility study runner; resolve the current checkout deterministically."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from esc50_sound.runner import *  # noqa: F401,F403
from esc50_sound.runner import main

if __name__=='__main__':
    main()
