# ESC-50 portable classifier

Read MODEL_CARD.md before use. The bundle includes one selected model, the inspectable esc50_sound package, dependency specifications and file hashes.

Install third-party dependencies if needed: `python -m pip install -r requirements.txt`. A globally installed copy of this project is unnecessary. From this directory run:

```sh
python predict.py --checkpoint checkpoint.pt --audio YOUR_SHORT_CLIP.wav
```

Use a recording at most five seconds long. Shorter recordings are padded; silence abstains. Scores are uncalibrated and the classifier does not detect arbitrary unknown sounds. No dataset audio is included. `bundle_manifest.json` lists SHA256 hashes for every bundled file.
