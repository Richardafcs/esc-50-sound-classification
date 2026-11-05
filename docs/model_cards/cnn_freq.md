# Frequency-preserving CNN

Seed 101 checkpoint from the completed ESC-50 study. Validation accuracy: 72.61% on 398 source-strict development clips. The three-seed family averages are reported separately in the [results notebook](../../notebooks/ESC50_Results.ipynb).

The frontend uses 16 kHz mono audio, 64 log-Mel bands, a 512-point FFT and a 256-sample hop. Training-only normalization is saved with the weights. Input clips longer than five seconds must be deliberately trimmed. Scores are uncalibrated and do not provide general unknown-sound rejection.

From the repository root:

```sh
python -m esc50_sound predict --checkpoint models/cnn/checkpoint.pt --audio YOUR_SHORT_CLIP.wav
```

The notebook separates historical fold-5 results from current development comparisons. All official folds have historical exposure; new-source generalization and field-noise reliability are not established. ESC-50 is distributed overall under CC BY-NC 3.0. Retain the official license and recording credits.
