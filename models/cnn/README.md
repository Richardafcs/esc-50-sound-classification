# CNN baseline

`checkpoint.pt` contains the seed-101 frequency-preserving CNN, with **49,058 parameters**, the fitted normalization statistics and the ESC-50 class mapping. This checkpoint achieves **72.61% development accuracy** on 398 clips. The model-family result is 72.61% ± 0.25 percentage points across three seeds.

The input is a single-channel log-Mel spectrogram derived from 16-kHz mono audio. Three convolutional stages produce 64 × 8 × 39 features; temporal averaging retains a 512-dimensional representation for the 50-class output. The FFT size is 512, hop length 256 and Mel-band count 64.

From the repository root, after installing the project:

```sh
python -m esc50_sound predict --checkpoint models/cnn/checkpoint.pt --audio YOUR_SHORT_CLIP.wav
```

The prediction command accepts recordings up to five seconds, averages channels, resamples to 16 kHz and right-pads shorter inputs. Longer recordings must be trimmed explicitly. A fixed near-silence energy rule can abstain; it is not an unknown-class detector.

See [methods](../../docs/methods.md), [evaluation](../../docs/evaluation.md) and the [results notebook](../../notebooks/01_results.ipynb). The scores are development measurements, not an untouched final-test result. ESC-50's [license](../../data/ESC-50/LICENSE) and [recording credits](../../data/ESC-50/meta/esc50.csv) apply to the source data.
