# CNN–BiGRU

`checkpoint.pt` contains the seed-101 CNN–BiGRU, with **251,810 parameters**, normalization statistics and the class mapping. This checkpoint achieves **75.13% development accuracy** on 398 clips. The three-seed model-family mean is **73.87% ± 1.26 percentage points**.

The convolutional encoder is shared with the CNN baseline. Its output is rearranged into 39 time positions with 512 features each. A bidirectional GRU with hidden size 64 per direction produces 128-dimensional contextual features. Temporal averaging and a linear layer produce the 50 class logits.

From the repository root, after installing the project:

```sh
python -m esc50_sound predict --checkpoint models/crnn/checkpoint.pt --audio YOUR_SHORT_CLIP.wav
```

The frontend uses 16-kHz mono audio, 64 log-Mel bands, a 512-point FFT and a 256-sample hop. The prediction command accepts clips up to five seconds and right-pads shorter recordings. Bidirectional processing uses the complete clip, so this model is not a causal streaming classifier.

See [methods](../../docs/methods.md), [evaluation](../../docs/evaluation.md) and the [results notebook](../../notebooks/01_results.ipynb). The observed gain over CNN does not isolate temporal modeling from added capacity. ESC-50's [license](../../data/ESC-50/LICENSE) and [recording credits](../../data/ESC-50/meta/esc50.csv) apply to the source data.
