# Portable models

| Model | Directory | Role |
| --- | --- | --- |
| CNN (`cnn_freq`) | [cnn](cnn/) | Historically frozen default, seed101 |
| CNN + BiGRU / CRNN (`crnn_freq`) | [crnn](crnn/) | Recurrent comparison model, seed101 |

Each directory contains the checkpoint, model definitions, prediction entry point, model card, and file-hash manifest. Read the corresponding `MODEL_CARD.md` before use. From the project root:

```sh
python models/cnn/predict.py --checkpoint models/cnn/checkpoint.pt --audio YOUR_SHORT_CLIP.wav
python models/crnn/predict.py --checkpoint models/crnn/checkpoint.pt --audio YOUR_SHORT_CLIP.wav
```

These bundles support WAV inference with their own frontend and stored normalization. Their internal manifests and weights retain the original identity. Softmax scores are uncalibrated and the models do not reliably detect arbitrary unknown sounds.

Classical baseline implementations are in [baseline.py](../src/esc50_sound/baseline.py) and [cepstral.py](../src/esc50_sound/cepstral.py). The latest matrix [summary evidence](../reports/evidence/matrix_20261028/) is separate from these frozen portable deliveries. Bulk matrix checkpoints and duplicate standalone package assemblies remain local and are ignored by Git.
