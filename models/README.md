# Models

| Model | Artifact |
| --- | --- |
| Frequency-preserving CNN | [Checkpoint and card](cnn/) |
| CNN + BiGRU | [Checkpoint and card](crnn/) |
| Frozen PANNs + linear classifier | [Linear head and card](panns_linear/) |

CNN/CRNN checkpoints use the shared source implementation. From the repository root:

```sh
python -m esc50_sound predict --checkpoint models/cnn/checkpoint.pt --audio YOUR_SHORT_CLIP.wav
python -m esc50_sound predict --checkpoint models/crnn/checkpoint.pt --audio YOUR_SHORT_CLIP.wav
```

The PANNs head requires the official external encoder. The [results notebook](../notebooks/ESC50_Results.ipynb) performs a fresh extraction and fit, displays all model comparisons and documents the frontend differences. Softmax scores are uncalibrated; these classifiers do not reliably detect unknown sound classes.
