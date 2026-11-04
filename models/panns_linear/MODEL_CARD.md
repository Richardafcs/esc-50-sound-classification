# Frozen PANNs + linear ESC-50 classifier

A single linear probe trained on 1,200 source-strict ESC-50 development recordings. It uses 2,048-dimensional frozen `Cnn14_16k` embeddings from the [PANNs author implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn). The official checkpoint is [Cnn14_16k_mAP=0.438.pth](https://zenodo.org/records/3987831). The backbone is not redistributed here.

- Validation: 363/398 correct, accuracy 91.21%, macro-F1 91.10%, 295 recording sources.
- Encoder: 81,033,231 loaded parameters, frozen and in eval mode. Source revision `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`.
- Input to this head: the exact 2,048-dimensional author-model embedding, not the 527 AudioSet output probabilities or a spectrogram.
- Head: train-only StandardScaler plus L2 multinomial logistic regression, C=1, L-BFGS; 102,450 coefficients and biases. One fit, 137 iterations, normal convergence.
- `linear_head.npz`: fitted training means/scales, coefficients, intercepts and class IDs. `class_names.json`: ESC-50 label mapping.
- `predict_embeddings.py`: NumPy inference on saved embedding NPZ files. It reproduces all 398 original validation predictions without rerunning the encoder.

From the project root, with the local feature cache available:

```sh
python models/panns_linear/predict_embeddings.py \
  --features results/runs/panns_20261105/val_embeddings.npz
```

For waveform-to-feature reproduction, use [the research runner](../../scripts/run_panns_linear_probe.py) with the exact official encoder, checkpoint and audio configuration in the [frozen protocol](../../reports/panns_20261105/protocol.json). The runner performs one complete extraction and fit; it is not a deployment service. Its output guard prevents accidentally replacing a completed run.

All folds have historical exposure and AudioSet/ESC-50 overlap has not been comprehensively excluded. This is a stronger clean-development candidate, not an independently confirmed replacement for a deployed system. Encoder capacity, frontend and external pretraining differ from the scratch-trained baselines. Noise robustness, calibration, unknown-sound rejection and deployment latency remain unevaluated. The head alone is small; the complete encoder is much larger than the earlier CNN.

[Full English report](../../reports/panns_20261105/report_en.md) · [Chinese explanation](../../reports/panns_20261105/report_zh.md). Retain ESC-50 licensing and recording credits; the small head export does not grant rights to the external backbone or dataset.
