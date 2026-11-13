# Frozen PANNs with a linear classifier

The clean-trained reference uses 2,048-dimensional embeddings from the official [PANNs Cnn14_16k encoder](https://github.com/qiuqiangkong/audioset_tagging_cnn), pretrained on AudioSet. Its [checkpoint](https://zenodo.org/records/3987831) is obtained separately. The loaded model has 81,033,231 parameters and remains frozen in evaluation mode.

The downstream head is a training-fitted StandardScaler followed by L2 multinomial logistic regression: C=1, L-BFGS, tolerance 1e-6 and maximum 3,000 iterations. The original fit converged in 137 iterations. It contains **102,450 coefficients and biases** and achieves **363/398 correct predictions: 91.21% accuracy and 91.10% macro-F1**.

| File | Contents |
| --- | --- |
| `linear_head.npz` | Training means and scales, coefficients, intercepts and class IDs |
| `class_names.json` | ESC-50 label names |
| `predict_embeddings.py` | NumPy inference on saved embeddings |

The head expects the encoder's embedding, not its 527 AudioSet output probabilities or a spectrogram. Given an NPZ containing `embeddings` and `filenames`:

```sh
python models/panns_linear/predict_embeddings.py --features PATH_TO_EMBEDDINGS.npz
```

The [main notebook](../../notebooks/01_results.ipynb) and [extraction runner](../../scripts/run_panns_linear_probe.py) provide waveform-to-feature computation. The [experiment protocol](../../reports/results/panns/protocol.json) records the exact encoder configuration. A matched Gaussian 10-dB panel gives 69.85% accuracy; a separate six-category external subset gives 45.17%. See the [evaluation protocol](../../docs/evaluation.md) for their interpretation.

This is a complete-system comparison with the scratch CNN: external training data, capacity, architecture and frontend all differ. AudioSet/ESC-50 overlap has not been comprehensively excluded. The small head does not make the full encoder a small inference system. Dataset and encoder licensing remain separate; retain the [ESC-50 license](../../data/ESC-50/LICENSE) and [recording credits](../../data/ESC-50/meta/esc50.csv).
