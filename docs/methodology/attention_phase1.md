# Phase 1 — Attention-CRNN implementation and research protocol

**Status (2026-10-28):** Three architectures implemented; unit/integration validation described in `docs/reviews/attention_cloud_validation.json`. **No real ESC-50 Attention models have been trained** in this package. Original baseline weights and historical conclusions have not been changed.

## What is new

In `src/esc50_sound/models.py`, register three models using the legacy CNN+BiGRU backbone:

| New name | Mechanism | Added parameters | Total parameters |
|---|---|---:|---:|
| `crnn_freq_ta` | Score each recurrent token, softmax over time, weighted aggregation | 129 | 251,939 |
| `crnn_freq_ca` | Squeeze time/frequency for 64 CNN feature channels; `64 → 8 → 64` gate | 1,096 | 252,906 |
| `crnn_freq_tca` | Channel gate before BiGRU plus temporal attention after BiGRU | 1,225 | 253,035 |

Unchanged: `crnn_freq` is 251,810 parameters; `[B,1,64,313] → [B,64,8,39] → [B,39,512] → BiGRU [B,39,128]` with mean pooling. No changes were made to the original feature extractor, training loop, evaluation API, historical experiment files, legacy model names, or saved checkpoints.

### Mechanistic design

Temporal Attention Pooling receives BiGRU output `H[B,T,128]` and computes

`alpha = softmax(Linear(128,1)(H), dim=time)` and `pooled = sum(alpha * H, dim=time)`.

The Linear score is initialized to zero, making `alpha=1/T` at initialization. Channel Attention receives `F[B,64,8,39]`, computes global channel means and a two-layer MLP, and multiplies `F` by `2*sigmoid(MLP(g))`. The final projection is zero initialized, yielding gate=1 at initialization. It learns both suppression and amplification. This is an **SE-style implementation informed by Zhang et al. (2019)**, not a layer-exact replication of the paper.

The optional model forward keyword `return_attention=True` returns `(logits, {"temporal": [B,39], "channel": [B,64]})`; absent mechanisms are omitted. The standard `forward(x)` still returns only logits and is compatible with existing training/evaluation/inference code. **Attention weights are not sufficient evidence of causal explanations.**

### Why not directly compare held-out accuracy?

Historical researchers already examined ESC-50 Fold 5 and folds 1–4 in the original project. These models may be evaluated on the old folds for **exploratory/development comparison**, but the results are *not* an independent confirmatory test. The prior 3-training-fold + 1-validation-fold + 1-test-fold source-strict policy is retained. It does not match published standard 4-fold-training benchmarks.

To claim unbiased external generalization, use a genuinely independent, source-audited test dataset and decide the model/parameters first. Never use the sealed test fold for tuning learning rate, architecture, augmentation, or stopping. All model contrast reports must disclose multiple seeds, paired predictions, and uncertainty.

## How to run on a machine with full ESC-50 WAV data and a compatible GPU

```bash
python -m pip install -e '.[dev]'
python -m esc50_sound prepare  # downloads pinned official audio if permitted and builds feature cache
python -m pytest -q tests/test_attention.py

# Repeat for MODEL = crnn_freq, crnn_freq_ta, crnn_freq_ca, crnn_freq_tca
# and SEED = 101, 202, 303, using unique output paths.
python -m esc50_sound train \
  --cache data/esc50_features.npz \
  --model crnn_freq_tca \
  --output runs/attention_v1/crnn_freq_tca_seed101 \
  --epochs 60 --max-seconds 3600 \
  --batch-size 32 --lr 0.001 --scheduler cosine \
  --seed 101 --train-folds 1,2,3 --val-fold 4 --test-fold 5 \
  --device cuda
```

`--max-seconds` is a **hard training cap**; the run must complete all 60 epochs to join equal-exposure comparisons. Use validation outputs (`validation.json`, `validation_predictions.csv`) for preliminary analysis. The `evaluate` command checks the sealed test authorization, so it must not be executed during model selection.

## Preanalysis of hypotheses

- **H_T:** Learned temporal weights improve validation macro-F1/accuracy vs the matched `crnn_freq` with fixed input/optimizer/training budget.
- **H_C:** Learned channel gates improve the same outcomes.
- **H_TC:** Joint attention has gains beyond their individual effects. Interaction on the accuracy scale: `A(tca) - A(ta) - A(ca) + A(base)`.
- **Costs/limits:** Parameter-count similarity does not guarantee matched optimization, FLOPs, or latency. One-fold, three-seed results cannot establish general statistical significance. Recording negative results is appropriate.

## Quality gates

- Tests in `tests/test_attention.py`: output/weights shapes, exact model parameter counts, initial identity/equivalence, legacy state-dict compatibility, gradients, token-order intervention, checkpoint round-trip.
- Additional smoke training uses a **synthetic** cache with valid metadata provenance; it is strictly for end-to-end compatibility, not an audio-classification result.
- Do not replace any historical `runs/`, `results/`, `deliverables/`, or notebook outputs with new tests.

## Build and execute a complete 12-run GPU grid (optional)

A deterministic generator now creates the *commands* for four model variants × three seeds; it does not run them or unlock the test fold:

```bash
python scripts/make_attention_grid.py --output configs/attention_gpu_grid_v1.json --device cuda
python scripts/legacy/runner.py run-grid --config configs/attention_gpu_grid_v1.json \
  --output results/attention_gpu_grid_v1 --base-dir .
```

Check that a compatible CUDA GPU and `data/esc50_features.npz` are available before launching. `run-grid` executes the **training** commands only; it preserves registered commands/logs. Each run has a 3,600-second process cap and a 3,500-second training cap; partial runs must be excluded from equal-budget comparisons. The generated run paths must not already exist. No paid cloud resources are provisioned by this tool.
