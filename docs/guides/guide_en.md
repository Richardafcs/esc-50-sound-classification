# ESC-50 study: explanation and practical guide

Prepared 28 October 2026. This guide explains the supplied implementation. Read the [current results report](../../reports/matrix_20261028/report_en.md) for what the new Colab execution actually completed and measured. A configured experiment, a passing software test, an old saved result and a newly executed training run are different kinds of evidence.

## 1. What question does the project answer?

Given a five-second recording, predict one of 50 environmental-sound classes, such as dog, rain, door knock or airplane. The model produces **one label for the whole clip**. It does not locate individual events in time, separate overlapping sound sources or reliably recognize an unknown class outside ESC-50.

The study asks whether performance changes when we change the classifier, training augmentation or audio representation. Those questions require controlled comparisons: changing several things at once cannot establish which one caused a score difference.

ESC-50 contains 2,000 recordings: 40 per class, organized into five official folds. This project's source-strict development protocol uses folds 1–3 for training, fold 4 for validation and fold 5 as the designated test role. Metadata reveals some shared original-source IDs across folds. Applying the implemented source priority leaves **1,200 training clips, 398 validation clips and 400 test-role clips**; two validation clips are excluded. A shared source ID is a reason to avoid source leakage, not proof that two clips are identical audio.

Training examples update the weights. Validation examples choose the checkpoint and compare candidate recipes. Test examples should answer a final, separately specified question. **The historical project already accessed fold 5 and rotated through other folds.** The current matrix does not evaluate fold 5; new validation results cannot restore an untouched confirmation set. This is also not the standard four-training-fold ESC-50 benchmark, so published benchmark numbers are not directly comparable.

## 2. How does sound become model input?

The main architecture/augmentation track follows this path:

```text
5-second WAV → mono, 16 kHz → 80,000 samples
             → STFT power → 64 Mel bands → natural logarithm
             → normalization fitted on training clips only
             → tensor [batch, 1, 64, 313]
```

A waveform is amplitude over time. The STFT analyzes successive short windows, exposing the frequency content of each local interval. Its window is 512 samples (32 ms), hop 256 samples (16 ms), with centered padding. The Mel filterbank combines nearby frequencies into 64 bands. Logarithms compress the large power range. Normalization puts each input band on a scale estimated from training recordings, without fitting to validation or test examples.

Here `[batch, 1, 64, 313]` means batch size, one input channel, 64 frequency bands and 313 time frames. It is a time–frequency representation, so image-style convolutions can detect local acoustic patterns. “Channel” inside the CNN later means a learned feature map; it does not mean a stereo microphone channel.

The shared encoder has three convolution → BatchNorm → ReLU → max-pooling stages, with 16, 32 and 64 learned channels. For the main 64-band input, its output is `[batch, 64, 8, 39]`.

## 3. What do CNN, GRU and attention change?

| Matrix model | Actual computation after the shared encoder | Question |
| --- | --- | --- |
| `cnn_freq` | Average over time, retain channel × frequency positions, linear classifier | How far does a small CNN go? |
| `crnn_freq` | Flatten each time step to 512 features; bidirectional GRU; average 39 recurrent outputs; classifier | Does sequential context help? |
| `crnn_freq_ta` | Same GRU; learned temporal weighting replaces mean pooling | Does choosing which moments matter help? |
| `crnn_freq_ca` | Learned channel gates before the GRU; mean pooling afterward | Does reweighting learned feature maps help? |
| `crnn_freq_tca` | Channel gates plus temporal attention | Does their combination help? |

The bidirectional GRU reads the **whole recorded clip** in both temporal directions. Its hidden size is 64 per direction, giving `[batch, 39, 128]` recurrent features. This is an offline clip classifier; bidirectionality does not establish causal streaming behavior.

Temporal attention assigns a score to each recurrent token, applies a softmax over time and forms a weighted average. With weights `α_t`, the pooled vector is `Σ_t α_t h_t`, where `α_t ≥ 0` and `Σ_t α_t = 1`. Initialization gives equal scores, so it begins as mean pooling. For example, weights 0.1, 0.8 and 0.1 emphasize the middle of three tokens; equal weights recover the ordinary average. Attention weights show how this mechanism combines representations, not a guaranteed explanation of the true sound source.

Channel attention averages each encoder channel over frequency and time, passes the 64 summaries through a 64 → 8 → 64 MLP, then uses **2 × sigmoid** gates. Gates start at 1, preserving the encoder features initially; their possible range is 0–2, not 0–1. This is a lightweight SE-style adaptation, not a verbatim reproduction of a cited attention paper. Channel attention alone changes emphasis, while temporal attention changes the clip-level time aggregation. Neither structure is guaranteed to improve accuracy.

## 4. What are the 65 tasks?

| Group | Tasks | Meaning |
| --- | ---: | --- |
| A: traditional baselines | 2 | Log-Mel temporal mean/std + multinomial logistic regression; DCT cepstral mean/std + linear SVM |
| B: architecture | 15 | The five models above × seeds 101, 202 and 303 |
| C: augmentation | 9 | CNN with SpecAugment, Gaussian noise or waveform Mixup × three seeds |
| D: features and sampling | 36 | Twelve representations/sample-rate conditions × three seeds, using a separate frequency-pooled `cnn` |
| E: robustness | 3 | Validation stress tests for clean CNN, clean GRU and noise-trained CNN, using seed-101 checkpoints |

A baseline establishes whether deep-model complexity adds useful predictive power. The logistic baseline reduces 64 Mel bands to 128 features: one temporal mean and one temporal standard deviation per band. The SVM uses 20 orthonormal DCT-II coefficients of the 64 Log-Mel bands, then their temporal mean/std: 40 features. It is a project-specific cepstral baseline, not a reproduction of Piczak's SVM system.

The main neural recipe uses 60 epochs, batch size 32, AdamW with initial learning rate 0.001 and weight decay 0.0001, cosine decay and gradient clipping. The selected checkpoint has the lowest validation cross-entropy, not necessarily the highest validation accuracy. Seeds change initialization and training randomness; three seeds help reveal instability, but are not three independent datasets.

**SpecAugment** masks selected normalized frequency/time regions during training. **Noise augmentation** adds Gaussian noise to each training waveform with probability 0.5 and added-noise SNR sampled from 10–30 dB. **Mixup** combines two training waveforms with a Beta(0.4, 0.4) mixing weight and uses the corresponding weighted sum of two cross-entropies. For example, a 0.7 dog + 0.3 rain mixture has a soft training target; calling it only “dog” would discard the label mixture. Validation stays clean for checkpoint selection.

Group E separately measures clean audio, synthetic noise at 20/10/0 dB SNR and gain changes of −6/+6 dB. At 0 dB the added noise and original recording have equal RMS power. Noise is scaled per clip without peak clipping. This controlled white-noise test is not evidence of robustness to all microphones, rooms, traffic noise or real-world deployment conditions.

## 5. Why twelve feature cases?

Group D crosses four representations with three conditions:

| Condition | Sample rate | Analyzed upper frequency | FFT / hop |
| --- | ---: | ---: | ---: |
| SR16–F8 | 16 kHz | 8 kHz | 512 / 256 samples |
| SR32–F8 | 32 kHz | 8 kHz | 1,024 / 512 samples |
| SR32–F16 | 32 kHz | 16 kHz | 1,024 / 512 samples |

All keep five-second duration, 32 ms windows, 16 ms hops and 313 frames. The representations are 64 Log-Mel bands, 32 Log-Mel bands, 40 DCT-II coefficients of 64 Log-Mel bands (`mfcc40`), and 64 linearly spaced triangular log-power bands. The MFCC convention has no liftering or special replacement of coefficient 0.

Comparing SR16–F8 with SR32–F8 holds analyzed bandwidth at 8 kHz; it does not add high-frequency information. Comparing SR32–F8 with SR32–F16 adds the upper band but also changes filter spacing at fixed band count. These are useful controlled contrasts with explicit limits, not a universal claim that higher sample rate is better.

This track uses `cnn`, which pools frequency globally and therefore keeps its parameter count fixed when the input has 32, 40 or 64 rows. It is **different from `cnn_freq`** in group B. Compare feature choices within group D; a B-versus-D difference changes both representation and model. Group-D checkpoints also require their own feature frontend and are not accepted as ordinary legacy `predict` inputs.

## 6. How should I read a result?

**Accuracy** is correctly classified clips divided by all evaluated clips. If 280 of 398 validation clips are correct, accuracy is 70.35%; one clip changes it by about 0.251 percentage points.

**Macro-F1** computes each class's F1 and averages all 50 classes equally. For a class, precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and F1 is `2TP/(2TP+FP+FN)`; an undefined zero-denominator class receives zero in this implementation. Macro-F1 helps reveal classes that a model neglects even when overall accuracy looks reasonable. Both measures range from 0 to 1 in JSON and are usually displayed as percentages.

A difference from 70% to 72% is **2 percentage points**, not 2%. Report all registered seeds, their mean and spread, clean and corrupted scores, and failed or partial runs. Choosing the best seed alone exaggerates performance. Repeated selection on the same 398 clips can overfit the development process even though training never backpropagates through validation.

## 7. Where is the relevant code and evidence?

| Location | Responsibility |
| --- | --- |
| `notebooks/templates/ESC50_Colab_Research_Console.ipynb` | Plan, pilot and selected-task controller |
| `configs/colab_study_matrix_v1.json` | Registered task roster; matches the supplied standalone JSON |
| `src/esc50_sound/audio.py`, `data.py` | Audio frontend, data acquisition, split and normalization |
| `models.py`, `training.py`, `augmentation.py` | Architectures, optimization and training-only transformations |
| `baseline.py`, `cepstral.py` | Traditional classifiers |
| `feature_ablation.py` and `scripts/feature_ablation.py` | Separate feature track and its command line |
| `evaluation.py`, `inference.py` | Dataset metrics versus prediction on one new clip |
| `study.py` | Controlled robustness and diagnostic measurements |
| `results/matrix_20261028/` | This execution's retained evidence; consult the results report for exact contents |
| `results/historical/final_study/`, old notebooks, model bundles | Historical research evidence; not automatically new executions |

The scheduler, registry and platform modules are optional research infrastructure. They organize work; their presence does not establish improved classification. PANNs, AST, BEATs, CLAP and reinforcement learning are listed as unavailable/future work, without integrated executable tasks or scores.

## 8. Running and recovering experiments

From the project root, a direct main-track example is:

```bash
python -m pip install -e '.[dev,baseline,report]'
python -m esc50_sound prepare --root data/ESC-50 --cache data/esc50_features.npz
python -m esc50_sound preflight --root data/ESC-50 --cache data/esc50_features.npz
python -m esc50_sound train --model crnn_freq_ta --seed 101 --epochs 60 \
  --scheduler cosine --device cuda --max-seconds 3600 \
  --output runs/new_ta_seed101
```

In the console, `plan` previews without training; `pilot` uses a separate short-run namespace; `selected` executes the chosen registered tasks. A two-epoch pilot verifies operation, not final classification quality. Run cells top to bottom, set explicit task/time caps and preserve each result directory.

For interrupted **new-format main training**, use:

```bash
python -m esc50_sound resume-train --output runs/new_ta_seed101 --max-seconds 3600
```

`last.pt` contains the last complete epoch and optimizer/random states; `best.pt` is the validation-selected inference model. Resume uses `last.pt`, requires compatible code/data/runtime and cannot turn an old inference-only checkpoint into a resumable run. Feature-track runs do not support epoch resume; preserve the incomplete directory and start a new run identity after inspection.

Colab local `/content` storage is temporary. Download completed logs, predictions and weights during execution; if using Drive, verify that writes and recovery work before long batches. A saved notebook alone does not preserve every runtime file. The current results report states what was actually exported. For scientific conclusions, rely on complete training records plus recomputed predictions—not merely a “completed” label.

## 9. Primary references and reading scope

Dataset definitions come from the [official ESC-50 repository](https://github.com/karolpiczak/ESC-50); the GRU interface is documented in [PyTorch 2.11](https://docs.pytorch.org/docs/2.11/generated/torch.nn.GRU.html). [Squeeze-and-Excitation](https://arxiv.org/abs/1709.01507) supplies the channel-recalibration idea; [SpecAugment](https://arxiv.org/abs/1904.08779) and [mixup](https://arxiv.org/abs/1710.09412) are the original augmentation papers. Their primary abstract/metadata pages were checked in this round; implementation details here are established from the actual source. The channel gates, masking without time warping, and waveform Mixup are specific adaptations, not claims of full paper reproduction. These references do not establish performance gains in this project.


![A real validation waveform and Log-Mel representation](../../reports/figures/matrix_20261028/frontend_example.png)

The figure shows one validation recording before train-fitted normalization. The vertical spectrogram axis is the Mel-band index, not a CNN channel.
