# Methods

The study compares learned representations, classifier capacity and training exposure for environmental sound classification. Its main reference is a frequency-preserving CNN trained from scratch. Classical classifiers, a separate feature/sampling study, encoded-sequence interventions, frozen PANNs transfer, external recordings and input-policy diagnostics provide complementary comparisons. [Evaluation settings](evaluation.md), [numerical results](../reports/report.md) and [executed figures](../notebooks/01_results.ipynb) are documented separately.

## Audio representation

The scratch models use five-second mono waveforms at 16 kHz, or 80,000 samples. Audio channels are averaged and other sample rates are converted using polyphase resampling. The frontend applies a Hann-windowed STFT with a 512-point FFT, a 256-sample hop and centered padding. The power spectrum is integrated into 64 HTK-scale triangular Mel bands covering 0–8 kHz. A floor of 1e-10 precedes the natural logarithm.

For batch size B, the input shape is **B × 1 × 64 × 313**. Mean and standard deviation are calculated separately for each Mel band over training clips and time frames. The same saved statistics normalize development and inference inputs. The resulting values are log spectral features, not calibrated sound-pressure levels.

The short-clip prediction command accepts recordings up to five seconds and right-pads shorter inputs. It rejects longer recordings so the caller chooses the analyzed window explicitly. A fixed near-silence RMS rule can abstain; this is not general unknown-class detection.

## CNN baseline

Three convolutional stages use the following operations:

| Stage | Convolution | Following operations | Output shape |
| --- | --- | --- | --- |
| 1 | 1→16 channels, 3×3, padding 1, no bias | BatchNorm, ReLU, 2×2 max pooling | B × 16 × 32 × 156 |
| 2 | 16→32 channels, 3×3, padding 1, no bias | BatchNorm, ReLU, 2×2 max pooling | B × 32 × 16 × 78 |
| 3 | 32→64 channels, 3×3, padding 1, no bias | BatchNorm, ReLU, 2×2 max pooling | B × 64 × 8 × 39 |

The frequency-preserving CNN averages the final features over the 39 time positions and flattens the remaining channel and frequency axes. Dropout with probability 0.2 and a 512→50 linear layer produce class logits. The encoder has 23,408 parameters; the complete baseline has **49,058**.

The code also retains older globally pooled CNN variants for historical feature experiments. Those models are not the numerical baseline of the main comparison.

## Recurrent and attention variants

The CRNN uses the same encoder. Its output is rearranged into **B × 39 × 512** tokens and passed to a one-layer bidirectional GRU with hidden size 64 per direction. Mean aggregation of the 128-dimensional recurrent outputs, dropout and a 128→50 linear layer produce the logits. The model has **251,810 parameters**. Bidirectional processing assumes the complete clip is available.

Temporal attention replaces the recurrent mean with a learned weighted sum. A shared linear scorer assigns one logit to each recurrent position; softmax across time gives weights summing to one. Zero-initialized scorer parameters make the initial operation equivalent to mean pooling.

Channel attention acts before the GRU. It averages the 64 encoder channels across frequency and time, then uses a 64→8→64 MLP with ReLU and twice-sigmoid gates. The final gating layer is initialized to zero, giving a scaling factor of one initially. This module adds **1,096 parameters**. Temporal, channel and combined attention are lightweight adaptations, not exact reproductions of every configuration in the related literature.

## Capacity controls

Two scratch-model controls use the same CNN encoder architecture, trained jointly with their heads and have **252,036 total parameters** each. Both use a 512→406→50 MLP with ReLU and dropout. A pooled MLP transforms the time-averaged 512-dimensional representation. A tokenwise MLP transforms each encoded time position before averaging its class logits. They use the 60-epoch scratch-training recipe and seeds 101/202/303. These controls match parameter count approximately; they do not match recurrence, computation or optimization behavior exactly.

This is the first of two MLP comparisons. It changes the classifier on jointly trained CNN features; the later PANNs-head comparison changes an activation on fixed pretrained embeddings. Their results address different questions.

## Training and augmentation

The main scratch-model comparison uses seeds **101, 202 and 303**, 60 epochs, batch size 32 and AdamW with learning rate 0.001 and weight decay 0.0001. Cosine scheduling spans the declared 60 epochs. The global gradient norm is clipped at 5. Models minimize multiclass cross-entropy on raw logits; [checkpoint selection](evaluation.md) uses development cross-entropy.

Augmentations are applied during training:

| Variant | Implementation |
| --- | --- |
| Gaussian noise | Per-example probability 0.5; waveform noise scaled to an added-noise SNR sampled uniformly from 10–30 dB |
| Waveform Mixup | Convex combinations of two training waveforms, with the same coefficient weighting their cross-entropy losses |
| Time/frequency masking | One frequency mask of width 0–8 and one time mask of width 0–24 per example; normalized feature values are set to zero |

Gaussian noise is mean-centered and scaled by the foreground and noise RMS. At SNR s, its amplitude multiplier is `RMS(foreground) / RMS(noise) × 10**(-s/20)`. Mixtures are not clipped or peak-normalized. The SNR measures added interference relative to the original waveform, which can already contain ambient noise. The masking implementation does not include time warping.

## Classical reference classifiers

The temporal-statistics logistic classifier summarizes each of the 64 log-Mel bands by its mean and population standard deviation over time, producing 128 features. These summaries are standardized with training-only statistics. A 50-class affine softmax model is fitted with SciPy L-BFGS-B to mean cross-entropy plus an L2 weight penalty; the intercept is unpenalized. The L2 grid is 0.001, 0.01 and 0.1, selected by minimum development cross-entropy. The saved experiment selects 0.001 and reaches 42.21% accuracy.

The cepstral reference applies an orthonormal DCT-II across the 64 log-Mel bands and retains 20 coefficients. Their temporal means and standard deviations form 40 features. After training-only standardization, LinearSVC fits a linear classifier using C in {0.01, 0.1, 1, 10}. Selection maximizes development accuracy, then macro-F1, then favors smaller C. The saved experiment selects C=1 and reaches 40.20% accuracy. This is a DCT-of-log-Mel statistical representation, not a reproduction of Piczak's feature extraction and SVM experiment.

## Feature and sampling study

The feature study crosses three sample-rate/passband pairs—16 kHz/8 kHz, 32 kHz/8 kHz and 32 kHz/16 kHz—with four representations: 64-band log-Mel, 32-band log-Mel, 40 DCT coefficients of 64 log-Mel bands, and 64-band log-linear filterbank features. All cases retain 32-ms windows and 16-ms hops, giving 313 time frames per five-second clip. The DCT variant has no liftering or separate energy replacement for coefficient zero.

The resulting **12 configurations × three seeds** use a different **26,658-parameter globally pooled CNN**. Its channel and frequency averaging produces a 64-dimensional classifier input, so changing the number of frequency bins does not change parameter count. Training uses 60 epochs, AdamW, cosine scheduling, training-only normalization and minimum development-loss selection. The study evaluates representation changes within this model; its lower accuracy is not a matched estimate of the cost of replacing the main 49,058-parameter baseline's frontend.

## Encoded-sequence interventions

The temporal-order diagnostic holds six fitted CNN/CRNN models fixed: two architectures and three training seeds. Each clip is encoded once into 64 × 8 × 39 features. Identity, reversal, a circular shift of 19 positions and two fixed random permutations are applied to the 39 encoded time positions before classification. No waveform is reversed and no model is retrained.

For the CNN, averaging the same encoded tokens is permutation-invariant in real arithmetic. The implementation records floating-point residuals separately from prediction agreement. For the CRNN, the permuted tokens pass through the existing GRU, so the diagnostic measures dependence on the encoded sequence order. Such dependence does not by itself establish that a particular natural acoustic cue explains the CRNN's clean-accuracy difference.

## Frozen PANNs reference

The pretrained reference uses the official AudioSet-pretrained **Cnn14_16k** implementation and checkpoint. Its frontend uses a 512-sample window, **160-sample hop**, 64 Mel bands and a 50–8,000 Hz frequency range. The scratch frontend's 256-sample hop is not substituted into the pretrained encoder.

All encoder parameters have gradients disabled, and the model is in evaluation mode during extraction. Its **2,048-dimensional embedding**, rather than its original 527 AudioSet output probabilities, is supplied to a StandardScaler fitted on training embeddings and an L2 multinomial logistic classifier. Settings are C=1, L-BFGS, maximum 3,000 iterations and tolerance 1e-6. The original head converged in 137 iterations and has **102,450 coefficients and biases**. The loaded frozen model contains **81,033,231 parameters**.

The [matched-head comparison](../notebooks/02_transfer_heads.ipynb) tests `2048→49→50` heads with either an identity or ReLU activation. Both have 102,901 parameters, matched initialization and minibatch ordering within each seed, and 100 fixed AdamW epochs with learning rate 0.001, weight decay 0.0001 and batch size 32. They have no dropout or learning-rate scheduler. Final-epoch weights are evaluated without development checkpoint selection. This comparison isolates the activation within that parameterization; ordinary L-BFGS logistic regression is a separate optimization reference.

## Noise-adapted linear heads

The adaptation studies keep the PANNs encoder and clean-training scaler fixed. Each of the same 1,200 foreground training clips contributes clean weight 0.50 and two augmented variants of weight 0.25 each. Total training weight remains 1,200, retaining the classifier's effective regularization scale. Both stages use C=1 L2 multinomial logistic regression, L-BFGS, tolerance 1e-6 and maximum 3,000 iterations. In a weighted-mean loss formulation, the quadratic penalty is `||W||² / (2*C*sum(sample_weight))`; keeping C alone fixed would not preserve this balance if the total training weight changed.

The first stage fits a clean reference and a Gaussian-adapted head. The Gaussian head uses two waveform-level variants per target at SNRs drawn uniformly from 10–30 dB. The second stage fits one recorded-background head, reusing those exact per-target SNR assignments. Its six DEMAND training recordings are DKITCHEN, NFIELD, OMEETING, PRESTO, SPSQUARE and TBUS. Each target receives two distinct recordings, with 200 uses per background per variant and 40 uses per ordered distinct background pair. A separate random generator samples valid five-second crop positions independently of the foreground label.

A distinct set of six recordings supplies the later evaluation grid. The controlled change is the complete training-noise distribution, including spectrum, temporal structure and possible competing events; diversity alone is not isolated. Neither stage fine-tunes the encoder. See the [Gaussian](../notebooks/05_gaussian_adaptation.ipynb) and [recorded-noise](../notebooks/06_recorded_noise.ipynb) notebooks.

## External recordings and duration handling

The external study uses the manually verified test partition of **FSDnoisy18k v1.0**. Six category mappings connect Clapping, Engine, Fireworks, Rain, Wind and Walk_or_footsteps to their ESC-50 labels. Five recording IDs found in ESC-50 are excluded, leaving 259 recordings. Existing PANNs and three noise-CNN fits are evaluated without fitting, added interference or output-class masking.

Each complete recording is decoded, averaged to mono and resampled to 16 kHz. Consecutive non-overlapping five-second windows cover the audio; the final fragment is right-padded. Each window produces probabilities over all 50 labels. These probabilities are averaged within the recording with weights equal to the actual unpadded sample counts, and one argmax produces the recording label. The saved panel contains 534 windows. This is probability aggregation, not logit averaging or majority voting.

The duration diagnostic holds the original PANNs scaler and head fixed on 105 recordings shorter than five seconds. Right-padding and left-padding produce 80,000-sample inputs. Native-duration inference uses the unpadded waveform and processes one recording at a time to avoid batch padding. Its minimum input is 4,960 samples (0.31 seconds): centered STFT with hop 160 must produce at least 32 frames for the encoder's five temporal downsampling stages. Three shorter recordings are excluded only from the native comparison, leaving 102 paired recordings; they remain in the original external benchmark.

The diagnostic first replays the original right-padded probabilities and includes ten identical five-second input controls. Native duration changes boundary behavior and temporal aggregation as well as the amount of zero input, so a prediction difference does not isolate one padding mechanism. See the [external](../notebooks/03_external_recordings.ipynb) and [duration](../notebooks/04_input_duration.ipynb) notebooks.

## Implementation

Core code is in [models.py](../src/esc50_sound/models.py), [audio.py](../src/esc50_sound/audio.py), [training.py](../src/esc50_sound/training.py) and [augmentation.py](../src/esc50_sound/augmentation.py). [Experiment runners](../scripts/README.md) handle pretrained extraction and supplementary evaluations. Saved model folders contain usage instructions and available weights.
