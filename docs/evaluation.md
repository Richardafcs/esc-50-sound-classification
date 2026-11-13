# Evaluation protocol

## Dataset roles

The main comparison uses the official ESC-50 fold assignments with metadata source exclusions:

| Role | Fold allocation | Retained clips |
| --- | --- | ---: |
| Training | Folds 1–3, excluding held-out source identifiers | 1,200 |
| Development | Fold 4, excluding fold-five source identifiers | 398 |
| Designated test partition | Fold 5 | 400 |

The development population represents **295 original recording sources**. Two fold-four clips share source identifiers with fold five and are excluded. Training-fitted spectrogram normalization and embedding standardization are reused unchanged for evaluation. The available metadata establishes source separation for these roles, rather than a comprehensive perceptual-duplicate or pretraining-overlap analysis.

All official folds had historical exposure during the project. The headline CNN, CRNN and PANNs results are therefore **development results**, not an untouched final-test benchmark. Older fold-five scores and fixed-recipe five-fold characterizations remain in the notebook as historical results and are kept separate from the main comparison.

## Comparison levels

The project contains several related experiments rather than one ranking in which every row changes the same factor:

| Experiment | Reference and changed factor | Evaluation unit |
| --- | --- | --- |
| Classical references | Log-Mel statistics/logistic and cepstral statistics/SVM | Development clip |
| Scratch architecture and augmentation | 49,058-parameter CNN; recurrence, attention or training augmentation | Development clip, three fixed fits |
| Scratch MLP controls | Approximately capacity-matched pooled/tokenwise heads on jointly trained CNN features | Development clip, three fixed fits |
| Feature/sampling study | 26,658-parameter globally pooled CNN; 12 frontend configurations | Development clip, three fixed fits per configuration |
| Encoded-order diagnostic | Fixed CNN/CRNN; post-encoder token permutations | Paired development clip |
| PANNs reference and head ablation | Frozen pretrained embeddings; logistic classifier or matched Identity/ReLU heads | Development clip |
| Noise-adaptation studies | Fixed encoder/scaler/objective; changed training interference | Foreground clip across fixed noise conditions |
| External and duration studies | Existing models and input policies; no fits | External recording |

The original 65-task matrix contains 60 neural fits, two classical tasks and three robustness tasks. Each classical task includes its own hyperparameter search rather than a single optimizer fit. Later PANNs experiments and recovered scratch MLP controls belong to separate stages. Shared CNN/CRNN reference fits are not counted again as independent replications.

## Fits and checkpoint selection

Scratch models use three fixed seeds: **101, 202 and 303**. Each main fit runs for 60 epochs, and its checkpoint is the epoch with minimum development cross-entropy; an exact tie retains the earlier epoch. Family results are the mean and sample standard deviation over the three fits. An exported seed-101 checkpoint can differ from that mean.

The original frozen PANNs reference uses one converged logistic-regression fit. It does not receive a three-seed error bar. The later matched linear/ReLU comparison uses three seeds per head and evaluates the final weights after 100 fixed epochs, without checkpoint selection. Repeated extraction of a fixed head is not an independent training replication.

Classical references use their own development selection rules: minimum cross-entropy over the temporal-statistics logistic L2 grid, and maximum accuracy followed by macro-F1 and smaller C for the cepstral SVM. Their results include hyperparameter selection and are not three-seed neural means. The feature study uses minimum development cross-entropy within each of its 36 fits; its different pooled model prevents direct attribution of score differences to the frontend alone.

## Metrics

- **Accuracy** is the proportion of clips whose predicted label matches the target.
- **Macro-F1** averages the per-class F1 scores equally over the 50 classes in the main task, with zero-division values set to zero.
- **Confusion matrices and class results** identify uneven performance and recurring errors. With only seven or eight development clips per class, these slices are exploratory.
- **Percentage-point differences** subtract accuracies on their percentage scale. A change from 72.61% to 73.87% is +1.26 percentage points.

The reported seed standard deviations describe variation between the three fixed fits. They are not confidence intervals for accuracy on a new population. Softmax probabilities are not calibrated, and the task does not evaluate general unknown-class rejection.

## Paired uncertainty estimates

Model comparisons use predictions on the same recordings. The paired analysis records both corrections and newly introduced errors. Current development source-bootstrap intervals use **4,000 resamples**, drawing `src_file` groups with replacement and keeping all associated clips together. Each draw uses the actual sampled clip count as its denominator. Multi-seed comparisons average fixed-fit differences within clips before source resampling. The older fold-five comparison retains its original 2,000-resample calculation and separate 400-clip, 310-source population.

These intervals describe foreground source-sampling uncertainty conditional on the fitted models and specified evaluation conditions. They do not cover all possible training seeds, noise draws, background environments or accumulated model-selection decisions. Reusing one foreground clip across noise conditions does not create independent target recordings. Secondary comparisons are descriptive unless a particular protocol specifies otherwise.

## Gaussian-noise panels

Two panels use different corruption realizations and must not be mixed:

| Panel | Compared systems | Relevant 10-dB accuracy |
| --- | --- | --- |
| Original three-seed robustness panel | Clean CNN, BiGRU and noise-trained CNN | Clean CNN 23.37%; noise-trained CNN 61.31% |
| Later matched transfer panel | Frozen PANNs reference and the same three noise-CNN fits | PANNs 69.85%; noise-trained CNN 61.56% |

Within each panel, systems receive matched corrupted recordings. The original panel includes clean audio, Gaussian SNRs of 20, 10 and 0 dB, and gains of ±6 dB. The later transfer panel uses the same waveform tensor before each system's own frontend. Full-window signal and added-noise RMS determine SNR; mixtures are not clipped or peak-normalized.

Absolute noisy accuracy and loss relative to clean accuracy answer different questions. In the matched transfer panel, PANNs has higher noisy accuracy but loses more percentage points from its stronger clean result. See [the matched comparison](../reports/results/transfer_heads/README.md).

## Encoded-order diagnostic

Each fixed fit is evaluated under identity, reverse, circular shift and two shuffled sequences of the same encoder outputs. Effects are accuracy under the intervention minus accuracy under identity, paired by clip and averaged across the three fixed fits. Source-bootstrap intervals condition on those fits and permutations.

The mean-pooled CNN's encoded-token invariance is a mathematical property of its aggregation; numerical residuals and prediction agreement are reported separately. BiGRU sensitivity to reversal demonstrates dependence on this representation's order. It is not a waveform robustness experiment, proof of a useful natural-sound mechanism, or evidence that order alone caused the CNN–BiGRU accuracy difference.

## Recorded-background evaluations

The first noise-transfer experiment compares clean and Gaussian-adapted PANNs heads on **three DEMAND recordings**—NRIVER, OHALLWAY and TCAR—with five-second windows starting at 30 and 150 seconds, all at 10 dB. Its primary result equally averages the six condition differences within each foreground clip and then averages over clips. The unchanged original logistic head is an additional reference, not another independent clean-model replication.

The later training-distribution comparison uses **six training recordings and six distinct evaluation recordings**. Its reserved evaluation recordings are DLIVING, NPARK, OOFFICE, PCAFETER, STRAFFIC and TMETRO, with five-second windows starting at 60 and 210 seconds, crossed with SNRs of 0, 10 and 20 dB. The previously used NRIVER, OHALLWAY and TCAR are excluded from this stage. The recorded-grid mean equally weights all 36 conditions. The saved clean and Gaussian-adapted heads are reused; one recorded-adapted head is fitted.

The resulting 14,328 foreground/condition pairs contain only **398 distinct foreground clips** and six evaluation background recordings. Macro-F1 is computed within each condition before averaging, rather than pooling repeated targets. Source-bootstrap intervals retain the fixed background grid. They do not estimate uncertainty over a population of acoustic environments.

Recorded backgrounds can contain other sound events. Evaluation measures recovery of the designated original foreground label, not recognition of every audible event. Equal waveform SNR does not imply equal perceptual difficulty. [Recorded-noise results](../reports/results/recorded_noise/README.md) report both the recorded-panel improvement and the Gaussian-noise tradeoff.

## External recordings and input duration

The external FSDnoisy18k subset contains **259 recordings in six mapped categories** after excluding five known ESC-50 recording IDs. All 50 model outputs remain active. Each system averages its 534 window-probability vectors within their recordings, weighted by unpadded duration, before taking one label per recording. A short final fragment therefore has less weight than a full five-second window. Aggregate accuracy is calculated once per recording; six-category macro recall and macro-F1 are identified separately from the main 50-class metrics. Predictions outside the six mapped labels are errors, not discarded observations.

The paired 4,000-resample bootstrap uses unique Freesound recording IDs, equivalent to row resampling in this panel, and compares PANNs correctness with the mean correctness of three fixed noise-CNN fits. It does not capture shared-uploader or pretraining uncertainty. The labels are manually verified; this does not mean the audio is acoustically noise-free. The subset, category mapping and full-recording aggregation differ from the ESC-50 five-second evaluation, so an accuracy decrease cannot be attributed solely to dataset shift.

The duration diagnostic starts from 105 short recordings and uses the **same 102 architecture-compatible recordings** for the primary right-padding/native-duration comparison. Three clips below the encoder's 4,960-sample minimum are excluded only from the native contrast. Right- and left-padding are also reported on all 105 clips, separately from the paired 102-clip results. Baseline replay covers all 105 original padded predictions; ten identical five-second inputs provide implementation-path controls, not additional independent test recordings.

Its paired bootstrap resamples recording IDs. Native duration changes many predictions, but its +5.88-point accuracy difference over right-padding has an interval spanning zero. The default inference policy remains unchanged, and no selectively combined 259-recording score replaces the original external benchmark. See [external recordings](../notebooks/03_external_recordings.ipynb) and [input duration](../notebooks/04_input_duration.ipynb).

## Computational cost

Parameter counts describe model size, not a matched runtime comparison. Scratch MLP controls approximately match the GRU's parameter budget; PANNs linear/ReLU heads match each other exactly, but neither comparison guarantees equal FLOPs, memory traffic or latency. The PANNs classifier's 102,450 trainable parameters exclude its approximately 81-million-parameter frozen model.

Saved execution times retain their original scope. PANNs extraction/fit timings exclude the historical AudioSet pretraining; later adaptation timings reuse existing features or heads and exclude setup/downloads. These stages have different workloads and cannot establish that one full system trains or deploys faster. Any historical warm-latency measurements are limited to their saved device, batch and preprocessing scope. No end-to-end real-time or energy-efficiency comparison is established.

## Interpreting the comparison

CNN–BiGRU changes both sequence processing and parameter count. Approximate MLP capacity controls help contextualize its result without proving a single temporal mechanism. Channel attention adds only +0.50 points over BiGRU, with a paired interval spanning zero. PANNs changes external training data, capacity, architecture and frontend simultaneously, so its gain is a complete-system result rather than an isolated causal effect of pretraining.

The project establishes comparative results under the stated development conditions. It does not establish calibrated confidence, general field-noise reliability, causal explanations for individual errors or real-time deployment performance.
