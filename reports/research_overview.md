# ESC-50 research overview

[Complete executed notebook](../notebooks/ESC50_Results.ipynb) · [Executed transfer follow-up](../notebooks/ESC50_Transfer_Followup.ipynb)

| Track | Executed evidence | Finding | Scope |
| --- | --- | --- | --- |
| Classical baselines | Logistic regression and cepstral-statistics Linear SVM | 42.21% and 40.20% development accuracy | Feature representations differ |
| CNN / BiGRU | Three fixed fits each | 72.61% versus 73.87% mean accuracy | Small development effect, not universal recurrence benefit |
| Earlier MLP capacity controls | Pooled and tokenwise MLP, three 60-epoch fits each | 72.28% and 69.68% | Restored records; not PANNs heads |
| Attention | Three variants × three seeds | Channel attention 74.37%; +0.50 pp over BiGRU, interval spans zero | No stable advantage established |
| Augmentation | Noise, masking and waveform Mixup | Clean accuracy 73.12%, 71.86%, 71.61% | Configuration-specific, including negative results |
| Feature / sampling | Twelve settings × three seeds, separate pooled CNN | Log-Mel32 improves over 64 in this track | Different model; not a general frontend rule |
| Historical synthetic robustness | Three families × three seeds × six conditions | At 10 dB noise-trained CNN 61.31% versus clean-trained CNN 23.37% | Legacy draws, kept separate |
| Temporal interventions | Six fixed models × five encoded-token interventions | BiGRU reversal −5.03 pp; CNN labels unchanged | Post-encoder sensitivity, not raw-audio causality |
| Frozen PANNs transfer | One fixed LR head, extraction/fit reproduced | 91.21% accuracy /91.10% macro-F1 | External pretraining/capacity/frontend confounded |
| New matched robustness | PANNs LR plus three noise-CNN fits, clean/10 dB | 69.85% versus 61.56%; +8.29 pp [1.83, 14.74] | Higher absolute noisy accuracy, larger clean-to-noisy loss |
| New matched PANNs heads | Six fixed 100-epoch fits, Identity versus ReLU | Linear 91.12%/68.76%; ReLU 89.78%/67.34%, clean/noisy | No gain for this ReLU configuration |
| External mapped FSD evaluation | Four frozen systems, 259 recording IDs, six target categories, all 50 outputs | PANNs 45.17% versus noise-CNN mean 20.08%; difference +25.10pp | Known ESC-50 IDs excluded; pretraining and perceptual overlap unresolved |
| Short-recording input policy | Same 102 eligible recordings, right/native/left, zero fits | Native 33.33% versus right 27.45%; +5.88pp [−0.98,+12.75]; many prediction flips | Sensitivity established; accuracy benefit and mechanism uncertain |
| Gaussian readout adaptation | Two converged heads, fixed encoder/objective | Gaussian 10 dB 69.85%→83.92%; initial recorded mean +1.05pp [−0.54,2.59] | Matched-noise benefit, uncertain transfer on the initial panel |
| Recorded readout adaptation | One converged head; six reserved recordings × two windows × three SNRs | Recorded grid 84.79% versus Gaussian head 83.43%; +1.36pp [0.41,2.32]; Gaussian 10 dB −10.30pp | Conditional training-distribution effect with a substantial cross-noise tradeoff |

The original 65-task matrix includes 60 neural fits, two classical fits and three stress tasks. Earlier CNN/BiGRU references overlap later records and are not extra independent replications. The six recovered earlier MLP fits and six new transfer-head fits are separate stages.

The evaluated research CNN has 49,058 parameters. The separate team implementation is outside this study.

## Current decision

The controlled noise-adaptation cycle is complete. Recorded-background exposure improves designated-foreground recovery on the reserved recorded panel, while giving up much of the Gaussian head's matched-noise benefit. Report this tradeoff alongside clean performance, the original architecture comparison and all negative head/input-policy findings. No universal default, isolated acoustic mechanism or blind foreground-generalization claim follows.

The current delivery is the completed course checkpoint. Use the [English course report](checkpoint_20261105/Checkpoint_Report_EN.md) as the primary narrative. The [extended research scope](research_scope_and_next_steps.md) and [latest experimental evidence](noise_distribution_20261107/README.md) remain supplementary; no additional research is scheduled for this delivery.

## Reading and implementation

The [focused noise-distribution source review](noise_distribution_20261107/literature_review.md) adds primary robustness and recorded-interference context.

The [26-publication literature map](checkpoint_20261105/literature_review.md) distinguishes reading depth, original task, and implemented adaptation. PANNs' frozen one-/three-layer transfer experiments motivate the classifier question; this project uses a different 16 kHz checkpoint, split and optimization recipe. Published benchmark scores are not included as our measured predictions.

[Earlier MLP evidence](mlp_controls_20261105/README.md) · [New matched experiment](transfer_followup_20261105/README.md) · [Original matrix tables](evidence/matrix_20261028/analysis/)

[External recording evaluation and error analysis](external_fsd_20261105/README.md).

The bounded input-policy diagnostic is complete. Keep the original classifier and right-padding default; do not select a native/left policy from this exposed panel. See the [complete diagnostic evidence](duration_probe_20261105/README.md).
