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

The original 65-task matrix includes 60 neural fits, two classical fits and three stress tasks. Earlier CNN/BiGRU references overlap later records and are not extra independent replications. The six recovered earlier MLP fits and six new transfer-head fits are separate stages.

The evaluated research CNN has 49,058 parameters. The original team-repository CNN has 1,368,386 parameters and no committed result evidence; no research score is assigned to it.

## Current decision

Retain the existing PANNs logistic classifier as the strongest observed transfer system among the tested heads. Keep noise augmentation's matched-Gaussian result and all MLP/attention negative findings in the report. Describe absolute noisy accuracy, clean-to-noisy loss and conditional error transitions separately. Avoid additional architecture/width/learning-rate searches on the same 398 recordings.

The next substantive question requires compatible new recording sources and realistic noise. If such evaluation data are unavailable, retain development-only conclusions rather than relabeling an exposed fold as independent confirmation.

## Reading and implementation

The [26-publication literature map](checkpoint_20261105/literature_review.md) distinguishes reading depth, original task, and implemented adaptation. PANNs' frozen one-/three-layer transfer experiments motivate the classifier question; this project uses a different 16 kHz checkpoint, split and optimization recipe. Published benchmark scores are not included as our measured predictions.

[Earlier MLP evidence](mlp_controls_20261105/README.md) · [New matched experiment](transfer_followup_20261105/README.md) · [Original matrix tables](evidence/matrix_20261028/analysis/)
