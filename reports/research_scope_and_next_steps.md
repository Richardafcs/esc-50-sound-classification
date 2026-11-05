# Research Question, Findings and Next Phase

Updated 6 November 2026, after completing the matched PANNs noise-adaptation experiment.

## Research identity

This is a comparative environmental-sound classification study that has developed into a focused investigation of noise adaptation. Its central question is:

**Which model and training choices improve environmental sound classification under the available labelled-data regime, and how far do their benefits transfer across acoustic conditions?**

Three subquestions organize the work:

1. Do recurrent modelling and attention provide useful gains over the independently implemented CNN, considering the available capacity controls?
2. How does a frozen pretrained audio system compare with the evaluated scratch-trained alternatives?
3. Which improvements from noise-aware training remain evident when the interference distribution changes?

The architecture experiments answer the original CNN/RNN brief. The controlled readout experiments provide a sharper continuation question. This organization is retrospective and consistent with the project's evolution; it does not imply that every experiment was preregistered together.

The numerical baseline is the independent 49,058-parameter research CNN. The separate team implementation is outside this study. The supplied course checkpoint is substantially covered, but course completion is not a reason to terminate a worthwhile research question. Conversely, a request to continue research is not evidence that additional model complexity is useful.

## What has been established

| Finding | Evidence | Claim boundary |
| --- | --- | --- |
| The evaluated recurrent/attention modifications produce modest development differences | CNN 72.61%, BiGRU 73.87%, channel attention 74.37%, over three fixed fits; attention's additional gain is uncertain | No universal architecture ranking, isolated temporal mechanism or proof that attention cannot help |
| The pretrained system has the largest observed clean-accuracy advantage | Frozen PANNs plus the historical linear head reaches 91.21%; the external six-category panel gives 45.17% versus noise-CNN mean 20.08% | Pretraining, capacity, frontend and architecture change together; external coverage and pretraining independence are limited |
| Gaussian sensitivity can be substantially reduced without updating PANNs | Matched clean versus Gaussian-adapted heads give 69.85% versus 83.92% at 10 dB; difference +14.07 pp, conditional interval [10.20, 18.32] | Shows a correctable component of readout error in this setting, not an invariant representation or information-loss mechanism |
| Broad recorded-background benefit remains unresolved | Six-condition recorded mean 89.07% versus 90.12%; primary difference +1.05 pp, conditional interval [-0.54, 2.59] | An uncertain difference is neither equivalence nor proof of failed transfer |

The matched heads change clean accuracy from 91.21% to 91.96%, a +0.75 pp difference with interval [-0.51, 2.06]. This does not establish clean superiority or noninferiority. The original default classifier remains available; the adapted head is supported as a candidate for the tested Gaussian condition, not as a universally preferred replacement.

Sources: [architecture comparison](checkpoint_20261105/model_comparison.csv), [matched transfer/head study](transfer_followup_20261105/README.md), [external evaluation](external_fsd_20261105/README.md), [completed noise-adaptation experiment](noise_transfer_20261106/README.md).

## What the new result changes

The missing frozen-PANNs/noise-adaptation comparison has now been completed. Repeating the same request for more attention, another MLP or a larger encoder would bypass the question left by that experiment.

The recorded panel was much less damaging to the clean-trained head than Gaussian noise at the same 10 dB: its clean-to-recorded mean loss is approximately 2.14 pp, compared with 21.36 pp for Gaussian. These are descriptive differences computed from the saved accuracies. Equal waveform SNR does not make the tested conditions equally difficult. This does not invalidate the fixed-10-dB comparison, prove a ceiling, or identify why transfer is uncertain.

The panel contains only three background recordings and six fixed segments, crossed with the same 398 target clips. It does not provide 2,388 independent target recordings or a representative sample of all acoustic environments. Target-source bootstrap intervals condition on the chosen backgrounds and fitted models. Backgrounds can contain competing events; the label denotes the designated foreground recording, not all events in a scene.

All official ESC-50 folds and the existing external panel have historical exposure. New-target generalization and conditional transfer to new background sources are separate claims. The latter can be studied without inventing a requirement to collect an untouched full-50-class dataset. The former requires additional target-source evidence, with its own label and exposure protocol.

## The next research question

**With the representation and linear-classifier objective fixed, how does Gaussian versus recorded-background training exposure affect recognition under previously unseen recorded interference?**

This is a question about the **training-noise distribution**. A recorded-versus-Gaussian comparison also changes spectral structure, temporal structure and possibly diversity. It cannot isolate diversity or coverage breadth alone. It is an established-method comparison, not a proposed new architecture.

### Minimal controlled continuation

| System | Status | Role |
| --- | --- | --- |
| Clean-trained PANNs linear head | Completed | Reference |
| Gaussian-adapted PANNs linear head | Completed | Existing noise-training comparator |
| Recorded-background-adapted PANNs linear head | Not fitted | New intervention |

Keep the frozen encoder, class mapping, target split, clean-training scaler, classifier family, regularization and total per-target training weight fixed. Match the augmentation weight, number of training variants and training SNR exposure to the Gaussian comparator. Only the training-noise distribution should intentionally change. Existing training-exposure records make the Gaussian SNR assignments inspectable.

Reserve new background recordings/environments for evaluation before fitting the new head or viewing its predictions. Different channels or time cuts from a training recording do not constitute independent environments. The six already-exposed diagnostic segments must not be relabelled as a fresh evaluation panel.

Use a predefined SNR grid and report the full degradation curves, clean results and background-specific results. Do not choose evaluation SNR from final accuracy in order to force equal damage. Any optional severity calibration must use separately designated calibration data and define a different estimand explicitly.

Before execution, fix the background-source roles, foreground-label semantics, mixture rule, SNR grid, primary weighting over targets/conditions, interval scope, convergence handling and stopping rule. No arbitrary practical gain threshold or acceptable clean-loss margin is supplied here. Exact background choices and the next execution protocol have not yet been fixed, and no new head is trained during this assessment.

### Decision value and stopping

- **A recorded-training advantage on held-out backgrounds:** supports a useful training-distribution intervention within this fixed system. It does not prove that diversity alone caused the gain, that the encoder is invariant, or that the result holds for new target sources.
- **An uncertain or unfavorable difference:** means this particular intervention has not demonstrated an advantage under the specified protocol. It does not establish equivalence, irreversible information loss or the necessity of fine-tuning.
- **A clean/noisy trade-off:** must remain visible; choosing a default requires an actual operating objective rather than a favorable pooled score.

Complete one fixed comparison and report the result regardless of direction. Do not expand architectures, backgrounds or SNR conditions in response to the evaluation score. Encoder fine-tuning becomes a separate future question if evidence and an intended use justify changing the representation. New-target-only inference would address another generalization axis; it is not a substitute for this controlled contrast.

## How the existing research should be presented

The main report should establish the baseline and evaluation population, summarize the architectural comparisons, introduce the pretrained alternative, present controlled noise training/readout adaptation, and explain the transfer limits. The recorded-background primary interval belongs in the main argument alongside the positive Gaussian result.

Capacity controls should appear where they constrain architectural interpretation. Full frontend sweeps, attention variants, head variants, token-order interventions, duration/padding diagnostics and segment-level tables belong in supporting sections. All **36 scientific figures** remain in the [complete executed notebook](../notebooks/ESC50_Results.ipynb); the figure count is an evidence inventory, not a count of independent findings.

Report consolidation and the next protocol can progress together. The immediate deliverable is one current research argument and one answerable continuation question, not another unconnected experiment. The remaining strategic choice is how much weight a later phase should give to new-target generalization versus conditional noise transfer; the current priority is the latter because it permits a controlled intervention with the existing representation.

No claim of sample efficiency across dataset sizes, measured deployment efficiency, state-of-the-art performance or algorithmic novelty is made. [PANNs Section V](https://arxiv.org/html/1912.10211) already describes frozen-feature transfer and fine-tuning; this project evaluates concrete systems and their limits under its stated protocols.
