# Research Question, Completed Findings and Future Directions

> Current delivery scope: complete the supplied COMP9444 course checkpoint. The [course report](checkpoint_20261105/Checkpoint_Report_EN.md) is the primary deliverable. The material below records completed extensions and possible future questions; these are not active tasks or additional course requirements.


Updated 7 November 2026, after the controlled training-noise-distribution experiment.

## Research identity

**Which model and training choices improve environmental sound classification under the available labelled-data regime, and how far do their benefits transfer across acoustic conditions?**

The project began with CNN/RNN environmental-sound classification and developed into a comparative study of architecture, pretrained representations and noise adaptation. The independently implemented 49,058-parameter CNN is the original numerical baseline. The latest controlled experiment uses the existing Gaussian-adapted PANNs readout as its primary comparator. These are different comparison levels, not conflicting baseline definitions. The separate team implementation is outside this study.

Three questions organize the completed evidence:

1. Do recurrence and attention provide useful gains over the independent CNN, considering the available capacity controls?
2. How does a frozen pretrained audio system compare with the tested scratch-trained alternatives?
3. How does the training-interference distribution affect recognition when the evaluation interference changes?

This organization reflects the actual evolution of the project. The experiments were not all preregistered together. The contribution is empirical comparison and characterization of transfer limits, not a novel architecture or a state-of-the-art benchmark claim.

## What the completed studies establish

| Stage | Observed evidence | Interpretation boundary |
| --- | --- | --- |
| CNN, BiGRU and attention | Three-fit mean accuracy 72.61%, 73.87% and 74.37%; attention adds an uncertain 0.50 pp | Modest development differences, not universal architectural or temporal-mechanism claims |
| Capacity and head controls | Pooled/token MLP means 72.28%/69.68%; matched PANNs ReLU heads did not improve the evaluated linear alternative | No evidence requiring added nonlinearity in these configurations; not a universal MLP rejection |
| Frozen PANNs alternative | Historical linear head reaches 91.21% accuracy and 91.10% macro-F1 | Full-system comparison changes pretraining, capacity, frontend and architecture together; no isolated pretraining effect |
| External six-category diagnostic | PANNs 45.17% versus noise-CNN mean 20.08% on 259 mapped recording IDs, all 50 outputs active | Narrow exposed external panel; not an independent full-50-class or pretraining-disjoint benchmark |
| Duration/input intervention | Native versus right-padding difference +5.88 pp, interval [−0.98, 12.75] on 102 eligible recordings | Input sensitivity, without established accuracy benefit or a causal padding mechanism |
| Matched Gaussian adaptation | Frozen-PANNs clean versus Gaussian heads: 69.85% versus 83.92% at shared Gaussian 10 dB; +14.07 pp [10.20, 18.32] | A readout-level intervention repairs errors under this tested noise condition; it does not create a noise-invariant encoder |
| Initial recorded transfer | Gaussian-adapted minus clean mean +1.05 pp [−0.54, 2.59] across six fixed 10-dB segments | Uncertain transfer on that panel, not equivalence or proof of failure |
| New recorded-background adaptation | Recorded versus Gaussian heads: 84.79% versus 83.43% across the reserved recorded grid; +1.36 pp [0.41, 2.32] | Positive conditional training-distribution effect, accompanied by a substantial Gaussian tradeoff |

Intervals refer to their respective stated protocols. They do not all describe the same uncertainty or population. The historical architecture fits, augmentation fits, head fits and inference-only diagnostics are not interchangeable replications.

Sources: [complete executed notebook](../notebooks/ESC50_Results.ipynb), [original comparison](checkpoint_20261105/model_comparison.csv), [head controls](transfer_followup_20261105/README.md), [external evaluation](external_fsd_20261105/README.md), [duration diagnostic](duration_probe_20261105/README.md), [Gaussian adaptation](noise_transfer_20261106/README.md), [new distribution comparison](noise_distribution_20261107/README.md).

## The decisive new comparison

The latest question was fixed before its new fit and predictions: with the representation and linear objective unchanged, does replacing Gaussian training interference with recorded backgrounds improve recognition under other recorded backgrounds?

Exactly one new recorded-adapted logistic head was fitted. It shares the frozen PANNs encoder, clean-training scaler, 1,200 foreground targets, C=1 objective and total training weight with the existing Gaussian head. It reuses the exact saved per-target training SNR values. Both adapted heads allocate 0.50 weight to clean and 0.25 to each of two waveform variants; total weight is 1,200. The noise family is the intended intervention.

Training and evaluation use six different DEMAND recordings each, with one recording from each of six metadata categories per role. Previously exposed NRIVER, OHALLWAY and TCAR are excluded. Evaluation uses two fixed windows per recording and 0/10/20 dB, fully crossed with the same 398 development targets. The 14,328 mixtures contain 398 foreground clips from 295 sources, not 14,328 independent recordings. Full settings are in the [fixed protocol](noise_distribution_20261107/protocol.json).

| Evaluation | Clean-trained LR | Gaussian-adapted LR | Recorded-adapted LR |
| --- | ---: | ---: | ---: |
| Clean | 91.21% | 91.96% | 91.71% |
| Gaussian 10 dB | 69.85% | 83.92% | 73.62% |
| Reserved recorded grid | 82.88% | 83.43% | 84.79% |

**The main finding is a distribution-dependent tradeoff.** Recorded training improves the fixed recorded panel over Gaussian training by **+1.36 pp [0.41, 2.32]**, but loses **10.30 pp [6.95, 13.84]** at Gaussian 10 dB relative to the Gaussian head. Their clean difference is −0.25 pp [−1.86, 1.45], which does not establish clean equivalence or noninferiority. The recorded head retains some Gaussian benefit over clean training (+3.77 pp); it simply does not retain the Gaussian-trained head's full benefit.

Recorded-grid mean per-condition macro-F1 also improves, from 83.78% to 85.18%, and mean NLL falls from 0.772 to 0.699. All six recording-level mean differences are positive; leave-one-recording-out point estimates range from +1.09 to +1.52 pp. These are consistency diagnostics within the selected panel, not independent confirmations or environmental-population inference. The apparently larger gain at lower SNR is descriptive, not a confirmed interaction.

The paired primary interval resamples 295 foreground source groups after retaining all conditions within clips. It conditions on the fitted heads, augmentation realization and selected background panel. It does not cover new training draws or future environments. Foreground targets remain historically exposed; the background holdout concerns adaptation-source identities, not verified independence from AudioSet pretraining.

## What remains unexplained

Gaussian-to-recorded substitution changes spectrum, temporal structure, event content and source diversity together. The intervention therefore does not isolate any one of them. PSD summaries characterize the selected background signals; they do not prove energetic masking, informational masking or semantic shortcut reliance. A frozen encoder also precludes claiming that the new head learned an invariant representation.

Backgrounds can contain other ESC-50 events. The task is recovery of the designated foreground label, including at 0 dB, rather than complete polyphonic tagging or a guarantee that one event is perceptually dominant. No listening-based neutrality screening was performed. Equal full-window SNR does not mean equal perceptual difficulty, and these additive mixtures do not reproduce all aspects of naturally recorded scenes.

The largest observed performance differences come from choosing the pretrained system and changing the training exposure, while the tested attention and nonlinear-head changes have less convincing incremental evidence. This motivates the study's emphasis; it is not proof that attention or fine-tuning cannot help.

## Completion and worthwhile continuation

The original checkpoint now has a completed baseline analysis, implemented improvements and alternatives, appropriate metrics, paired comparisons and a discussion of favorable and unfavorable findings. This experimental cycle is complete. All **40 scientific figures** remain in the complete executed notebook, with the new four also available as editable plotting code in the dedicated notebook. No additional fit or model search was triggered by the result.

Completion of this cycle does not mean every research question is resolved. The next question depends on the intended contribution:

- **Portability:** Evaluate the three existing heads on independently sourced foregrounds and additional background recordings. This tests whether the observed tradeoff persists beyond exposed targets. A carefully mapped partial-class study can be useful if its claims remain partial; a fresh 50-class dataset is not a blanket prerequisite for further research.
- **Mechanism:** Compare original recorded backgrounds with phase-randomized, PSD-matched surrogates while fixing SNR, source assignment and training objective. This can investigate sensitivity beyond the preserved power spectrum, but does not isolate semantic or perceptual masking. Reusing current panels would be exploratory, rather than newly blinded confirmation.
- **Engineering tradeoff:** Mixed Gaussian/recorded training is now a motivated candidate, but requires an explicit target operating distribution and matched augmentation budget. The present results do not show that it preserves both advantages. There is no justified universal default without such an objective.

Another attention sweep, MLP search or encoder fine-tuning run would answer a different question and needs its own hypothesis and controls. None is required to finish the current report, and an unfavorable or uncertain result is not by itself evidence that a larger model is necessary.

The [primary-source review](noise_distribution_20261107/literature_review.md) records relevant prior augmentation, transfer and robustness studies, reading depth and mismatches with this experiment. It establishes context without importing published scores as this project's evidence.
