# Research Scope, Evidence, and Next Decision

## Completed extension — 6 November 2026

The two matched head fits and fixed six-condition recorded-noise evaluation have now completed on Colab. The primary adapted-minus-clean difference is **+1.05 pp [-0.54, +2.59]**, conditional 95% source-bootstrap interval; clean change is **+0.75 pp**. [Executed evidence](noise_transfer_20261106/README.md). Background eligibility and mixture semantics are recorded in the final protocol. The design discussion below records the earlier planning stage; statements that fits or backgrounds were not yet executed/selected refer to that stage.

## Working decision

Continue the independently implemented comparative environmental-sound classification study, consistent with the original COMP9444 brief. Use the research CNN and its traceable predictions as the baseline. Develop a bounded frozen-PANNs noise-transfer extension. The separate team CNN/parallel-CNN–BiGRU experiment is outside this study; its scores and artifacts are not prerequisites.

This document organizes an evolving exploratory project retrospectively. It is not a new preregistration, a claim that all experiments were designed together, or an additional experimental result.

## Central research question

**Under the stated ESC-50 evaluation protocols, how do CNN/RNN architectures, targeted noise augmentation, and frozen pretrained audio features compare in classification accuracy, noise sensitivity, and performance on other recordings?**

The original plan centered on CNN → CRNN → Attention-CRNN → pretrained audio models. The core questions remain:

1. **Architecture:** Does recurrent temporal modelling improve the evaluated CNN system, and does attention add further value?
2. **Training exposure:** How does noise-aware training affect clean accuracy and performance under specified perturbations?
3. **Pretrained systems:** What advantages and limits arise when using frozen pretrained features instead of the evaluated scratch-trained systems?

A fixed small task-labelled training partition does not establish sample efficiency across dataset sizes. Parameter counts do not establish deployment latency or energy efficiency. Frozen-feature classification and fine-tuning are established transfer strategies, as described in [PANNs, Section V](https://arxiv.org/html/1912.10211); their use here is not a demonstrated novel algorithm.

## Experimental scope

The baseline is the independently implemented 49,058-parameter research CNN, with source-strict 1,200 training / 398 development recordings from 295 development recording IDs. Its recurrent, attention, augmentation and frozen-PANNs comparisons constitute this study. All official folds have historical exposure; the development panel is not a new untouched test set.

The separate team implementation and its results are outside the quantitative scope of this study.

## What the extended research can currently support

| Question | Existing finding | Boundary |
| --- | --- | --- |
| Architecture within the compact research family | CNN 72.61%, BiGRU 73.87%, channel attention 74.37%; attention's additional approximately 0.50-point gain is uncertain | Not a universal conclusion about architectures or an isolated temporal mechanism; not the original team-model comparison |
| Targeted noise training | In the historical matched Gaussian panel, 10 dB accuracy is 61.31% for noise-trained CNN versus 23.37% for clean-trained CNN | Perturbation-specific benefit, not general field-noise robustness; gain changes show different behavior |
| Frozen pretrained alternative | PANNs achieves 91.21% clean development accuracy and 69.85% on the new shared 10 dB panel versus noise-CNN mean 61.56% | A full-system comparison, not an isolated causal pretraining effect; PANNs loses more accuracy from clean input |
| External boundary | PANNs achieves 45.17% versus noise-CNN mean 20.08% on 259 mapped FSD recordings with known ESC-50 recording IDs excluded | Six-category external comparison with all 50 outputs active, not full-50-class independent confirmation or a directly comparable decline from 91.21% |
| Input-policy diagnostic | Native duration changes predictions and gives an uncertain +5.88-point difference on 102 eligible short recordings | Explanatory follow-up on an exposed panel; no validated replacement policy or identified padding mechanism |

Sources: [model comparison](checkpoint_20261105/model_comparison.csv), [Gaussian follow-up](followup_20261104/README.md), [matched PANNs comparison](transfer_followup_20261105/README.md), [external evaluation](external_fsd_20261105/README.md), [duration diagnostic](duration_probe_20261105/README.md). Intervals remain conditional on the fixed systems and stated evaluation populations.

## Research narrative and supporting material

The main narrative should establish the baseline/protocol, compare architectural systems, evaluate targeted noise augmentation, introduce the pretrained alternative, and explain the limits of the resulting performance claims. All quantitative comparisons in this narrative refer to the independent research study and must identify their evaluation population.

Capacity controls should be summarized where architectural interpretations require them. Full frontend sweeps, attention variants, head experiments, encoded-token interventions, detailed duration probes and the remaining figures belong in supporting sections. All 33 scientific figures remain available in the [complete executed notebook](../notebooks/ESC50_Results.ipynb); they are an evidence inventory, not 33 independent conclusions.

Against the supplied course checkpoint, the project contains baseline analysis, implemented alternatives, metrics and comparative discussion. Its quantitative narrative is grounded in the independent study, without attributing its results to the separate team model. A publishable new-method claim would require a separately justified contribution and suitable prospective evidence; neither experiment count nor higher development accuracy supplies that automatically.

## Strongest prospective extension: noise-aware frozen PANNs

Only after fixing the current research argument, the most consequential missing combination is:

| System | Clean training exposure | Noise-aware training exposure |
| --- | --- | --- |
| Research CNN | Completed | Completed |
| Frozen PANNs with a downstream classifier | Completed | Not evaluated |

A focused new question would be: **Does train-only noise exposure improve a classifier on frozen PANNs features under held-out noise recordings, while retaining clean performance?**

Keep the encoder and classifier family fixed, and compare defined head-training regimes. This can examine noise-aware head adaptation within PANNs. It does not isolate pretraining from architecture, study encoder fine-tuning, or by itself constitute a novel method. The cross-system table is a set of system/condition combinations, not a fully controlled causal factorial experiment.

Before fitting anything, define the target noise conditions, training/evaluation noise-source separation, target-recording exposure, mixture label semantics, classifier/scaler treatment, fitting budget, clean/noisy endpoints, uncertainty units and decision rule. A practical gain margin or acceptable clean loss must follow the project objective rather than be invented after results.

**Holdout must match the claim.** Held-out noise recordings/environments mixed with already studied ESC-50 targets can answer a conditional noise-transfer question. Generalization to new target recordings requires previously unexposed target sources. Neither automatically establishes pretraining independence or field deployment performance. [DEMAND](https://zenodo.org/records/1227121) is an available source of recorded environmental backgrounds to assess for a future protocol; no such mixture experiment has been executed in this review.

## Current sequencing

**Coherent independent comparative study → finalize the foreground/noise evaluation specification → execute the bounded clean-versus-noise-adapted frozen-PANNs comparison.**

No new training, model inference or noise-data download was performed during this global review.


## Deeper decision: transferable linear readout adaptation

The next phase should test **whether waveform-noise exposure during head training produces a transferable improvement with a fixed PANNs representation**. This is narrower than explaining all external failures, proving intrinsic robustness, or identifying where information is destroyed.

Three explanations remain compatible with the existing results:

- **Readout mismatch:** noisy embeddings contain usable class information, but the clean-trained classifier uses unsuitable boundaries.
- **Limited accessible information:** the fixed representation makes classification harder under corruption; some information might remain nonlinearly accessible.
- **Training/evaluation mismatch:** adaptation works on its training corruption but does not transfer to a different interference distribution.

Successful head adaptation can demonstrate a correctable component of system error. Unsuccessful adaptation cannot prove information loss. Clean-trained ReLU heads previously failing to improve performance do not establish that every noise-trained nonlinear head would fail. Larger clean-to-noisy losses also do not isolate a mechanism: systems start at different clean accuracies and have different sets of correct examples.

### Primary design decision: two matched head fits

| Head | Training exposure | Role |
| --- | --- | --- |
| Historical logistic head | Existing clean training embeddings | Preserved reference; never silently replaced |
| C: matched clean refit | Same 1,200 clean training embeddings | Comparator under the current explicitly recorded implementation |
| G: mixed Gaussian exposure | Same targets, clean plus waveform-corrupted training embeddings | Test the head-training intervention |

There are **two proposed new fits**, not a new architecture search. The same official Cnn14_16k encoder remains frozen and in evaluation mode. Preserve the 2,048-dimensional embedding, 50-class mapping, 16 kHz/5-second waveform frontend, and a single scaler fitted on clean training embeddings. The saved clean cache exists with shape `(1200, 2048)`; access to the actual source WAVs and encoder assets must be established before execution.

A noisy-only head would address the separate question of competing clean/noisy training objectives. It is not indispensable for the primary C-versus-G comparison, and is not automatically added after an unfavorable outcome. Noisy-only failure would still not prove information loss.

### Matched objective and implementation consequence

Let `x_i` be a training waveform of 80,000 samples, `y_i` its class, and `f` the fixed encoder. Let `T` be the fixed clean-training scaler. Then `z_i = T(f(x_i))` is a 2,048-vector. Corrupted features must be computed as `z_ik = T(f(x_i + a_ik n_ik))`; adding Gaussian noise directly to a cached embedding is a different intervention.

The classifier has `W` of shape `(50, 2048)` and `b` of shape `(50,)`, with probabilities `softmax(W z + b)`. Let `ell` denote cross-entropy. The proposed objectives are:

\[
J_C = \frac{1}{N}\sum_{i=1}^{N}\ell(Wz_i+b,y_i)
      + \frac{\lambda}{2}\lVert W\rVert_F^2,
\]

\[
J_G = \frac{1}{N}\sum_{i=1}^{N}\left[
\frac12\ell(Wz_i+b,y_i)
+\frac{1}{2K}\sum_{k=1}^{K}\ell(Wz_{ik}+b,y_i)
\right]+\frac{\lambda}{2}\lVert W\rVert_F^2.
\]

Here `N=1200`. Proposed fixed choices are `K=2`, independent zero-mean Gaussian training draws with target SNR uniformly drawn from 10–30 dB, and 50/50 total clean/noisy weight. These choices define one bounded intervention; they are not claimed optimal. Fix and record the actual random draws before evaluation.

Use the historical solver settings: multinomial L2 logistic regression, `C=1`, L-BFGS, `max_iter=3000`, `tol=1e-6`, no class reweighting and no intercept penalty. The local installed implementation is scikit-learn 1.9.1. Its source and [official objective](https://scikit-learn.org/stable/modules/linear_model.html#multinomial-case) use an average weighted loss plus `||W||²/(2 C S)`, where `S` is total sample weight. Give each clean row weight 1 for C; for G, give each original clean row weight 1/2 and each noisy variant weight `1/(2K)`. Both have `S=N`, so both have `lambda=1/(C N)`.

**Sanity checks:** G's per-target weights sum to one. If all corrupted features equal their clean counterparts, its objective reduces exactly to C. Concatenating all variants with unit weights would instead change effective regularization. These are algebraic checks, not additional fitted controls or a unit-test suite. Convexity of the head objective does not make all coefficient representations unique, nor make repeated optimizer seeds independent evidence. Convergence status must be recorded; a nonconverged fit cannot be explained as a scientific negative result.

### Waveform mixing and task meaning

For a nonzero-power target `x` and nonzero-power, mean-centered interference segment `n`, use full-window powers `P_x = mean(x²)` and `P_n = mean(n²)`. At desired SNR `s` dB, set:

\[
a=\sqrt{\frac{P_x}{P_n 10^{s/10}}},\qquad \widetilde{x}=x+a n.
\]

The added-noise power is then `P_x / 10^(s/10)`. At 10 dB it is one tenth of target power. This defines **added interference relative to the original recording**, not an estimate of its unknown event-only SNR. Do not silently clip or normalize mixtures afterward. Silent/nonfinite inputs need an explicit eligibility policy before outcomes are inspected.

The proposed task is recovery of the designated target recording's class in an additive mixture. Natural backgrounds can contain other ESC-50 events; they are not semantically empty noise. Check candidate audio and mixture intelligibility before model predictions, specify handling of salient competing events, and retain the resulting eligibility record. Any resulting claim is conditional on that foreground-recovery task. Do not select or discard backgrounds because they produce favorable model scores.

### Evaluation and the limits of its uncertainty

**Primary endpoint:** G-minus-C accuracy on the same target recordings under a fixed 10 dB recorded-background panel, averaging paired results equally over the preselected backgrounds and noise realizations before aggregating targets. All recorded evaluation backgrounds are excluded from training, which uses Gaussian noise only. Their selection must precede model outcomes.

**Secondary endpoints:** clean accuracy change, 50-class macro-F1, repairs/new errors, and the historical shared 10 dB Gaussian panel as descriptive context. A Gaussian-only gain supports corruption-matched adaptation, not transfer to recorded backgrounds. Since no acceptable clean-loss margin has been supplied, report clean and noisy changes jointly; do not claim clean noninferiority or automatically replace the deployed/default head.

Paired source-group bootstrap intervals can condition on fixed head fits and fixed background draws. Average repeated conditions within each target before resampling all clips from a sampled `src_file`. Extra mixtures do not create extra independent target recordings. A small fixed panel of environments does not support an interval for arbitrary future environments; show environment-specific results instead of hiding them in one pooled score. Real-world deployment and previously unseen target-source generalization remain untested.

Noise assignment must be independent of target labels. A 50-way classifier always returns a class on a noise-only input, so that fact alone is not shortcut evidence. Noise-only predictions can describe background bias; only a defined association with assigned target labels could investigate label-linked shortcuts, and small accidental associations are not conclusive.

### Interpretation and stopping

| Outcome | Defensible conclusion | Next decision |
| --- | --- | --- |
| G improves on recorded backgrounds with the clean change reported | Some transferable improvement is achievable by changing only the linear head's training exposure | Preserve both models; assess the measured trade-off before replacement |
| G improves on Gaussian evaluation but not recorded backgrounds | Benefit is limited to the tested corruption family | Report the boundary; no automatic new head search |
| G improves noisy performance but lowers clean performance | An observed accuracy trade-off, with no established acceptable loss | Do not declare uniform superiority |
| G fails to improve under either condition after a converged fit | This fixed intervention is insufficient | Do not infer information destruction or automatically start encoder fine-tuning |

The empirical question is directional and scoped. No arbitrary practical gain or clean-loss threshold is invented. A point estimate or interval crossing zero does not establish equivalence. One completed fixed C/G comparison is the stopping point for this phase; new noise policies, nonlinear heads or encoder updates require a new question rather than an outcome-driven expansion.

### Relation to prior work and execution status

[PANNs, Section V](https://arxiv.org/html/1912.10211) already describes frozen-feature classification and encoder fine-tuning. [SoundCLR](https://arxiv.org/abs/2103.01929) describes transfer learning with strong waveform/spectrogram augmentation, but its abstract does not establish this fixed-readout design. A particularly close [author report on frozen YAMNet and Gaussian augmentation](https://nhsjs.com/2026/robustness-of-ai-based-environmental-sound-recognition-under-controlled-gaussian-noise/) uses an MLP and repeated stratified splits. Its reported results have not been independently reproduced here, and its scores are not directly comparable with ours. These sources rule out presenting the general combination as a new method; this is not a systematic novelty review.

[DEMAND's author documentation](https://zenodo.org/records/1227121/files/DEMAND.pdf?download=1) describes environmental recordings suitable for considering as interference. Each environment's microphone channels capture the same scene and are not independent recordings. Exact backgrounds, time segments, random assignments and foreground-label eligibility have **not yet been selected**, so this document is a concrete design decision, not a frozen execution protocol or a completed experiment.

This continuation performed analysis, source inspection and protocol design only. No new head fit, encoder inference, noise download, model selection or GitHub publication was performed.
