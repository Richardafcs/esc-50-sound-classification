# Research and development completion contract

## Final requirement audit — 28 October 2026

The planned research and offline prototype are complete. Negative outcomes are retained: recurrent gains are modest and synthetic-noise robustness is poor. These results do not establish field-deployment readiness. The original prospective contract below is preserved as historical scope.

| Requirement | Observed proof | Status and limits |
|---|---|---|
| Data/frontend | 2,000-WAV format/duplicate audit (optional local evidence: `results/historical/waveform_audit.json`); pinned metadata/cache records in [final study](../../results/historical/final_study); schema/source-split tests; original-WAV prediction agreement | Met for the declared dataset/frontend; perceptual near-duplicate detection incomplete and disclosed |
| Real training correctness | Training-only real-audio memorization (optional local evidence: `results/historical/real_audio_overfit_v2/real_audio_overfit.json`);21 completed 60-epoch runs; finite-gradient checks and eval-mode histories | Met; compact checkpoints support inference, not exact resume |
| Credible baselines | Converged three-value L2 search and [final baseline](../../results/historical/final_study/results/statistics_baseline); trained CNN/CRNN and controls | Met; deterministic statistical fit reported separately from neural seed variation |
| Recurrence question | Four candidates×three seeds; development selection (optional local evidence: `results/historical/development_selection.json`); capacity/token controls and order probes (optional local evidence: `results/historical/final_study/results/order`) | Met within scope: +1.26 pp development gain fails the registered 3 pp preference; no causal temporal-reasoning claim |
| Generalization | [Freeze](../../configs/confirmation_v1.json) precedes first access; 20 audited neural evaluations (optional local evidence: `results/historical/final_evaluation_audit/report.md`); baseline recomputation; [paired source bootstrap](../../reports/evidence/baseline_20261028/comparison.json) | Met; fold 5 confirmation separated from later characterization, no posttest selection |
| Broader context | Ten model/fold records (optional local evidence: `results/historical/final_study/results/cross_validation.json`), all 400 testclips/fold, exact registered roles | Met as fixed-recipe one-seed characterization; folds 1–4 development history disclosed |
| Robustness/input behavior | Registered noise/gain measurements (optional local evidence: `results/historical/final_study/results/stress`); input-error/silence tests; no held-out abstention | Measurement complete; noise robustness failed, including6.25% CNN accuracy at 0 dB |
| Efficiency | Latency records (optional local evidence: `results/historical/latency`),100 samples/model,2 threads,10 warmups; parameters/checkpoint sizes; standalone CLI check (optional local evidence: `results/historical/portable_bundle_verification.json`) | Met for warmed Apple M4 workload; model loading/CLI startup measured separately, no service/CUDA inference claim |
| Working delivery | Two portable bundles (optional local evidence: `releases/README.md`); class mapping/frontend/source/model cards; Colab notebook (optional local evidence: `notebooks/archive/ESC50_Research_and_Inference.ipynb`) and executed integration copy (optional local evidence: `notebooks/archive/ESC50_Research_and_Inference_output.ipynb`) | Met; default 60-epoch template verified with a two-epoch integration override; full research runs separately retained |
| Research communication | [Report](../../reports/baseline_20261028/report_en.md), [method](../methodology/method.md), [primary-source reading](../methodology/literature_deep_read.md), six [figures](../../reports/figures/baseline_20261028), failure analysis and [review closure](../reviews/current_review.md) | Met; no novelty, SOTA or institutional-affiliation claim |
| Independent verification | Separate internal verification (optional local evidence: `results/historical/independent_verification.json`), all 12repeated neural+baseline metric recomputations, both primary models match400originaltestWAV predictions;153 currenttests | Met as internal independent execution; external human replication not claimed |

The selected default is CNN seed 101, fixed before test access. Portable prediction, frozen study evidence and a runnable notebook are delivered. The report gives exact scope and limitations; no learner mastery rating was changed.

---

## Original prospective contract

27 October 2026. User requested continuing the full project to completion. This contract preserves the attached brief's task: a deep learning environmental-sound classifier on ESC-50 using CNN and recurrent modeling, with measured accuracy, robustness and efficiency. It does not promise novelty or a particular unmeasured accuracy. A negative recurrence result remains a valid research outcome; an untrained or unverified deliverable does not.

## Required end state and proof

| Requirement | Required authoritative evidence |
|---|---|
| Correct and portable data/frontend pipeline | Pinned archive/metadata/cache identities; source-ID split audit; frontend/schema/hash rejection tests; waveform-to-feature consistency |
| Real training implementation verified | Real-audio subset memorization, finite gradients; comparable training/validation readouts; checkpoint and inference equivalence |
| Credible baselines | Train-only feature-statistics classifier; trained CNN and CRNN; full logs and disclosed search budget |
| Answer the recurrence question within scope | Same data exposure/checkpoint rule; repeated seeds; parameter-comparable nonrecurrent head; encoder-token order ablation and invariant nonrecurrent control |
| Valid generalization evaluation | Freeze recipe/model-selection rule before any new fold-5 prediction; paired per-clip predictions; accuracy/macro-F1/per-class failures; appropriate source-group descriptive uncertainty; no further tuning to that holdout |
| Broader benchmark context | Predeclared five-fold fixed-recipe assessment after development, retaining all folds and explicitly labeling prior use of folds 1–4; never claim five untouched confirmation datasets |
| Robustness tested | Predeclared additive-noise/SNR conditions and gain/silence/invalid-input behavior; separate simulated stress tests from real-world robustness claims |
| Efficiency measured | Actual device/environment; fixed workload, warmup and repetitions; feature+model latency and model-only latency, distribution/parameter/file-size costs |
| Working user deliverable | Portable selected checkpoint, classes/frontend/source/model card; CLI inference on a real WAV; clear errors; self-contained Colab notebook; saved runnable evidence and reproducibility commands |
| Research communication | Full-method primary-source reading, architecture/shape/assumption explanation, result tables/curves/confusion, failure analysis, limitations and explicit response to review |
| Independent verification | Separate reviewer re-runs supported checks and selected inference/recomputation; completion matrix linked to evidence; unresolved issues disclosed and repaired where required |

## Experimental sequencing and confirmation boundary

Development remains source-strict train1–3, validation4, holdout5. Current two-epoch pilots are preserved unchanged. First complete correctness/provenance repairs and real-audio checks. Then measure longer curves before freezing the final training duration, regularization and candidate roster. Any added architecture or augmentation must be registered with its reason and reported as development selection.

Primary comparison constraint: identical training clips and maximum epoch/data-exposure budget. Other resource dimensions (parameters, actual updates, time, inference latency) are outcomes. The first longer pilot uses 40 epochs, seed 17, batch32, AdamW0.001, no augmentation, identical preprocessing and validation-loss checkpoint selection for CNN/CRNN. This is a development measurement, not final selection or confirmation. If time stops a run, its actual exposure is reported and equal-exposure comparisons require completion/restart with proper provenance.

Before repeats/final evaluation, write a frozen study configuration including seed roster, maximum epochs, model candidates, tuning attempts, checkpoint rule and decision criterion. Default practical preference: retain the simpler/faster model unless the repeated development comparison provides at least a 3 percentage-point mean accuracy gain for the more costly candidate; disclose seed variation and macro-F1. This is a project decision threshold, not a scientific law or a claim that smaller gains do not exist. Stronger mechanism claims require the capacity/order controls. Final selection cannot use test outcomes.

Fold5 is the current confirmation holdout. Later fixed-recipe cross-validation has a disclosed development history because folds 1–4 already informed choices. Report those results as benchmark characterization, separately from current held-out confirmation. Do not present a partially nested procedure as a fully nested evaluation of the entire adaptive development process.

## Resource controls

Use existing local CPU and already authorized Colab resources; no purchases, paid API calls or persistent service. Local pilot measurements are approximately5 seconds/epoch before extra diagnostics. First longer runs have 40 epochs and a 10-minute per-run training cap. A first development tranche is bounded to 30 local CPU wall-clock minutes, including tuning/failed runs, with evidence retained. Each expansion is justified from measured time and the remaining acceptance requirements and gets an explicit updated run/time cap; the user has authorized continuing the task, not buying resources. Temporary data/checkpoint storage target below5GiB; monitor before larger ensembles or fold studies. Stop on nonfinite values, data/provenance mismatch or unavailable dependencies; repair the smallest fault and preserve failed-run records.

User subsequently explicitly authorized Colab GPU use and asked to prefer Colab. The main development screen now runs on an observed TeslaT4 (torch2.11.0+cu130), after48 uploaded correctness checks passed. Source bundle identity is recorded in `notebooks/archive/ESC50_GPU_Study.ipynb`. The registered screen uses six models (both frequency-averaged and frequency-preserving regimes, each CNN/CRNN/capacity control), seed 17,60 epochs, cosine schedule, no augmentation, identical actual training clips/batch/optimizer settings. It follows three-epoch cost pilots. Per-run training cap600 seconds; the initial notebook's1800second screen budget is checked between jobs, so a final job can extend past that boundary. Its streaming timeout also depends on child output. These guard limitations were discovered by independent review and are retained as first-screen limitations; `runner.py` implements a silence-independent watchdog and remaining-global-budget accounting for the next stage. No test predictions are part of this screen. No upgrade or purchase was made.

Completion is audited requirement by requirement. No amount of generated documentation, agreement between agents, or exhausted budget substitutes for the evidence above.
