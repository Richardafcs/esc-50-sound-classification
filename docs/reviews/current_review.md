# Research review before final confirmation

Snapshot before the final study completed. For current findings and deliverables, see [research.md](../../reports/baseline_20261028/report_en.md).

28 October 2026. Internal review of existing code, configurations and saved experiment artifacts. This review does not constitute external replication. No real held-out test inference was run for this review.

## Assessment

The project has progressed from an implementation pilot to a credible exploratory study of small, from-scratch sound classifiers. It is not yet a completed research result or a verified final delivery. The strongest evidence concerns the usefulness of the frequency-preserving model family under one development setup. The benefit of recurrence, its mechanism, final generalization, robustness and deployment cost remain unresolved.

The appropriate research question is narrow: under the declared ESC-50 split and training budget, does a BiGRU improve on the strongest registered nonrecurrent head enough to justify its cost? A negative answer is useful. The current architecture families do not by themselves establish an original algorithmic contribution. The literature audit is targeted historical context, not a comprehensive novelty or current state-of-the-art assessment.

## Results that actually exist

All figures below are validation accuracy on the same 398-clip development subset, selected by minimum validation cross-entropy. They are not official five-fold ESC-50 scores.

| Evidence | Model | Seed | Accuracy | Evidence available locally |
|---|---|---:|---:|---|
| Statistical baseline | Temporal mean/std + logistic classifier | Deterministic fit | 42.21% | Model, search record, row predictions |
| First 60-epoch GPU screen | Frequency-averaged CNN | 17 | 51.51% | Executed notebook and complete curves |
| First 60-epoch GPU screen | Frequency-averaged CNN+GRU | 17 | 54.52% | Executed notebook and complete curves |
| First 60-epoch GPU screen | Frequency-preserving CNN | 17 | 69.35% | Executed notebook and complete curves |
| First 60-epoch GPU screen | Frequency-preserving CNN+GRU | 17 | 75.38% | Executed notebook and complete curves |
| First 60-epoch GPU screen | Frequency-preserving parameter-matched pooled MLP | 17 | 66.33% | Executed notebook and complete curves |
| Repeated-study persistence probe | Frequency-preserving CNN | 101 | 72.61% | Recovered checkpoint, manifest, curves and row predictions |

The first screen's checkpoint archive and row predictions were not recovered before its temporary runtime ended. Its saved stdout supports the reported screen scores and curves, but cannot support a new paired-error analysis or checkpoint re-evaluation. One actual repeated-study checkpoint has been recovered successfully; there is no completed three-seed recurrent/control comparison in the saved local evidence inspected here.

## Findings in priority order

### 1. Architecture selection is not settled

The seed 17 recurrent model exceeds the corresponding small CNN by 6.03 percentage points. That is an exploratory observation. The recovered CNN seed 101 differs from the screen CNN seed 17 by 3.27 points; this is not an estimate of a seed distribution, and runtime differences also limit a seed-only interpretation. It demonstrates the weakness of choosing from individual runs. Do not compare the seed 17 CRNN directly with seed 101 CNN as if they were a controlled pair. Complete the registered model/seed roster and report every seed before choosing.

### 2. The mechanism claim needs stronger controls

The frequency-preserving variants improve markedly in the first screen, but retaining frequency also changes head dimensions and parameter counts. The pattern motivates the representation choice; it does not isolate frequency identity as the sole cause. A matched parameter count also does not match nonlinear placement, optimization, compute or inductive bias. The registered tokenwise MLP is a necessary stronger competitor. Encoder-token permutation probes sensitivity to an intervention; a drop alone does not prove useful temporal reasoning, because the intervention changes the input distribution seen by the head. Pooled versus tokenwise heads also differ in dropout granularity.

### 3. Generalization and practical value remain unmeasured

Fold5 has not supplied a real-model test result in the inspected artifacts. Repeated inspection and selection on fold 4 makes development scores optimistic for a final performance claim. Freeze the candidate selection and measurement settings before fold 5. Later rotating folds 1–4 cannot make those folds untouched again; call that fixed-recipe benchmark characterization and disclose the development history. With roughly eight validation clips per class, individual class rankings are unstable. Report training variation separately from source-group sampling uncertainty; do not pool repeated predictions as independent recordings.

### 4. Evidence persistence and project navigation still need work

Losing the first screen weights was a material execution failure. The recovered seed 101 capsule proves the replacement persistence route works. Keep that simple solution and recover artifacts promptly; do not build another provenance framework. Preserve dataset/split identity, run recipe, source snapshot, checkpoint and predictions. Repeated hashing of the same assets and more approval-style gates do not answer the research question.

The small src package is a substantial improvement: audio, data, models, training, evaluation and inference have visible responsibilities. At review entry, root classify.py/diagnostics.py duplicated packaged responsibilities, research helpers remained at root, and the README foregrounded the obsolete two-epoch CPU route. The duplicate production implementations have now been archived, their active entry points delegate to the package, and the README has been rewritten around the current project. Research helper placement and final delivery documentation still need a final pass. Use one supported CLI and a short path from reproduction to results. Archive frozen historical source without maintaining two current implementations. Dependency ranges support installation but do not replace the already recorded exact execution environments.

The review reran the local suite and initially reproduced 150 passes and two failures from stale monkeypatch targets. The targets were updated to the modules that now own evaluation and training. After the import and compatibility migration, the final rerun passed all 152 tests in 8.26 seconds. This is a test migration, with no change to model mathematics or historical results. A green suite establishes the covered implementation behaviors; it does not validate research superiority.

### 5. The scientific contribution needs a completed argument

At present the project can support a well-controlled empirical study and a practical classifier. It cannot support claims of a novel architecture, state-of-the-art performance, broadly robust sound recognition, or established recurrent superiority. Failure analysis should explain which categories benefit, which remain confused, whether the advantage survives stronger nonrecurrent controls, and the accuracy/latency tradeoff. Additional model families should follow a specific unresolved question; adding more architectures now would expand search without closing the current argument.

## What to do next

1. Complete the registered repeated development study with saved checkpoints and row predictions; keep the original frozen training source for that comparison.
2. Apply the registered augmentation rule and select the model from development evidence only. Retain the CNN+GRU implementation even if a nonrecurrent default wins.
3. Freeze and run held-out evaluation, order/noise/gain probes and actual CPU end-to-end latency. Report limits and failures alongside averages.
4. Complete the declared broader fold characterization, clearly separating it from confirmation.
5. Finish package integration, one runnable Colab entry, portable selected models, model cards and one coherent research report. Perform a clean-environment inference/reproduction check.

A completed deliverable should answer the research question with traceable results and an understandable implementation. The next effort should primarily produce those missing measurements, rather than additional scaffolding or documents.

## Evidence

- ../results/gpu_screen_summary.json
- ../results/gpu_screen_recovered_curves/evidence_scope.json
- ../results/recovered_probe/recovery.json and its recovered run artifacts
- ../results/persistence_probe_verified.json
- ../configs/repeated_development_v2.json
- ../configs/decision_clarification_v2.json
- ../src/esc50_sound/models.py and ../src/esc50_sound/inference.py
- literature_deep_read.md and completion_plan.md

## Actions following the review

The official Colab CLI0.7.5 is installed and authenticated with explicit user approval for its requested permissions. It connected to the observed T4 runtime without allocating another VM or restarting the active one. The runtime contained no study files, confirming the old ephemeral filesystem was unavailable. The saved frozen source and recovered seed 101 model were restored; the regenerated GPU cache matched that model, and the statistical baseline reproduced 42.21% validation accuracy. Remaining registered repeats are launched through `scripts/run_colab_repeats.py`, which exports ordinary downloadable archives after each seed. This is execution progress, not yet a completed repeat result. No SSH shell was opened: the account reports0.00 compute-unit balance, and the official FAQ restricts SSH on free managed runtimes. CLI code execution and file retrieval are the supported route used here.

## Closure after the frozen study

All registered development runs, the held-out evaluation, probes and five-fold characterization are now complete. The final [research report](../../reports/baseline_20261028/report_en.md) retains the negative findings: limited recurrent benefit, validation-to-test drop and severe synthetic-noise degradation.

| Review issue | Resolution / remaining limit |
|---|---|
| Single-seed selection | Four models × three seeds completed; all results reported; CNN selected before test |
| Missing mechanism controls | Matched pooled/tokenwise heads and five token permutations executed; no causal temporal-reasoning claim |
| Unmeasured generalization | Twelve repeated neural holdout evaluations plus baseline, then eight additional fixed-recipe fold runs |
| Lost first-screen weights | Failure disclosed; all final selected weights and row predictions recovered and saved locally |
| Duplicate implementation and stale entry point | One package owns active algorithms; compatibility launchers and historical source are labeled |
| Unmeasured robustness and efficiency | Noise/gain/silence/input and warmed CPU measurements completed; robustness remains weak |
| Verification | 153 current tests; separate reviewer recalculated metrics and reproduced both primary models on all 400 test WAVs |
| Novelty and external validity | No novel-method or field-deployment claim; single dataset, limited seeds and prior fold use remain limits |
