# ESC-50 Colab Research Console · handoff guide

## What is included

- `notebooks/templates/ESC50_Colab_Research_Console.ipynb` — interactive Colab controller, safe plan mode by default.
- `configs/colab_study_matrix_v1.json` — 65 registered runnable development tasks with stage, model, seed, augmentation and requirements.
- Existing `scripts/feature_ablation.py`, `src/esc50_sound/cli.py`, `study.py`, and current model/augmentation modules; no new training code path was invented.

## Execution matrix

| Group | Runnable tasks | Scope |
| --- | ---: | --- |
| A_baselines | 2 | Log-Mel temporal stats + logistic; DCT cepstral + Linear SVM |
| B_architecture | 15 | CNN, BiGRU, TA, CA, TCA × 3 seeds |
| C_augmentation | 9 | SpecAugment, waveform Gaussian noise, waveform Mixup × 3 seeds. Clean control is reused from group B CNN |
| D_feature_sampling | 36 | 12 prespecified feature/sample-rate cases × 3 seeds, separate frequency-pooled CNN |
| E_robustness | 3 | Development-validation stress of two clean controls and one noise-trained CNN |
| F_advanced | 0 | PANNs/AST/BEATs/CLAP/RL not integrated; future work only |
| G_confirmation | 0 | Held-out test and final CV require a separately frozen study protocol |

The study does not require running all 65 tasks; select groups based on the research questions and compute budget.

## Colab steps

1. Download the **Colab full-project ZIP** and the standalone `.ipynb` notebook.
2. Open the `.ipynb` with Google Colab and choose a GPU runtime if available.
3. Keep `RUN_MODE='plan'`, then **Runtime > Run all** to preview the full experiment matrix without downloading or training.
4. Set `RUN_MODE='pilot'`, `TASK_IDS=['B_cnn_freq_s101']`, `MAX_TASKS_PER_SESSION=1`; run all cells. When asked, upload the full-project ZIP (or provide its Drive path).
5. The data installer downloads the official ESC-50 source at the project's pinned revision unless you set `DATASET_ZIP_PATH` to an existing archive. Network access is required unless a dataset ZIP is already available.
6. After the pilot, set `RUN_MODE='selected'` and `TASK_IDS` or `GROUPS_TO_RUN`, increase the task cap deliberately, and rerun cells in order.
7. With Drive enabled, results and `last.pt` are stored in `MyDrive/ESC50_Research_Colab`; base caches can be reused. If Colab changes the underlying environment, the project's strict resume guard may refuse to resume an incomplete run. Completed runs are skipped. **The feature-ablation runner does not support epoch-resume.**
8. Read development validation results and paired feature contrasts at the end of the notebook. No part of this notebook automatically evaluates held-out Fold 5.

## Important scientific limitations

- Main model and augmentation experiments share original log-Mel 16 kHz representation and the source-strict train/validation protocol.
- Feature/sampling experiments use a frequency-pooled `cnn`, **not the frequency-preserving `cnn_freq`** used in architecture studies. Compare within each controlled track, not across them as a one-variable change.
- The source-strict split uses train folds 1–3, validation fold 4, reserved test fold 5; it is not the canonical four-fold training benchmark. The historical project had prior test exposure.
- Pilot runs use 2 epochs by default and have separate output directories; they are operational smoke tests, not final research results.
- Training logs and `run.json` must show completed epochs before scores are included. An unfinished ablation run must be inspected, not overwritten.
- Drive mounted filesystem can be slower or behave differently from a local POSIX disk. Always begin with a short pilot in Colab. If it does not support the project's file operations reliably, switch to local execution plus explicit result export rather than force-running the full grid.
- Storing full WAV archive, legacy cache and optional 12 feature caches can require several GB of storage.

## Validations performed when constructing this delivery

- The notebook's plan-only mode was executed top to bottom with `nbclient`, without Colab, Drive, external downloads or training.
- All notebook Python cells passed compilation, and the notebook JSON passed `nbformat.validate`.
- The study JSON was checked for 65 unique task identities and stage counts.
- **Actual Colab end-to-end training has not been run here**; it requires network/Drive access and official audio.
