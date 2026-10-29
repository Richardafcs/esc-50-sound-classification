# Script navigation

The installed application entry point is `python -m esc50_sound`. Run repository scripts from the project root.

| Task | Entry points |
| --- | --- |
| Audit the current complete study | `python scripts/analyze_colab_matrix.py --require-complete` |
| Plot the current study | `python scripts/plot_colab_matrix.py` |
| Historical figure regeneration | `python scripts/make_figures.py --svg --include-cv` |
| Feature experiment commands | `python scripts/feature_ablation.py --help` |
| Evaluation / platform / registry | `research_evaluation.py`, `research_platform.py`, `model_registry.py` |
| Notebook generation | `build_notebook.py`, `build_matrix_execution.py`, `build_bilingual_notebook.py` |
| Report generation | `write_matrix_reports.py`, `build_english_pdf_reports.py`, `verify_english_pdf_reports.py` |
| Frozen-era compatibility | [legacy](legacy/): old launchers and original diagnostic/report utilities |

Only explicitly invoked training scripts train models. Historical Colab orchestration in the optional local `archive/orchestration/` collection retains the exact original source and runtime assumptions; it belongs with frozen releases rather than current launch commands. New implementation belongs in `src/esc50_sound/`, not in root-level Python files.

## Fresh-checkout scope

The application, platform catalog, prediction bundles and full test suite work from the shared repository. Full-study audit/plotting/report-generation commands above require the original local run collection, which is excluded from Git. Read the shared PDFs, summary CSVs, and figure originals under `reports/` without that collection. `verify_english_pdf_reports.py` uses the shared report evidence; install `Pillow`, `pypdf`, and `pdfplumber` to run it.
