# ESC-50 English report set

Prepared 29 October 2026 from saved, reviewed project evidence. No model was retrained or newly evaluated for these PDFs.

- **ESC50_Complete_Study_Report_EN.pdf**: 18 pages; the latest 65-task development study; all 62 fitted-task rows, all 25 aggregate paired contrasts, and seven full-page figures including the audio/Log-Mel illustration.
- **ESC50_CNN_CRNN_Baseline_Report_EN.pdf**: 18 pages; the complete historical CNN/CRNN comparison; repeated seeds, paired errors, synthetic stress and fixed-recipe fold rotation; eight full-page figures including both primary confusion matrices and a 50-class ID key.

Click figure references or use PDF bookmarks to reach the figure atlas. Historical and latest results retain separate scopes; do not treat old test exposure as fresh independent confirmation. Source labels and project-evidence paths are retained in each PDF.

The shared `reports/figures/` directories includes all 15 PNG figures and all 13 available SVG originals. `figure_manifest.json` maps them to PDF pages and original project paths. `report_verification.json` records page checks, embedded-image pixel comparisons and result-row coverage.

Rebuild from the project with `scripts/build_english_pdf_reports.py`; check with `scripts/verify_english_pdf_reports.py`. The authoring runtime requires ReportLab/Pillow and the verifier requires pypdf/pdfplumber.

Regenerating the reports requires the original local experiment collection. Verifying the published PDFs uses the curated evidence included in this repository.
