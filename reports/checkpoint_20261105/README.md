# Model improvement checkpoint - 5 November 2026

[English checkpoint PDF](../pdf/ESC50_Model_Improvement_Checkpoint_EN.pdf) · [Editable report](Checkpoint_Report_EN.md) · [Literature review](literature_review.md)

The report addresses all five checkpoint requirements using completed experiments: baseline error analysis, BiGRU and attention implementations, waveform augmentation, an alternative frozen PANNs approach, quantitative evaluation and comparison. It contains nine selected figures; the companion [final research PDF](../pdf/ESC50_Final_Research_Report_EN.pdf) and [PANNs PDF](../pdf/ESC50_PANNs_Transfer_Report_EN.pdf) retain all 22 scientific figures across their separate evaluation scopes.

## Results

| System | Development accuracy | Macro-F1 |
| --- | ---: | ---: |
| Research CNN | 72.61% ± 0.25 pp | 71.39% ± 0.58 pp |
| CNN + BiGRU | 73.87% ± 1.26 pp | 72.97% ± 1.36 pp |
| BiGRU + channel attention | 74.37% ± 1.15 pp | 73.48% ± 1.00 pp |
| Frozen PANNs + linear classifier | 91.21% | 91.10% |

Neural summaries use three fixed fits; PANNs uses one fixed head fit. All comparisons use the same 398 development clips. The original team-repository CNN is a different implementation with no committed evaluation evidence; the research CNN score must not be attributed to it. See [baseline identity](baseline_identity.json).

Noise augmentation improves CNN accuracy at 10 dB Gaussian noise from 23.37% to 61.31%. PANNs noise robustness remains unmeasured. Channel attention has a small uncertain gain; further attention training is deferred.

## Evidence and reproduction

- [Nine-system clean comparison](model_comparison.csv)
- [Six-condition robustness comparison](robustness_comparison.csv)
- [26-publication screening matrix](literature_matrix.csv): reading depth is explicit; screening is not a claim of full-paper reading.
- [Figure manifest](figure_manifest.json)
- [AI-assisted discussion and final review](review_notes.md)

From the research project root, install the report dependencies and render the saved manuscript and figures:

```sh
python -m pip install -e '.[baseline,report,transfer]'
python scripts/build_checkpoint_report.py
```

This document-generation command does not train or evaluate a model. Shared-evidence PANNs analysis can be recomputed with `python scripts/analyze_panns_transfer.py`; extraction and new fitting additionally require the official PANNs source/checkpoint and ESC-50 WAV files.

All folds have historical exposure. These are development findings, not fresh held-out confirmation. Differences between PANNs and scratch models combine pretraining, capacity, frontend and classifier changes. Bootstrap intervals condition on the fitted models and recordings. See the report for full qualifications.
