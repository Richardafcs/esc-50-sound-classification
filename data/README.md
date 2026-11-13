# Data

The project uses [ESC-50](https://github.com/karolpiczak/ESC-50): 2,000 five-second recordings, 50 classes and 40 examples per class. The original [dataset README](ESC-50/README.md), [license](ESC-50/LICENSE) and [metadata with recording credits](ESC-50/meta/esc50.csv) are retained here.

From the repository root:

```sh
python -m esc50_sound prepare
```

This downloads the configured ESC-50 version into `data/ESC-50/audio/` and creates `data/esc50_features.npz` with the scratch-model log-Mel features. Audio and feature caches are excluded from version control.

| Role | Official folds | Clips |
| --- | --- | ---: |
| Training | 1–3 | 1,200 |
| Development | 4, after source exclusions | 398 |
| Designated test partition | 5 | 400 |

Two fold-four clips share metadata source identifiers with fold five and are excluded from development evaluation. Normalization statistics are fitted on training data only. All official folds had historical exposure during the project; the headline comparisons are development results. The [evaluation protocol](../docs/evaluation.md) explains these roles.

Supplementary notebooks obtain external FSD recordings and DEMAND backgrounds from their original sources. These recordings and the official PANNs backbone are not redistributed. ESC-50 is distributed overall under CC BY-NC 3.0; retain the original attribution and recording credits when using its audio.
