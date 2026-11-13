# Encoded temporal-order interventions

The experiment changes the order of the CNN encoder's **39 time tokens** before recurrent processing. It compares reversal, a 19-position circular shift and two fixed shuffles with the original token order, using CNN and CNN–BiGRU fits for seeds 101, 202 and 303.

| BiGRU intervention | Accuracy change | Conditional 95% source interval |
| --- | ---: | ---: |
| Reverse | −5.03 pp | [−7.54, −2.60] |
| Circular shift by 19 | −1.42 pp | [−2.94, 0.08] |
| Shuffle 17 | −1.09 pp | [−2.64, 0.42] |
| Shuffle 29 | −0.08 pp | [−1.47, 1.32] |

The CNN's temporal-mean aggregation is invariant to reordering the same encoded tokens; its predictions remain unchanged. The BiGRU is order-sensitive, especially under reversal. This intervention changes an internal sequence distribution and does not establish that natural temporal structure caused the model's original accuracy advantage. It also does not imply CNN invariance to reversing the raw waveform, which can change the encoded features.

Effects are averaged across the three fixed fits on 398 development clips from 295 sources. Intervals use 4,000 paired source-group bootstrap draws and do not estimate training-seed population uncertainty. Class-level contrasts are exploratory.

[Results notebook](../../../notebooks/01_results.ipynb) · [Protocol](protocol.json) · [Metrics](metrics.csv) · [Paired changes](paired_errors.csv) · [Class results](class_results.csv) · [Summary](summary.json) · [Figure](crnn_accuracy_change.png)
