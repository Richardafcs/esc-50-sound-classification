# Classical, neural and audio-feature comparisons

The main study compares a small frequency-preserving CNN with recurrent processing, attention and training augmentation. Training uses folds 1–3 and development evaluation uses 398 source-separated fold-four clips. Scratch results are means over seeds 101, 202 and 303.

| Model | Development accuracy | Macro-F1 |
| --- | ---: | ---: |
| CNN baseline | 72.61% | 71.39% |
| CNN–BiGRU | 73.87% | 72.97% |
| BiGRU + channel attention | 74.37% | 73.48% |
| BiGRU + temporal attention | 72.28% | 71.52% |
| BiGRU + temporal/channel attention | 72.86% | 71.69% |
| CNN + Gaussian noise | 73.12% | 71.99% |
| CNN + masking | 71.86% | 70.75% |
| CNN + Mixup | 71.61% | 70.65% |

Recurrence gives a modest observed improvement. Channel attention adds 0.50 percentage points over BiGRU, but its paired source-bootstrap interval includes zero. Masking and Mixup do not improve this configuration's clean accuracy.

The original matrix contains 60 neural-fit tasks, two classical-reference tasks and three stress-evaluation tasks. Feature/sampling experiments use a separate 26,658-parameter pooled CNN and should not be treated as architectural ablations of the 49,058-parameter baseline.

## Classical references

Log-Mel temporal mean/standard-deviation summaries with multinomial logistic regression give **42.21% accuracy / 38.86% macro-F1**. DCT20 cepstral summaries with a linear SVM give **40.20% / 37.50%**. Each classical task includes its own regularization search; these are reference systems with different feature representations.

## Separate audio-feature study

Twelve configurations cross sample rate and maximum frequency with Log-Mel32, Log-Mel64, log-linear64 and 40 cepstral coefficients. Each uses three seeds and the **26,658-parameter globally pooled CNN**. At 16 kHz, Log-Mel32 averages 55.61% accuracy versus 51.59% for Log-Mel64. The best observed configuration is 32-kHz/16-kHz-bandwidth Log-Mel32 at 57.12%.

These scores describe the separate frontend track, not the main frequency-preserving CNN. The [full report](../../report.md) explains its relation to the architecture, capacity and temporal-order results.

[Results notebook](../../../notebooks/01_results.ipynb) · [Run metrics](analysis/runs.csv) · [Group summaries](analysis/groups.csv) · [Paired effects](analysis/paired_effects.csv) · [Stress results](analysis/stress.csv) · [Training environment](runtime_environment.json)
