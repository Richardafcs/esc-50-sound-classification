# Gaussian training exposure and recorded-background transfer

This experiment asks whether Gaussian augmentation during linear-head fitting improves recognition under recorded interference. The PANNs encoder is frozen; the scaler, classifier objective and total training weight are matched.

| Linear classifier | Clean accuracy | Gaussian 10 dB | Recorded mean accuracy | Mean recorded macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Clean-trained | 91.21% | 69.85% | 89.07% | 89.15% |
| Gaussian-adapted | 91.96% | 83.92% | 90.12% | 90.21% |

Gaussian exposure improves matched Gaussian accuracy by **14.07 percentage points**. The primary recorded-background difference is **+1.05 pp**, with a conditional 95% source-bootstrap interval of **[−0.54, 2.59]**; transfer to this panel remains uncertain.

Both heads use 2,048-dimensional features, the clean-training StandardScaler and L2 multinomial logistic regression with `C=1`. Each of the 1,200 training clips contributes total weight one: the adapted objective assigns 0.50 to clean and 0.25 to each of two Gaussian variants with SNRs from 10–30 dB. Preserving total weight also preserves effective regularization.

Evaluation mixes each of 398 development clips with DEMAND channel-one recordings NRIVER, OHALLWAY and TCAR at 30–35 and 150–155 seconds, using 10 dB added-noise SNR. These are **2,388 target/condition pairs from 398 foreground clips**, not 2,388 independent recordings. The target is the designated foreground class; other recognizable events may be present in the background.

The 4,000 paired source-bootstrap draws condition on the fitted heads and selected segments. The target folds have historical exposure, and the backgrounds are a small fixed panel. These results do not establish general environmental robustness. The later [recorded-noise experiment](../recorded_noise/README.md) uses different training and evaluation recordings.

[Notebook](../../../notebooks/05_gaussian_adaptation.ipynb) · [Protocol](protocol.json) · [Metrics](metrics.csv) · [Paired effects](paired_effects.csv) · [Predictions](predictions.csv) · [Training exposures](training_exposure.csv) · [Fit records](head_fits.csv) · [Backgrounds](background_segments.csv) · [Summary](summary.json)

The two head fits and evaluation took 174.90 seconds on Colab Tesla T4, excluding setup/downloads. Sources: [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn), [ESC-50](https://github.com/karolpiczak/ESC-50), [DEMAND](https://zenodo.org/records/1227121). DEMAND recordings are licensed CC BY-SA 3.0; raw audio is not redistributed here.
