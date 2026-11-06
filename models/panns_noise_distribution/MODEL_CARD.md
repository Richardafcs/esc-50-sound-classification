# Recorded-background-adapted PANNs readout

Frozen official Cnn14_16k features, clean-training standardization and a 50-class L2 multinomial logistic head. The NPZ contains mean, scale, coef, intercept and classes; it requires the original official encoder.

Training: 1,200 ESC-50 targets, clean weight 0.50 plus two recorded-background variants of weight 0.25 each; six reserved training DEMAND recordings; exact prior Gaussian SNR assignments. [Protocol](../../reports/noise_distribution_20261107/protocol.json).

Primary recorded-minus-Gaussian result: +1.36 pp, conditional interval [+0.41,+2.32]. [Full results and limitations](../../reports/noise_distribution_20261107/README.md). Not designated a universal default. Inference matches existing PANNs head code: standardize embedding with saved mean/scale, apply coef/intercept, map argmax through classes. No raw DEMAND recordings are redistributed.
