# Related work: training-noise distributions

The experiment compares Gaussian and recorded-background augmentation while keeping the PANNs encoder and downstream classification objective fixed. Its contribution is a controlled comparison within this course project, rather than a new augmentation method. Published benchmark scores use different datasets and evaluation protocols and are not included in the project's model ranking.

## Environmental sound augmentation

McLoughlin et al., [*Robust Sound Event Classification Using Deep Neural Networks*](https://kar.kent.ac.uk/51341/1/machine_hearing.pdf) (2015), investigate multi-condition training with recorded NOISEX-92 backgrounds and reserve speech babble for evaluation. Their work establishes relevant precedent for evaluating unfamiliar interference. It does not use the present Gaussian-trained versus recorded-trained frozen-classifier contrast.

Salamon and Bello, [*Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification*](https://arxiv.org/html/1608.04363v2) (2017), mix training examples with four recorded acoustic scenes and report class-dependent augmentation effects on UrbanSound8K. Their backgrounds are screened to exclude the ten target categories. Our 50-class designated-foreground task permits competing events in the background, and its full-window SNR rule differs from their mixture-weight formulation.

Guzhov et al., [*ESResNe(X)t-fbsp: Learning Robust Time-Frequency Transformation of Audio*](https://arxiv.org/html/2104.11587) (2021), compare learned frequency B-spline and STFT frontends under Gaussian noise and other perturbations. This motivates holding the frontend fixed in the present exposure comparison; changing it would introduce another explanation for the observed robustness differences.

## Pretrained representations

Kong et al., [*PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition*](https://arxiv.org/html/1912.10211) (2020), evaluate frozen audio embeddings and fine-tuning for downstream classification. Our implementation uses the official later-code-version `Cnn14_16k` checkpoint and a regularized logistic classifier. The encoder remains frozen throughout the noise-distribution experiment, allowing the comparison to focus on downstream training exposure.

[AST](https://arxiv.org/html/2104.01778v3) and [BEATs](https://arxiv.org/html/2212.09058) provide alternative pretrained representations, but they are not implemented here. Their published clean ESC-50 results cannot establish an advantage for a particular representation on this project's reserved-background task. Architecture, pretraining data, frontend and optimization would all require consideration in such a comparison.

## Background recordings and task definition

Thiemann, Ito and Vincent's [DEMAND corpus](https://zenodo.org/records/1227121) supplies environmental recordings from a sixteen-channel microphone array. Channels are simultaneous measurements of a recording scene, not independent environments. This experiment therefore separates complete recording identities between training and evaluation, using channel one at 16 kHz. The recordings are licensed CC BY-SA 3.0.

[MUSAN](https://arxiv.org/html/1510.08484) is another possible interference corpus, but labels such as "noise" do not guarantee absence of recognizable foreground events. Similarly, [ESC50Mix](https://zenodo.org/records/7913138) studies sound mixtures and embedding behaviour, a related setting with a different tagging interpretation. In this project, every mixture retains the designated original foreground label; the metric does not measure recognition of every audible source.

Goulão et al., [*Training Environmental Sound Classification Models for Real-World Deployment in Edge Devices*](https://link.springer.com/article/10.1007/s42452-024-05803-7) (2024), distinguish ESC-50 benchmarking from a multilabel street deployment. That distinction also matters here: additive mixtures and a small reserved background panel do not establish deployment reliability.

## Consequences for the experiment

Three design choices follow from this context:

1. **Separate recording identities.** Training and evaluation use different DEMAND recordings. Different channels, crops or SNRs alone would not create independent environmental sources.
2. **Report the complete noise panel.** Equal waveform SNR does not imply equal perceptual difficulty. The analysis includes every predefined recording, segment and SNR, rather than selecting equally damaging or favourable conditions after observing accuracy.
3. **Match the classifier objective.** Each target has total training weight one in both augmented objectives: 0.50 clean plus two variants weighted 0.25 each. The total remains 1,200.

The third choice matters because the [scikit-learn 1.6.1 L-BFGS implementation](https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/linear_model/_logistic.py) scales the L2 coefficient penalty in its weighted-mean loss by `1 / (C * sum_weights)`. Holding `C` fixed while increasing total sample weight would change effective regularization. Matching the total weight avoids that additional difference.

Gaussian and recorded interference still differ jointly in spectral, temporal and semantic content. A positive recorded-panel result therefore identifies a training-distribution effect, not the isolated effect of diversity or one acoustic mechanism. The [protocol](protocol.json) specifies the recordings, assignment, SNRs and aggregation; the [results](README.md) report the measured tradeoff and conditional uncertainty.
