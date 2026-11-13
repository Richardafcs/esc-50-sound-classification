# Related Work

Environmental sound classification has been approached through handcrafted audio descriptors, learned spectrogram features, recurrent aggregation and large pretrained representations. This project compares selected approaches on its stated ESC-50 development partition. It is an empirical course study, not a new architecture or an exact reproduction of the cited methods.

## Environmental-sound benchmarks and CNNs

[Piczak's ESC dataset paper (2015)](https://doi.org/10.1145/2733373.2806390) establishes the dataset context. The [official ESC-50 repository](https://github.com/karolpiczak/ESC-50) documents its classes, recording metadata and predefined folds. Our evaluation additionally applies the metadata source exclusions described in the [report](../reports/report.md).

[Piczak's CNN paper (2015)](https://doi.org/10.1109/MLSP.2015.7324337) motivates learning representations from spectrograms. Our baseline uses a small full-clip CNN that retains encoded frequency positions before temporal averaging. Its architecture and training recipe are independent of the original paper.

[Boddapati et al. (2017)](https://doi.org/10.1016/j.procs.2017.08.250) provide context for treating audio-derived representations as inputs to image recognition networks. Their AlexNet/GoogLeNet approach is distinct from our scratch CNN and the audio-pretrained PANNs reference. We do not report an AlexNet or GoogLeNet experiment.

## Augmentation and mixed examples

[Salamon and Bello (2017)](https://arxiv.org/abs/1608.04363) study deep CNNs and waveform augmentation, with main experiments on UrbanSound8K. Their findings motivate examining augmentation on environmental sounds, but their scores and augmentation effects cannot be transferred directly to our ESC-50 partition.

Our project evaluates Gaussian waveform interference, waveform Mixup and time/frequency masking. [Zhang et al.'s mixup paper (ICLR 2018)](https://arxiv.org/abs/1710.09412) describes convex combinations of examples and targets. [Tokozume et al.'s between-class sound learning (ICLR 2018)](https://arxiv.org/abs/1711.10282) is related sound-mixing work with a different formulation. We implement generic waveform Mixup rather than claiming to reproduce the latter's complete recipe.

[SpecAugment (Park et al., 2019)](https://arxiv.org/abs/1904.08779) introduces spectrogram augmentation for speech recognition. Our adaptation includes time and frequency masks, without time warping. Its negative clean result is specific to the tested environmental-sound configuration.

## Recurrence and attention

Our CNN–BiGRU extension processes a sequence of encoded spectrogram features before clip-level pooling. [Zhang et al. (2019)](https://doi.org/10.1109/ACCESS.2019.2939495) motivate temporal and channel attention for environmental-sound classification. We implement lightweight temporal pooling and channel gating as separate comparisons. These adaptations do not reproduce all of the paper's architecture, optimization or ablations, so their results neither confirm nor refute the full published method.

## Pretrained audio representations

[PANNs (Kong et al., 2020)](https://arxiv.org/abs/1912.10211) investigates supervised AudioSet pretraining and transfer. We use the official later-code-version **Cnn14_16k** checkpoint from the [author implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn), with the encoder frozen and a regularized linear classifier fitted to ESC-50 embeddings. This preserves the pretrained frontend and differs from both full fine-tuning and the paper's other transfer configurations.

Other representation families provide context: [BYOL-A](https://arxiv.org/abs/2103.06695), [AST](https://arxiv.org/abs/2104.01778), [PaSST](https://arxiv.org/abs/2110.05069), [HTS-AT](https://arxiv.org/abs/2202.00874), [BEATs](https://arxiv.org/abs/2212.09058), [CLAP](https://arxiv.org/abs/2206.04769), [M2D](https://arxiv.org/abs/2404.06095) and [EAT](https://arxiv.org/abs/2401.03497). These were not implemented in this project. Their published results use different pretraining, adaptation and evaluation procedures and are not directly ranked against our development scores.

## Evaluation and scope

The implemented comparisons cover scratch CNN/GRU/attention models, augmentation, and frozen pretrained features. Model capacity, frontend settings and external data differ across these groups. The PANNs gain is therefore a full-system result rather than an isolated estimate of pretraining's effect. Likewise, a benefit under Gaussian interference does not establish reliability under other backgrounds.

The [literature catalog](literature_matrix.csv) lists 26 references, their relationship to the project and the source material available for each entry. Contextual entries based on abstracts or publication records do not support detailed implementation or quantitative reproduction claims. The numerical findings in the report come from this project's saved experiments.
