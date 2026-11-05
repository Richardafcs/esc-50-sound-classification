# Short-recording input-policy diagnostic

[Executed notebook](../../notebooks/ESC50_Duration_Diagnostic.ipynb) · [Fixed protocol](protocol.json) · [All metrics](metrics.csv) · [Paired effects](paired_effects.csv) · [Prediction transitions](transitions.csv)

## Findings

| Input policy | Same eligible recordings | Correct | Accuracy |
| --- | ---: | ---: | ---: |
| Original right-zero-pad to five seconds | 102 | 28 | 27.45% |
| Native duration, no padding | 102 | 34 | 33.33% |
| Left-zero-pad to five seconds | 102 | 27 | 26.47% |

Native minus right is **+5.88 percentage points**, conditional 95% recording-bootstrap interval **[−0.98, +12.75]**. The interval includes zero; improved accuracy is not established. Native input changes 46 predictions, repairing ten errors and introducing four. Left-padding changes 50 predictions on the same 102 recordings, with a net loss of one correct recording.

All 105 short recordings remain in the original right/left comparison: 29 versus 27 correct. Three recordings below the predetermined architecture minimum are excluded only from native inference. The original **45.17% external benchmark on 259 recordings remains unchanged**. No hybrid benchmark or replacement input policy is selected.

## Method and scope

The [official Cnn14_16k implementation](https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py) uses a centered STFT with hop 160 samples and five factor-two temporal pooling stages. At least 32 frames are required: `1 + floor(N/160) >= 32`, giving `N >= 4960` samples, or 0.31 seconds at 16 kHz. [Three exclusions](native_exclusions.csv) were determined from duration before new policy predictions; no adaptive native padding is introduced.

The encoder, scaler, logistic classifier and all 50 output classes are frozen. Waveform content and amplitude are retained. The primary comparison pairs exactly 102 recording IDs, with 4,000 fixed-seed bootstrap resamples. Secondary placement and class results are descriptive without multiplicity adjustment.

All 105 original right-padded labels reproduce, with maximum probability difference 0.00003013. Ten unchanged full five-second inputs have zero prediction changes across the fixed/native paths and maximum probability difference 0.00001458. These are implementation controls, not evidence of an acoustic mechanism. The saved baseline probability matrix is included as compact numerical evidence for replay.

The one-pass experiment completed on Colab Tesla T4 in **23.98 seconds**, excluding setup/downloads, with **zero fits**. [Execution summary](summary.json), [baseline replay](baseline_replay.json), [identical-input controls](negative_controls.csv), [all predictions](predictions.csv), [per-class recall](per_class.csv), [short-recording manifest](short_recordings.csv), and [control manifest](full_window_controls.csv) retain the complete evidence. Raw audio and the large pretrained backbone are excluded from Git.

## Research decision

Input duration and padding placement materially affect predictions, but their accuracy benefits are uncertain. Native processing changes convolution boundaries and temporal pooling statistics simultaneously. Neither comparison identifies one internal mechanism or establishes that padding explains the external short-recording deficit.

Global pooling is invariant to permutation of unchanged encoded tokens. Moving padding in a raw waveform can change those tokens through STFT boundaries, strided downsampling and convolution context; it is a different intervention from the earlier encoded-token experiment.

This FSD panel was already exposed when the diagnostic was proposed. Retain the original benchmark and default inference policy; report all fixed comparisons and close this bounded experiment. Any replacement policy requires a separately planned evaluation on untouched recordings. No additional padding variants, classifier tuning or architecture search follows these results.
