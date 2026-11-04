"""Assemble the executed transfer-study report from saved predictions and figures."""
from pathlib import Path
import csv,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_final_research_report import FinalBuilder,FinalDoc,layout
from reportlab.platypus import Spacer,PageBreak,KeepTogether

OUT=ROOT/'reports/panns_20261105'
SOURCE=OUT/'report_en.md'
PDF=ROOT/'reports/pdf/ESC50_PANNs_Transfer_Report_EN.pdf'


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


class TransferBuilder(FinalBuilder):
    def table(self, headers, rows):
        start=len(self.story)
        super().table(headers,rows)
        if len(rows)<=4:
            self.story[start:]=[KeepTogether(self.story[start:])]


class TransferDoc(FinalDoc):
    def header(self,c,doc):
        w,h=c._pagesize;c.saveState();c.setStrokeColor(layout.LINE)
        c.line(layout.M,h-34,w-layout.M,h-34)
        c.setFont('BodyBold',8);c.setFillColor(layout.TEAL)
        c.drawString(layout.M,h-26,'ESC-50 / PANNs TRANSFER STUDY')
        c.setFillColor(layout.MUTED);c.setFont('Body',8)
        c.drawRightString(w-layout.M,h-26,'5 November 2026')
        c.line(layout.M,31,w-layout.M,31)
        c.drawString(layout.M,20,'Development validation | external pretraining | fixed linear head')
        c.drawRightString(w-layout.M,20,str(doc.page));c.restoreState()


def main():
    s=json.loads((OUT/'comparison_summary.json').read_text());run=json.loads((OUT/'run_summary.json').read_text())
    paired=list(csv.DictReader((OUT/'paired_errors.csv').open()))
    classes=list(csv.DictReader((OUT/'per_class_comparison.csv').open()))
    primary=s['comparisons']['Channel Attention'];timing=run['timing_seconds']
    systems=[['CNN, scratch-trained','72.61 +/- 0.25','Three existing fixed fits'],['CNN + BiGRU, scratch-trained','73.87 +/- 1.26','Three existing fixed fits'],['CRNN + Channel Attention, scratch-trained','74.37 +/- 1.15','Three existing fixed fits'],['Frozen PANNs + linear head',f"{100*s['panns_accuracy']:.2f}",'One fixed encoder and one head fit']]
    effects=[[name,f"{item['delta_accuracy_pp']:+.2f}",f"[{item['source_bootstrap_low_pp']:+.2f}, {item['source_bootstrap_high_pp']:+.2f}]"] for name,item in s['comparisons'].items()]
    ca_rows=[[r['seed'],r['b_only_correct'],r['a_only_correct'],r['net_correct']] for r in paired if r['reference']=='Channel Attention']
    class_table=[[r['category'],r['n'],r['panns_correct'],f"{100*float(r['panns_accuracy']):.1f}",f"{100*float(r['Channel Attention_accuracy_mean']):.1f}"] for r in classes]
    text=f'''# ESC-50 Frozen PANNs Transfer Study - 5 November 2026

## 1. Main finding and decision

A frozen AudioSet-pretrained PANNs Cnn14_16k representation with one fixed logistic-regression head classified **363 of 398 validation clips correctly: 91.21% accuracy and 91.10% macro-F1**. No encoder fine-tuning or hyperparameter search was performed.

The primary paired difference against the mean accuracy of the three existing Channel Attention fits was **+{primary['delta_accuracy_pp']:.2f} percentage points (pp)**, with a conditional 95% source-bootstrap interval of **[{primary['source_bootstrap_low_pp']:+.2f}, {primary['source_bootstrap_high_pp']:+.2f}] pp**. This exceeds the execution-time-frozen +3 pp practical screening threshold. That threshold is a project decision rule, not a statistical theorem.

The result supports retaining frozen pretrained audio features plus a linear head as the stronger clean-validation candidate in this project. It does not independently confirm generalization, isolate the effect of pretraining from model capacity or frontend changes, or establish noise robustness. The existing scratch-trained CNN and CRNN remain useful controlled baselines.

## 2. Question, protocol and evaluation scope

The question was fixed before any new feature extraction or head-fitting result: does this externally pretrained system outperform the existing Channel Attention reference on the same development recordings?

The [execution-time-frozen protocol](protocol.json) specifies one embedding extraction, one deterministic head fit, C=1 without search, the reference fits, source resampling and the stopping rule. This is not external preregistration. The broader project has previously exposed every official ESC-50 fold, including the current development-validation fold.

| Partition | Role | Clips |
| --- | --- | --- |
| Official folds 1-3 | Train-only scaler and linear-head fit; exclude sources appearing in validation or test roles | 1,200 |
| Official fold 4 | Development validation; exclude sources appearing in fold 5 | 398 from 295 sources |
| Official fold 5 | Metadata used only to exclude source overlap; no waveform decoding or model scoring in this phase | 400 designated clips |

The dataset revision remains 33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6. All compared validation predictions are aligned by filename and target. Source grouping addresses known recording IDs; perceptual duplicates and AudioSet/ESC-50 pretraining overlap have not been comprehensively excluded.

## 3. Frozen encoder and linear classifier

The [PANNs author repository](https://github.com/qiuqiangkong/audioset_tagging_cnn) provides the Cnn14_16k implementation and its [official checkpoint](https://zenodo.org/records/3987831). The source revision used here was d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4; the checkpoint identifier is Cnn14_16k_mAP=0.438.pth. Its filename's mAP refers to AudioSet, not this ESC-50 accuracy.

| Component | Fixed implementation |
| --- | --- |
| Waveform | Mono, scipy resample_poly to 16 kHz; exactly five seconds; no peak normalization |
| Encoder frontend | Author model's internal Log-Mel: FFT/window 512, hop 160, 64 bands, 50-8,000 Hz |
| Representation | 2,048-dimensional embedding after fc1 ReLU; eval mode disables dropout and augmentation |
| Encoder state | Strict state-dictionary loading; all parameters frozen; inference_mode |
| Scaling | StandardScaler fitted only on the 1,200 training embeddings |
| Head | Multiclass logistic regression; C=1, L2 penalty, lbfgs, max_iter=3,000, tol=1e-6 |
| Repetition | One fixed head fit; no nominal seed repetitions, tuning, augmentation or encoder fine-tuning |

For one clip, let e be its 2,048-dimensional frozen embedding. Training-set means and scales produce z=(e-mu_train)/sigma_train. The head computes 50 logits Wz+b, then predicts the largest logit. W has shape [50, 2048], and b has shape [50]. Thus the head has 102,450 fitted coefficients and biases. Standardization remains an affine transformation of the embedding.

Only W and b are optimized by the regularized multinomial logistic objective. The encoder is nonlinear but frozen; this is a linear probe of its representation. The loaded author model contains {run['loaded_model_parameters']:,} parameters, with zero trainable encoder parameters. Its external pretraining computation is excluded from this phase's timing. The much larger encoder means that a simple head does not imply a small end-to-end model.

## 4. Clean-validation comparison

{table(['System','Accuracy (%), mean +/- sample SD where repeated','Scope'],systems)}

The grey reference systems each summarize three existing fits, not three independent audio datasets. PANNs reports one fit; a three-seed SD is neither estimated nor fabricated. References are individual-fit means, not an ensemble classifier.

![System comparison: scratch-trained references and the fixed pretrained system](system_comparison.png)

## 5. Paired uncertainty and changes in errors

The primary estimand is the new system's clip accuracy minus the arithmetic mean of the three fixed Channel Attention accuracies. For each clip, the new correctness indicator is compared with the mean of its three reference correctness indicators. The analysis resamples 295 src_file groups with replacement 4,000 times using RNG seed 20261105. Every sampled source retains all constituent clips; each resample divides by its actual clip count. The same draws are shared across systems. The resampled denominator ranged from 360 to 450 clips.

{table(['Reference three-fit mean','PANNs difference (pp)','Conditional 95% source interval (pp)'],effects)}

These intervals condition on the fixed pretrained encoder, fitted head, reference fits and development dataset. They do not include future training randomness, pretraining uncertainty, historical model-selection bias or independent-dataset uncertainty. They are not simultaneous confidence intervals; CNN and BiGRU comparisons are secondary.

![Paired accuracy gains with conditional source-bootstrap intervals](paired_accuracy_gains.png)

For each Channel Attention fit, the table distinguishes repaired reference errors from correct reference predictions that PANNs loses:

{table(['CA seed','Errors repaired','New errors','Net additional correct'],ca_rows)}

PANNs improves on each reference fit, while also introducing some new errors. Of the 41 clips previously misclassified by CNN, BiGRU, Channel Attention and noise-trained CNN in all three fits, PANNs correctly classifies 26 and still misses 15. This is a descriptive analysis of a previously selected error set, not an independent benchmark.

## 6. Remaining failures

There are 35 remaining validation errors. Wind and washing_machine each have 4/8 correct predictions; footsteps has 5/8. Twenty-eight classes have every available validation clip classified correctly. Each class contains only seven or eight clips, so neither perfect nor low class scores establish class-wide generalization. No new listening study or label audit was performed.

The complete 50-class table is retained in Appendix A and the row-normalized confusion matrix shows all categories.

![All-class row-normalized confusion matrix for the fixed pretrained system](confusion_matrix.png)

## 7. Execution, compatibility repair and delivery verification

Execution used Google Colab Tesla T4 through the official Colab CLI and its SSH transport. Python {run['environment']['python']}, PyTorch {run['environment']['torch']}, NumPy {run['environment']['numpy']}, scipy {run['environment']['scipy']}, scikit-learn {run['environment']['sklearn']}, torchlibrosa {run['environment']['torchlibrosa']} and librosa {run['environment']['librosa']} were recorded in the [run summary](run_summary.json).

| Measured stage | Wall time |
| --- | --- |
| Decode, resample and extract training embeddings | {timing['train_features']:.2f} s |
| Decode, resample and extract validation embeddings | {timing['val_features']:.2f} s |
| Fit the single linear head | {timing['linear_fit']:.2f} s |
| Main program, including model loading and saving outputs | {timing['total']:.2f} s |

These times exclude package installation, data and checkpoint downloads, prior compatibility repair, external pretraining and report production. They are not comparable to scratch-training-loop times as total model-development costs. L-BFGS converged after 137 iterations with no recorded convergence warning.

The first attempt stopped before extraction because the legacy official checkpoint contains NumPy metadata that newer PyTorch weights-only loading does not allow by default. The repair used a targeted NumPy data-type allowlist while retaining weights_only=True and strict model loading; the [failed attempt](checkpoint_loading_attempt1.log) is retained. This compatibility repair changed no model, head hyperparameter or score-selection rule.

Saved normalization and linear coefficients were replayed on all 398 original validation embeddings using the portable NumPy head: **zero prediction changes, 363 correct**. This verifies the exported head on saved features; it is not a second encoder extraction or independent replication. Predictions and probability vectors were downloaded before the Colab T4 session was terminated. No resource upgrade or purchase was made.

## 8. Interpretation and stopping decision

This fixed pretrained system is a substantially better clean-validation candidate under the present protocol. The observed gain is much larger than the earlier +0.50 pp Channel Attention difference, making pretrained representations a more productive practical direction than another attention-module search in this project.

However, the comparison changes external training data, encoder size, learned representation and frontend together. It cannot identify which change causes the gain. Unknown pretraining overlap and repeated development exposure also prevent a claim of fresh independent generalization. PANNs was not evaluated on the Gaussian-noise or gain conditions in this phase; the previous noise-trained CNN findings do not transfer automatically to it.

The fixed experiment is complete and stops here, with no fine-tuning, alternate checkpoint, regularization sweep or additional seed fits. Any future robustness or independent-data study needs its own stated evaluation scope. The previous 65-task study and its conclusions remain a separate record.

## 9. Evidence and reproduction

[Protocol](protocol.json), [paired summary](comparison_summary.json), [per-fit paired errors](paired_errors.csv), [all fixed reference predictions](reference_predictions.csv), [per-class comparison](per_class_comparison.csv), [clip predictions](validation_predictions.csv), [probabilities](validation_probabilities.csv), [run log](run.log), and [portable-head replay](portable_head_replay.json) are retained. Code: [single extraction and fit](../../scripts/run_panns_linear_probe.py) and [offline comparison and plotting](../../scripts/analyze_panns_transfer.py). The [model card](../../models/panns_linear/MODEL_CARD.md) describes the small exported head and its required external encoder.

Two existing Chrome ChatGPT roles supplied an implementation draft and protocol/result review. Local integration, actual Colab execution and recorded predictions supplied the experimental evidence. Role agreement is not external human review or independent replication.

## Appendix A. All 50 validation classes

PANNs counts concern one fit. CA accuracy is the mean of three fixed fits on the same class clips. It is not based on three times as many independent recordings.

{table(['Class','Clips','PANNs correct','PANNs accuracy (%)','CA mean accuracy (%)'],class_table)}
'''
    SOURCE.write_text(text)
    b=TransferBuilder('PANNs',SOURCE)
    b.story.extend([Spacer(1,40),b.para('EXECUTED TRANSFER STUDY / 5 NOVEMBER 2026','kicker'),layout.AccentRule(),Spacer(1,24),b.para('Frozen PANNs + Linear Classification','title'),b.para('ESC-50 development-validation results','subtitle'),b.para('91.21% accuracy | 91.10% macro-F1','h1'),b.para('363 / 398 clips correct. One fixed feature extraction and one linear-head fit on Colab Tesla T4.'),Spacer(1,20),b.para('Primary paired gain over Channel Attention: +16.83 pp','h2'),b.para('Conditional 95% source-bootstrap interval: +12.61 to +21.35 pp.'),Spacer(1,16),b.para('All three new scientific figures and all 50 class results are included. External pretraining, larger encoder capacity and a different frontend distinguish this system from the scratch-trained references. Independent generalization and noise robustness are not established.'),b.para('A separate research phase following the original 65-task study. Actual run artifacts support the results; ChatGPT roles assisted code drafting and review.','small'),PageBreak()])
    b.markdown(text);b.story.append(PageBreak());b.evidence();b.emit_figures()
    TransferDoc(PDF,'PANNs','ESC-50 Frozen PANNs Transfer Study').build(b.story)
    (OUT/'pdf_figures.json').write_text(json.dumps(b.figures,indent=2))
    print(json.dumps({'pdf':str(PDF),'figures':len(b.figures),'class_rows':len(classes),'tables':b.table_count}))

if __name__=='__main__':main()
