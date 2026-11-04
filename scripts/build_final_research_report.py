"""Assemble the final reviewed manuscript, saved result ledgers and 19 figures.

Document production only: no training, inference, test suite, or file hashing.
"""
from pathlib import Path
import csv
import html
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_english_pdf_reports as layout
from project_layout import project_path
from reportlab.platypus import Paragraph, Spacer, PageBreak
from reportlab.lib import colors

FINAL = ROOT / 'reports/final_20261105'
SOURCE = FINAL / 'Final_Research_Report_EN.md'
PDF = ROOT / 'reports/pdf/ESC50_Final_Research_Report_EN.pdf'


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                      ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in records])


def prepare_source():
    text = SOURCE.read_text().split('\n## Appendix A.')[0]
    groups = {r['model']: r for r in read_csv(ROOT/'reports/evidence/matrix_20261028/analysis/groups.csv') if r['group']=='B_architecture'}
    old = '| Model | Parameters | Accuracy, mean ± SD |\n|---|---:|---:|\n'
    if old in text:
        a = text.index(old); b = text.index('\n\n', a)
        labels = [('cnn_freq','CNN'),('crnn_freq','CNN-BiGRU'),('crnn_freq_ta','Temporal Attention (TA)'),('crnn_freq_ca','Channel Attention (CA)'),('crnn_freq_tca','Temporal + Channel Attention')]
        parameters = dict(cnn_freq=49058,crnn_freq=251810,crnn_freq_ta=251939,crnn_freq_ca=252906,crnn_freq_tca=253035)
        values=[]
        for model,label in labels:
            r=groups[model]
            values.append([label,f'{parameters[model]:,}',f"{100*float(r['accuracy_mean']):.2f} ± {100*float(r['accuracy_sample_sd']):.2f}%",f"{100*float(r['macro_f1_mean']):.2f} ± {100*float(r['macro_f1_sample_sd']):.2f}%"])
        text=text[:a]+table(['Model','Parameters','Accuracy, mean ± SD','Macro-F1, mean ± SD'],values)+text[b:]
    classic='The classical baselines also used different feature families: Log-Mel statistics with logistic regression achieved 42.21% accuracy and 38.86% macro-F1, while DCT-cepstral statistics with a linear SVM achieved 40.20% accuracy and 37.50% macro-F1. Their differences cannot be attributed exclusively to classifier choice.'
    text=text.replace(classic,'')
    if '### 3.0 Classical Baselines' not in text:
        text=text.replace('## 3. Results\n','## 3. Results\n\n### 3.0 Classical Baselines\n\n'+table(['Method','Accuracy','Macro-F1'],[['Log-Mel statistics + logistic regression','42.21%','38.86%'],['DCT cepstral statistics + linear SVM','40.20%','37.50%']])+'\n\nThe feature families and selection procedures differ; the score difference is not a classifier-only comparison.\n')
    if 'Each source resample retains' not in text:
        text=text.replace('These intervals do not characterize the population of possible training seeds.','These intervals do not characterize the population of possible training seeds. Each source resample retains every constituent clip and uses its actual, varying clip denominator; the same source draws are shared across paired models and fixed seeds. The paired-analysis RNG seed is 20261104; the temporal-intervention RNG seed is 20261105.')
    if 'Four representations were crossed' not in text:
        text=text.replace('Twelve feature-and-sampling configurations were evaluated across three training seeds, yielding 36 training runs.','Twelve feature-and-sampling configurations were evaluated across three training seeds, yielding 36 training runs. Four representations were crossed with three sampling/bandwidth settings: Log-Mel 64, Log-Mel 32, 40-coefficient DCT cepstral features, and Log-Linear 64. FFT windows and hops were held at 32 ms and 16 ms across sampling rates.')
    text=text.replace('Their BiGRU correctness counts changed under encoder-token reversal as follows:','Their BiGRU correctness counts changed under encoder-token reversal as follows. Each count concerns three fitted models on the same single recording, not three independent recordings:')
    if 'Perceptual near-duplicate' not in text:
        text=text.replace('**Incomplete independent confirmation.**','**Source grouping and near duplicates.** Source-strict grouping addresses known recording IDs; it does not rule out perceptual near-duplicates with different IDs. A complete perceptual near-duplicate audit was not performed.\n\n**Incomplete independent confirmation.**')
    records=json.loads((ROOT/'reports/pdf/figure_manifest.json').read_text())
    figures=[]
    text+='\n\n## Appendix A. Development-study and follow-up figure atlas\n\nThese figures retain their original evaluation scopes. In particular, the original robustness chart uses seed101 only; the completed three-seed result is shown separately. SD across fits is not a dataset confidence interval.\n\n'
    for f in records[0]['figures']:
        path=project_path(f['path'])
        title=f['title']
        if 'robustness.png' in str(path):title='Original seed101 robustness comparison: one fitted checkpoint per model'
        figures.append(dict(title=title,path=str(path.relative_to(ROOT)),scope='Original development matrix'))
    for title,path in [
        ('Completed three-seed Gaussian-noise and gain results; bars show sample SD, not confidence intervals',FINAL/'three_seed_robustness.png'),
        ('Feature-study loss trajectories: Log-Mel 32 and Log-Mel 64, three fixed seeds',ROOT/'reports/followup_20261104/convergence/validation_loss_curves.png'),
        ('Four selected real-audio error cases: waveform and Log-Mel evidence, not causal explanations',ROOT/'reports/followup_20261104/error_cases/error_case_atlas.png'),
        ('CRNN encoder-token intervention effects with conditional source-bootstrap intervals',ROOT/'reports/temporal_20261105/crnn_accuracy_change.png')]:
        figures.append(dict(title=title,path=str(path.relative_to(ROOT)),scope='November inference/analysis follow-up'))
    import os
    for f in figures:text+=f"![{f['title']}]({os.path.relpath(ROOT/f['path'],FINAL)})\n\n"
    text+='\n## Appendix B. Historical baseline figures: separate evaluation scope\n\nThe following eight figures belong to the earlier CNN/CRNN study, including historical fold-5 evaluation and fixed-recipe fold rotation. They are retained for completeness and provenance, not presented as new independent confirmation or merged into the current development averages.\n\n'
    for f in records[1]['figures']:
        if f['path'].endswith('right_confusion.png'):path=ROOT/'reports/figures/baseline_20261028/cnn_confusion.png'
        elif f['path'].endswith('left_confusion.png'):path=ROOT/'reports/figures/baseline_20261028/crnn_confusion.png'
        else:path=project_path(f['path'])
        title='Historical study - '+f['title']
        figures.append(dict(title=title,path=str(path.relative_to(ROOT)),scope='Historical baseline study; not new confirmation'))
        text+=f"![{title}]({os.path.relpath(path,FINAL)})\n\n"
    runs=[r for r in read_csv(ROOT/'reports/evidence/matrix_20261028/analysis/runs.csv') if r['kind'] in ('baseline','supervised','feature_ablation')]
    text+='\n## Appendix C. Original fitted-task ledger\n\nThe original matrix has 62 fitted tasks (60 neural fits and two classical fits) plus three original stress tasks. The following are the original development-validation records; follow-up inference adds no fitted models.\n\n'+table(['Task ID','Accuracy (%)','Macro-F1 (%)','Selected epoch'],[[r['id'],f"{float(r['accuracy'])*100:.2f}",f"{float(r['macro_f1'])*100:.2f}",r['best_epoch'] or 'not applicable'] for r in runs])+'\n'
    robust=read_csv(ROOT/'reports/followup_20261104/robustness_runs.csv')
    text+='\n## Appendix D. Completed three-seed robustness ledger\n\nAll rows use the same 398 validation recordings. Training seeds identify fixed fits, not independent audio datasets.\n\n'+table(['Model','Seed','Condition','Accuracy (%)','Correct','Clips'],[[r['variant'],r['seed'],r['condition'],f"{float(r['accuracy'])*100:.2f}",r['correct'],398] for r in robust])+'\n'
    temporal=read_csv(ROOT/'reports/temporal_20261105/metrics.csv')
    text+='\n## Appendix E. Temporal-intervention ledger\n\nChanged predictions are relative to each checkpoint\'s own identity predictions. The intervention reorders encoded tokens, not waveform samples. The first float32 absolute-tolerance failure and corrected numerical interpretation are retained in the project records.\n\n'+table(['Model','Seed','Condition','Accuracy (%)','Changed predictions','Clips'],[[r['model'],r['seed'],r['condition'],f"{float(r['accuracy'])*100:.2f}",r['changedpredcount'],398] for r in temporal])+'\n'
    text+='''\n## Appendix F. Sources, analysis and completion status

- [Official ESC-50 dataset and credits](https://github.com/karolpiczak/ESC-50). Dataset license and recording metadata remain in the project.
- [Original matrix summary tables](../evidence/matrix_20261028/analysis/).
- [Completed prediction and robustness analysis](../followup_20261104/report_en.md).
- [Temporal-order probe and numerical implementation record](../temporal_20261105/README.md).
- [New attention paired effects](attention_paired_effects.json).
- [Model-decision meeting and manuscript review](../../docs/research_meetings/2026-11-05_model_decision.md).
- [Initial attention methods, with completed-training update](../../docs/methodology/attention_phase1.md).
- [PANNs author repository](https://github.com/qiuqiangkong/audioset_tagging_cnn) and [AST author repository](https://github.com/YuanGongND/ast) document external-pretraining routes. No such models were downloaded, trained or evaluated here; published benchmark protocols and extra data differ from this study.

This report closes the current executed research scope. It preserves the original study and all available figure originals. Further model-performance optimization is a separately proposed phase, not an unperformed requirement hidden inside this completion claim. The report and computational analyses are agent-produced; role-based ChatGPT discussion is not independent human peer review. No new training, test-fold inference, file-hash verification or test-suite expansion was performed to produce this final document.
'''
    SOURCE.write_text(text)
    for n,f in enumerate(figures,1):f['number']=n
    (FINAL/'figure_atlas.json').write_text(json.dumps(figures,indent=2)+'\n')
    return text,figures,len(runs),len(robust),len(temporal)


class FinalBuilder(layout.Builder):
    def figure(self,path,title,note=''):
        path=Path(path)
        record=dict(number=len(self.figures)+1,title=layout.clean(title),path=str(path.relative_to(ROOT)),note=layout.clean(note or title))
        self.figures.append(record)
        key='fig-'+self.key+'-'+str(record['number'])
        self.story.append(Paragraph('<link href="#'+key+'" color="#167D8D"><b>Figure '+str(record['number'])+'. '+html.escape(layout.clean(title))+'</b></link> - full-page figure atlas.',layout.ST['small']))


class FinalDoc(layout.ReportDoc):
    def header(self,c,doc):
        w,h=c._pagesize;c.saveState();c.setStrokeColor(layout.LINE);c.line(layout.M,h-34,w-layout.M,h-34)
        c.setFont('BodyBold',8);c.setFillColor(layout.TEAL);c.drawString(layout.M,h-26,'ESC-50 / FINAL RESEARCH REPORT')
        c.setFont('Body',8);c.setFillColor(layout.MUTED);c.drawRightString(w-layout.M,h-26,'5 November 2026')
        c.line(layout.M,31,w-layout.M,31);c.drawString(layout.M,20,'Development and historical evidence are labeled separately')
        c.drawRightString(w-layout.M,20,str(doc.page));c.restoreState()


def main():
    text,figures,fits,robust,temporal=prepare_source()
    b=FinalBuilder('FINAL',SOURCE)
    b.story.extend([Spacer(1,44),b.para('COMPLETED STUDY / 5 NOVEMBER 2026','kicker'),layout.AccentRule(),Spacer(1,24),b.para('Environmental Sound Classification on ESC-50','title'),b.para('Architecture, robustness and representation','subtitle'),b.para('Final research report with original fitted-task results, completed three-seed robustness, encoder-token interventions, and all 19 scientific figures.'),Spacer(1,20),b.para('65 original tasks  |  60 neural fits  |  19 figures','h2'),Spacer(1,14),b.para('Every current score is development-validation evidence. Historical fold-5 and fold-rotation figures are retained in a separate appendix. No new independent confirmation is claimed.'),b.para('Manuscript drafted and reviewed through role-based ChatGPT discussions; existing experiment artifacts supplied the results. No additional attention training was required to complete this study.','small'),PageBreak()])
    b.markdown(text);b.evidence();b.emit_figures();doc=FinalDoc(PDF,'FINAL','Environmental Sound Classification on ESC-50 - Final Research Report');doc.build(b.story)
    print(json.dumps(dict(pdf=str(PDF),figures=len(b.figures),original_fitted_rows=fits,robustness_rows=robust,temporal_rows=temporal,tables=b.table_count),indent=2))


if __name__=='__main__':main()
