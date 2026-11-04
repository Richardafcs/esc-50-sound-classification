"""Render the checkpoint manuscript and its nine saved scientific figures.

Document production only. No training, inference, tests or file-hash checks.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_final_research_report import FinalBuilder,layout
from reportlab.platypus import Spacer,PageBreak

SOURCE=ROOT/'reports/checkpoint_20261105/Checkpoint_Report_EN.md'
OUTPUT=ROOT/'reports/pdf/ESC50_Model_Improvement_Checkpoint_EN.pdf'

class CheckpointDoc(layout.ReportDoc):
    def header(self,c,doc):
        w,h=c._pagesize;c.saveState();c.setStrokeColor(layout.LINE)
        c.line(layout.M,h-34,w-layout.M,h-34)
        c.setFont('BodyBold',8);c.setFillColor(layout.TEAL)
        c.drawString(layout.M,h-26,'COMP9444 / MODEL IMPROVEMENT CHECKPOINT')
        c.setFont('Body',8);c.setFillColor(layout.MUTED)
        c.drawRightString(w-layout.M,h-26,'5 November 2026')
        c.line(layout.M,31,w-layout.M,31)
        c.drawString(layout.M,20,'ESC-50 | development validation | completed experiments and scoped comparisons')
        c.drawRightString(w-layout.M,20,str(doc.page));c.restoreState()


def main():
    b=FinalBuilder('CHECKPOINT',SOURCE)
    b.story.extend([Spacer(1,40),b.para('COMP9444 / RESEARCH CHECKPOINT','kicker'),layout.AccentRule(),Spacer(1,25),b.para('Improving Environmental Sound Classification on ESC-50','title'),b.para('Model improvement, alternative approaches and comparative evaluation','subtitle'),b.para('CNN / BiGRU / attention | augmentation | frozen PANNs','h2'),Spacer(1,16),b.para('Research CNN: 72.61% mean accuracy. Frozen PANNs + linear head: 91.21% accuracy and 91.10% macro-F1 on the same 398 development-validation clips.'),b.para('A direct response to the five checkpoint requirements, with nine scientific figures, exact metric tables, paired comparisons, literature-informed choices and explicit limitations.'),Spacer(1,12),b.para('The original team-repository CNN is a different implementation with no committed result evidence; its performance is not inferred from our research CNN. PANNs introduces external pretraining and a much larger frozen model.','small'),b.para('Existing experiments supply the results. AI-assisted review is documented; no human-team discussion or independent replication is fabricated.','small'),PageBreak()])
    b.markdown(SOURCE.read_text())
    b.story.append(PageBreak());b.evidence();b.emit_figures()
    CheckpointDoc(OUTPUT,'CHECKPOINT','ESC-50 Model Improvement Checkpoint').build(b.story)
    (SOURCE.parent/'figure_manifest.json').write_text(json.dumps(b.figures,indent=2))
    print(json.dumps({'pdf':str(OUTPUT),'figures':len(b.figures),'tables':b.table_count}))

if __name__=='__main__':main()
