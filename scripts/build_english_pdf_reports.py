"""Build the two English reports from reviewed project text and existing figures."""
from pathlib import Path
import csv,json,re,html,textwrap,hashlib
from collections import OrderedDict
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate,PageTemplate,Frame,Paragraph,Spacer,Table,TableStyle,PageBreak,NextPageTemplate,Image,Preformatted,KeepTogether,Flowable
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.lib.pagesizes import A4,landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

P=Path(__file__).resolve().parents[1];OUT=P / 'reports/pdf';TMP=P/'.cache/pdf';OUT.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
FONT=Path('/System/Library/Fonts/Supplemental')
for name,file in [('Body','Arial.ttf'),('BodyBold','Arial Bold.ttf'),('BodyItalic','Arial Italic.ttf'),('BodyBoldItalic','Arial Bold Italic.ttf'),('Mono','Courier New.ttf')]:
 pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBoldItalic')
NAVY=colors.HexColor('#153047');TEAL=colors.HexColor('#167D8D');INK=colors.HexColor('#23323E');MUTED=colors.HexColor('#576775');LIGHT=colors.HexColor('#F1F6F8');LINE=colors.HexColor('#D8E2E8')
W,H=A4;M=44;CW=W-2*M
ST={
 'body':ParagraphStyle('body',fontName='Body',fontSize=10,leading=14.5,textColor=INK,spaceAfter=8,splitLongWords=True),
 'h1':ParagraphStyle('h1',fontName='BodyBold',fontSize=18,leading=23,textColor=NAVY,spaceBefore=16,spaceAfter=9,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=12,leading=16,textColor=TEAL,spaceBefore=13,spaceAfter=7,keepWithNext=True),
 'small':ParagraphStyle('small',fontName='Body',fontSize=8,leading=11,textColor=MUTED,spaceAfter=6,splitLongWords=True),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.5,leading=11.5,textColor=INK,splitLongWords=True),
 'th':ParagraphStyle('th',fontName='BodyBold',fontSize=8.5,leading=11.5,textColor=colors.white,splitLongWords=True),
 'code':ParagraphStyle('code',fontName='Mono',fontSize=7.8,leading=11,textColor=INK,backColor=LIGHT,borderPadding=9,spaceAfter=10),
 'figtitle':ParagraphStyle('figtitle',fontName='BodyBold',fontSize=15,leading=19,textColor=NAVY,spaceAfter=12),
 'title':ParagraphStyle('title',fontName='BodyBold',fontSize=34,leading=40,textColor=NAVY,spaceAfter=20),
 'subtitle':ParagraphStyle('subtitle',fontName='Body',fontSize=19,leading=25,textColor=TEAL,spaceAfter=22),
 'kicker':ParagraphStyle('kicker',fontName='BodyBold',fontSize=10,leading=14,textColor=TEAL,spaceAfter=18),
}

def clean(s):
 for ch in '\u2010\u2011\u2012\u2013\u2014\u2212':s=s.replace(ch,'-')
 s=s.replace('Data / 数据','Data').replace('Plot code / 绘图代码','Plot code').replace('WAV→logits','WAV-to-logits')
 return re.sub(r'[\u4e00-\u9fff]','',s)

class AccentRule(Flowable):
 def __init__(self,width=58):Flowable.__init__(self);self.width=width;self.height=6
 def draw(self):self.canv.setFillColor(TEAL);self.canv.rect(0,1,self.width,4,fill=1,stroke=0)

class ReportDoc(BaseDocTemplate):
 def __init__(self,path,report_id,title):
  super().__init__(str(path),pagesize=A4,leftMargin=M,rightMargin=M,topMargin=49,bottomMargin=45,title=title,author='ESC-50 research project | agent-produced internal verification')
  self.report_id=report_id;self.short_title=title;self.section_counter=0
  for name,size in [('portrait',A4),('landscape',landscape(A4))]:
   w,h=size;frame=Frame(M,45,w-2*M,h-94,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id=name)
   self.addPageTemplates(PageTemplate(name,[frame],onPage=self.header,pagesize=size))
 def header(self,c,doc):
  w,h=c._pagesize;c.saveState();c.setStrokeColor(LINE);c.line(M,h-34,w-M,h-34)
  c.setFont('BodyBold',8);c.setFillColor(TEAL);c.drawString(M,h-26,'ESC-50  /  '+self.report_id)
  c.setFillColor(MUTED);c.setFont('Body',8);c.drawRightString(w-M,h-26,'English results report | 29 October 2026')
  c.line(M,31,w-M,31);c.drawString(M,20,'Research evidence - development and historical scopes are labeled')
  c.drawRightString(w-M,20,str(doc.page));c.restoreState()
 def afterFlowable(self,flow):
  if isinstance(flow,Paragraph) and getattr(flow,'_bookmark',None):
   key=flow._bookmark;self.canv.bookmarkPage(key);self.canv.addOutlineEntry(flow.getPlainText(),key,level=0,closed=False)

class Builder:
 def __init__(self,key,source):self.key=key;self.source=source;self.refs=OrderedDict();self.figures=[];self.story=[];self.heading_count=0;self.table_count=0
 def inline(self,s):
  s=clean(s);slots=[]
  def link(m):
   label,target=m.groups();label=clean(label).strip(' /')
   if not label:return ''
   if target.startswith(('http://','https://')):value='<link href="'+html.escape(target,quote=True)+'" color="#167D8D">'+html.escape(label)+'</link>'
   else:
    path=(self.source.parent/target.split('#')[0]).resolve()
    if path not in self.refs:self.refs[path]='P'+str(len(self.refs)+1).zfill(2)
    value=html.escape(label)+' <super><font size="7">['+self.refs[path]+']</font></super>'
   token='ZZLINK'+str(len(slots))+'ZZ';slots.append(value);return token
  s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,s)
  s=html.escape(s,quote=False)
  s=re.sub(r'`([^`]+)`',lambda m:'<font name="Mono" size="8">'+m.group(1)+'</font>',s)
  s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
  for i,v in enumerate(slots):s=s.replace('ZZLINK'+str(i)+'ZZ',v)
  return s
 def para(self,text,style='body'):return Paragraph(self.inline(text),ST[style])
 def heading(self,text,level=2):
  self.heading_count+=1;p=self.para(text,'h1' if level==2 else 'h2');p._bookmark=self.key+'-s'+str(self.heading_count);self.story.append(p)
 def cover(self,title,subtitle,scope,stats):
  self.story.extend([Spacer(1,44),self.para('ENVIRONMENTAL SOUND CLASSIFICATION / REPORT '+self.key,'kicker'),AccentRule(),Spacer(1,24),self.para(title,'title'),self.para(subtitle,'subtitle'),self.para(scope),Spacer(1,19)])
  data=[[self.para(value,'h1') for label,value in stats],[self.para(label,'small') for label,value in stats]]
  table=Table(data,colWidths=[CW/len(stats)]*len(stats));table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),14),('TOPPADDING',(0,0),(-1,0),9),('BOTTOMPADDING',(0,1),(-1,1),12)]));self.story.extend([table,Spacer(1,22),self.para('Prepared 29 October 2026. Main recorded experiments: 28 October 2026.','small'),self.para('All figures are embedded at their original resolution in the full-page figure atlas. Click any figure reference to jump to its page. Full tables and scientific limitations are retained. Companion-project evidence paths are indexed at the end.','small'),self.para('This report documents internal agent-produced execution and verification. It does not certify external human replication, learner mastery, state of the art or production readiness.','small'),PageBreak()])
 def widths(self,headers,rows):
  h=' '.join(headers).lower();n=len(headers)
  if headers[0]=='Severity':v=[35,220,252]
  elif headers[0]=='Group':v=[42,48,417]
  elif 'accuracy: 101' in h:v=[143,128,80,82,74]
  elif 'paired contrast' in h:v=[160,135,108,104]
  elif headers[0]=='Case':v=[225,141,141]
  elif 'zero-recall' in h:v=[154,66,287]
  elif 'cpu wav' in h:v=[181,111,111,104]
  elif n==6:v=[129,36,83,83,91,85]
  elif 'train / validation' in h:v=[136,69,140,81,81]
  elif headers[0]=='Task ID':v=[261,78,78,90]
  elif headers[0]=='Contrast (right minus left)':v=[285,114,108]
  else:
   lens=[max(len(clean(row[j])) for row in [headers]+rows) for j in range(n)];v=[max(11,min(x,50))**.75 for x in lens]
  return [CW*x/sum(v) for x in v]
 def table(self,headers,rows):
  self.table_count+=1
  table=Table([[self.para(x,'th') for x in headers]]+[[self.para(x,'cell') for x in row] for row in rows],colWidths=self.widths(headers,rows),repeatRows=1,hAlign='LEFT')
  commands=[('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.8,TEAL),('LINEBELOW',(0,1),(-1,-1),.3,LINE)]
  for i in range(1,len(rows)+1):
   if i%2==0:commands.append(('BACKGROUND',(0,i),(-1,i),LIGHT))
  table.setStyle(TableStyle(commands));self.story.extend([Spacer(1,3),table,Spacer(1,11)])
 def figure(self,path,title,note=''):
  path=Path(path);assert path.is_file(),path
  n=len(self.figures)+1;record=dict(number=n,title=clean(title),path=str(path.relative_to(P)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),note=clean(note or title))
  self.figures.append(record)
  key='fig-'+self.key+'-'+str(n)
  self.story.append(Paragraph('<link href="#'+key+'" color="#167D8D"><b>Figure '+str(n)+'. '+html.escape(clean(title))+'</b></link> - see the full-page figure atlas.',ST['small']))
 def emit_figures(self):
  for record in self.figures:
   path=P/record['path'];iw,ih=PILImage.open(path).size
   template='portrait' if iw/ih<1.45 else 'landscape';size=A4 if template=='portrait' else landscape(A4)
   maxw=size[0]-2*M;frameh=size[1]-94
   heading=self.para('Figure '+str(record['number'])+'. '+record['title'],'figtitle')
   heading._bookmark='fig-'+self.key+'-'+str(record['number'])
   caption=self.para(record['note'],'small')
   source=self.para('Source: `'+record['path']+'`. Original saved figure; no experiment was rerun for this PDF.','small')
   heights=[f.wrap(maxw,10000)[1]+getattr(f,'getSpaceAfter',lambda:0)() for f in [heading,caption,source]]
   maxh=frameh-sum(heights)-17
   scale=min(maxw/iw,maxh/ih);im=Image(str(path),width=iw*scale,height=ih*scale);im.hAlign='CENTER'
   self.story.extend([NextPageTemplate(template),PageBreak(),heading,im,Spacer(1,9),caption,source])
 def markdown(self,text):
  lines=text.splitlines();i=0
  while i<len(lines):
   s=lines[i].strip()
   if not s:i+=1;continue
   if s.startswith('# '):i+=1;continue
   if s.startswith('##'):
    level=len(s)-len(s.lstrip('#'));self.heading(s[level:].strip(),level);i+=1;continue
   if s.startswith('```'):
    block=[];i+=1
    while i<len(lines) and not lines[i].strip().startswith('```'):block.append(clean(lines[i]));i+=1
    wrapped=[]
    for line in block:
     chunks=textwrap.wrap(line,width=95,subsequent_indent='  ',break_long_words=False,break_on_hyphens=False) or ['']
     for j,chunk in enumerate(chunks):wrapped.append(chunk+(' '+chr(92) if j<len(chunks)-1 else ''))
    self.story.append(Preformatted('\n'.join(wrapped),ST['code']));i+=1;continue
   m=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',s)
   if m:
    title,target=m.groups();self.figure((self.source.parent/target).resolve(),title);i+=1;continue
   if s.startswith('|') and i+1<len(lines) and re.match(r'^\|[\s:|\-]+\|$',lines[i+1].strip()):
    headers=[c.strip() for c in s.strip('|').split('|')];rows=[];i+=2
    while i<len(lines) and lines[i].strip().startswith('|'):
     row=[c.strip() for c in lines[i].strip().strip('|').split('|')];assert len(row)==len(headers),(headers,row);rows.append(row);i+=1
    self.table(headers,rows);continue
   if s.startswith('- '):self.story.append(self.para('• '+s[2:]));i+=1;continue
   para=[s];i+=1
   while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![','```','- ')):
    para.append(lines[i].strip());i+=1
   self.story.append(self.para(' '.join(para)))
 def evidence(self):
  self.heading('Project evidence index',2);self.story.append(self.para('The paths below refer to the accompanying ESC-50 project. They remain useful provenance references when the PDF is shared, but the project files are not embedded in this PDF. Public web references in the report are clickable.'))
  for path,label in self.refs.items():
   if path.exists():
    try:rel=str(path.relative_to(P))
    except ValueError:rel=path.name
   else:rel='Historical reference: '+path.name+' (superseded or not retained in this working copy)'
   self.story.append(self.para('['+label+'] `'+rel+'`','small'))
 def build(self,filename,title):
  self.evidence();self.emit_figures();doc=ReportDoc(OUT/filename,self.key,title);doc.build(self.story)
  return dict(filename=filename,source=str(self.source.relative_to(P)),figures=self.figures,tables=self.table_count,headings=self.heading_count)


def rows(file):
 with file.open(newline='') as f:return list(csv.DictReader(f))


def main():
 records=[]
 src=P / 'reports/matrix_20261028/report_en.md';b=Builder('01',src)
 b.cover('Environmental sound classification','Complete Colab matrix results',
  'The latest executed study: architectures, training augmentation, representations, sampling rates and synthetic stress. Every score in the main results is development validation. Historical test exposure prevents a new independent-confirmation claim.',
  [('Audited tasks','65 / 65'),('Neural training runs','60 × 60 epochs'),('Embedded figures','7')])
 text=src.read_text().replace('[English content guide](guide_en.md) · [中文报告](report_zh.md)','[Detailed English content guide](guide_en.md)')
 b.markdown(text)
 b.heading('Appendix A. Real audio and Log-Mel input',2)
 b.story.append(b.para('The example is the validation recording `4-102844-A-49.wav` (hand saw). It illustrates the waveform and the natural-log-power Mel representation used by the main models. Amplitude is a digital signal value, not calibrated sound-pressure level; the Mel axis is an index over nonuniform frequency bands.'))
 b.figure(P / 'results/matrix_20261028/runtime_verification/frontend_example.png','A real validation waveform and Log-Mel representation','Five-second hand-saw validation example; visualization only, not an extra training or test result.')
 b.heading('Appendix B. All fitted-task validation results',2)
 b.story.append(b.para('All 62 fitted tasks are listed: two classical models and 60 neural runs. Each neural run completed 60 epochs; the displayed checkpoint is selected by the earliest minimum validation cross-entropy. The three remaining E tasks evaluate already fitted checkpoints; their six conditions each are reported in Section 6.'))
 rr=[r for r in rows(P / 'results/matrix_20261028/analysis/runs.csv') if r['kind']!='robustness'];assert len(rr)==62
 b.table(['Task ID','Accuracy (%)','Macro-F1 (%)','Selected epoch'],[[r['id'],f"{float(r['accuracy'])*100:.2f}",f"{float(r['macro_f1'])*100:.2f}",r['best_epoch'] or 'not applicable'] for r in rr])
 b.heading('Appendix C. All aggregated paired contrasts',2)
 b.story.append(b.para('Each row averages three within-seed contrasts on the same 398 clips. The delta is right minus left, except the separately defined attention interaction TCA - TA - CA + original GRU. SD is training-seed spread, not a dataset confidence interval.'))
 effects=rows(P / 'results/matrix_20261028/analysis/paired_effects.csv');assert len(effects)==25
 b.table(['Contrast (right minus left)','Accuracy delta: mean ± SD (pp)','Macro-F1 delta (pp)'],[[r['effect']+'; '+r['left']+' → '+r['right'],f"{float(r['mean_delta_accuracy_pp']):+.3f} ± {float(r['sample_sd_delta_accuracy_pp']):.3f}",f"{float(r['mean_delta_macro_f1_pp']):+.3f}"] for r in effects])
 # Literal line break markup is handled below without embedding arbitrary source HTML.
 records.append(b.build('ESC50_Complete_Study_Report_EN.pdf','Complete Colab matrix results'))
 src=P / 'reports/baseline_20261028/report_en.md';b=Builder('02',src)
 b.cover('CNN and CRNN baselines','Complete historical study results',
  'The earlier frozen baseline comparison: repeated seeds, reserved-fold results, paired errors, order probes, synthetic stress and fixed-recipe fold rotation. Historical test results remain historical; this PDF export performs no new evaluation.',
  [('Full neural runs','21'),('Repeated training seeds','3'),('Embedded figures','8')])
 b.story.extend([b.para('Scope note for this 29 October export','h2'),b.para('The body preserves the 28 October completed study and its then-current 153-test verification scope. The later matrix study raised project checks to 274 and re-executed additional development tasks; those belong to Report 01. The two latency tables also use different hardware/workloads. Keep their scopes separate.')])
 text=src.read_text().replace("'.[dev,report]'","'.[dev,baseline,report]'")
 b.markdown(text)
 b.heading('Appendix A. Full primary-checkpoint confusion matrices',2)
 b.story.append(b.para('Both matrices use the historical primary seed101 checkpoints on the 400 fold-5 clips. Rows are true classes, columns are predicted classes; cells show within-true-class normalized proportions. They supplement the named-confusion figure and were not recomputed for this PDF.'))
 for label,file in [('CNN','right_confusion.png'),('CRNN (CNN + BiGRU)','left_confusion.png')]:
  b.figure(P / 'results/historical/holdout_primary_comparison'/file,label+' primary fold-5 confusion matrix','Historical primary seed101 checkpoint; 400 clips, eight per class. In the original comparison files, left is CRNN and right is CNN. The plotted class IDs follow the official metadata.')
 b.story.append(PageBreak())
 b.heading('Appendix B. Class-ID key for the confusion matrices',2)
 b.story.append(b.para('Class IDs are shared by both models and follow the pinned official ESC-50 metadata. Underscores in category identifiers are shown as spaces for readability.'))
 metadata=rows(P/'data/ESC-50/meta/esc50.csv');classes={int(r['target']):r['category'].replace('_',' ') for r in metadata}
 b.table(['ID','Category','ID','Category'],[[str(i),classes[i],str(i+25),classes[i+25]] for i in range(25)])
 records.append(b.build('ESC50_CNN_CRNN_Baseline_Report_EN.pdf','CNN and CRNN historical baseline results'))
 (TMP/'pdf_figure_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 print(json.dumps([dict(file=r['filename'],figures=len(r['figures']),tables=r['tables']) for r in records],indent=2))

if __name__=='__main__':main()
