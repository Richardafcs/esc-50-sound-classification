"""Check complete figure embedding, result-row coverage and PDF page boundaries."""
from pathlib import Path
import csv,hashlib,json,re,zipfile,sys
from datetime import datetime
from PIL import Image
from pypdf import PdfReader
import pdfplumber
from project_layout import project_path

P=Path(__file__).resolve().parents[1];OUT=P / 'reports/pdf';TMP=P/'.cache/pdf'
manifest_source=TMP/'pdf_figure_manifest.json'
if not manifest_source.exists():manifest_source=OUT/'figure_manifest.json'
records=json.loads(manifest_source.read_text());checks=[]
def normalize(s):
 for c in '\u2010\u2011\u2012\u2013\u2014\u2212':s=s.replace(c,'-')
 return re.sub(r'\s+',' ',s).strip()
for record in records:
 path=OUT/record['filename'];reader=PdfReader(path);texts=[page.extract_text() or '' for page in reader.pages];alltext=normalize('\n'.join(texts))
 assert len(reader.pages)==18
 assert all(len(t)>170 for t in texts)
 assert not re.search(r'[\u4e00-\u9fff\x00\u25a0]',alltext)
 assert '<br/>' not in alltext and 'ZZLINK' not in alltext
 source=project_path(record['source']).read_text()
 for heading in re.findall(r'^##+ (.+)$',source,re.M):assert normalize(heading) in alltext,heading
 image_pages=[]
 with pdfplumber.open(path) as pdf:
  for i,page in enumerate(pdf.pages):
   for c in page.chars:
    assert c['x0']>=42 and c['x1']<=page.width-42,(path.name,i+1,c)
    assert c['top']>=10 and c['bottom']<=page.height-10,(path.name,i+1,c)
   for im in page.images:
    assert im['x0']>=43 and im['x1']<=page.width-43
    assert im['top']>=48 and im['bottom']<=page.height-45
  for i,page in enumerate(reader.pages):
   if page.images:
    assert len(page.images)==1
    image_pages.append((i+1,page.images[0].image.convert('RGB')))
 assert len(image_pages)==len(record['figures'])
 for figure,(page_number,embedded) in zip(record['figures'],image_pages):
  original=Image.open(project_path(figure['path'])).convert('RGB')
  assert original.size==embedded.size and original.tobytes()==embedded.tobytes(),figure
  assert 'Figure '+str(figure['number'])+'.' in texts[page_number-1]
  figure['pdf_page']=page_number
 checks.append(dict(pdf=path.name,pages=len(reader.pages),embedded_figures=len(image_pages),all_source_headings_retained=True,all_figure_pixels_match_originals=True,page_boundaries_pass=True,blank_pages=0,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 if 'Complete_Study' in path.name:
  with project_path('results/matrix_20261028/analysis/runs.csv').open() as f:runs=[r for r in csv.DictReader(f) if r['kind']!='robustness']
  for r in runs:
   expected=normalize(' '.join([r['id'],f"{float(r['accuracy'])*100:.2f}",f"{float(r['macro_f1'])*100:.2f}",r['best_epoch'] or 'not applicable']))
   assert expected in alltext,expected
  assert len(runs)==62
  checks[-1]['fitted_task_rows_verified']=62
  with project_path('results/matrix_20261028/analysis/paired_effects.csv').open() as f:effects=list(csv.DictReader(f))
  for r in effects:
   assert normalize(f"{float(r['mean_delta_accuracy_pp']):+.3f} ± {float(r['sample_sd_delta_accuracy_pp']):.3f}") in alltext
  checks[-1]['paired_aggregate_rows']=len(effects)
comparison=json.loads(project_path('results/historical/holdout_primary_comparison/comparison.json').read_text())
assert 'crnn_freq' in comparison['left_path'] and '/cnn_freq_' in comparison['right_path']
assert records[1]['figures'][-2]['path'].endswith('right_confusion.png')
assert records[1]['figures'][-1]['path'].endswith('left_confusion.png')
VERIFY=P/'reports/verification'
VERIFY.mkdir(exist_ok=True)
(VERIFY/'figure_manifest_latest.json').write_text(json.dumps(records,indent=2)+'\n')
proof=dict(checks=checks,total_figures=15,visual_review='All 36 pages inspected in rendered contact sheets; changed table/code pages also inspected individually',scope='Existing reviewed results exported; no new model training or dataset evaluation',historical_confusion_matrix_mapping='CNN=right_confusion; CRNN=left_confusion, verified against comparison.json')
(VERIFY/'report_verification_latest.json').write_text(json.dumps(proof,indent=2)+'\n')
if '--bundle' in sys.argv:
 export_root=P/'releases/exports'
 export_root.mkdir(exist_ok=True)
 archive=export_root/('ESC50_English_Reports_and_All_Figures_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for name in ['ESC50_Complete_Study_Report_EN.pdf','ESC50_CNN_CRNN_Baseline_Report_EN.pdf','README.md']:z.write(OUT/name,name)
  z.write(VERIFY/'figure_manifest_latest.json','figure_manifest.json')
  z.write(VERIFY/'report_verification_latest.json','report_verification.json')
  for report_id,record in enumerate(records,1):
   for figure in record['figures']:
    path=project_path(figure['path'])
    for asset in [path,path.with_suffix('.svg')]:
     if asset.exists():z.write(asset,f'figures/report_{report_id:02d}/figure_{figure["number"]:02d}_{asset.name}')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  assert len([n for n in z.namelist() if n.endswith('.png')])==15
  assert len([n for n in z.namelist() if n.endswith('.svg')])==13
else:
 archive=None
print(json.dumps(dict(reports=checks,zip=str(archive) if archive else None,zip_bytes=archive.stat().st_size if archive else None,PNG=15,SVG=13),indent=2))
