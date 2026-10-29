from pathlib import Path
import nbformat
p=Path(__file__).resolve().parents[1]
template=p / 'notebooks/templates/ESC50_Colab_Research_Console.ipynb'
j=nbformat.read(template,as_version=4)
# Include the requested stress outcomes instead of leaving them in logs only.
extra='''
if EXECUTE_SETUP:
    stress_rows=[]
    for task in TASKS:
        if task['kind']!='robustness':continue
        out=output_for(task)
        if not completed(task,out,PILOT_EPOCHS if RUN_MODE=='pilot' else FULL_EPOCHS):continue
        for condition,metrics in json.loads((out/'stress.json').read_text())['conditions'].items():
            stress_rows.append(dict(task=task['id'],condition=condition,accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],n=metrics['n']))
    if stress_rows:display(pd.DataFrame(stress_rows))
'''
if 'stress_rows=[]' not in j.cells[14].source:
 j.cells[14].source+=extra
 nbformat.validate(j);nbformat.write(j,template)
j.cells[0].source='''# ESC-50: complete development matrix / 完整开发实验矩阵

This notebook executes the supplied 65-task matrix on Colab. / 本 notebook 在 Colab 实际执行所提供的65项任务。

Main experiments use 60 epochs and seeds 101/202/303; separate two-epoch pilot results are excluded. Results are development-validation measurements, not independent confirmation. / 正式实验60轮；2轮试跑独立保存，不纳入正式结果。这些是开发验证结果，不是独立确认。

Storage in this execution uses the Colab local disk and an atomic export ZIP after every completed task, downloaded to the local project during execution. No Drive mount or paid upgrade is needed. / 本次使用Colab本地磁盘，每项任务完成后导出ZIP并及时下载，无需新增Drive挂载或付费升级。
'''
s=j.cells[2].source
replacements={'RUN_MODE = "plan"':'RUN_MODE = "selected"','GROUPS_TO_RUN = ["A_baselines"]':'GROUPS_TO_RUN = ["A_baselines", "B_architecture", "C_augmentation", "D_feature_sampling", "E_robustness"]','MAX_TASKS_PER_SESSION = 1':'MAX_TASKS_PER_SESSION = 65','USE_GOOGLE_DRIVE = True':'USE_GOOGLE_DRIVE = False','PERSIST_BASE_CACHE = True':'PERSIST_BASE_CACHE = False','DEVICE = "auto"':'DEVICE = "cuda"','MAX_SECONDS_PER_TRAIN = 10800':'MAX_SECONDS_PER_TRAIN = 600','MAX_SESSION_MINUTES = 90':'MAX_SESSION_MINUTES = 120'}
for a,b in replacements.items():
 assert a in s,a
 s=s.replace(a,b)
j.cells[2].source=s
archive='''
# Export each verified task atomically so a disconnect does not erase completed evidence.
from datetime import datetime,timezone
EXPORTS=Path('/content/esc50_matrix_exports')
EXPORTS.mkdir(exist_ok=True)
_progress=[]
_original_execute_task=execute_task

def execute_task(task):
    begin=time.monotonic()
    result=_original_execute_task(task)
    if task['kind']=='feature_ablation':
        out=RUNS/task['group']/'main'/task['model']/task['case']/('seed'+str(task['seed']))
    else:out=output_for(task)
    target=EXPORTS/(task['id']+'.zip')
    temporary=target.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as bundle:
        for f in sorted(out.rglob('*')):
            if f.is_file():bundle.write(f,str(f.relative_to(STORAGE)))
        for f in sorted(LOGS.glob(task['id']+'*.log')):bundle.write(f,str(f.relative_to(STORAGE)))
    temporary.replace(target)
    _progress.append(dict(id=task['id'],group=task['group'],kind=task['kind'],status=result,seconds=time.monotonic()-begin,archive=target.name,timestamp_utc=datetime.now(timezone.utc).isoformat()))
    temporary=EXPORTS/'progress.tmp'
    temporary.write_text(json.dumps(_progress,indent=2))
    temporary.replace(EXPORTS/'progress.json')
    print('EXPORTED',task['id'],'|',len(_progress),'/',len(SELECTED),flush=True)
    return result
'''
j.cells.insert(12,nbformat.v4.new_code_cell(archive))
for c in j.cells:
 if c.cell_type=='code':c.outputs=[];c.execution_count=None
nbformat.validate(j)
nbformat.write(j,p / 'notebooks/templates/ESC50_Matrix_Full_20261028.ipynb')
print('Created full execution notebook')
