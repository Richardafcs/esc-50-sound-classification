from pathlib import Path
import json,csv,torch
p=Path(__file__).resolve().parents[1]
new=p / 'results/matrix_20261028/execution/runs/B_architecture'
old=p / 'results/historical/repeated_development/runs'
rows=[]
for model in ['cnn_freq','crnn_freq']:
 for seed in [101,202,303]:
  a=new/f'B_{model}_s{seed}'
  b=old/f'{model}_seed{seed}_60ep'
  old_checkpoint=p / 'results/historical/repeated_development/models'/f'{model}_seed{seed}_60ep.pt'
  if not(a/'best.pt').is_file() or not old_checkpoint.is_file():continue
  aa=torch.load(a/'best.pt',map_location='cpu',weights_only=True)
  bb=torch.load(old_checkpoint,map_location='cpu',weights_only=True)
  pa=list(csv.DictReader((a/'validation_predictions.csv').open()));pb=list(csv.DictReader((b/'validation_predictions.csv').open()))
  rows.append(dict(model=model,seed=seed,predictions_equal=pa==pb,best_epoch_equal=aa['epoch']==bb['epoch'],weights_equal=aa['model'].keys()==bb['model'].keys() and all(torch.equal(aa['model'][k],bb['model'][k]) for k in aa['model'])))
out=p / 'results/matrix_20261028/baseline_regression.json';out.write_text(json.dumps(dict(scope='New runs versus historical saved development runs; not fresh heldout evidence',runs=rows),indent=2))
print(json.dumps(rows,indent=2))
