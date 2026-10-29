"""Build a self-contained, readable Colab notebook from the actual package."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))


markdown('''# ESC-50: research and inference

This notebook contains the project's inspectable Python package. It prepares the pinned ESC-50 dataset, trains a frequency-preserving CNN and CNN+GRU, fits a cheap statistical baseline, and predicts one **validation** recording. No test predictions are made here.

The completed three-seed development study selected the small CNN under a predeclared practical preference. Its held-out fold 5 mean accuracy was 63.08%; CNN+GRU reached 63.92%. These saved study results are separate from the runs produced below. This is a small from-scratch research prototype, with substantial error remaining; scores are uncalibrated.

Use **Runtime → Change runtime type → GPU** when available, then **Run all**. The default GPU run trains 60 epochs per neural model. CPU use is limited to a two-epoch correctness demonstration. No upgrade, Drive mount or external API is needed. ESC-50 is distributed overall under CC BY-NC 3.0; retain its credits and license.''')
code('''import os, sys, subprocess, json, time
from pathlib import Path
import torch

ROOT = Path('/content/esc50_package_demo')
ROOT.mkdir(exist_ok=True)
(ROOT / 'src/esc50_sound').mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'numpy>=2,<3', 'scipy>=1.14,<2', 'soundfile>=0.13,<1'], check=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = int(os.environ.get('ESC50_EPOCHS', '60'))
if DEVICE == 'cpu' and EPOCHS > 2:
    EPOCHS = 2
    print('GPU unavailable: using two epochs to verify the pipeline, not to estimate final accuracy.')
DATA_ROOT = os.environ.get('ESC50_DATA_ROOT', str(ROOT / 'data/ESC-50'))
CACHE = os.environ.get('ESC50_CACHE', str(ROOT / 'data/esc50_features.npz'))
RUN_ROOT = ROOT / 'results/runs' / time.strftime('%Y%m%d_%H%M%S', time.gmtime())
RUN_ROOT.mkdir(parents=True, exist_ok=False)
os.environ['PYTHONPATH'] = str(ROOT / 'src')
print({'device': DEVICE, 'gpu': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None,
       'epochs': EPOCHS, 'torch': torch.__version__, 'run_directory': str(RUN_ROOT)})''')

markdown('''## Package source

Each cell writes one ordinary source file. Audio, data, models, optimization, evaluation and user inference have separate responsibilities. The model receives `[B,1,64,313]` log-mel features; its shared encoder returns `[B,64,8,39]`. The CNN averages time while retaining frequency positions. The hybrid sends 39 ordered 512-dimensional tokens to a bidirectional GRU. Both return 50 logits.''')
order = ('__init__', 'audio', 'records', 'data', 'models', 'checkpoints',
         'evaluation', 'training', 'baseline', 'inference', 'cli', '__main__')
for name in order:
    path = f'src/esc50_sound/{name}.py'
    code(f'%%writefile {path}\n' + (ROOT / path).read_text())
for name in ('pyproject.toml', 'requirements.txt'):
    code(f'%%writefile {name}\n' + (ROOT / name).read_text())

markdown('''## Prepare data and verify the implementation

The dataset contains 2,000 five-second recordings in 50 balanced categories. Development trains on folds 1–3, selects checkpoints on source-repaired fold 4 (398 clips), and reserves fold 5. Normalization is fitted only on training clips. The synthetic smoke check below tests implementation behavior; its accuracy is not an ESC-50 result.''')
code('''def run(label, *arguments, timeout=600):
    result = subprocess.run([sys.executable, '-m', 'esc50_sound', *map(str, arguments)],
                            capture_output=True, text=True, timeout=timeout)
    (RUN_ROOT / f'{label}.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        print(result.stderr)
        raise RuntimeError(f'{label} failed; inspect its saved log.')
    print(f'{label} completed')

run('smoke', 'smoke', '--output', RUN_ROOT / 'smoke.json', timeout=120)
run('prepare', 'prepare', '--root', DATA_ROOT, '--cache', CACHE)''')

markdown('''## Train comparable models

Both models see the same training clips, batch size, optimizer and epoch ceiling. A cosine schedule decays the learning rate from 0.001. Selection uses the earliest minimum validation cross-entropy, not the highest observed accuracy. These two single-seed runs illustrate reproduction; they do not replace the recorded three-seed comparison or establish recurrent superiority.''')
code('''summary = []
for model in ('cnn_freq', 'crnn_freq'):
    output = RUN_ROOT / model
    run(model, 'train', '--cache', CACHE, '--model', model, '--output', output,
        '--epochs', EPOCHS, '--scheduler', 'cosine', '--seed', 101,
        '--device', DEVICE, '--max-seconds', 300, timeout=480)
    record = json.loads((output / 'run.json').read_text())
    if record['status'] != 'completed':
        raise RuntimeError('Partial run preserved; it cannot enter the equal-exposure comparison.')
    metric = json.loads((output / 'validation.json').read_text())
    summary.append({'model': model, 'accuracy': metric['accuracy'], 'macro_f1': metric['macro_f1'],
                    'parameters': record['parameters'], 'selected_epoch': record['best_epoch']})
print(json.dumps({'epochs': EPOCHS, 'split': 'validation only', 'models': summary}, indent=2))''')
code('''run('statistical_baseline', 'baseline', '--cache', CACHE,
    '--output', RUN_ROOT / 'statistics_baseline', timeout=120)
print(json.loads((RUN_ROOT / 'statistics_baseline/validation.json').read_text())['accuracy'])''')

markdown('''## Predict a validation WAV

The default CNN was selected from the completed study before test evaluation. Prediction rejects recordings longer than five seconds, pads shorter ones, and abstains on near-silence. Silence handling is not general unknown-sound detection. The five softmax scores are not calibrated confidence.''')
code('''sys.path.insert(0, str(ROOT / 'src'))
from esc50_sound.inference import AudioClassifier
checkpoint = RUN_ROOT / 'cnn_freq/best.pt'
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
filename = saved['manifest']['split']['validation']['names'][0]
torch.set_num_threads(2)
result = AudioClassifier(checkpoint).classify(Path(DATA_ROOT) / 'audio' / filename)
print(json.dumps(result, indent=2))''')

markdown('''## Save the run

The ZIP contains source, configuration, checkpoints, logs and predictions. It excludes the dataset and feature cache. Download it from Colab's Files panel before the temporary runtime ends. The project's selected trained model bundles and complete research evidence are separate deliverables.''')
code('''import zipfile
archive = ROOT / f'esc50_demo_{RUN_ROOT.name}.zip'
files_to_save = [p for p in RUN_ROOT.rglob('*') if p.is_file()]
files_to_save += list((ROOT / 'src/esc50_sound').glob('*.py'))
files_to_save += [ROOT / 'pyproject.toml', ROOT / 'requirements.txt']
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as output:
    for path in files_to_save:
        output.write(path, path.relative_to(ROOT))
print('Download from the Files panel:', archive)
print('Fold5 remains unevaluated by this notebook.')''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'name': 'python3', 'display_name': 'Python3'},
    'language_info': {'name': 'python'}, 'colab': {'name': 'ESC50_Research_and_Inference.ipynb'}})
nbf.validate(notebook)
destination = ROOT / 'notebooks/templates/ESC50_Research_and_Inference.ipynb'
destination.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, destination)
print(destination)
