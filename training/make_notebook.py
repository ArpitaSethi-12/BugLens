"""Builds BugLens_training.ipynb from modeling.py, train.py and the probe set,
so the Colab notebook always matches the app code. Run: python training/make_notebook.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from app.samples import probe_set  # noqa: E402


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


modeling = open(os.path.join(ROOT, "app", "modeling.py")).read()
train = open(os.path.join(HERE, "train.py")).read()
probe = json.dumps(probe_set())

cells = [
    md("""
# BugLens training: CodeBERT and CodeT5 for bug detection

Fine-tunes two code models on the CodeXGLUE Defect Detection dataset and packs the
trained models plus all evaluation results into one zip for the BugLens app.

**Before you start (Colab)**
1. `Runtime > Change runtime type > T4 GPU`, then Save.
2. `Runtime > Run all`.
3. Allow Google Drive access when asked, so a disconnect does not lose finished work. If your Drive is full, set `USE_DRIVE = False` in the second code cell.

**On Kaggle:** Settings > Accelerator > GPU T4, and Internet on. Then Run all.

Expected time on a T4: about 35 to 45 min for CodeBERT and 15 to 25 min for CodeT5.
At the end the browser downloads `buglens_trained.zip`.
"""),
    code("""
# Set QUICK = True only for a 5 minute smoke test (tiny subset, numbers will be meaningless)
QUICK = False
EPOCHS = 5
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""),
    code("""
# Works on Colab and Kaggle. Set USE_DRIVE = False if your Google Drive is full.
USE_DRIVE = True
import os, sys
IN_COLAB = 'google.colab' in sys.modules or os.path.exists('/content')
IN_KAGGLE = os.path.exists('/kaggle/working')
OUT = '/kaggle/working/BugLens' if IN_KAGGLE else '/content/BugLens'
if IN_COLAB and USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        test_dir = '/content/drive/MyDrive/BugLens'
        os.makedirs(test_dir, exist_ok=True)
        with open(f'{test_dir}/_write_test', 'wb') as f:
            f.write(os.urandom(1_000_000))  # fails here if Drive is full
        os.remove(f'{test_dir}/_write_test')
        OUT = test_dir
    except Exception as e:
        print('Drive not usable, saving inside this session instead:', e)
os.makedirs(OUT, exist_ok=True)
print('Saving to', OUT)
"""),
    code("""
!pip -q install "datasets>=2.18" scikit-learn
import transformers, torch, datasets
print('torch', torch.__version__, '| transformers', transformers.__version__, '| datasets', datasets.__version__)
open(f'{OUT}/versions.txt', 'w').write(f'torch {torch.__version__}\\ntransformers {transformers.__version__}\\n')
"""),
    md("## Code\nThe two files below are identical to `app/modeling.py` and `training/train.py` in the project."),
    code("%%writefile modeling.py\n" + modeling),
    code("%%writefile train.py\n" + train),
    code("open('probe_set.json', 'w').write(r'''" + probe + "''')\nprint('probe set written')"),
    md("## Train CodeBERT (125M parameters)"),
    code("""
extra = '--max_train_samples 1500 --epochs 1' if QUICK else f'--epochs {EPOCHS}'
!python train.py --model codebert --probe_file probe_set.json --out_dir "$OUT" $extra
"""),
    md("## Train CodeT5-small encoder (about 35M parameters used)"),
    code("""
!python train.py --model codet5 --probe_file probe_set.json --out_dir "$OUT" $extra
"""),
    md("## Results summary"),
    code("""
import json, glob
for p in sorted(glob.glob(f'{OUT}/results/*.json')):
    r = json.load(open(p))
    t = r['test']
    print(f"{r['model']:<9} acc {t['accuracy']:.4f}  f1 {t['f1']:.4f}  auc {t['auc']:.4f}  mcc {t['mcc']:.4f}  "
          f"epochs {r['config']['epochs_run']}  train {r['train_seconds']/60:.1f} min")
"""),
    md("## Download\nUnzip `buglens_trained.zip` inside the BugLens project folder. It contains `models/` and `results/`."),
    code("""
import shutil
shutil.copy(f'{OUT}/versions.txt', f'{OUT}/results/versions.txt')
if os.path.exists(f'{OUT}/buglens_trained.zip'):
    os.remove(f'{OUT}/buglens_trained.zip')  # do not zip an older zip
ZIP_DIR = '/kaggle/working' if IN_KAGGLE else '/content'
zip_path = shutil.make_archive(f'{ZIP_DIR}/buglens_trained', 'zip', OUT, '.')
print('Zip ready:', zip_path, round(os.path.getsize(zip_path) / 1e6), 'MB')
if OUT.startswith('/content/drive'):
    shutil.copy(zip_path, f'{OUT}/buglens_trained.zip')
    print('A copy is also in Drive/BugLens')
if IN_KAGGLE:
    print('Download it from the Output panel on the right: /kaggle/working/buglens_trained.zip')
else:
    from google.colab import files
    files.download(zip_path)
    print('If the download does not start, open the folder icon on the left and download buglens_trained.zip')
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
out = os.path.join(HERE, "BugLens_training.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
