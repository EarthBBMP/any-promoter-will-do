"""Generate colab_train.ipynb.

Written as a generator rather than hand-authored JSON so the cell sources stay
readable and correctly escaped.

    python make_colab_notebook.py
"""

import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "colab_train.ipynb"


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": list(_join(lines))}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": list(_join(lines)),
    }


def _join(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


CELLS = [
    md(
        "# ASPINE: retraining sweep on GPU",
        "",
        "Retrains the released ASPINE architecture from scratch across several seeds and",
        "optimizers, to measure **how often it collapses to constant mean-prediction**.",
        "",
        "None of the results in `results/FINDINGS.md` depend on this notebook -- those are",
        "measured on the released weights with a forward pass. This answers one extra",
        "question: is the released training result reproducible from a cold start?",
        "",
        "---",
        "",
        "### Before you start",
        "",
        "1. Run `python prepare_colab_upload.py` locally to produce **`aspine_colab.zip`** (~85 MB).",
        "2. Upload that file to the **top level of your Google Drive** (My Drive).",
        "3. Here: **Runtime -> Change runtime type -> T4 GPU** (or A100), then run the cells in order.",
    ),
    md("## 1. Check the runtime"),
    code(
        "import subprocess, sys",
        "",
        "print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',",
        "                      '--format=csv,noheader'],",
        "                     capture_output=True, text=True).stdout.strip() or 'NO GPU DETECTED')",
        "",
        "import tensorflow as tf, keras",
        "print(f'tensorflow {tf.__version__}   keras {keras.__version__}')",
        "print('GPUs visible to TF:', tf.config.list_physical_devices('GPU'))",
        "",
        "if not tf.config.list_physical_devices('GPU'):",
        "    print('\\n!! No GPU. Runtime -> Change runtime type -> T4 GPU, then rerun this cell.')",
        "if int(keras.__version__.split('.')[0]) < 3:",
        "    print('\\n!! Keras 3 is required (PyDataset). Colab normally ships it.')",
    ),
    md(
        "## 2. Mount Drive and unpack",
        "",
        "The archive mirrors the local repository layout, so `aspine_bench.config` resolves",
        "its paths on Colab exactly as it does locally -- nothing needs patching.",
    ),
    code(
        "from google.colab import drive",
        "drive.mount('/content/drive')",
    ),
    code(
        "import zipfile, pathlib, shutil",
        "",
        "ZIP = pathlib.Path('/content/drive/MyDrive/aspine_colab.zip')",
        "ROOT = pathlib.Path('/content/aspine')",
        "",
        "assert ZIP.exists(), (",
        "    f'{ZIP} not found. Upload aspine_colab.zip to the top level of My Drive, '",
        "    'or edit the ZIP path above.'",
        ")",
        "",
        "if ROOT.exists():",
        "    shutil.rmtree(ROOT)",
        "ROOT.mkdir(parents=True)",
        "",
        "with zipfile.ZipFile(ZIP) as archive:",
        "    archive.extractall(ROOT)",
        "",
        "for path in sorted(ROOT.rglob('*')):",
        "    if path.is_file():",
        "        print(f'{path.stat().st_size / 1e6:8.1f} MB  {path.relative_to(ROOT)}')",
    ),
    md("## 3. Sanity check: reproduce the published number first\n",
       "\n",
       "Before spending GPU time on training, confirm the released weights still give\n",
       "R2 = 0.330 here. If this disagrees with the local run, stop and investigate --\n",
       "everything downstream would be suspect."),
    code(
        "%cd /content/aspine/aspine-newinml",
        "!python run_benchmark.py --stage tabular",
        "!python run_benchmark.py --stage cnn",
        "!python run_benchmark.py --stage report",
    ),
    md(
        "## 4. The sweep",
        "",
        "Six runs: seeds 0/1/2 x {SGD as released, Adam}.",
        "",
        "`DeadRunDetector` stops a run as soon as its predictions go constant, so collapsed",
        "runs cost ~3 epochs instead of 250. Surviving runs train until early stopping.",
        "",
        "`results/retrained.csv` is rewritten after **every** run, so a disconnect never",
        "loses finished work.",
    ),
    code(
        "!python train_cnn.py \\",
        "    --seed 0 --seed 1 --seed 2 \\",
        "    --optimizer sgd --optimizer adam \\",
        "    --max-epochs 250",
    ),
    md("## 5. Results"),
    code(
        "import pandas as pd",
        "",
        "df = pd.read_csv('/content/aspine/aspine-newinml/results/retrained.csv')",
        "cols = ['seed', 'optimizer', 'collapsed', 'r2', 'rmse', 'pred_std', 'epochs_run', 'minutes']",
        "display(df[cols])",
        "",
        "print(f\"collapsed: {int(df['collapsed'].sum())}/{len(df)} runs\")",
        "",
        "alive = df[~df['collapsed']]",
        "if len(alive) > 1:",
        "    print(f\"R2 across surviving runs: {alive['r2'].mean():.4f} +/- {alive['r2'].std(ddof=1):.4f}\")",
        "elif len(alive) == 1:",
        "    print(f\"one surviving run: R2 = {alive['r2'].iloc[0]:.4f}\")",
        "else:",
        "    print('No run escaped collapse -- report this as a reproducibility finding.')",
        "print('released weights: R2 = 0.330')",
    ),
    md(
        "## 6. Save results back to Drive",
        "",
        "Colab wipes `/content` on disconnect. Run this before you close the tab.",
    ),
    code(
        "import shutil, pathlib",
        "",
        "dest = pathlib.Path('/content/drive/MyDrive/aspine_results')",
        "dest.mkdir(parents=True, exist_ok=True)",
        "",
        "src = pathlib.Path('/content/aspine/aspine-newinml/results')",
        "for path in src.iterdir():",
        "    if path.is_file():",
        "        shutil.copy2(path, dest / path.name)",
        "        print('saved', path.name)",
        "",
        "print(f'\\nAll results copied to {dest}')",
    ),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}


def main() -> None:
    OUTPUT.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
