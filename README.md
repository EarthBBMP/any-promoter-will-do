# Any Promoter Will Do

Code for *Any Promoter Will Do: Auditing a Sequence-to-Expression Model on
Patient-Derived Data*, ICBINB-BIO workshop (I Can't Believe It's Not Better:
Failure Modes of AI in Biology), NeurIPS 2026.

The paper audits ASPINE, a convolutional model that predicts gene expression
from patient-derived promoter sequence. This repository reproduces the
published numbers and adds the baselines, retraining sweep and promoter
interventions that the original release did not include.

## Data

The dataset, model definition and trained checkpoint were released by Team
Bangkok-NMH (iGEM 2024) under CC BY 4.0:
<https://2024.igem.wiki/bangkok-nmh/description>. They are not redistributed
here and are read unmodified.

The release was published on the iGEM GitLab at
<https://gitlab.igem.org/2024/software-tools/bangkok-nmh> (the page currently
requires a sign-in). `aspine_bench/config.py` looks for it one directory above
this repository, so clone this repository *inside* your copy of the release:

```
cd bangkok-nmh
git clone https://github.com/EarthBBMP/any-promoter-will-do.git
cd any-promoter-will-do
```

which gives

```
bangkok-nmh/                    original iGEM release
  data/Aspine/                  train_aspine.csv, valid_aspine.csv, test_aspine.csv
  data/Saved_Weights/           model_weight.keras (the released checkpoint)
  xpresso_module/               model definition, imported by train_cnn.py
  any-promoter-will-do/         this repository
```

To keep the files elsewhere, set `ASPINE_DATA_DIR` (directory holding the three
CSVs), `ASPINE_WEIGHTS` (path to `model_weight.keras`) and, optionally,
`ASPINE_RESULTS_DIR` (output directory, default `results/`). `train_cnn.py`
also needs `xpresso_module` importable, so add the release to `PYTHONPATH` in
that case.

## Findings

- The released checkpoint's R² reproduces (0.330 against a reported 0.329).
  The reported RMSE does not (we measure 0.391, not 0.440).
- The released checkpoint does not significantly beat gradient boosting on
  eight scalar mRNA half-life features (p = 0.385). Retrained models reach
  R² = 0.352 ± 0.018; two of three significantly beat it, and their mean
  prediction does so with ΔR² = +0.052, p = 0.003.
- That advantage survives destroying the promoter. Swapping promoters between
  genes, dinucleotide-shuffling and mononucleotide-shuffling each change R² by
  less than 0.010, against 0.029 for removing the promoter entirely. Shifting
  the input window by thousands of base pairs costs nothing in seven of eight
  model-window combinations.
- The architecture collapses to constant mean-prediction in 3 of 6 cold
  starts, through a two-unit ReLU bottleneck.

See the paper for intervals and the full analysis.

## Usage

```
pip install -r requirements.txt

# Released checkpoint and baselines
python run_benchmark.py --stage tabular   # seconds; no sequence data loaded
python run_benchmark.py --stage cnn       # ~10 s; forward pass over released weights
python run_benchmark.py --stage kmer      # ~1 min; streams the 277 MB training split
python run_benchmark.py --stage report    # assembles results/results.{csv,md}

# Retraining sweep (3 seeds x 2 optimisers, ~16 GPU-minutes total)
python train_cnn.py --seed 0 --seed 1 --seed 2 --optimizer sgd --optimizer adam
python analyze_retrained.py               # retrained models vs GBM, paired bootstrap

# Promoter interventions
python ablate_sequence.py                 # released checkpoint: swap and zero
python ablate_shuffles.py                 # all models: swap, di-/mononucleotide shuffle, zero
python ablate_windows.py                  # all models: window shifted +6,500 and -3,500 bp

# Figures
python make_figures.py --out-dir figures

# Tests
python -m pytest tests/ -q
```

`train_cnn.py` with no arguments runs a single seed-0 SGD model, which is one
of the runs that collapses; pass the seeds and optimisers as above for the full
sweep. It can also be run on a hosted GPU with `colab_train.ipynb`.

`analyze_retrained.py`, `ablate_shuffles.py`, `ablate_windows.py` and
`make_figures.py` read the sweep from `--sweep-dir`, which defaults to
`aspine_results/`, the checkpoints and predictions from our own run. To use
models you retrained locally, pass `--sweep-dir results`.

Every stage caches its predictions to `results/preds_*.npy`. These are not
committed, so on a fresh clone run the four `run_benchmark.py` stages before
`make_figures.py`; after that, tables and figures can be rebuilt without
recomputation.

## Layout

```
aspine_bench/
  config.py             paths, seed, bootstrap settings
  data.py               memory-efficient loading, one-hot encoding, k-mer features
  metrics.py            R², RMSE, correlations, bootstrap and paired-bootstrap CIs
  baselines.py          mean / ridge / GBM baselines, tuned on the validation split
  cnn.py                loads and runs the released .keras weights
  shuffles.py           mononucleotide and dinucleotide (Altschul-Erikson) shuffles
run_benchmark.py        benchmark driver (staged)
ablate_sequence.py      promoter swap and zero interventions, released checkpoint
ablate_shuffles.py      in-distribution shuffles, all models
ablate_windows.py       shifted-window interventions, all models
train_cnn.py            retraining sweep, with early stop on collapsed runs
analyze_retrained.py    analysis of retrained models
colab_train.ipynb       retraining notebook for a hosted GPU
make_colab_notebook.py  builds the notebook
prepare_colab_upload.py packages inputs for the notebook
make_figures.py         paper figures
results/                benchmark and intervention tables
aspine_results/         retraining sweep: checkpoints, predictions, training histories
tests/                  unit tests for encoders and metrics
```

## Notes on the reimplemented encoder

`aspine_bench.data.one_hot` differs from the original `utils.encode_sequence`
in three ways, all deliberate:

- It slices to the model's input window *before* encoding rather than after.
  The original allocated the full 20,000 bp float32 tensor and then discarded
  half of it (~4.4 GB for the training split, vs ~0.55 GB here).
- It encodes via a lookup table into `int8` instead of a Python double loop
  into `float32`.
- It raises on ragged input. The original sized the entire batch from the
  first sequence, silently zero-padding shorter sequences and raising
  `IndexError` on longer ones.

Behaviour on valid, equal-length input is identical, including the treatment
of `N` as an all-zero column.

## Licence

Code in this repository: MIT, see `LICENSE`. The ASPINE dataset, model and
weights remain under their original CC BY 4.0 licence from Team Bangkok-NMH.
