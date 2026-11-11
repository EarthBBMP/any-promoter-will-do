# ASPINE re-evaluation and baselines

Reproducible evaluation of the ASPINE gene-expression model, with the baseline
suite and ablations the original release did not include.

This directory is **new work built on a public artifact**. The underlying
dataset, model architecture and trained weights were released by Team
Bangkok-NMH (iGEM 2024) under CC BY 4.0 and are read, unmodified, from the
parent repository. Nothing here retrains the model.

## Why this exists

The original release reported `R² = 0.329, RMSE = 0.440` in prose. No
evaluation code was shipped: there is no `r2_score`, RMSE, scipy or
scikit-learn anywhere in that repository, and its notebook stops at a scatter
plot. This package makes those numbers reproducible, and adds the comparisons
needed to interpret them.

## Findings

See [`results/FINDINGS.md`](results/FINDINGS.md). In short: the R² reproduces
(0.330), the RMSE does not (0.391, not 0.440), and the promoter CNN does not
significantly outperform eight scalar half-life features — it is statistically
indifferent to which gene's promoter it is shown.

## Usage

```bash
pip install -r requirements.txt

python run_benchmark.py --stage tabular   # seconds; no sequence data loaded
python run_benchmark.py --stage cnn       # ~10 s; forward pass over released weights
python run_benchmark.py --stage kmer      # ~1 min; streams the 277 MB training split
python run_benchmark.py --stage report    # assembles results/results.{csv,md}

python ablate_sequence.py                 # promoter shuffle / zero ablation
python -m pytest tests/ -q
```

Every stage caches its predictions to `results/preds_*.npy`, so the report can
be rebuilt without recomputation.

## Layout

```
aspine_bench/
  config.py         paths, seed, bootstrap settings
  data.py           memory-efficient loading, one-hot encoding, k-mer features
  metrics.py        R², RMSE, correlations, bootstrap and paired-bootstrap CIs
  baselines.py      mean / ridge / GBM baselines, tuned on the validation split
  cnn.py            loads and runs the released .keras weights
run_benchmark.py    benchmark driver (staged)
ablate_sequence.py  promoter-corruption ablation
results/            predictions, tables, FINDINGS.md
tests/              unit tests for encoders and metrics
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
