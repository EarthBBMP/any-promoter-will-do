"""Analyse the retraining sweep: does a well-trained CNN beat the tabular baseline?

The released repository ships one checkpoint. Evaluating only that checkpoint
turns out to understate the architecture: it sits at the low end of what
retraining achieves, and conclusions drawn from it alone are not safe.

This script answers two questions the sweep raises:

  1. Do the surviving retrained models significantly outperform gradient
     boosting on the eight half-life features? (The released weights do not.)
  2. Do they use *gene-specific* promoter information, or merely the statistics
     of real promoter sequence? Shuffling promoters across genes preserves the
     distribution while destroying identity; zeroing destroys both.

Expects the sweep artifacts (produced by train_cnn.py, optionally on Colab) in
--sweep-dir, and the baseline predictions from run_benchmark.py in results/.

    python analyze_retrained.py
    python analyze_retrained.py --sweep-dir path/to/sweep
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bench import config, data, metrics


def verdict(delta: Dict[str, float]) -> str:
    return "SIGNIFICANT" if (delta["ci_low"] > 0 or delta["ci_high"] < 0) else "not significant"


def compare(y, a, b, label_a: str, label_b: str) -> Dict[str, float]:
    d = metrics.paired_bootstrap_delta(
        y, a, b, "r2", n_resamples=config.N_BOOTSTRAP, seed=config.SEED
    )
    print(
        f"  {label_a:30} vs {label_b:24} ΔR² = {d['delta']:+.4f}  "
        f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  p={d['p_two_sided']:.3f}  {verdict(d)}"
    )
    return d


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", default="sweep")
    args = parser.parse_args()

    sweep = Path(args.sweep_dir)
    if not sweep.exists():
        raise SystemExit(f"{sweep} not found. Run train_cnn.py, or pass --sweep-dir.")

    y = np.load(sweep / "y_test.npy")
    local_y = config.RESULTS_DIR / "y_test.npy"
    if local_y.exists() and not np.array_equal(y, np.load(local_y)):
        raise SystemExit("Sweep test labels differ from the local run; refusing to compare.")

    gbm = np.load(sweep / "preds_halflife_gbm.npy")
    released = np.load(sweep / "preds_aspine_cnn.npy")

    runs = pd.read_csv(sweep / "retrained.csv")
    survivors = runs[~runs["collapsed"]]

    print(f"Test set n={len(y)}   collapsed {int(runs['collapsed'].sum())}/{len(runs)} runs")
    print(
        f"R² across survivors: {survivors['r2'].mean():.4f} ± {survivors['r2'].std(ddof=1):.4f}"
        f"   (released weights: {metrics.all_metrics(y, released)['r2']:.4f})\n"
    )

    print("=== 1. Against gradient boosting on the 8 half-life features ===\n")
    preds: List[Tuple[str, np.ndarray]] = [("released weights", released)]
    for _, row in survivors.iterrows():
        tag = f"seed{int(row['seed'])}_{row['optimizer']}"
        path = sweep / f"preds_retrained_{tag}.npy"
        if path.exists():
            preds.append((f"retrained {tag}", np.load(path)))

    for label, p in preds:
        compare(y, p, gbm, label, "GBM 8 features")

    ensemble = np.mean([p for label, p in preds if label != "released weights"], axis=0)
    compare(y, ensemble, gbm, "mean of survivors", "GBM 8 features")

    print("\n=== 2. Does the model use gene-specific promoter information? ===\n")

    from tensorflow.keras.models import load_model

    test = (
        pd.read_csv(
            str(config.TEST_CSV), usecols=["SEQ", data.LABEL] + data.HALFLIFE_FEATURES
        )
        .dropna()
        .reset_index(drop=True)
    )
    halflife = test[data.HALFLIFE_FEATURES].to_numpy(np.float32)
    promoter = data.one_hot(test["SEQ"].tolist()).astype(np.float32)
    order = np.random.default_rng(config.SEED).permutation(len(promoter))

    checkpoints = [("released weights", config.WEIGHTS_PATH)]
    for _, row in survivors.iterrows():
        tag = f"seed{int(row['seed'])}_{row['optimizer']}"
        path = sweep / f"retrained_{tag}.keras"
        if path.exists():
            checkpoints.append((f"retrained {tag}", path))

    print(f"  {'model':26} {'real':>8} {'shuffled':>10} {'zeroed':>9}")
    variants = {}
    for label, path in checkpoints:
        model = load_model(str(path), compile=False)
        real = model.predict([promoter, halflife], batch_size=64, verbose=0).ravel().astype(np.float64)
        shuffled = model.predict([promoter[order], halflife], batch_size=64, verbose=0).ravel().astype(np.float64)
        zeroed = model.predict([np.zeros_like(promoter), halflife], batch_size=64, verbose=0).ravel().astype(np.float64)
        variants[label] = (real, shuffled, zeroed)
        print(
            f"  {label:26} {metrics.all_metrics(y, real)['r2']:>8.3f} "
            f"{metrics.all_metrics(y, shuffled)['r2']:>10.3f} "
            f"{metrics.all_metrics(y, zeroed)['r2']:>9.3f}"
        )

    print("\n  Real vs shuffled promoters (identity destroyed, distribution kept):")
    for label, (real, shuffled, _) in variants.items():
        compare(y, real, shuffled, label, "same model, shuffled")

    best = max(variants, key=lambda k: metrics.all_metrics(y, variants[k][0])["r2"])
    real, _, zeroed = variants[best]
    print(f"\n  Decomposition on the best model ({best}):")
    compare(y, real, gbm, "full promoter", "GBM 8 features")
    compare(y, zeroed, gbm, "promoter zeroed", "GBM 8 features")
    compare(y, real, zeroed, "full promoter", "promoter zeroed")


if __name__ == "__main__":
    main()
