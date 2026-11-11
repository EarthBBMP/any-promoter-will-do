"""In-distribution promoter ablations across every trained model.

Our zeroing intervention is confounded: an all-zero tensor is not a
promoter-free genome but an impossible one, and three of four models fail on it
outright rather than degrading. These controls remove that confound. Each hands
the network a real DNA sequence of the same length, so the model stays in
distribution and only the arrangement of the promoter changes.

Five conditions per model:

  real            the gene's own promoter
  across-genes    another gene's real promoter (destroys gene identity,
                  preserves everything about the sequence distribution)
  mononucleotide  bases permuted within the sequence (preserves composition,
                  destroys dinucleotides, motifs and position)
  dinucleotide    Altschul-Erikson shuffle (preserves all 16 dinucleotide
                  counts including CpG, destroys higher-order and positional
                  structure)
  zeroed          all-zero tensor, reported for continuity with earlier results
                  and as an out-of-distribution reference point

Each condition is compared to `real` by paired bootstrap.

    python ablate_shuffles.py

This is forward-pass only: no training, no GPU required.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bench import config, data, metrics, shuffles


def load_models(sweep: Path):
    from tensorflow.keras.models import load_model

    found = [("released", config.WEIGHTS_PATH)]
    runs = pd.read_csv(sweep / "retrained.csv")
    for _, row in runs[~runs["collapsed"]].iterrows():
        tag = f"seed{int(row['seed'])}_{row['optimizer']}"
        path = sweep / f"retrained_{tag}.keras"
        if path.exists():
            found.append((tag, path))

    return [(name, load_model(str(p), compile=False)) for name, p in found]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", default="sweep")
    args = parser.parse_args()

    rng = np.random.default_rng(config.SEED)
    lo, hi = data.WINDOW

    test = (
        pd.read_csv(
            str(config.TEST_CSV),
            usecols=["SEQ", data.LABEL] + data.HALFLIFE_FEATURES,
        )
        .dropna()
        .reset_index(drop=True)
    )
    y = test[data.LABEL].to_numpy(np.float64)
    halflife = test[data.HALFLIFE_FEATURES].to_numpy(np.float32)
    windows = [s[lo:hi] for s in test["SEQ"]]

    print(f"Building shuffled promoter sets for {len(windows)} test genes")
    started = time.time()
    mono = [shuffles.mononucleotide_shuffle(s, rng) for s in windows]
    dinu = [shuffles.dinucleotide_shuffle(s, rng) for s in windows]
    print(f"  done in {time.time() - started:.0f}s")

    # Confirm the shuffles preserved what they are supposed to preserve, on a
    # sample. A silent failure here would invalidate every number below.
    idx = rng.integers(0, len(windows), size=25)
    assert all(sorted(mono[i]) == sorted(windows[i]) for i in idx), "composition changed"
    assert all(
        shuffles.dinucleotide_counts(dinu[i]) == shuffles.dinucleotide_counts(windows[i])
        for i in idx
    ), "dinucleotide counts changed"
    print("  verified: composition and dinucleotide counts preserved")

    order = rng.permutation(len(windows))
    encoded = {
        "real": data.one_hot(windows, window=None).astype(np.float32),
        "mononucleotide": data.one_hot(mono, window=None).astype(np.float32),
        "dinucleotide": data.one_hot(dinu, window=None).astype(np.float32),
    }
    encoded["across-genes"] = encoded["real"][order]
    encoded["zeroed"] = np.zeros_like(encoded["real"])

    conditions = ["real", "across-genes", "dinucleotide", "mononucleotide", "zeroed"]
    rows = []

    for name, model in load_models(Path(args.sweep_dir)):
        print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
        preds = {
            c: model.predict([encoded[c], halflife], batch_size=64, verbose=0)
            .ravel()
            .astype(np.float64)
            for c in conditions
        }

        print(f"  {'condition':16} {'R2':>8} {'RMSE':>7}   vs real: {'dR2':>8} {'95% CI':>20} {'p':>7}")
        for c in conditions:
            m = metrics.all_metrics(y, preds[c])
            if c == "real":
                print(f"  {c:16} {m['r2']:>8.4f} {m['rmse']:>7.3f}")
                rows.append({"model": name, "condition": c, "r2": m["r2"], "rmse": m["rmse"]})
                continue
            d = metrics.paired_bootstrap_delta(
                y, preds["real"], preds[c], "r2",
                n_resamples=config.N_BOOTSTRAP, seed=config.SEED,
            )
            sig = "sig" if (d["ci_low"] > 0 or d["ci_high"] < 0) else "ns"
            print(
                f"  {c:16} {m['r2']:>8.4f} {m['rmse']:>7.3f}            "
                f"{d['delta']:>+8.4f} [{d['ci_low']:>+7.4f},{d['ci_high']:>+7.4f}] "
                f"{d['p_two_sided']:>7.3f} {sig}"
            )
            rows.append({
                "model": name, "condition": c, "r2": m["r2"], "rmse": m["rmse"],
                "delta_r2": d["delta"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                "p": d["p_two_sided"],
            })

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "shuffle_ablation.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
