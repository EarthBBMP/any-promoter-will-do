"""Does the model need a promoter at all, or merely 10 kb of DNA?

The shuffle controls in ``ablate_shuffles.py`` destroy the arrangement of the
promoter while keeping it a promoter. This one keeps real, unmodified genomic
sequence and moves the window somewhere else entirely.

Stored sequences span TSS +/- 10,000 bp, and the released model consumes
positions [3500:13500], that is -6,500 to +3,500. Two alternative windows of the
same length are available from the same stored strings without touching any
other data:

  downstream    [10000:20000], 0 to +10,000, mostly gene body
  far upstream  [0:10000],     -10,000 to 0, upstream of the promoter

Both are real DNA of identical length and identical provenance; neither is the
promoter window the model was trained on. If performance is unchanged, the
network is not reading a promoter, it is reading DNA.

    python ablate_windows.py

Forward passes only; no training, no GPU required.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bench import config, data, metrics

WINDOWS = {
    "promoter": (3500, 13500),
    "downstream": (10000, 20000),
    "far upstream": (0, 10000),
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", default="sweep")
    args = parser.parse_args()
    sweep = Path(args.sweep_dir)

    from tensorflow.keras.models import load_model

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
    seqs = test["SEQ"].tolist()

    encoded = {
        name: data.one_hot([s[lo:hi] for s in seqs], window=None).astype(np.float32)
        for name, (lo, hi) in WINDOWS.items()
    }

    models = [("released", config.WEIGHTS_PATH)]
    runs = pd.read_csv(sweep / "retrained.csv")
    for _, row in runs[~runs["collapsed"]].iterrows():
        tag = f"seed{int(row['seed'])}_{row['optimizer']}"
        path = sweep / f"retrained_{tag}.keras"
        if path.exists():
            models.append((tag, path))

    rows = []
    print(f"Test set n={len(y)}; each window is 10,000 bp of real, unmodified DNA\n")
    print(f"  {'model':12} {'window':14} {'R2':>8} {'RMSE':>7}   vs promoter")

    for name, path in models:
        model = load_model(str(path), compile=False)
        preds = {
            w: model.predict([encoded[w], halflife], batch_size=64, verbose=0)
            .ravel()
            .astype(np.float64)
            for w in WINDOWS
        }
        for w in WINDOWS:
            m = metrics.all_metrics(y, preds[w])
            row = {"model": name, "window": w, "r2": m["r2"], "rmse": m["rmse"]}
            if w == "promoter":
                print(f"  {name:12} {w:14} {m['r2']:>8.4f} {m['rmse']:>7.3f}")
            else:
                d = metrics.paired_bootstrap_delta(
                    y, preds["promoter"], preds[w], "r2",
                    n_resamples=config.N_BOOTSTRAP, seed=config.SEED,
                )
                row.update({"delta_r2": d["delta"], "ci_low": d["ci_low"],
                            "ci_high": d["ci_high"], "p": d["p_two_sided"]})
                sig = "sig" if (d["ci_low"] > 0 or d["ci_high"] < 0) else "ns"
                print(f"  {name:12} {w:14} {m['r2']:>8.4f} {m['rmse']:>7.3f}   "
                      f"{d['delta']:>+8.4f} [{d['ci_low']:>+7.4f},{d['ci_high']:>+7.4f}] "
                      f"p={d['p_two_sided']:.3f} {sig}")
            rows.append(row)
        print()

    out = config.RESULTS_DIR / "window_control.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
