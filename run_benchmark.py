"""Run the ASPINE benchmark: released model plus baselines, with bootstrap CIs.

Stages can be run independently because they differ by orders of magnitude in
cost:

    python run_benchmark.py --stage tabular   # seconds, no sequence data
    python run_benchmark.py --stage cnn       # ~1 min, forward pass only
    python run_benchmark.py --stage kmer      # ~10 min, streams the 277 MB split
    python run_benchmark.py --stage report    # assemble results table

Predictions are cached to results/preds_<name>.npy so the report can be rebuilt
without recomputing anything.
"""

import argparse
import json
import sys
import time
from typing import Dict

import numpy as np
import pandas as pd

from aspine_bench import baselines, config, data, metrics

PRED_DIR = config.RESULTS_DIR
DISPLAY_NAMES = {
    "mean": "Train-mean (floor)",
    "halflife_ridge": "Half-life features only, ridge",
    "halflife_gbm": "Half-life features only, GBM",
    "kmer_ridge": "6-mer composition only, ridge",
    "kmer_halflife_ridge": "6-mer + half-life, ridge",
    "aspine_cnn": "ASPINE CNN (released weights)",
}


def _save(name: str, preds: np.ndarray, info: Dict) -> None:
    np.save(PRED_DIR / f"preds_{name}.npy", preds)
    (PRED_DIR / f"info_{name}.json").write_text(json.dumps(info, indent=2))
    print(f"  saved {name}: n={len(preds)}  {info}")


def load_tabular_splits():
    return (
        data.load_tabular(str(config.TRAIN_CSV)),
        data.load_tabular(str(config.VALID_CSV)),
        data.load_tabular(str(config.TEST_CSV)),
    )


def stage_tabular() -> None:
    print("[tabular] loading splits (SEQ column skipped)")
    train, valid, test = load_tabular_splits()

    xs = {
        split_name: df[data.HALFLIFE_FEATURES].to_numpy(dtype=np.float64)
        for split_name, df in [("train", train), ("valid", valid), ("test", test)]
    }
    ys = {
        split_name: df[data.LABEL].to_numpy(dtype=np.float64)
        for split_name, df in [("train", train), ("valid", valid), ("test", test)]
    }
    np.save(PRED_DIR / "y_test.npy", ys["test"])

    print("[tabular] mean")
    _save("mean", baselines.predict_mean(ys["train"], len(ys["test"])), {})

    print("[tabular] ridge")
    preds, info = baselines.predict_ridge(
        xs["train"], ys["train"], xs["valid"], ys["valid"], xs["test"]
    )
    _save("halflife_ridge", preds, info)

    print("[tabular] gbm (grid over validation split)")
    preds, info = baselines.predict_gbm(
        xs["train"], ys["train"], xs["valid"], ys["valid"], xs["test"]
    )
    _save("halflife_gbm", preds, info)


def stage_cnn() -> None:
    from aspine_bench import cnn

    print("[cnn] loading released weights")
    model = cnn.load_released_model()
    window = cnn.expected_promoter_length(model)
    print(f"[cnn] model expects {window} bp promoter window")

    print("[cnn] loading test split with sequences")
    test = pd.read_csv(
        str(config.TEST_CSV),
        usecols=["gene_id", data.LABEL, "SEQ"] + data.HALFLIFE_FEATURES,
    ).dropna().reset_index(drop=True)

    halflife = test[data.HALFLIFE_FEATURES].to_numpy(dtype=np.float64)
    y_test = test[data.LABEL].to_numpy(dtype=np.float64)
    np.save(PRED_DIR / "y_test.npy", y_test)

    start = time.time()
    preds = cnn.predict(model, test["SEQ"].tolist(), halflife)
    print(f"[cnn] forward pass over {len(preds)} genes in {time.time()-start:.1f}s")

    _save("aspine_cnn", preds, {"window_bp": window, "n_params": int(model.count_params())})


def stage_kmer(k: int = 6) -> None:
    print(f"[kmer] streaming training split for {k}-mer features")

    def featurise(csv_path):
        feats, tabs = [], []
        for sequences, tab in data.iter_sequence_chunks(str(csv_path)):
            feats.append(data.kmer_frequencies(sequences, k=k))
            tabs.append(tab)
        return np.vstack(feats), pd.concat(tabs, ignore_index=True)

    start = time.time()
    train_k, train_tab = featurise(config.TRAIN_CSV)
    print(f"[kmer] train {train_k.shape} in {time.time()-start:.1f}s")
    valid_k, valid_tab = featurise(config.VALID_CSV)
    test_k, test_tab = featurise(config.TEST_CSV)
    print(f"[kmer] valid {valid_k.shape}  test {test_k.shape}")

    ys = {
        "train": train_tab[data.LABEL].to_numpy(dtype=np.float64),
        "valid": valid_tab[data.LABEL].to_numpy(dtype=np.float64),
        "test": test_tab[data.LABEL].to_numpy(dtype=np.float64),
    }

    print("[kmer] ridge on composition alone")
    preds, info = baselines.predict_ridge(
        train_k, ys["train"], valid_k, ys["valid"], test_k
    )
    _save("kmer_ridge", preds, {**info, "k": k, "n_features": train_k.shape[1]})

    print("[kmer] ridge on composition + half-life")
    combo = {
        "train": np.hstack([train_k, train_tab[data.HALFLIFE_FEATURES].to_numpy(np.float64)]),
        "valid": np.hstack([valid_k, valid_tab[data.HALFLIFE_FEATURES].to_numpy(np.float64)]),
        "test": np.hstack([test_k, test_tab[data.HALFLIFE_FEATURES].to_numpy(np.float64)]),
    }
    preds, info = baselines.predict_ridge(
        combo["train"], ys["train"], combo["valid"], ys["valid"], combo["test"]
    )
    _save("kmer_halflife_ridge", preds, {**info, "k": k})


def stage_report() -> None:
    y_test = np.load(PRED_DIR / "y_test.npy")
    rows = []

    for name, label in DISPLAY_NAMES.items():
        path = PRED_DIR / f"preds_{name}.npy"
        if not path.exists():
            continue
        preds = np.load(path)
        row = {"model": label, "key": name}
        row.update(metrics.all_metrics(y_test, preds))
        point, lo, hi = metrics.bootstrap_ci(
            y_test, preds, "r2", n_resamples=config.N_BOOTSTRAP, seed=config.SEED
        )
        row["r2_ci_low"], row["r2_ci_high"] = lo, hi
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(PRED_DIR / "results.csv", index=False)

    lines = [
        "| Model | R² (var. explained) | 95% CI | RMSE | Pearson r | Pearson r² | Spearman ρ |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['model']} | {r['r2']:.3f} | [{r['r2_ci_low']:.3f}, {r['r2_ci_high']:.3f}] | "
            f"{r['rmse']:.3f} | {r['pearson_r']:.3f} | {r['pearson_r2']:.3f} | {r['spearman_r']:.3f} |"
        )
    table = "\n".join(lines)
    (PRED_DIR / "results.md").write_text(table, encoding="utf-8")

    print(f"\nTest set: n={len(y_test)}, std={y_test.std():.4f}\n")
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["tabular", "cnn", "kmer", "report", "all"],
        default="all",
    )
    parser.add_argument("--k", type=int, default=6, help="k-mer size")
    args = parser.parse_args()

    # Windows consoles default to a legacy codepage (cp874 here) that cannot
    # encode the symbols used in the results table; the files are UTF-8 either way.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(config.SEED)

    if args.stage in ("tabular", "all"):
        stage_tabular()
    if args.stage in ("cnn", "all"):
        stage_cnn()
    if args.stage in ("kmer", "all"):
        stage_kmer(k=args.k)
    if args.stage in ("report", "all"):
        stage_report()


if __name__ == "__main__":
    main()
