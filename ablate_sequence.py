"""Does the released ASPINE CNN actually use the promoter sequence?

The baseline table shows the CNN does not significantly outperform eight
tabular half-life features. That is consistent with two very different
explanations:

  (a) the promoter sequence carries signal, but the model extracts only what
      the tabular features already encode; or
  (b) the model has learned to ignore the promoter branch entirely.

These are distinguishable by intervention. We re-run the forward pass with the
promoter input corrupted in ways that destroy any gene-specific sequence
signal while leaving the half-life features untouched:

  * ``shuffled`` - promoters permuted across genes, so each gene is paired with
    another gene's real promoter. Preserves the sequence distribution exactly.
  * ``zeroed``   - all-zero promoter, the encoding used for ambiguous bases.

If performance is unchanged under these corruptions, the promoter contributes
nothing gene-specific to the prediction and explanation (b) holds.
"""

import sys

import numpy as np
import pandas as pd

from bench import cnn, config, data, metrics


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rng = np.random.default_rng(config.SEED)

    model = cnn.load_released_model()
    test = (
        pd.read_csv(
            str(config.TEST_CSV),
            usecols=["gene_id", data.LABEL, "SEQ"] + data.HALFLIFE_FEATURES,
        )
        .dropna()
        .reset_index(drop=True)
    )

    y = test[data.LABEL].to_numpy(dtype=np.float64)
    halflife = test[data.HALFLIFE_FEATURES].to_numpy(dtype=np.float64)
    sequences = test["SEQ"].tolist()

    promoter = data.one_hot(sequences).astype(np.float32)

    # Permute promoters across genes. Any gene that keeps its own promoter is
    # rotated onward, so no gene is left paired with its real sequence.
    order = rng.permutation(len(sequences))
    fixed = np.flatnonzero(order == np.arange(len(order)))
    if len(fixed) == 1:
        swap = (fixed[0] + 1) % len(order)
        order[[fixed[0], swap]] = order[[swap, fixed[0]]]
    elif len(fixed) > 1:
        order[fixed] = np.roll(order[fixed], 1)

    variants = {
        "real promoter": promoter,
        "promoters shuffled across genes": promoter[order],
        "promoter zeroed out": np.zeros_like(promoter),
    }

    lines = [
        f"Test set: n={len(y)}",
        "",
        "| Promoter input | R² | RMSE | Pearson r |",
        "|---|---|---|---|",
    ]

    predictions = {}
    for name, promoter_variant in variants.items():
        preds = (
            model.predict(
                [promoter_variant, halflife.astype(np.float32)],
                batch_size=64,
                verbose=0,
            )
            .ravel()
            .astype(np.float64)
        )
        predictions[name] = preds
        m = metrics.all_metrics(y, preds)
        lines.append(
            f"| {name} | {m['r2']:.3f} | {m['rmse']:.3f} | {m['pearson_r']:.3f} |"
        )

    real = predictions["real promoter"]

    lines += ["", "### Movement in predictions when the promoter is destroyed", ""]
    for name, preds in predictions.items():
        if name == "real promoter":
            continue
        delta = np.abs(preds - real)
        corr = float(np.corrcoef(preds, real)[0, 1])
        lines.append(
            f"- {name}: mean |Δprediction| = {delta.mean():.5f}, "
            f"max = {delta.max():.5f}, correlation with real = {corr:.5f}"
        )

    lines += [
        "",
        "### Paired bootstrap: real vs corrupted promoter",
        "",
        "| Comparison | ΔR² | 95% CI | p | Verdict |",
        "|---|---|---|---|---|",
    ]
    for name, preds in predictions.items():
        if name == "real promoter":
            continue
        d = metrics.paired_bootstrap_delta(
            y, real, preds, "r2", n_resamples=config.N_BOOTSTRAP, seed=config.SEED
        )
        verdict = (
            "**significant**"
            if (d["ci_low"] > 0 or d["ci_high"] < 0)
            else "not significant"
        )
        lines.append(
            f"| real vs {name} | {d['delta']:+.4f} | "
            f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] | "
            f"{d['p_two_sided']:.3f} | {verdict} |"
        )

    report = "\n".join(lines)
    (config.RESULTS_DIR / "ablation.md").write_text(report, encoding="utf-8")
    print("\n" + report)

    np.save(
        config.RESULTS_DIR / "preds_cnn_shuffled_promoter.npy",
        predictions["promoters shuffled across genes"],
    )
    np.save(
        config.RESULTS_DIR / "preds_cnn_zeroed_promoter.npy",
        predictions["promoter zeroed out"],
    )


if __name__ == "__main__":
    main()
