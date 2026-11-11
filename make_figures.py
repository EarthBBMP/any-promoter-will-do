"""Generate the two figures for the paper.

Figure 1: every model on one axis with bootstrap confidence intervals.
Figure 2: the promoter intervention - real vs shuffled vs zeroed.

Both are written as vector PDF so they stay sharp in the LaTeX build.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from aspine_bench import config, data, metrics

# Validated categorical slots (light surface). Order is fixed, never cycled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#dcdcd8"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def load_predictions(sweep: Path):
    y = np.load(sweep / "y_test.npy")
    preds = {
        "Train mean": np.load(sweep / "preds_mean.npy"),
        "6-mer only (ridge)": np.load(config.RESULTS_DIR / "preds_kmer_ridge.npy"),
        "Half-life only (ridge)": np.load(sweep / "preds_halflife_ridge.npy"),
        "6-mer + half-life (ridge)": np.load(config.RESULTS_DIR / "preds_kmer_halflife_ridge.npy"),
        "Half-life only (GBM)": np.load(sweep / "preds_halflife_gbm.npy"),
        "CNN, released weights": np.load(sweep / "preds_aspine_cnn.npy"),
        "CNN, retrained (best)": np.load(sweep / "preds_retrained_seed1_adam.npy"),
    }
    return y, preds


def figure_baselines(y, preds, out: Path) -> None:
    """Horizontal bars: model names are long, and rank is easier to read vertically."""

    names, points, lows, highs = [], [], [], []
    for name, p in preds.items():
        point, low, high = metrics.bootstrap_ci(
            y, p, "r2", n_resamples=config.N_BOOTSTRAP, seed=config.SEED
        )
        names.append(name)
        points.append(point)
        lows.append(point - low)
        highs.append(high - point)

    order = np.argsort(points)
    names = [names[i] for i in order]
    points = np.array(points)[order]
    err = np.vstack([np.array(lows)[order], np.array(highs)[order]])

    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    positions = np.arange(len(names))

    # The two CNN rows are the subject of the paper; the rest are reference points.
    colors = [ORANGE if "CNN" in n else BLUE for n in names]
    ax.barh(positions, points, height=0.62, color=colors, zorder=3)
    ax.errorbar(
        points, positions, xerr=err, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=4
    )

    # Anchor each label outside the whisker, not at the bar end, or the value
    # sits on top of the error bar it is meant to annotate.
    upper = points + err[1]
    lower = points - err[0]
    for pos, value, hi, lo in zip(positions, points, upper, lower):
        if value >= 0:
            ax.text(hi + 0.014, pos, f"{value:.3f}", va="center", ha="left",
                    fontsize=7.5, color=INK)
        else:
            ax.text(lo - 0.014, pos, f"{value:.3f}", va="center", ha="right",
                    fontsize=7.5, color=INK)

    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Test $R^2$ (variance explained), with 95% bootstrap CI")
    ax.set_xlim(-0.16, 0.52)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        handles=[
            Patch(facecolor=ORANGE, label="Promoter CNN"),
            Patch(facecolor=BLUE, label="Baseline"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=7.5,
    )

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def figure_ablation(table: pd.DataFrame, out: Path) -> None:
    """
    Grouped bars: five promoter conditions across four trained models.

    The four in-distribution conditions form an ordered progression of how much
    sequence structure has been destroyed, so they take an ordinal blue ramp
    (darkest = intact). Zeroing is categorically different, an input the network
    could never see, so it takes a contrasting hue rather than a ramp step.
    """

    order = ["real", "across-genes", "dinucleotide", "mononucleotide", "zeroed"]
    pretty = {
        "real": "Real promoter",
        "across-genes": "Swapped between genes",
        "dinucleotide": "Dinucleotide-shuffled",
        "mononucleotide": "Mononucleotide-shuffled",
        "zeroed": "Zeroed (off-distribution)",
    }
    # Ordinal ramp, intact -> most destroyed; orange marks the off-distribution case.
    colours = ["#184f95", "#2a78d6", "#5598e7", "#86b6ef", ORANGE]

    models = [m for m in
              ["released", "seed1_sgd", "seed2_sgd", "seed1_adam"]
              if m in set(table["model"])]
    names = {"released": "Released\nweights", "seed1_sgd": "Retrained\nseed 1, SGD",
             "seed2_sgd": "Retrained\nseed 2, SGD", "seed1_adam": "Retrained\nseed 1, Adam"}

    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    x = np.arange(len(models))
    width = 0.165

    for i, (cond, colour) in enumerate(zip(order, colours)):
        offset = (i - 2) * width
        vals = [
            float(table[(table.model == m) & (table.condition == cond)]["r2"].iloc[0])
            for m in models
        ]
        ax.bar(x + offset, vals, width * 0.9, label=pretty[cond], color=colour, zorder=3)

    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=2)
    # Band showing where all in-distribution conditions land, to make the
    # "nothing moves" result legible at a glance.
    intact = table[table.condition != "zeroed"]["r2"]
    ax.axhspan(intact.min(), intact.max(), color="#2a78d6", alpha=0.07, zorder=1)
    ax.annotate("every in-distribution condition", xy=(len(models) - 0.42, intact.max()),
                xytext=(0, 5), textcoords="offset points", fontsize=6.4, color=MUTED, ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([names[m] for m in models], fontsize=7.5)
    ax.set_ylabel("Test $R^2$")
    ax.set_ylim(-0.36, 0.52)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.22), columnspacing=1.1, handlelength=1.2)

    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", default="aspine_results")
    parser.add_argument(
        "--out-dir",
        default=str(config.REPO_ROOT / "Formatting_Instructions_For_NeurIPS_2026 (3)"),
    )
    args = parser.parse_args()

    sweep = Path(args.sweep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y, preds = load_predictions(sweep)
    figure_baselines(y, preds, out_dir / "fig_baselines.pdf")

    ablation = pd.read_csv(config.RESULTS_DIR / "shuffle_ablation.csv")
    figure_ablation(ablation, out_dir / "fig_ablation.pdf")

    print("\nCIs used in Figure 1:")
    for name, p in preds.items():
        point, low, high = metrics.bootstrap_ci(
            y, p, "r2", n_resamples=config.N_BOOTSTRAP, seed=config.SEED
        )
        print(f"  {name:30} {point:+.4f}  [{low:+.4f}, {high:+.4f}]")


if __name__ == "__main__":
    main()
