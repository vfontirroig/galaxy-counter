"""
Histogram of the per-galaxy MSE of every trained version (one per lambda_geo),
per direction.

For each version it reads <root>/<version>/pixel_metrics/per_galaxy.csv, written
by pixel_metrics.py (column mse_mean_vs_target, in the normalised units of the
target survey). Versions without a per_galaxy.csv are skipped with a warning.

Three views, each split by direction (Euclid target, COSMOS target), since the
MSE is in each target survey's own normalised units and the two are not
comparable with each other:
  mse_all.png                   every lambda_geo overlaid on shared bins, one
                                panel per direction
  mse_all_panels_<survey>.png   one small panel per lambda_geo, its histogram
                                over the average of all lambda_geo in grey,
                                with its median dashed; one figure per direction
  mse_all_boxes.png             one box per lambda_geo on the lambda_geo axis:
                                median, quartiles and 5-95% whiskers; one panel
                                per direction
Histograms use log-spaced bins on a log axis when a direction's values span two
orders of magnitude or more, as pixel_metrics.py's histograms do; the boxes then
use a log axis too.

Usage:
    python experiments/euclid-cosmos/visualization/mse_all_plot.py \
        --root    /n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1 \
        --out-dir /n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/mse_all
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#CONFIG

VERSIONS = ['v14','v5','v6','v4','v7','v8','v9','v10','v11','v12','v13','v15']
LAMBDA_GEO = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.05]

# One colour per lambda_geo, in LAMBDA_GEO order: the same as fid_all_plot.py.
LAMBDA_COLORS = ["#37cda0", "#2956c4", "#a94620", "#6880f8", "#463d7a", "#d76821",
                 "#c1aaff", "#7c5d7b", "#ee9e10", "#7b2431", "#a1809f", "#4b0339"]

RUN_NAME = "pixel_metrics"    # subfolder written by run_pixel_metrics.sh
COLUMN = "mse_mean_vs_target"
SURVEYS = {"euclid": "Euclid target", "cosmos": "COSMOS target"}


def load_all(root):
    """One row per (version, galaxy): version, lambda_geo, idx, anchor_survey, mse."""
    frames = []
    for lam, version in sorted(zip(LAMBDA_GEO, VERSIONS)):
        path = os.path.join(root, version, RUN_NAME, "per_galaxy.csv")
        if not os.path.exists(path):
            print(f"[WARN] {version} (lambda_geo={lam}): no {RUN_NAME}/per_galaxy.csv, skipped")
            continue
        df = pd.read_csv(path, usecols=["idx", "anchor_survey", COLUMN])
        df.insert(0, "lambda_geo", lam)
        df.insert(0, "version", version)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No version under {root} has {RUN_NAME}/per_galaxy.csv.")
    return pd.concat(frames, ignore_index=True)


def shared_bins(values):
    """40 bins over all versions of one panel; log-spaced when values span >= 2 decades."""
    values = values[np.isfinite(values)]
    positive = values[values > 0]
    if positive.max() / positive.min() >= 100:
        return np.logspace(np.log10(positive.min()), np.log10(positive.max()), 41), True
    return np.linspace(values.min(), values.max(), 41), False


def plot_all(df, path):
    """One panel per direction, every lambda_geo overlaid on that panel's bins."""
    # keyed by the lambda value, so a skipped version never repaints the others
    color_of = dict(zip(LAMBDA_GEO, LAMBDA_COLORS))

    surveys = [s for s in SURVEYS if s in set(df["anchor_survey"])]
    fig, axes = plt.subplots(len(surveys), 1, figsize=(8, 4.5 * len(surveys)), squeeze=False)
    for ax, survey in zip(axes[:, 0], surveys):
        sub = df[df["anchor_survey"] == survey]
        bins, log = shared_bins(sub[COLUMN].to_numpy())
        handles = []
        for lam, group in sub.groupby("lambda_geo"):
            values = group[COLUMN].to_numpy()
            ax.hist(values, bins=bins, histtype="step", color=color_of[lam], linewidth=2)
            # hist() legend entries are filled patches; lines match the step outlines
            handles.append(Line2D([], [], color=color_of[lam], linewidth=2,
                                  label=f"λ = {lam:g} {group['version'].iloc[0]} "
                                        f"(median: {np.median(values):.3g})"))
        if log:
            ax.set_xscale("log")
        ax.set_xlabel("MSE, mean generation vs target (normalised units)")
        ax.set_ylabel("Galaxies")
        ax.set_title(f"{SURVEYS[survey]} (N = {sub.groupby('lambda_geo').size().max()} per version)",
                     fontsize=11)
        ax.legend(handles=handles, title="Geometric weight", fontsize=8, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_panels(df, survey, path, ncols=4):
    """One small panel per lambda_geo for one direction, each over the all-lambda average."""
    color_of = dict(zip(LAMBDA_GEO, LAMBDA_COLORS))
    sub = df[df["anchor_survey"] == survey]
    bins, log = shared_bins(sub[COLUMN].to_numpy())
    groups = list(sub.groupby("lambda_geo"))
    # average histogram over versions, the same reference behind every panel
    reference = np.mean([np.histogram(g[COLUMN], bins=bins)[0] for _, g in groups], axis=0)

    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.4 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    for ax in axes.flat[len(groups):]:
        ax.axis("off")
    for ax, (lam, group) in zip(axes.flat, groups):
        values = group[COLUMN].to_numpy()
        ax.stairs(reference, bins, fill=True, color="#d0d4da")
        ax.hist(values, bins=bins, histtype="step", color=color_of[lam], linewidth=2)
        ax.axvline(np.median(values), color=color_of[lam], linewidth=1, linestyle="--")
        if log:
            ax.set_xscale("log")
        ax.set_title(f"λ = {lam:g} ({group['version'].iloc[0]}) · median {np.median(values):.3g}",
                     fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("MSE, mean generation vs target (normalised units)")
    fig.supylabel("Galaxies")
    fig.suptitle(f"{SURVEYS[survey]} · grey: average of all λ", fontsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_boxes(df, path):
    """One box per lambda_geo on a categorical lambda_geo axis, one panel per direction."""
    color_of = dict(zip(LAMBDA_GEO, LAMBDA_COLORS))
    # categorical positions: 0.05 would otherwise be squeezed between 0.0 and 0.1
    lambdas = sorted(df["lambda_geo"].unique())
    pos = {lam: i for i, lam in enumerate(lambdas)}
    version_of = dict(zip(df["lambda_geo"], df["version"]))

    surveys = [s for s in SURVEYS if s in set(df["anchor_survey"])]
    fig, axes = plt.subplots(len(surveys), 1, figsize=(10, 3.6 * len(surveys)),
                             sharex=True, squeeze=False)
    for ax, survey in zip(axes[:, 0], surveys):
        sub = df[df["anchor_survey"] == survey]
        groups = list(sub.groupby("lambda_geo"))
        boxes = ax.boxplot([g[COLUMN].to_numpy() for _, g in groups],
                           positions=[pos[lam] for lam, _ in groups], widths=0.6,
                           whis=(5, 95), showfliers=False,
                           boxprops=dict(color="#444444"), whiskerprops=dict(color="#444444"),
                           capprops=dict(color="#444444"))
        for median, (lam, _) in zip(boxes["medians"], groups):
            median.set(color=color_of[lam], linewidth=2.5)
        if shared_bins(sub[COLUMN].to_numpy())[1]:
            ax.set_yscale("log")
        ax.set_ylabel("MSE (normalised units)")
        ax.set_title(f"{SURVEYS[survey]} · median, quartiles, 5–95% whiskers", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[-1, 0]
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{lam:g}\n{version_of[lam]}" for lam in lambdas], fontsize=9)
    ax.set_xlabel("Geometric weight λ")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",    required=True,
                   help="Folder holding one subfolder per version, each with pixel_metrics/per_galaxy.csv")
    p.add_argument("--out-dir", default="mse_all")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_all(args.root)
    summary = (df.groupby(["version", "lambda_geo", "anchor_survey"], sort=False)[COLUMN]
                 .agg(n="size", mean="mean", median="median", std="std").reset_index())
    for row in summary.itertuples():
        print(f"  {row.version:>3} | lambda_geo {row.lambda_geo:<4g} | {row.anchor_survey:<6} target | "
              f"N {row.n} | MSE mean {row.mean:.4g}  median {row.median:.4g}")

    summary_path = os.path.join(args.out_dir, "mse_all.csv")
    summary.to_csv(summary_path, index=False, float_format="%.6g")
    print(f"Saved {summary_path}")

    plot_all(df, os.path.join(args.out_dir, "mse_all.png"))
    for survey in SURVEYS:
        if survey in set(df["anchor_survey"]):
            plot_panels(df, survey, os.path.join(args.out_dir, f"mse_all_panels_{survey}.png"))
    plot_boxes(df, os.path.join(args.out_dir, "mse_all_boxes.png"))


if __name__ == "__main__":
    main()
