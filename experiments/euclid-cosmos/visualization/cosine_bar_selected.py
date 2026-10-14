"""
Bar plot of the Euclid-vs-COSMOS cosine similarity for a few galaxies, one bar
per galaxy.

Reads per_galaxy.csv from cosine_similarity.py, which covers the test split
only. By default it draws --n random test galaxies (reproducible via --seed);
pass --indices / --indices-file to pick the galaxies yourself instead.

Usage:
    python experiments/euclid-cosmos/visualization/cosine_bar_selected.py \
        --per-galaxy /n03data/fontirro/.../cosine_similarity/per_galaxy.csv \
        --out cosine_bar_random.png

    # hand-picked galaxies instead of random ones
    ... --indices 1203 4410 5872
    ... --indices-file my_galaxies.txt   # .npy, or .txt/.csv with one index per line
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAR_COLOR = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
AXIS_COLOR = "#898781"
GRID_COLOR = "#e4e3df"


def read_indices(args):
    indices = list(args.indices or [])
    if args.indices_file:
        if args.indices_file.endswith(".npy"):
            indices += np.load(args.indices_file).astype(int).tolist()
        else:
            indices += np.loadtxt(args.indices_file, dtype=int, delimiter=",",
                                  ndmin=1).tolist()
    return list(dict.fromkeys(indices))  # drop duplicates, keep order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-galaxy", required=True, help="per_galaxy.csv from cosine_similarity.py")
    p.add_argument("--n", type=int, default=8, help="Random test galaxies to draw (default: 8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--indices", type=int, nargs="+",
                   help="h5 indices to plot instead of random galaxies")
    p.add_argument("--indices-file", default=None,
                   help=".npy, or .txt/.csv with one h5 index per line")
    p.add_argument("--variant", default="enc1_flat_global",
                   help="enc{1,2}_{flat,pooled}_{raw,global,per_survey} (default: enc1_flat_global)")
    p.add_argument("--out", default="cosine_bar_selected.png")
    args = p.parse_args()

    df = pd.read_csv(args.per_galaxy).set_index("dataset_idx")
    column = f"{args.variant}_cos_matched"
    if column not in df.columns:
        raise SystemExit(f"No column {column} in {args.per_galaxy}; check --variant.")

    wanted = read_indices(args)
    if wanted:
        selected = df.reindex(wanted)
        missing = selected.index[selected[column].isna()].tolist()
        if missing:
            print(f"[WARN] {len(missing)} indices are not in the test split and are skipped: {missing}")
        selected = selected.dropna(subset=[column])
        if selected.empty:
            raise SystemExit("None of the given indices are in per_galaxy.csv.")
        label = f"{len(selected)} selected galaxies"
    else:
        selected = df.sample(n=min(args.n, len(df)), random_state=args.seed)
        label = f"{len(selected)} random test galaxies (seed {args.seed})"

    values = selected[column].to_numpy()
    ids = selected.index.to_numpy()
    print(f"{'dataset_idx':>11}  cosine")
    for idx, v in zip(ids, values):
        print(f"{idx:>11}  {v:+.4f}")
    print(f"mean {values.mean():+.4f}, median {np.median(values):+.4f}")

    x = np.arange(len(values))
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(values) + 2), 3.8))
    ax.bar(x, values, width=0.6, color=BAR_COLOR, zorder=2)
    for xi, v in zip(x, values):
        ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3 if v >= 0 else -3),
                    textcoords="offset points", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, color=INK_SECONDARY)
    ax.axhline(0, color=AXIS_COLOR, linewidth=0.8, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in ids], rotation=45 if len(ids) > 10 else 0)
    ax.set_xlabel("Galaxy (h5 dataset_idx)", color=INK_SECONDARY)
    ax.set_ylabel("Cosine similarity\n(Euclid vs COSMOS)", color=INK_SECONDARY)
    ax.set_ylim(min(0, values.min() - 0.1), 1.05)
    ax.set_title(f"{label} · {args.variant}", color=INK_PRIMARY, fontsize=11)
    ax.tick_params(colors=INK_SECONDARY)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
