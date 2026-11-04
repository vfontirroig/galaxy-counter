"""
Bar chart of the FID of every trained version (one per lambda_geo), per direction.

For each version it reads <root>/<version>/fid/fid.csv, written by compute_fid.py
(rows anchor_survey, n, fid). Versions without a fid.csv are skipped with a
warning.

One panel per direction (Euclid target, COSMOS target), sharing the lambda_geo
axis. Each panel has its own y-scale: the two FIDs are measured against
different real sets, so their values are not comparable with each other.

Usage:
    python experiments/euclid-cosmos/visualization/fid_all.py \
        --root    /n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1 \
        --out-dir /n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/fid_all
"""

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#CONFIG

VERSIONS = ['v14','v5','v6','v4','v7','v8','v9','v10','v11','v12','v13','v15']
LAMBDA_GEO = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.05]

# One colour per lambda_geo, in LAMBDA_GEO order: the first 11 are cos_sim_all's.
# The 12th (v15) was found by the same search: it keeps every pair at OKLab
# Delta E >= 12.7 under normal vision and simulated protan/deutan/tritan colour
# blindness (Machado et al. 2009), and clears 2:1 contrast on white.
LAMBDA_COLORS = ["#37cda0", "#2956c4", "#a94620", "#6880f8", "#463d7a", "#d76821",
                 "#c1aaff", "#7c5d7b", "#ee9e10", "#7b2431", "#a1809f", "#4b0339"]

RUN_NAME = "fid"    # subfolder written by run_compute_fid.sh
SURVEYS = {"euclid": "Euclid target", "cosmos": "COSMOS target"}


def load_all(root):
    """One row per (version, direction): version, lambda_geo, anchor_survey, n, fid."""
    frames = []
    for lam, version in sorted(zip(LAMBDA_GEO, VERSIONS)):
        path = os.path.join(root, version, RUN_NAME, "fid.csv")
        if not os.path.exists(path):
            print(f"[WARN] {version} (lambda_geo={lam}): no {RUN_NAME}/fid.csv, skipped")
            continue
        df = pd.read_csv(path)
        df.insert(0, "lambda_geo", lam)
        df.insert(0, "version", version)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No version under {root} has {RUN_NAME}/fid.csv.")
    return pd.concat(frames, ignore_index=True)


def plot_all(df, path):
    """One bar per lambda_geo, one panel per direction, on shared categorical x."""
    # categorical positions: 0.05 would otherwise be squeezed between 0.0 and 0.1
    lambdas = sorted(df["lambda_geo"].unique())
    pos = {lam: i for i, lam in enumerate(lambdas)}
    version_of = dict(zip(df["lambda_geo"], df["version"]))
    # keyed by the lambda value, so a skipped version never repaints the others
    color_of = dict(zip(LAMBDA_GEO, LAMBDA_COLORS))

    surveys = [s for s in SURVEYS if s in set(df["anchor_survey"])]
    fig, axes = plt.subplots(len(surveys), 1, figsize=(8, 3 * len(surveys)),
                             sharex=True, squeeze=False)
    for ax, survey in zip(axes[:, 0], surveys):
        sub = df[df["anchor_survey"] == survey]
        ax.bar([pos[lam] for lam in sub["lambda_geo"]], sub["fid"],
               width=0.8, color=[color_of[lam] for lam in sub["lambda_geo"]])
        ax.set_ylabel("FID")
        ax.set_title(f"{SURVEYS[survey]} (N = {sub['n'].iloc[0]})", fontsize=11)
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
                   help="Folder holding one subfolder per version, each with fid/fid.csv")
    p.add_argument("--out-dir", default="fid_all")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_all(args.root)
    for row in df.itertuples():
        print(f"  {row.version:>3} | lambda_geo {row.lambda_geo:<4g} | "
              f"{row.anchor_survey:<6} target | N {row.n} | FID {row.fid:.4g}")

    summary_path = os.path.join(args.out_dir, "fid_all.csv")
    df.to_csv(summary_path, index=False, float_format="%.6g")
    print(f"Saved {summary_path}")

    plot_all(df, os.path.join(args.out_dir, "fid_all.png"))


if __name__ == "__main__":
    main()
