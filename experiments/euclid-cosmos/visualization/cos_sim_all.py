"""
Histogram of the Euclid-vs-COSMOS cosine similarity of known pairs, for every
lambda_geo in [0, 1] with step 0.1 (one trained version per value).

For each version it reads the embeddings written by export_embeddings.py
(<ckpt-root>/<version>/embeddings/embeddings.npz) and that version's own test
split, and computes the matched cosine exactly as cosine_similarity.py does, so
the values agree with its per_galaxy.csv. Versions whose embeddings have not been
exported yet are skipped with a warning.

All histograms are overlaid on one axis, one colour-blind-safe colour (and,
past the eighth, a dashed line) per lambda_geo.

Usage:
    python experiments/euclid-cosmos/visualization/cos_sim_all.py \
        --ckpt-root /n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1 \
        --out-dir   /n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/cos_sim_all
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cosine_similarity import ENCODERS, REPS, CENTERINGS, analyse, load_splits

# cosine_similarity sets gray axes on import; this plot keeps them black
plt.rcParams.update({
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

#CONFIG

VERSIONS = ['v14','v5','v6','v4','v7','v8','v9','v10','v11','v12','v13']
LAMBDA_GEO = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Okabe & Ito (2008) colour-blind-safe palette. Among the 8-colour palettes
# checked it separates best under simulated protan/deutan/tritan vision when
# every pair must be distinguishable, as overlaid lines require. The yellow is
# faint on white, but any darker yellow becomes indistinguishable from the
# orange for colour-blind readers, so it is kept and lines are drawn 2 px wide.
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73",
             "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def matched_cosine(version_dir, encoder, rep, centering, seed):
    """Matched cosine for every test pair of one trained version."""
    data = np.load(os.path.join(version_dir, "embeddings", "embeddings.npz"))
    euclid = data[f"euclid_emb{encoder}"].astype(np.float64)
    cosmos = data[f"cosmos_emb{encoder}"].astype(np.float64)
    train, test = load_splits(version_dir, len(euclid))
    _, per_galaxy = analyse(euclid, cosmos, train, test, rep, centering,
                            np.random.default_rng(seed))
    return per_galaxy["cos_matched"]


def lambda_styles():
    """(colour, linestyle) per lambda_geo, fixed by the lambda value itself so
    a skipped version never repaints the others.

    Eight colours cannot be told apart beyond eight series, so lambdas past the
    eighth reuse the palette from the start with a dashed line: every lambda
    keeps a unique colour + style pair."""
    styles = {}
    for i, lam in enumerate(sorted(LAMBDA_GEO)):
        color = OKABE_ITO[i % len(OKABE_ITO)]
        styles[lam] = (color, "-" if i < len(OKABE_ITO) else "--")
    return styles


def plot_all(results, encoder, rep, centering, path):
    """Every lambda_geo overlaid on one axis, on shared bins."""
    styles = lambda_styles()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-1, 1, 81)
    for lam, _, values in results:
        color, linestyle = styles[lam]
        ax.hist(values, bins=bins, histtype="step", color=color,
                linestyle=linestyle, linewidth=2)
    ax.set_xlabel("Cosine similarity values")
    ax.set_ylabel("Number of pairs")
    #ax.set_title(f"Known pairs, test split · encoder {encoder} ({rep} · {centering})",
    #             fontsize=12)
    # hist() legend entries are patches; lines show the dash pattern too
    handles = [Line2D([], [], color=styles[lam][0], linestyle=styles[lam][1],
                      linewidth=2, label=f"λ = {lam:.1f} ({version})")
               for lam, version, _ in results]
    ax.legend(handles=handles, title="λ_geo", fontsize=9,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True,
                   help="Folder holding one subfolder per version (v4 ... v14)")
    p.add_argument("--out-dir",   default="cos_sim_all")
    p.add_argument("--encoder",   choices=ENCODERS, default="1")
    p.add_argument("--rep",       choices=REPS, default="flat")
    p.add_argument("--centering", choices=CENTERINGS, default="global")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results, rows = [], []
    for lam, version in sorted(zip(LAMBDA_GEO, VERSIONS)):
        version_dir = os.path.join(args.ckpt_root, version)
        if not os.path.exists(os.path.join(version_dir, "embeddings", "embeddings.npz")):
            print(f"[WARN] {version} (lambda_geo={lam}): no embeddings/embeddings.npz, skipped")
            continue
        values = matched_cosine(version_dir, args.encoder, args.rep, args.centering,
                                args.seed)
        results.append((lam, version, values))
        rows.append({"version": version, "lambda_geo": lam, "n_test": len(values),
                     "mean": values.mean(), "median": np.median(values),
                     "std": values.std()})
        print(f"  {version:>3} | lambda_geo {lam:.1f} | {len(values)} pairs | "
              f"mean {values.mean():+.3f}  median {np.median(values):+.3f}")

    if not results:
        raise SystemExit(f"No version under {args.ckpt_root} has exported embeddings.")

    stem = f"cosine_hist_lambda_enc{args.encoder}_{args.rep}_{args.centering}"
    summary_path = os.path.join(args.out_dir, f"{stem}.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Saved {summary_path}")

    plot_all(results, args.encoder, args.rep, args.centering,
             os.path.join(args.out_dir, f"{stem}.png"))


if __name__ == "__main__":
    main()
