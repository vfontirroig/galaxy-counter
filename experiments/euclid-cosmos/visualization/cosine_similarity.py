"""
Cosine similarity between the Euclid and COSMOS embeddings of the same galaxy.

Reads the full-dimensional embeddings written by export_embeddings.py (row i is
h5 index i, Euclid and COSMOS of the same pair) and, on the test split only,
asks whether a galaxy's Euclid embedding is closer to its own COSMOS embedding
than to other galaxies':

  - matched cosine   cos(euclid[i], cosmos[i])
  - random baseline  cos(euclid[i], cosmos[j]), j != i drawn at random
  - retrieval rank   where cosmos[i] lands when every test COSMOS embedding is
                     ranked by cosine to euclid[i] (0 = it is the closest), and
                     the same in the COSMOS -> Euclid direction

Each metric is computed for every combination of
  encoder    1 (trained to align surveys via lambda_geometric) and 2 (reference)
  rep        flat   — the (seq_len * embed_dim) vector the Sinkhorn loss sees
             pooled — tokens averaged to (embed_dim,), insensitive to position
  centering  raw        — no mean subtracted
             global     — one mean over both surveys (train split) subtracted
             per_survey — each survey's own train mean subtracted; this removes
                          the survey offset itself, so read it separately

Usage:
    python experiments/euclid-cosmos/visualization/cosine_similarity.py \
        --embeddings /n03data/fontirro/.../embeddings/embeddings.npz \
        --ckpt-dir   /n03data/fontirro/.../test-1-phase1/v3 \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5 \
        --out-dir    /n03data/fontirro/.../cosine_similarity
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMBED_DIM = 16  # cross_attention_dim in train.py

ENCODERS = ["1", "2"]
REPS = ["flat", "pooled"]
CENTERINGS = ["raw", "global", "per_survey"]

# Reference dataviz palette: slots 1-2 for the two surveys, gray for the baseline
EUCLID_COLOR = "#2a78d6"
COSMOS_COLOR = "#eb6834"
BASELINE_COLOR = "#898781"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": BASELINE_COLOR,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 2,
})


def load_splits(ckpt_dir, n):
    """Train/test pair indices from the files train.py saved. Train is
    everything not in test or val, and is only used to estimate means."""
    test = np.load(os.path.join(ckpt_dir, "test_indices.npy"))
    val = np.load(os.path.join(ckpt_dir, "val_indices.npy"))
    if max(test.max(), val.max()) >= n:
        raise ValueError(
            f"Split indices go up to {max(test.max(), val.max())} but the embeddings "
            f"have only {n} rows — they were exported from a different h5, or with "
            f"--n-samples. Re-export every pair from the h5 used for training.")
    train = np.setdiff1d(np.arange(n), np.concatenate([test, val]))
    return train, test


def represent(x, rep):
    if rep == "flat":
        return x
    return x.reshape(len(x), -1, EMBED_DIM).mean(axis=1)  # (N, embed_dim)


def unit(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def retrieval_ranks(queries, keys, chunk=2048):
    """For each query i, how many keys are more similar than keys[i].
    Both inputs are unit vectors; 0 means the true counterpart ranks first."""
    ranks = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), chunk):
        sims = queries[start:start + chunk] @ keys.T  # (chunk, N)
        rows = np.arange(sims.shape[0])
        matched = sims[rows, start + rows]
        ranks[start:start + chunk] = (sims > matched[:, None]).sum(axis=1)
    return ranks


def auc(matched, random_pairs):
    """Probability that a matched pair has higher cosine than a random pair."""
    below = np.searchsorted(np.sort(random_pairs), matched, side="left")
    return below.sum() / (len(matched) * len(random_pairs))


def analyse(euclid, cosmos, train, test, rep, centering, rng):
    euc = represent(euclid, rep)
    cos = represent(cosmos, rep)

    if centering == "global":
        mean = np.concatenate([euc[train], cos[train]]).mean(axis=0)
        euc, cos = euc - mean, cos - mean
    elif centering == "per_survey":
        euc, cos = euc - euc[train].mean(axis=0), cos - cos[train].mean(axis=0)

    euc_t, cos_t = euc[test], cos[test]
    norms_euc = np.linalg.norm(euc_t, axis=1)
    norms_cos = np.linalg.norm(cos_t, axis=1)
    euc_t, cos_t = unit(euc_t), unit(cos_t)

    n = len(test)
    matched = (euc_t * cos_t).sum(axis=1)
    partner = (np.arange(n) + rng.integers(1, n, size=n)) % n  # never itself
    random_pairs = (euc_t * cos_t[partner]).sum(axis=1)
    rank_e2c = retrieval_ranks(euc_t, cos_t)
    rank_c2e = retrieval_ranks(cos_t, euc_t)

    summary = {
        "matched_mean": matched.mean(),
        "matched_median": np.median(matched),
        "random_mean": random_pairs.mean(),
        "random_median": np.median(random_pairs),
        "gap_mean": matched.mean() - random_pairs.mean(),
        "auc_matched_vs_random": auc(matched, random_pairs),
        "top1_euclid_to_cosmos": (rank_e2c == 0).mean(),
        "top10_euclid_to_cosmos": (rank_e2c < 10).mean(),
        "median_rank_euclid_to_cosmos": np.median(rank_e2c),
        "top1_cosmos_to_euclid": (rank_c2e == 0).mean(),
        "top10_cosmos_to_euclid": (rank_c2e < 10).mean(),
        "median_rank_cosmos_to_euclid": np.median(rank_c2e),
        "norm_euclid_mean": norms_euc.mean(),
        "norm_euclid_std": norms_euc.std(),
        "norm_cosmos_mean": norms_cos.mean(),
        "norm_cosmos_std": norms_cos.std(),
    }
    per_galaxy = {
        "cos_matched": matched,
        "cos_random": random_pairs,
        "rank_euclid_to_cosmos": rank_e2c,
        "rank_cosmos_to_euclid": rank_c2e,
    }
    curves = {"matched": matched, "random": random_pairs,
              "rank_e2c": rank_e2c, "rank_c2e": rank_c2e,
              "norm_euc": norms_euc, "norm_cos": norms_cos}
    return summary, per_galaxy, curves


def plot_histograms(curves, encoder, path):
    fig, axes = plt.subplots(len(CENTERINGS), len(REPS), figsize=(9, 9),
                             sharex=True, squeeze=False)
    bins = np.linspace(-1, 1, 81)
    for r, centering in enumerate(CENTERINGS):
        for c, rep in enumerate(REPS):
            ax = axes[r, c]
            cv = curves[(encoder, rep, centering)]
            for key, label, color in [("random", "Random pairs", BASELINE_COLOR),
                                      ("matched", "Same galaxy", EUCLID_COLOR)]:
                ax.hist(cv[key], bins=bins, density=True, histtype="stepfilled",
                        color=color, alpha=0.25)
                ax.hist(cv[key], bins=bins, density=True, histtype="step",
                        color=color, linewidth=2, label=label)
            ax.set_title(f"{rep} · {centering}", fontsize=11)
            if r == len(CENTERINGS) - 1:
                ax.set_xlabel("Cosine similarity (Euclid vs COSMOS)")
            if c == 0:
                ax.set_ylabel("Density")
    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Encoder {encoder}: same-galaxy vs random-pair cosine (test split)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_ranks(curves, encoder, n_test, path):
    fig, axes = plt.subplots(len(CENTERINGS), len(REPS), figsize=(9, 9),
                             sharex=True, sharey=True, squeeze=False)
    for r, centering in enumerate(CENTERINGS):
        for c, rep in enumerate(REPS):
            ax = axes[r, c]
            cv = curves[(encoder, rep, centering)]
            for key, label, color in [("rank_e2c", "Euclid → COSMOS", EUCLID_COLOR),
                                      ("rank_c2e", "COSMOS → Euclid", COSMOS_COLOR)]:
                k = np.sort(cv[key]) + 1  # top-k at which each galaxy is retrieved
                ax.plot(k, np.arange(1, len(k) + 1) / len(k), color=color, label=label)
            # chance: the counterpart is uniformly placed among n_test candidates
            k_chance = np.arange(1, n_test + 1)
            ax.plot(k_chance, k_chance / n_test, color=BASELINE_COLOR,
                    linewidth=1, linestyle="--", label="Chance")
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_title(f"{rep} · {centering}", fontsize=11)
            if r == len(CENTERINGS) - 1:
                ax.set_xlabel("k")
            if c == 0:
                ax.set_ylabel("Top-k retrieval accuracy")
    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Encoder {encoder}: cross-survey retrieval by cosine (test split, "
                 f"{n_test} candidates)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_norms(curves, encoder, path):
    """Raw (uncentered) embedding norms per survey: cosine discards these, so a
    systematic gap here means cosine and Euclidean distance can disagree."""
    fig, axes = plt.subplots(1, len(REPS), figsize=(9, 3.5), squeeze=False)
    for c, rep in enumerate(REPS):
        ax = axes[0, c]
        cv = curves[(encoder, rep, "raw")]
        both = np.concatenate([cv["norm_euc"], cv["norm_cos"]])
        bins = np.linspace(both.min(), both.max(), 61)
        for key, label, color in [("norm_euc", "Euclid", EUCLID_COLOR),
                                  ("norm_cos", "COSMOS", COSMOS_COLOR)]:
            ax.hist(cv[key], bins=bins, density=True, histtype="step",
                    color=color, linewidth=2, label=label)
        ax.set_title(rep, fontsize=11)
        ax.set_xlabel("Embedding L2 norm")
        if c == 0:
            ax.set_ylabel("Density")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Encoder {encoder}: embedding norms (test split, uncentered)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True, help=".npz from export_embeddings.py")
    p.add_argument("--ckpt-dir",   required=True,
                   help="Training CKPT_DIR holding test_indices.npy / val_indices.npy")
    p.add_argument("--h5",         default=None,
                   help="Optional: the h5 the embeddings were exported from. Checks "
                        "that num_pairs matches the number of embedding rows.")
    p.add_argument("--out-dir",    default="cosine_similarity")
    p.add_argument("--seed",       type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.embeddings)
    n = len(data["euclid_emb1"])
    print(f"Loaded {n} pairs from {args.embeddings}")

    if args.h5 is not None:
        with h5py.File(args.h5, "r") as f:
            num_pairs = int(f.attrs["num_pairs"])
        if num_pairs != n:
            raise ValueError(f"{args.h5} has {num_pairs} pairs but the embeddings have "
                             f"{n} rows — they do not come from this h5.")

    train, test = load_splits(args.ckpt_dir, n)
    print(f"Split: {len(train)} train (for means) / {len(test)} test (for metrics)")

    rng = np.random.default_rng(args.seed)
    rows = []
    per_galaxy = {"dataset_idx": test}
    curves = {}
    for encoder in ENCODERS:
        euclid = data[f"euclid_emb{encoder}"].astype(np.float64)
        cosmos = data[f"cosmos_emb{encoder}"].astype(np.float64)
        for rep in REPS:
            for centering in CENTERINGS:
                summary, galaxy_cols, cv = analyse(euclid, cosmos, train, test,
                                                   rep, centering, rng)
                rows.append({"encoder": encoder, "rep": rep, "centering": centering,
                             **summary})
                for col, values in galaxy_cols.items():
                    per_galaxy[f"enc{encoder}_{rep}_{centering}_{col}"] = values
                curves[(encoder, rep, centering)] = cv
                print(f"  encoder {encoder} | {rep:6s} | {centering:10s} | "
                      f"matched {summary['matched_mean']:+.3f}  "
                      f"random {summary['random_mean']:+.3f}  "
                      f"AUC {summary['auc_matched_vs_random']:.3f}  "
                      f"top-1 E→C {summary['top1_euclid_to_cosmos']:.3f}")

    summary_path = os.path.join(args.out_dir, "summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Saved {summary_path}")

    # One row per test galaxy, keyed by dataset_idx so it joins to catalog/features
    galaxy_path = os.path.join(args.out_dir, "per_galaxy.csv")
    pd.DataFrame(per_galaxy).to_csv(galaxy_path, index=False, float_format="%.6f")
    print(f"Saved {galaxy_path}")

    for encoder in ENCODERS:
        plot_histograms(curves, encoder,
                        os.path.join(args.out_dir, f"cosine_hist_encoder{encoder}.png"))
        plot_ranks(curves, encoder, len(test),
                   os.path.join(args.out_dir, f"retrieval_encoder{encoder}.png"))
        plot_norms(curves, encoder,
                   os.path.join(args.out_dir, f"norms_encoder{encoder}.png"))


if __name__ == "__main__":
    main()
