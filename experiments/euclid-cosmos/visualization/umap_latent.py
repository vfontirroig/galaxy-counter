"""
UMAP visualization of the latent space from the trained Euclid x COSMOS model.

Encodes validation-set images through both encoders and plots:
  - encoder_1 (same-galaxy / physics): Euclid and COSMOS of the SAME galaxies
    should cluster together if the model learned survey-invariant features.
  - encoder_2 (same-instrument): shows instrument-specific structure.

Usage:
    python experiments/euclid-cosmos/visualization/umap_latent.py \
        --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-phase1/best-epoch=21-step=98000.ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs.h5 \
        --out        /n03data/fontirro/plots_model/euclid-cosmos-phase1/umap.png \
        --out-cutouts /n03data/fontirro/plots_model/euclid-cosmos-phase1/umap_cutouts.png \
        --n-samples  5000
"""

import os
import sys
import argparse
from functools import partial
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Subset
import umap

_here = os.path.dirname(__file__)
_experiment_dir = os.path.abspath(os.path.join(_here, ".."))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
sys.path.insert(0, _experiment_dir)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


# Highlighted-pair colours: Paul Tol's colour-blind-safe "muted" palette, minus
# its light blue, so no star shares a hue with the steelblue/darkorange clouds.
PAIR_COLORS = ["#CC6677", "#332288", "#DDCC77", "#117733",
               "#882255", "#44AA99", "#999933", "#AA4499"]


def _percentile_scale(arr):
    """Clip and rescale a 2-D float array to [0, 1] for display."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def _restyle_for_dark(fig, color="white"):
    """Recolour a figure's chrome so it reads on a dark background.

    Only the frame, text and marker outlines change. The data colours — the
    steelblue/darkorange survey clouds and the per-pair star colours — are left
    untouched, so the light and dark files show exactly the same points and stay
    comparable side by side.

    Figure-level texts (the suptitle) are recoloured, but per-axes annotations
    are not: the highlighted-pair numbers are deliberately tinted to match their
    star, and overriding them to white would throw away that pairing.
    """
    for text in fig.texts:                      # suptitle lives here
        text.set_color(color)

    for ax in fig.axes:
        ax.title.set_color(color)
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis="both", colors=color)
        for spine in ax.spines.values():
            spine.set_color(color)

        # The highlighted stars are outlined in black, which vanishes against a
        # dark slide. Only collections that actually have an outline are flipped,
        # so the background scatter (linewidth 0) is left alone.
        for coll in ax.collections:
            if np.any(coll.get_linewidths()):
                coll.set_edgecolor(color)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(color)
            legend.get_frame().set_facecolor("none")
            legend.get_frame().set_edgecolor(color)

    return fig


def _white_axis_labels(fig, color="white"):
    """Recolour only the x/y axis labels, leaving every other colour as drawn.

    For the cutouts grid, where the pair colours on the borders and titles are
    the data and must survive, but the row labels are plain black and would
    otherwise vanish against a dark slide.
    """
    for ax in fig.axes:
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
    return fig


def _save_both(fig, out_path, dpi=150, restyle=True):
    """Save `fig` twice: as it is, then again with a transparent background.

    The second file lands next to the first with a '_transparent' suffix, and is
    written last because the restyle mutates the figure in place.

    `restyle` selects how much of the figure is recoloured for the transparent
    copy: True applies _restyle_for_dark (all chrome to white), False keeps every
    colour exactly as drawn, and a callable is applied to the figure instead —
    for figures that need only part of their chrome whitened.
    """
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")

    base, ext = os.path.splitext(out_path)
    dark_path = f"{base}_transparent{ext}"
    if callable(restyle):
        restyle(fig)
    elif restyle:
        _restyle_for_dark(fig)
    fig.savefig(dark_path, dpi=dpi, bbox_inches="tight", transparent=True)
    print(f"Saved: {dark_path}")
    return dark_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--h5",          required=True)
    p.add_argument("--out",         default="umap.png")
    p.add_argument("--out-cutouts", default=None,
                   help="If given, save a separate figure with the highlighted galaxy cutouts.")
    p.add_argument("--indices",     default=None,
                   help="Optional .npy file of indices (e.g. test_indices.npy). "
                        "If omitted, a random sample is used.")
    p.add_argument("--n-samples",   type=int, default=5000,
                   help="Number of galaxy pairs to encode. Set to -1 to use all pairs (ignored if --indices given)")
    p.add_argument("--n-highlight", type=int, default=8,
                   help="Number of random pairs to highlight on encoder_1 plot")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create the folders the output files go into
    for path in (args.out, args.out_cutouts):
        if path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    # --- Build dataset subset ---
    dataset = EuclidCosmosDataset(args.h5)
    if args.indices is not None:
        indices = np.load(args.indices).tolist()
        print(f"Using {len(indices)} indices from {args.indices}")
    else:
        n = len(dataset) if args.n_samples == -1 else min(args.n_samples, len(dataset))
        indices = np.random.choice(len(dataset), size=n, replace=False).tolist()
        print(f"Using {'all' if args.n_samples == -1 else n} samples ({n} pairs)")

    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, dataset=dataset),
    )

    # --- Encode all images ---
    euclid_emb1_list, cosmos_emb1_list = [], []
    euclid_emb2_list, cosmos_emb2_list = [], []

    print("Encoding images...")
    with torch.no_grad():
        for euclid, cosmos, _, _, _ in loader:
            euclid = euclid.to(device)
            cosmos = cosmos.to(device)
            euclid_emb1_list.append(model.encoder_1(euclid).flatten(1).cpu())
            cosmos_emb1_list.append(model.encoder_1(cosmos).flatten(1).cpu())
            euclid_emb2_list.append(model.encoder_2(euclid).flatten(1).cpu())
            cosmos_emb2_list.append(model.encoder_2(cosmos).flatten(1).cpu())

    euclid_emb1 = torch.cat(euclid_emb1_list).numpy()
    cosmos_emb1 = torch.cat(cosmos_emb1_list).numpy()
    euclid_emb2 = torch.cat(euclid_emb2_list).numpy()
    cosmos_emb2 = torch.cat(cosmos_emb2_list).numpy()
    N = len(euclid_emb1)
    print(f"Encoded {N} pairs. Embedding dim: {euclid_emb1.shape[1]}")

    # --- UMAP ---
    umap_params = dict(n_neighbors=15, min_dist=0.1, n_components=2,
                       metric="euclidean", random_state=args.seed, n_jobs=1)

    print("Computing UMAP for encoder_1 (same-galaxy / physics)...")
    all_emb1  = np.concatenate([euclid_emb1, cosmos_emb1], axis=0)
    umap_emb1 = umap.UMAP(**umap_params).fit_transform(all_emb1)
    euc_u1, cos_u1 = umap_emb1[:N], umap_emb1[N:]

    print("Computing UMAP for encoder_2 (same-instrument)...")
    all_emb2  = np.concatenate([euclid_emb2, cosmos_emb2], axis=0)
    umap_emb2 = umap.UMAP(**umap_params).fit_transform(all_emb2)
    euc_u2, cos_u2 = umap_emb2[:N], umap_emb2[N:]

    # --- Pick random pairs to highlight ---
    rng = np.random.default_rng(args.seed)
    pair_ids   = rng.choice(N, size=min(args.n_highlight, N), replace=False)
    pair_colors = [PAIR_COLORS[k % len(PAIR_COLORS)] for k in range(len(pair_ids))]

    # --- UMAP plot --- encoder 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    kw = dict(s=22, alpha=0.5, rasterized=True)
    ax1.scatter(euc_u1[:, 0], euc_u1[:, 1], c="steelblue",  label="Euclid NIR-H", **kw)
    ax1.scatter(cos_u1[:, 0], cos_u1[:, 1], c="darkorange", label="COSMOS F150W", **kw)

    for k, (pid, color) in enumerate(zip(pair_ids, pair_colors)):
        label = str(k + 1)
        ax1.scatter(euc_u1[pid, 0], euc_u1[pid, 1], s=400, color=color,
                    marker="*", edgecolors="black", linewidths=0.8, zorder=5)
        ax1.scatter(cos_u1[pid, 0], cos_u1[pid, 1], s=400, color=color,
                    marker="*", edgecolors="black", linewidths=0.8, zorder=5)
        for x, y in [(euc_u1[pid, 0], euc_u1[pid, 1]),
                     (cos_u1[pid, 0], cos_u1[pid, 1])]:
            ax1.annotate(label, xy=(x, y), xytext=(4, 4), textcoords="offset points",
                         fontsize=16, color=color, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",  markersize=10, label="Euclid NIR-H"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange", markersize=10, label="COSMOS F150W"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markersize=20,
               markeredgecolor="black", label=f"{len(pair_ids)} highlighted pairs"),
    ]

    ax1.legend(handles=legend_handles, fontsize=15)
    ax1.set_title("encoder_1 — same galaxy (physics)", fontsize=21)
    ax1.set_xlabel("UMAP 1", fontsize=19)
    ax1.set_ylabel("UMAP 2", fontsize=19)

    # --- UMAP plot --- encoder 2

    ax2.scatter(euc_u2[:, 0], euc_u2[:, 1], c="steelblue",  label="Euclid NIR-H", **kw)
    ax2.scatter(cos_u2[:, 0], cos_u2[:, 1], c="darkorange", label="COSMOS F150W", **kw)

    for k, (pid, color) in enumerate(zip(pair_ids, pair_colors)):
        label = str(k + 1)
        ax2.scatter(euc_u2[pid, 0], euc_u2[pid, 1], s=400, color=color,
                    marker="*", edgecolors="black", linewidths=0.8, zorder=5)
        ax2.scatter(cos_u2[pid, 0], cos_u2[pid, 1], s=400, color=color,
                    marker="*", edgecolors="black", linewidths=0.8, zorder=5)
        for x, y in [(euc_u2[pid, 0], euc_u2[pid, 1]),
                     (cos_u2[pid, 0], cos_u2[pid, 1])]:
            ax2.annotate(label, xy=(x, y), xytext=(4, 4), textcoords="offset points",
                         fontsize=16, color=color, fontweight="bold")

    ax2.set_title("encoder_2 — same instrument", fontsize=21)
    ax2.set_xlabel("UMAP 1", fontsize=19)
    ax2.set_ylabel("UMAP 2", fontsize=19)
    ax2.legend(handles=legend_handles, fontsize=15)

    # Poster styling: thicker frame and ticks, larger tick numbers
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(2.5)
        ax.tick_params(axis="both", labelsize=16, width=2.5, length=7)

    fig.suptitle(f"Latent space UMAP  |  N = {N} galaxy pairs", fontsize=21)
    plt.tight_layout()
    _save_both(fig, args.out)
    plt.close(fig)

    # --- Separate cutouts figure ---
    if args.out_cutouts is not None:
        print("Loading cutout images for highlighted pairs...")
        hl_euclid_imgs, hl_cosmos_imgs, hl_ids = [], [], []
        for pid in pair_ids:
            e_img, c_img, meta = subset[pid]
            hl_euclid_imgs.append(_percentile_scale(e_img.squeeze(0).numpy()))
            hl_cosmos_imgs.append(_percentile_scale(c_img.squeeze(0).numpy()))
            hl_ids.append(meta["idx"])

        n_pairs = len(pair_ids)
        fig2, axes = plt.subplots(2, n_pairs, figsize=(2.5 * n_pairs, 5.5))
        if n_pairs == 1:
            axes = axes[:, np.newaxis]

        row_labels = ["Euclid NIR-H", "COSMOS F150W"]
        for k, (pid, color) in enumerate(zip(pair_ids, pair_colors)):
            for row, img in enumerate([hl_euclid_imgs[k], hl_cosmos_imgs[k]]):
                ax = axes[row, k]
                ax.imshow(img, cmap="plasma", origin="lower")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                if row == 0:
                    ax.set_title(f"Pair {k + 1}", color=color, fontsize=18, fontweight="bold")
                    ax.text(0.02, 0.98, f"idx={hl_ids[k]}", fontsize=20,
                            ha="left", va="top", color="magenta",
                            transform=ax.transAxes)

        for row, label in enumerate(row_labels):
            axes[row, 0].set_ylabel(label, fontsize=20)

        #fig2.suptitle("Highlighted galaxy cutouts", fontsize=15)
        plt.tight_layout()
        # Pair colours stay as drawn — the per-panel border and title colour is
        # what ties a cutout back to its star in the UMAP panel. Only the two
        # black row labels are whitened for the transparent copy.
        _save_both(fig2, args.out_cutouts, restyle=_white_axis_labels)
        plt.close(fig2)


if __name__ == "__main__":
    main()
