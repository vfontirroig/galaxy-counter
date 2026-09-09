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


# Blobs expected in each encoder's UMAP: the Euclid blob plus the two COSMOS
# blobs. Both encoders show this same structure, so it is fixed here rather than
# exposed as a flag.
N_BLOBS = 3


def _percentile_scale(arr):
    """Clip and rescale a 2-D float array to [0, 1] for display."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)


def _regions_from_graph(reducer, min_blobs=2, min_size=25):
    """Region label per point, taken from UMAP's own manifold graph.

    reducer.graph_ is the fuzzy simplicial set UMAP actually embedded. Two points
    sit in the same connected component of it iff UMAP saw them as part of one
    continuous manifold, so its components ARE the separated blobs — exactly,
    with no cluster count to choose and no centroid heuristics.

    UMAP itself returns no cluster labels (fit_transform gives coordinates only),
    so this is the closest thing to a native answer. It only helps when the graph
    is in fact disconnected, which depends on n_neighbors: more neighbours glue
    components together.

    Returns None unless it finds at least `min_blobs` regions, so the caller can
    fall back to KMeans. That floor matters because the graph tends to UNDER-
    segment: two visually separate blobs joined by a thin trail of points are one
    component, and accepting that would silently merge them.

    Components smaller than min_size are treated as stragglers and merged into
    label -1 rather than becoming their own "blob".
    """
    from scipy.sparse.csgraph import connected_components

    n_comp, raw = connected_components(reducer.graph_, directed=False)
    print(f"  UMAP graph has {n_comp} connected component(s)")
    if n_comp < min_blobs:
        return None

    keep = [c for c in range(n_comp) if (raw == c).sum() >= min_size]
    if len(keep) < min_blobs:
        print(f"  only {len(keep)} component(s) above min_size={min_size}, "
              f"need {min_blobs}")
        return None

    labels = np.full(len(raw), -1, dtype=int)
    for new, c in enumerate(keep):
        labels[raw == c] = new
    n_stray = int((labels == -1).sum())
    if n_stray:
        print(f"  {n_stray} point(s) in components below min_size -> label -1")
    return labels


def _assign_groups(coords, n_groups, seed):
    """Fallback when the UMAP graph is fully connected: KMeans on the 2-D coords.

    Less principled than _regions_from_graph — it will happily force `n_groups`
    blobs whether or not that many exist — but it always returns something, and
    unlike a hand-picked coordinate cut (e.g. "UMAP 1 < 5") it does not need
    re-eyeballing when the embedding moves.
    """
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_groups, n_init=10,
                  random_state=seed).fit_predict(coords)


def _renumber_left_to_right(labels, coords):
    """Relabel blobs 0..k-1 by centroid UMAP 1, leaving label -1 untouched.

    Both labelling routes hand back arbitrary numbering, so this makes "blob 0"
    mean "leftmost in the figure" either way, and keeps it stable across runs.
    """
    present = sorted({int(g) for g in labels} - {-1})
    order = sorted(present, key=lambda g: coords[labels == g, 0].mean())
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[int(g)] if g >= 0 else -1 for g in labels])


def _find_blobs(reducer, emb, N, seed, name):
    """Blob label per point for one encoder's UMAP: graph first, then KMeans.

    Both encoders are treated identically and expected to hold N_BLOBS regions.

    Returns (groups, n_groups). `groups` covers all 2N points — rows [0, N) are
    the Euclid views, rows [N, 2N) the COSMOS views of those same galaxies.
    """
    print(f"Finding {name} regions from UMAP's own manifold graph...")
    groups = _regions_from_graph(reducer, min_blobs=N_BLOBS)
    if groups is None:
        print(f"  graph found fewer than {N_BLOBS} regions; falling back to "
              f"KMeans k={N_BLOBS} on the 2-D coordinates")
        groups = _assign_groups(emb, N_BLOBS, seed)
    groups = _renumber_left_to_right(groups, emb)
    n_groups = int(groups.max()) + 1

    print(f"{name} UMAP split into {n_groups} blobs "
          f"(numbered left-to-right by UMAP 1):")
    for g in range(n_groups):
        n_euc = int((groups[:N] == g).sum())
        n_cos = int((groups[N:] == g).sum())
        owner = "Euclid" if n_euc > n_cos else "COSMOS"
        print(f"  blob {g}: {n_euc:5d} Euclid + {n_cos:5d} COSMOS -> {owner}-dominated")
    return groups, n_groups


def _write_group_csv(path, indices, groups, N, pts):
    """Write the per-galaxy blob assignment for one encoder.

    Both survey columns go in so crossovers are findable: a galaxy whose two
    views landed in different blobs has euclid_group != cosmos_group, and
    same_blob==1 means this encoder placed them together. A group of -1 means the
    point sat in a tiny off-manifold component, and two -1s do not count as
    "same" — they are unplaced, not together.
    """
    euc, cos = groups[:N], groups[N:]
    with open(path, "w") as fh:
        fh.write("dataset_idx,euclid_group,cosmos_group,same_blob,umap_1,umap_2\n")
        for i in range(N):
            same = int(euc[i] >= 0 and euc[i] == cos[i])
            fh.write(f"{indices[i]},{euc[i]},{cos[i]},{same},"
                     f"{pts[i, 0]:.6f},{pts[i, 1]:.6f}\n")
    n_together = int(((euc == cos) & (euc >= 0)).sum())
    print(f"Saved group assignment: {path}  ({N} galaxies, "
          f"{n_together} with both views in the same blob)")


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
    p.add_argument("--group-survey", choices=["cosmos", "euclid"], default="cosmos",
                   help="Which survey's galaxies to tabulate and show cutouts for "
                        "(default: cosmos). The blobs themselves are always found "
                        "using BOTH surveys' points, so a COSMOS galaxy sitting "
                        "inside Euclid's blob is labelled as being in that blob.")
    p.add_argument("--per-group", type=int, default=8,
                   help="Cutouts to show per group, most central first (default: 8)")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    reducer1  = umap.UMAP(**umap_params)   # kept: reducer1.graph_ defines the regions
    umap_emb1 = reducer1.fit_transform(all_emb1)
    euc_u1, cos_u1 = umap_emb1[:N], umap_emb1[N:]

    print("Computing UMAP for encoder_2 (same-instrument)...")
    all_emb2  = np.concatenate([euclid_emb2, cosmos_emb2], axis=0)
    reducer2  = umap.UMAP(**umap_params)   # kept: reducer2.graph_ defines the regions
    umap_emb2 = reducer2.fit_transform(all_emb2)
    euc_u2, cos_u2 = umap_emb2[:N], umap_emb2[N:]

    # --- Find each encoder's blobs (also used to label the main plot) ---
    # The two encoders are grouped independently: each has its own embedding, so
    # blob 1 of encoder_1 has nothing to do with blob 1 of encoder_2.
    groups1, n_groups1 = _find_blobs(reducer1, umap_emb1, N, args.seed, "encoder_1")
    groups2, n_groups2 = _find_blobs(reducer2, umap_emb2, N, args.seed, "encoder_2")

    # --- Pick random pairs to highlight ---
    rng = np.random.default_rng(args.seed)
    pair_ids   = rng.choice(N, size=min(args.n_highlight, N), replace=False)
    pair_colors = plt.cm.tab10(np.linspace(0, 1, len(pair_ids)))

    # --- UMAP plot --- encoder 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    kw = dict(s=22, alpha=0.5, rasterized=True)
    ax1.scatter(euc_u1[:, 0], euc_u1[:, 1], c="steelblue",  label="Euclid VIS", **kw)
    ax1.scatter(cos_u1[:, 0], cos_u1[:, 1], c="darkorange", label="COSMOS F150W", **kw)

    for k, (pid, color) in enumerate(zip(pair_ids, pair_colors)):
        label = str(k + 1)
        ax1.scatter(euc_u1[pid, 0], euc_u1[pid, 1], s=144, color=color,
                    marker="*", edgecolors="black", linewidths=0.4, zorder=5)
        ax1.scatter(cos_u1[pid, 0], cos_u1[pid, 1], s=144, color=color,
                    marker="*", edgecolors="black", linewidths=0.4, zorder=5)
        for x, y in [(euc_u1[pid, 0], euc_u1[pid, 1]),
                     (cos_u1[pid, 0], cos_u1[pid, 1])]:
            ax1.annotate(label, xy=(x, y), xytext=(4, 4), textcoords="offset points",
                         fontsize=10, color=color, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",  markersize=10, label="Euclid VIS"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="darkorange", markersize=10, label="COSMOS F150W"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markersize=12,
               markeredgecolor="black", label=f"{len(pair_ids)} highlighted pairs"),
    ]
    # Blob labels. Each encoder is grouped in its own embedding, so ax1 gets
    # encoder_1's numbers and ax2 encoder_2's — they are unrelated numberings.
    # Placed just above each blob's top edge (median x, max y) rather than at the
    # centroid, so they sit beside the points instead of on top of them.
    # Hand-placed positions in data coordinates, for blobs whose automatic label
    # crowds the axes or the legend. These are tied to THIS embedding, so clear
    # or revisit them if the UMAP layout changes (new checkpoint, rebuilt HDF5,
    # different seed). Blobs not listed fall back to the automatic placement.
    # blob_label_pos = {0: (-6.5, 12.0)}

    # for g in range(n_groups1):
    #     blob = umap_emb1[groups1 == g]
    #     if g in blob_label_pos:
    #         xy, offset = blob_label_pos[g], (0, 0)
    #     else:
    #         xy, offset = (np.median(blob[:, 0]), blob[:, 1].max()), (0, 8)
    #     ax1.annotate(str(g), xy=xy, xytext=offset, textcoords="offset points",
    #                  ha="center", va="bottom",
    #                  fontsize=15, color="black", zorder=7)

    ax1.legend(handles=legend_handles, fontsize=12)
    ax1.set_title("encoder_1 — same galaxy (physics)", fontsize=18)
    ax1.set_xlabel("UMAP 1", fontsize=15)
    ax1.set_ylabel("UMAP 2", fontsize=15)

    # --- UMAP plot --- encoder 2

    ax2.scatter(euc_u2[:, 0], euc_u2[:, 1], c="steelblue",  label="Euclid VIS", **kw)
    ax2.scatter(cos_u2[:, 0], cos_u2[:, 1], c="darkorange", label="COSMOS F150W", **kw)

    for k, (pid, color) in enumerate(zip(pair_ids, pair_colors)):
        label = str(k + 1)
        ax2.scatter(euc_u2[pid, 0], euc_u2[pid, 1], s=144, color=color,
                    marker="*", edgecolors="black", linewidths=0.4, zorder=5)
        ax2.scatter(cos_u2[pid, 0], cos_u2[pid, 1], s=144, color=color,
                    marker="*", edgecolors="black", linewidths=0.4, zorder=5)
        for x, y in [(euc_u2[pid, 0], euc_u2[pid, 1]),
                     (cos_u2[pid, 0], cos_u2[pid, 1])]:
            ax2.annotate(label, xy=(x, y), xytext=(4, 4), textcoords="offset points",
                         fontsize=10, color=color, fontweight="bold")

    # for g in range(n_groups2):
    #     blob = umap_emb2[groups2 == g]
    #     ax2.annotate(str(g), xy=(np.median(blob[:, 0]), blob[:, 1].max()),
    #                  xytext=(0, 8), textcoords="offset points",
    #                  ha="center", va="bottom",
    #                  fontsize=15, color="black", zorder=7)

    ax2.set_title("encoder_2 — same instrument", fontsize=18)
    ax2.set_xlabel("UMAP 1", fontsize=15)
    ax2.set_ylabel("UMAP 2", fontsize=15)
    ax2.legend(handles=legend_handles, fontsize=12)

    fig.suptitle(f"Latent space UMAP  |  N = {N} galaxy pairs", fontsize=15)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")

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

        row_labels = ["Euclid VIS", "COSMOS F150W"]
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
        plt.savefig(args.out_cutouts, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved cutouts: {args.out_cutouts}")

    # --- Assign every galaxy to the blob its point lands in, per encoder ---
    # Always written, next to --out. Blobs were already found above (they label
    # the main plot). They come from ALL 2N points, so they describe regions of
    # the shared space rather than one survey's clumps, and each galaxy gets the
    # blob its own point fell into — which is how a COSMOS galaxy can sit inside
    # Euclid's blob.
    #
    # euc_u1 and cos_u1 share row order, so row i of either is galaxy indices[i]
    n_groups = n_groups1
    euc_groups, cos_groups = groups1[:N], groups1[N:]
    groups = cos_groups if args.group_survey == "cosmos" else euc_groups
    pts = cos_u1 if args.group_survey == "cosmos" else euc_u1
    sizes = [int((groups == g).sum()) for g in range(n_groups)]

    out_dir = os.path.dirname(os.path.abspath(args.out))

    # One table per encoder. The blob numbers are per-encoder and unrelated
    # between the two files, so they are kept separate rather than joined —
    # join them on dataset_idx to compare.
    _write_group_csv(os.path.join(out_dir, "umap_groups_encoder_1.csv"),
                     indices, groups1, N, pts)
    _write_group_csv(os.path.join(out_dir, "umap_groups_encoder_2.csv"),
                     indices, groups2, N,
                     cos_u2 if args.group_survey == "cosmos" else euc_u2)

    # cutouts of the most central galaxies in each encoder_1 blob
    grid_path = os.path.join(out_dir, "umap_groups_encoder_1_cutouts.png")
    fig3, axes3 = plt.subplots(n_groups, args.per_group,
                               figsize=(2.4 * args.per_group, 2.7 * n_groups),
                               squeeze=False)
    for g in range(n_groups):
        members = np.flatnonzero(groups == g)
        if len(members):
            centroid = pts[members].mean(axis=0)
            central = members[np.argsort(
                np.linalg.norm(pts[members] - centroid, axis=1))]
        else:
            central = members  # no galaxies of this survey here; row stays blank
        for c in range(args.per_group):
            ax = axes3[g, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(central):
                ax.axis("off")
                continue
            pos = int(central[c])
            e_img, c_img, meta = subset[pos]
            img = c_img if args.group_survey == "cosmos" else e_img
            ax.imshow(_percentile_scale(img.squeeze(0).numpy()),
                      cmap="plasma", origin="lower")
            ax.set_title(f"idx={meta['idx']}", fontsize=10)
        n_euc, n_cos = int((euc_groups == g).sum()), int((cos_groups == g).sum())
        owner = "Euclid" if n_euc > n_cos else "COSMOS"
        axes3[g, 0].set_ylabel(f"blob {g} ({owner})\n{sizes[g]} {args.group_survey}",
                               fontsize=13, fontweight="bold")
    fig3.suptitle(f"encoder_1 blobs — {args.per_group} most central "
                  f"{args.group_survey} galaxies in each", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved group cutouts: {grid_path}")


if __name__ == "__main__":
    main()
