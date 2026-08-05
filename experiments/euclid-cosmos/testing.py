"""
Test the trained Euclid x COSMOS flow-matching model on the held-out test set.

Loads the test indices saved by train.py, generates images, and reports MSE.
Figure columns depend on direction:
  cosmos-to-euclid:
      [COSMOS input | Generated Euclid | Real Euclid | Residual]
  euclid-to-cosmos:
      [Euclid input | Generated COSMOS | Real COSMOS | Residual]

Usage:
    python experiments/euclid-cosmos/testing.py \
        --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-vis-f150w/test-4-phase1/best-epoch=00-step=100000.ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs_vis_f150w.h5 \
        --indices    /n03data/fontirro/checkpoints/euclid-cosmos-vis-f150w/test-4-phase1/test_indices.npy \
        --out        /n03data/fontirro/checkpoints/euclid-cosmos-vis-f150w/test-4-phase1/test_results.png \
        --direction  cosmos-to-euclid
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
from torch.utils.data import DataLoader, Subset

_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


def show_image(ax, img_tensor, title=None):
    img = img_tensor.squeeze().cpu().float().numpy()
    ax.imshow(img, cmap="plasma")
    if title:
        ax.set_title(title, fontsize=14)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5",         required=True)
    p.add_argument("--indices",    required=True, help="test_indices.npy saved by train.py")
    p.add_argument("--out",        default="test_results.png")
    p.add_argument("--direction",  default="cosmos-to-euclid",
                   choices=["cosmos-to-euclid", "euclid-to-cosmos"],
                   help="cosmos-to-euclid: COSMOS input, generate Euclid"
                        "euclid-to-cosmos: Euclid input, generate COSMOS")
    p.add_argument("--n-plot",     type=int, default=8,   help="Galaxy rows to show in figure")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-steps",  type=int, default=100, help="ODE integration steps")
    p.add_argument("--num-workers",type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  WARNING: no GPU found, running on CPU (will be slow)")

    print(f"Direction: {args.direction}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    # --- Build test dataset from saved indices ---
    test_indices = np.load(args.indices)
    print(f"Test set: {len(test_indices)} galaxies")
    dataset     = EuclidCosmosDataset(args.h5)
    test_subset = Subset(dataset, test_indices.tolist())
    loader      = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, dataset=dataset),
    )

    # --- Build a same-instrument neighbor pool from the whole test set ---
    # Sampling within-batch (as train.py's collate_fn does) breaks when a
    # batch has only one row (e.g. the last, leftover batch): there'd be no
    # "different" galaxy to pick, and the anchor would leak into its own
    # conditioning. Sampling from the full test set avoids that entirely.
    print("Building same-instrument neighbor pool from the full test set...")
    pool_imgs = []
    for idx in test_indices.tolist():
        euc, cos, _ = dataset[idx]
        pool_imgs.append(euc if args.direction == "cosmos-to-euclid" else cos)
    sameins_pool = torch.stack(pool_imgs) #images of the same instrument as the anchor.
    n_pool = sameins_pool.shape[0] #number of same-instrument galaxies.

    # --- Run inference over the full test set ---
    all_mse = []
    plot_input, plot_generated, plot_target, plot_metadata = [], [], [], []

    print("Running inference...")
    offset = 0
    with torch.no_grad():
        for batch_idx, (euclid, cosmos, _, masks, metadata) in enumerate(loader):
            euclid = euclid.to(device)
            cosmos = cosmos.to(device)
            masks  = masks.to(device)
            B      = euclid.shape[0] 

            if args.direction == "cosmos-to-euclid":
                anchor, cond = euclid, cosmos
            else:
                anchor, cond = cosmos, euclid

            # sameins: a different galaxy of the same instrument as the
            # anchor (what's being generated), drawn from the full test
            # set — mirrors train.py's collate_fn, minus the batch-size
            # dependency.
            if n_pool > 1:
                row_pos    = torch.arange(offset, offset + B)
                rand_local = torch.randint(0, n_pool - 1, (B,))
                rand_idx   = rand_local + (rand_local >= row_pos)
                sameins    = sameins_pool[rand_idx].unsqueeze(1).to(device)
            else:
                sameins = anchor.unsqueeze(1)
            offset += B

            generated = model.sample(
                cond_image_samegal=cond,
                cond_image_sameins=sameins,
                masks=masks,
                num_steps=args.num_steps,
            )

            mse = ((generated - anchor) ** 2).mean(dim=(1, 2, 3))
            all_mse.append(mse.cpu())

            if batch_idx == 0:
                plot_input     = cond.cpu()
                plot_generated = generated.cpu()
                plot_target    = anchor.cpu()
                plot_metadata  = metadata

    all_mse = torch.cat(all_mse)
    print(f"\n=== Test Results ({args.direction}) ===")
    print(f"N test galaxies : {len(all_mse)}")
    print(f"Mean MSE        : {all_mse.mean():.6f}")
    print(f"Median MSE      : {all_mse.median():.6f}")
    print(f"Std MSE         : {all_mse.std():.6f}")

    # --- Figure ---
    if args.direction == "cosmos-to-euclid":
        col_titles = ["COSMOS input", "Generated Euclid", "Real Euclid", "Residual (Gen - Real)"]
    else:
        col_titles = ["Euclid input", "Generated COSMOS", "Real COSMOS", "Residual (Gen - Real)"]

    n = min(args.n_plot, len(plot_input))
    ids = [m["idx"] for m in plot_metadata[:n]]
    fig, axes = plt.subplots(n, 4, figsize=(9, 2.5 * n))
    if n == 1:
        axes = axes[None, :]

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14)

    for i in range(n):
        show_image(axes[i, 0], plot_input[i])
        show_image(axes[i, 1], plot_generated[i])
        show_image(axes[i, 2], plot_target[i])

        residual = (plot_generated[i] - plot_target[i]).squeeze().float().numpy()
        #vmax = np.abs(residual).max()
        im = axes[i, 3].imshow(residual, cmap="coolwarm")
        axes[i, 3].axis("off")
        fig.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)

        axes[i, 0].text(0.02, 0.98, f"idx={ids[i]}", fontsize=18,
                        ha="left", va="top", color="magenta",
                        transform=axes[i, 0].transAxes)

    fig.suptitle(
        f"Test set ({args.direction})  |  Mean MSE = {all_mse.mean():.5f}  |  N = {len(all_mse)}",
        fontsize=18, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {args.out}")


if __name__ == "__main__":
    main()
