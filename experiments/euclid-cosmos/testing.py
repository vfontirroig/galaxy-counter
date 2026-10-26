"""
Test the trained Euclid x COSMOS flow-matching model on the held-out test set.

Loads the test indices saved by train.py, generates --n-samples independent
images per input (same conditioning, different noise), averages them, and
reports MSE of that mean against the target and against the input.
Figure columns depend on direction:
  cosmos-to-euclid:
      [COSMOS input | Generated #1..N | Mean | Real Euclid | Residual (Mean - Real)]
  euclid-to-cosmos:
      [Euclid input | Generated #1..N | Mean | Real COSMOS | Residual (Mean - Real)]

Per-galaxy metrics are written to --metrics (default: metrics.csv next to --out).

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
import pandas as pd
import torch
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_here = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


class OneDirectionDataset(EuclidCosmosDataset):
    """Every row in the same direction, whatever its index parity.

    Returns (anchor, cond, metadata) with metadata["anchor_survey"] set to the
    generated survey, so train.py's collate_fn picks the precomputed sky
    neighbour from that survey, as in training and fid_plot.py.
    """

    def __init__(self, hdf5_path, anchor_survey):
        super().__init__(hdf5_path)
        self.anchor_survey = anchor_survey

    def __getitem__(self, idx):
        euc, cos, metadata = super().__getitem__(idx)
        if self.anchor_survey == "cosmos":
            return cos, euc, {"idx": idx, "anchor_survey": "cosmos"}
        return euc, cos, metadata


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
    p.add_argument("--metrics",    default=None,
                   help="Per-galaxy metrics CSV (default: metrics.csv in the --out directory)")
    p.add_argument("--direction",  default="cosmos-to-euclid",
                   choices=["cosmos-to-euclid", "euclid-to-cosmos"],
                   help="cosmos-to-euclid: COSMOS input, generate Euclid"
                        "euclid-to-cosmos: Euclid input, generate COSMOS")
    p.add_argument("--n-plot",     type=int, default=8,   help="Galaxy rows to show in figure")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-steps",  type=int, default=100, help="ODE integration steps")
    p.add_argument("--n-samples",  type=int, default=3,   help="Generations per input, averaged into the mean")
    p.add_argument("--seed",       type=int, default=42,
                   help="Fixes the neighbour fallbacks and the sampling noise, "
                        "so reruns reproduce the same figure and metrics")
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--precision",  default="bf16-mixed", choices=["bf16-mixed", "32"],
                   help="Sampling precision. bf16-mixed matches training (train.py's "
                        "Trainer precision, which also covers its validation sampling) and "
                        "is several times faster; 32 is full fp32.")
    args = p.parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Created up front: the figure and CSV are only written after inference
    # over the whole test set, so a missing directory would lose that run.
    metrics_path = args.metrics or os.path.join(os.path.dirname(os.path.abspath(args.out)), "metrics.csv")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)

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
    # sameins is the precomputed same-instrument sky neighbour, read by
    # train.py's collate_fn exactly as in training and fid_plot.py.
    anchor_survey = "euclid" if args.direction == "cosmos-to-euclid" else "cosmos"
    dataset     = OneDirectionDataset(args.h5, anchor_survey)
    test_subset = Subset(dataset, test_indices.tolist())
    loader      = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, dataset=dataset),
    )

    # --- Run inference over the full test set ---
    rows = []
    plot_input, plot_generated, plot_mean, plot_target, plot_metadata = [], [], [], [], []

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=args.precision == "bf16-mixed")
    with torch.no_grad(), autocast:
        for batch_idx, (anchor, cond, sameins, masks, metadata) in enumerate(tqdm(loader, desc="Generating")):
            anchor  = anchor.to(device)
            cond    = cond.to(device)
            sameins = sameins.to(device)
            masks   = masks.to(device)

            # Same conditioning (cond + sameins) for every draw, so the only
            # difference between them is the sampling noise; averaging them
            # cancels that scatter.
            generations = torch.stack([
                model.sample(
                    cond_image_samegal=cond,
                    cond_image_sameins=sameins,
                    masks=masks,
                    num_steps=args.num_steps,
                )
                for _ in range(args.n_samples)
            ]).float()  # (n_samples, B, 1, H, W); metrics in fp32 whatever the precision
            mean_generation = generations.mean(dim=0)  # (B, 1, H, W)

            mse_target = ((mean_generation - anchor) ** 2).mean(dim=(1, 2, 3)).cpu()
            mse_input  = ((mean_generation - cond) ** 2).mean(dim=(1, 2, 3)).cpu()
            for i, m in enumerate(metadata):
                rows.append({
                    "idx": m["idx"],
                    "direction": args.direction,
                    "mse_mean_vs_target": mse_target[i].item(),
                    "mse_mean_vs_input": mse_input[i].item(),
                })

            if batch_idx == 0:
                plot_input     = cond.cpu()
                plot_sameins   = sameins[:, 0].cpu()
                plot_sameins_ids = [m["sameins_idx"] for m in metadata]
                plot_generated = generations.cpu()
                plot_mean      = mean_generation.cpu()
                plot_target    = anchor.cpu()
                plot_metadata  = metadata

    df = pd.DataFrame(rows)
    df.to_csv(metrics_path, index=False, float_format="%.6f")

    print(f"\n=== Test Results ({args.direction}, mean of {args.n_samples} generations) ===")
    print(f"N test galaxies : {len(df)}")
    for col, label in [("mse_mean_vs_target", "vs target"), ("mse_mean_vs_input", "vs input")]:
        print(f"MSE {label:<10}: mean {df[col].mean():.6f}  "
              f"median {df[col].median():.6f}  std {df[col].std():.6f}")
    print(f"Metrics saved: {metrics_path}")

    # --- Figure ---
    if args.direction == "cosmos-to-euclid":
        input_label, real_label = "COSMOS input", "Real Euclid"
    else:
        input_label, real_label = "Euclid input", "Real COSMOS"
    col_titles = (
        [input_label, "Same-instrument input"]
        + [f"Generated #{k + 1}" for k in range(args.n_samples)]
        + [f"Mean of {args.n_samples}", real_label, "Residual (Mean - Real)"]
    )
    n_cols = len(col_titles)

    n = min(args.n_plot, len(plot_input))
    ids = [m["idx"] for m in plot_metadata[:n]]
    fig, axes = plt.subplots(n, n_cols, figsize=(2.25 * n_cols, 2.5 * n))
    if n == 1:
        axes = axes[None, :]

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14)

    for i in range(n):
        show_image(axes[i, 0], plot_input[i])
        show_image(axes[i, 1], plot_sameins[i])
        for k in range(args.n_samples):
            show_image(axes[i, 2 + k], plot_generated[k, i])
        show_image(axes[i, -3], plot_mean[i])
        show_image(axes[i, -2], plot_target[i])

        residual = (plot_mean[i] - plot_target[i]).squeeze().float().numpy()
        #vmax = np.abs(residual).max()
        im = axes[i, -1].imshow(residual, cmap="coolwarm")
        axes[i, -1].axis("off")
        fig.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.04)

        axes[i, 0].text(0.02, 0.98, f"idx={ids[i]}", fontsize=16,
                        ha="left", va="top", color="magenta",
                        transform=axes[i, 0].transAxes)
        # -1: no precomputed neighbour, collate_fn fell back to a random one
        sam_label = f"idx={plot_sameins_ids[i]}" if plot_sameins_ids[i] != -1 else "idx=? (random)"
        axes[i, 1].text(0.02, 0.98, sam_label, fontsize=16,
                        ha="left", va="top", color="magenta",
                        transform=axes[i, 1].transAxes)

    fig.suptitle(
        f"Test set ({args.direction})  |  Mean MSE (mean vs target) = {df['mse_mean_vs_target'].mean():.5f}"
        f"  |  N = {len(df)}",
        fontsize=18, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {args.out}")


if __name__ == "__main__":
    main()
