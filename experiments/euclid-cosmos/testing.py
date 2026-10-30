"""
Plot generations of the trained Euclid x COSMOS flow-matching model for a
random selection of held-out test galaxies.

Draws --n-galaxies random galaxies from the test indices saved by train.py
(fixed by --seed), generates --n-samples independent images per input (same
conditioning, different noise) and plots them with their mean (no Mean column
when --n-samples is 1). Figure columns depend on direction:
  cosmos-to-euclid:
      [COSMOS input | Same-instrument input | Generated #1..N | Mean | Real Euclid]
  euclid-to-cosmos:
      [Euclid input | Same-instrument input | Generated #1..N | Mean | Real COSMOS]
Each generated cutout and the mean are labelled with their MSE against the
real target, in the normalised units of the target survey.

Usage:
    python experiments/euclid-cosmos/testing.py \
        --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-vis-f150w/test-4-phase1/best-epoch=00-step=100000.ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs_vis_f150w.h5 \
        --indices    /n03data/fontirro/checkpoints/euclid-cosmos-vis-f150w/test-4-phase1/test_indices.npy \
        --out        /n03data/fontirro/checkpoints/euclid-cosmos-vis-f150w/test-4-phase1/test_results.png \
        --direction  cosmos-to-euclid \
        --n-galaxies 8
"""

import os
import sys
import argparse
from functools import partial
import numpy as np
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


def show_metrics(ax, generated, target):
    """Write the cutout's MSE vs the target in its upper right."""
    mse = ((generated - target) ** 2).mean().item()
    ax.text(0.98, 0.98, f"MSE={mse:.3g}", fontsize=12,
            ha="right", va="top", color="white",
            transform=ax.transAxes)


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
    p.add_argument("--n-galaxies", type=int, default=8,   help="Random test galaxies to generate and plot")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-steps",  type=int, default=100, help="ODE integration steps")
    p.add_argument("--n-samples",  type=int, default=3,   help="Generations per input, averaged into the mean")
    p.add_argument("--seed",       type=int, default=42,
                   help="Fixes the galaxy selection, the neighbour fallbacks and the "
                        "sampling noise, so reruns reproduce the same figure")
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--precision",  default="bf16-mixed", choices=["bf16-mixed", "32"],
                   help="Sampling precision. bf16-mixed matches training (train.py's "
                        "Trainer precision, which also covers its validation sampling) and "
                        "is several times faster; 32 is full fp32.")
    args = p.parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Created up front so a missing directory doesn't lose the run after sampling.
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

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

    # --- Pick random test galaxies ---
    test_indices = np.load(args.indices)
    n = min(args.n_galaxies, len(test_indices))
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(test_indices, size=n, replace=False)
    print(f"Test set: {len(test_indices)} galaxies, plotting {n} random ones")

    # sameins is the precomputed same-instrument sky neighbour, read by
    # train.py's collate_fn exactly as in training and fid_plot.py.
    anchor_survey = "euclid" if args.direction == "cosmos-to-euclid" else "cosmos"
    dataset     = OneDirectionDataset(args.h5, anchor_survey)
    subset      = Subset(dataset, chosen.tolist())
    loader      = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, dataset=dataset),
    )

    # --- Generate ---
    plot_input, plot_sameins, plot_generated, plot_mean, plot_target = [], [], [], [], []
    plot_metadata = []

    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=args.precision == "bf16-mixed")
    with torch.no_grad(), autocast:
        for anchor, cond, sameins, masks, metadata in tqdm(loader, desc="Generating"):
            cond    = cond.to(device)
            sameins = sameins.to(device)
            masks   = masks.to(device)

            # Same conditioning (cond + sameins) for every draw, so the only
            # difference between them is the sampling noise.
            generations = torch.stack([
                model.sample(
                    cond_image_samegal=cond,
                    cond_image_sameins=sameins,
                    masks=masks,
                    num_steps=args.num_steps,
                )
                for _ in range(args.n_samples)
            ]).float()  # (n_samples, B, 1, H, W)

            plot_input.append(cond.cpu())
            plot_sameins.append(sameins[:, 0].cpu())
            plot_generated.append(generations.cpu())
            plot_mean.append(generations.mean(dim=0).cpu())
            plot_target.append(anchor)
            plot_metadata.extend(metadata)

    plot_input     = torch.cat(plot_input)
    plot_sameins   = torch.cat(plot_sameins)
    plot_generated = torch.cat(plot_generated, dim=1)
    plot_mean      = torch.cat(plot_mean)
    plot_target    = torch.cat(plot_target)

    # --- Figure ---
    if args.direction == "cosmos-to-euclid":
        input_label, real_label = "COSMOS input", "Real Euclid"
    else:
        input_label, real_label = "Euclid input", "Real COSMOS"
    col_titles = (
        [input_label, "Same-instrument input"]
        + [f"Generated #{k + 1}" for k in range(args.n_samples)]
        + ([f"Mean of {args.n_samples}"] if args.n_samples > 1 else [])
        + [real_label]
    )
    n_cols = len(col_titles)

    ids = [m["idx"] for m in plot_metadata]
    sameins_ids = [m["sameins_idx"] for m in plot_metadata]
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
            show_metrics(axes[i, 2 + k], plot_generated[k, i], plot_target[i])
        if args.n_samples > 1:
            show_image(axes[i, -2], plot_mean[i])
            show_metrics(axes[i, -2], plot_mean[i], plot_target[i])
        show_image(axes[i, -1], plot_target[i])

        axes[i, 0].text(0.02, 0.98, f"idx={ids[i]}", fontsize=16,
                        ha="left", va="top", color="white",
                        transform=axes[i, 0].transAxes)
        # -1: no precomputed neighbour, collate_fn fell back to a random one
        sam_label = f"idx={sameins_ids[i]}" if sameins_ids[i] != -1 else "idx=? (random)"
        axes[i, 1].text(0.02, 0.98, sam_label, fontsize=16,
                        ha="left", va="top", color="white",
                        transform=axes[i, 1].transAxes)

    fig.suptitle(
        f"Test set ({args.direction})"
        f"  |  {n} random galaxies",
        fontsize=18, y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {args.out}")


if __name__ == "__main__":
    main()
