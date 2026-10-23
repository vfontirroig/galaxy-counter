"""
Per-galaxy MSE and pixel Fréchet distance of the mean generation over the
Euclid x COSMOS test set.

For every test galaxy, generates --n-samples cutouts from its input (the
same-galaxy image in the other survey + the same-instrument neighbour), averages
them, and compares that mean with the real target (anchor):
  - MSE            pixel by pixel, mean generation vs target.
  - pixel Fréchet  between the distribution of the target's pixel values and
                   that of the mean generation's, each summarised as a Gaussian
                   (mean mu, standard deviation sigma):
                       (mu_target - mu_gen)^2 + (sigma_target - sigma_gen)^2
                   This is the FID formula in one dimension, on raw pixel values
                   instead of Inception features. It ignores where pixels are,
                   so it measures brightness levels and contrast, not structure.
Both are in the normalised units of the target survey.

The test set is loaded bidirectionally, as in training: even h5 rows have a
Euclid target (COSMOS input), odd rows a COSMOS target (Euclid input).

Writes to --out-dir:
  per_galaxy.csv  idx, anchor_survey, mse_mean_vs_target, pixel_frechet
  histograms.png  per-galaxy MSE | per-galaxy pixel Fréchet distance

Usage:
    python experiments/euclid-cosmos/visualization/fid_plot.py \
        --checkpoint /n03data/fontirro/euclid-cosmos/checkpoints/.../best-epoch=75-step=92000.ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5 \
        --indices    /n03data/fontirro/euclid-cosmos/checkpoints/.../test_indices.npy \
        --out-dir    /n03data/fontirro/euclid-cosmos/plots_model/.../fid
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
_experiment_dir = os.path.abspath(os.path.join(_here, ".."))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
sys.path.insert(0, _experiment_dir)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


def pixel_frechet(target, generated):
    """Per-image Fréchet distance between Gaussians fitted to the pixel values.

    target, generated: (B, C, H, W) -> (B,)
    """
    t, g = target.flatten(1), generated.flatten(1)
    mu_t, mu_g = t.mean(dim=1), g.mean(dim=1)
    sigma_t, sigma_g = t.std(dim=1, unbiased=False), g.std(dim=1, unbiased=False)
    return (mu_t - mu_g) ** 2 + (sigma_t - sigma_g) ** 2


def metric_hist(ax, values, xlabel, title):
    """Histogram of a non-negative per-galaxy metric.

    Log axis when the values span two orders of magnitude or more (a linear one
    would squash the bulk into the first bins); linear otherwise, where log
    tick labels would overlap.
    """
    positive = values[values > 0]
    if positive.max() / positive.min() >= 100:
        ax.hist(positive, bins=np.logspace(np.log10(positive.min()), np.log10(positive.max()), 40))
        ax.set_xscale("log")
        if len(positive) < len(values):
            print(f"[WARN] {len(values) - len(positive)} galaxies with {xlabel} == 0 "
                  f"are not shown on the log axis")
    else:
        ax.hist(values, bins=40)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Galaxies")
    ax.set_title(title)


def plot_histograms(per_galaxy, n_samples, path):
    fig, (ax_mse, ax_fd) = plt.subplots(1, 2, figsize=(12, 4.5))
    n = len(per_galaxy)
    metric_hist(ax_mse, per_galaxy["mse_mean_vs_target"],
             "MSE, mean generation vs target (normalised units)",
             f"Per-galaxy MSE (mean of {n_samples} generations, N={n})")
    metric_hist(ax_fd, per_galaxy["pixel_frechet"],
             "Pixel Fréchet distance, mean generation vs target",
             f"Per-galaxy pixel Fréchet distance (N={n})")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--h5",          required=True)
    p.add_argument("--indices",     required=True, help="test_indices.npy saved by train.py")
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--n-samples",   type=int, default=3,   help="Generations per input, averaged into the mean")
    p.add_argument("--num-steps",   type=int, default=100, help="ODE integration steps")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42,
                   help="Fixes the neighbour fallbacks and the sampling noise, "
                        "so reruns reproduce the same metrics")
    p.add_argument("--precision",   default="bf16-mixed", choices=["bf16-mixed", "32"],
                   help="Sampling precision. bf16-mixed matches training (train.py's "
                        "Trainer precision, which also covers its validation sampling) and "
                        "is several times faster; 32 is full fp32.")
    args = p.parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Created up front: everything is written after the full generation pass.
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    # --- Test set, both directions as in training ---
    test_indices = np.load(args.indices)
    dataset = EuclidCosmosDataset(args.h5, bidirectional=True)
    loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, dataset=dataset),
    )
    print(f"Test set: {len(test_indices)} galaxies")

    # --- Generate: mean of n_samples draws per input ---
    rows = []
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=args.precision == "bf16-mixed")
    with torch.no_grad(), autocast:
        for anchor, cond, sameins, masks, metadata in tqdm(loader, desc="Generating"):
            anchor  = anchor.to(device)
            cond    = cond.to(device)
            sameins = sameins.to(device)
            masks   = masks.to(device)

            # Same conditioning for every draw, so they differ only by the
            # sampling noise.
            mean_generation = torch.stack([
                model.sample(
                    cond_image_samegal=cond,
                    cond_image_sameins=sameins,
                    masks=masks,
                    num_steps=args.num_steps,
                )
                for _ in range(args.n_samples)
            ]).float().mean(dim=0)  # (B, 1, H, W); metrics in fp32 whatever the precision

            mse = ((mean_generation - anchor) ** 2).mean(dim=(1, 2, 3)).cpu()
            fd  = pixel_frechet(anchor, mean_generation).cpu()
            for i, m in enumerate(metadata):
                rows.append({"idx": m["idx"], "anchor_survey": m["anchor_survey"],
                             "mse_mean_vs_target": mse[i].item(),
                             "pixel_frechet": fd[i].item()})

    per_galaxy = pd.DataFrame(rows)
    per_galaxy.to_csv(os.path.join(args.out_dir, "per_galaxy.csv"), index=False, float_format="%.6f")

    print(f"\n=== Test Results (mean of {args.n_samples} generations, N = {len(per_galaxy)}) ===")
    for col, label in [("mse_mean_vs_target", "MSE"), ("pixel_frechet", "Pixel Fréchet")]:
        v = per_galaxy[col]
        print(f"{label:<14}: mean {v.mean():.6f}  median {v.median():.6f}  std {v.std():.6f}")

    plot_histograms(per_galaxy, args.n_samples, os.path.join(args.out_dir, "histograms.png"))
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
