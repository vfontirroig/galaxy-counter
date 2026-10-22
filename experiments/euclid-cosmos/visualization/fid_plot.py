"""
FID and MSE of the mean generation over the Euclid x COSMOS test set.

For every test galaxy, generates --n-samples cutouts from its input (the
same-galaxy image in the other survey + the same-instrument neighbour), averages
them, and compares that mean with the real target (anchor):
  - MSE  per galaxy, mean generation vs target, in the normalised units of the
         target survey.
  - FID  between the distribution of the real targets and that of the mean
         generations: Inception-v3 pool features (pytorch-fid weights). Also
         bootstrapped over galaxies (--n-bootstrap) to show its spread.

FID depends on the number of images (it is biased high at small N), and a
bootstrap resample repeats galaxies, so it holds fewer distinct images than the
full set: the bootstrap histogram sits above the full-set FID. Read its width
as the uncertainty and the full-set FID (fid.csv, dashed line) as the value.

The test set is loaded bidirectionally, as in training: even h5 rows have a
Euclid target (COSMOS input), odd rows a COSMOS target (Euclid input). Both
directions go into the same target and generated sets.

Writes to --out-dir:
  per_galaxy.csv     idx, anchor_survey, mse_mean_vs_target
  fid.csv            n, fid, bootstrap mean/std, stretch limits
  fid_bootstrap.csv  every bootstrap FID, so the histogram can be re-plotted
  histograms.png     MSE per galaxy | bootstrap FID

Inception weights: pytorch-fid downloads pt_inception-2015-12-05-6726825d.pth
to ~/.cache/torch/hub/checkpoints the first time. Compute nodes without
internet need it fetched once beforehand on a login node:
    python -c "from pytorch_fid.inception import InceptionV3; InceptionV3()"

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
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

_here = os.path.dirname(__file__)
_experiment_dir = os.path.abspath(os.path.join(_here, ".."))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
sys.path.insert(0, _experiment_dir)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn

# Stretch to [0, 1] before Inception, in percentiles of the real targets. The
# same lo/hi is applied to real and generated images, since the stretch itself
# changes the FID.
STRETCH_PERCENTILES = (0.5, 99.5)


def inception_features(images, inception, lo, hi, device, batch_size):
    """(N, 1, H, W) normalised cutouts -> (N, 2048) Inception pool features.

    Linearly stretched to [0, 1] with lo/hi, then copied to 3 channels;
    InceptionV3 resizes to 299x299 and rescales to [-1, 1] itself.
    """
    feats = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x = images[start:start + batch_size].to(device)
            x = ((x - lo) / (hi - lo)).clamp(0, 1).repeat(1, 3, 1, 1)
            feats.append(inception(x)[0].flatten(1).cpu())
    return torch.cat(feats).double().numpy()


def fid_from_features(real, gen):
    mu_r, sigma_r = real.mean(axis=0), np.cov(real, rowvar=False)
    mu_g, sigma_g = gen.mean(axis=0), np.cov(gen, rowvar=False)
    return float(calculate_frechet_distance(mu_r, sigma_r, mu_g, sigma_g))


def plot_histograms(per_galaxy, fid, fid_boot, n_samples, path):
    fig, (ax_mse, ax_fid) = plt.subplots(1, 2, figsize=(12, 4.5))

    # MSE is right-skewed over orders of magnitude: log-spaced bins on a log axis.
    mse = per_galaxy["mse_mean_vs_target"]
    mse_bins = np.logspace(np.log10(mse[mse > 0].min()), np.log10(mse.max()), 40)
    ax_mse.hist(mse, bins=mse_bins)
    ax_mse.set_xscale("log")
    ax_mse.set_xlabel("MSE, mean generation vs target (normalised units)")
    ax_mse.set_ylabel("Galaxies")
    ax_mse.set_title(f"Per-galaxy MSE (mean of {n_samples} generations, N={len(mse)})")

    ax_fid.hist(fid_boot["fid"], bins=20, label=f"{len(fid_boot)} bootstrap resamples")
    ax_fid.axvline(fid, color="k", linestyle="--", label=f"Full test set: {fid:.2f}")
    ax_fid.set_xlabel("FID, targets vs mean generations (Inception-v3)")
    ax_fid.set_ylabel("Bootstrap resamples")
    ax_fid.set_title("FID")
    ax_fid.legend()

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
    p.add_argument("--n-bootstrap", type=int, default=100, help="Bootstrap resamples of the FID")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42,
                   help="Fixes the neighbour fallbacks, the sampling noise and the bootstrap, "
                        "so reruns reproduce the same metrics")
    args = p.parse_args()
    pl.seed_everything(args.seed, workers=True)

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
    rows, targets, generated = [], [], []
    with torch.no_grad():
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
            ]).mean(dim=0)  # (B, 1, H, W)

            mse = ((mean_generation - anchor) ** 2).mean(dim=(1, 2, 3)).cpu()
            targets.append(anchor.cpu())
            generated.append(mean_generation.cpu())
            for i, m in enumerate(metadata):
                rows.append({"idx": m["idx"], "anchor_survey": m["anchor_survey"],
                             "mse_mean_vs_target": mse[i].item()})

    per_galaxy = pd.DataFrame(rows)
    per_galaxy.to_csv(os.path.join(args.out_dir, "per_galaxy.csv"), index=False, float_format="%.6f")

    # --- FID: target distribution vs mean-generation distribution ---
    real = torch.cat(targets)
    gen  = torch.cat(generated)
    n = len(real)
    lo, hi = np.percentile(real.numpy(), STRETCH_PERCENTILES)

    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    real_feats = inception_features(real, inception, lo, hi, device, args.batch_size)
    gen_feats  = inception_features(gen,  inception, lo, hi, device, args.batch_size)
    fid = fid_from_features(real_feats, gen_feats)

    # Paired bootstrap: each resample draws galaxies, keeping a galaxy's real
    # and generated images together.
    rng = np.random.default_rng(args.seed)
    boot_rows = []
    for b in tqdm(range(args.n_bootstrap), desc="Bootstrap FID"):
        idx = rng.integers(0, n, n)
        boot_rows.append({"replicate": b, "fid": fid_from_features(real_feats[idx], gen_feats[idx])})
    fid_boot = pd.DataFrame(boot_rows)

    pd.DataFrame([{"n": n, "fid": fid,
                   "fid_bootstrap_mean": fid_boot["fid"].mean(),
                   "fid_bootstrap_std": fid_boot["fid"].std(),
                   "stretch_lo": lo, "stretch_hi": hi}]).to_csv(
        os.path.join(args.out_dir, "fid.csv"), index=False, float_format="%.6f")
    fid_boot.to_csv(os.path.join(args.out_dir, "fid_bootstrap.csv"), index=False, float_format="%.6f")

    if n < 2048:
        print(f"[WARN] {n} galaxies < 2048 feature dims, so the covariances are "
              f"rank-deficient and this FID is biased high. Compare FIDs only at equal N.")

    mse = per_galaxy["mse_mean_vs_target"]
    print(f"\n=== Test Results (mean of {args.n_samples} generations, N = {n}) ===")
    print(f"MSE : mean {mse.mean():.6f}  median {mse.median():.6f}  std {mse.std():.6f}")
    print(f"FID : {fid:.3f}  (bootstrap {fid_boot['fid'].mean():.3f} +/- {fid_boot['fid'].std():.3f})")

    plot_histograms(per_galaxy, fid, fid_boot, args.n_samples,
                    os.path.join(args.out_dir, "histograms.png"))
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
