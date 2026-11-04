"""
Per-galaxy MSE and pixel Fréchet distance of the mean generation over the
Euclid x COSMOS test set, from cutouts already generated and saved as FITS
files by generate_test_set.py.

Each generated file is paired with its real target in the h5 by name,
<input>_<target>_idx_<idx>_n_<k>.fits (see compute_fid.py):
  cos_euc -> real target euclid_images_upscaled[idx]
  euc_cos -> real target cosmos_images_downscaled[idx]
Draws n_1 ... n_<--n-samples> of each galaxy are averaged (with --n-samples 1,
the single generated cutout is used as it is) and that mean is compared with
the real target (anchor):
  - MSE            pixel by pixel, mean generation vs target.
  - pixel Fréchet  between the distribution of the target's pixel values and
                   that of the mean generation's, each summarised as a Gaussian
                   (mean mu, standard deviation sigma):
                       (mu_target - mu_gen)^2 + (sigma_target - sigma_gen)^2
                   This is the FID formula in one dimension, on raw pixel values
                   instead of Inception features. It ignores where pixels are,
                   so it measures brightness levels and contrast, not structure.
Both are in the normalised units of the target survey: the FITS files and the
h5 hold preprocessed units, so the dataset's (x - mean) / std normalisation is
applied to both first, as in training. Those units differ between Euclid and
COSMOS, so pooling both directions gives a bimodal histogram. So also, each
divided by the variance of the target image's pixels:
  - normalised MSE            MSE / variance = 1 - R^2: the fraction of the
                              target's pixel variance the mean generation
                              leaves unexplained.
  - normalised pixel Fréchet  pixel Fréchet / variance
                                = ((mu_target - mu_gen) / sigma_target)^2
                                  + (1 - sigma_gen / sigma_target)^2
Both are unitless and unchanged by each survey's (x - mean) / std
normalisation, so comparable across surveys.

Galaxies missing any of the --n-samples draws are skipped, with a warning.

Writes to --out-dir:
  per_galaxy.csv             idx, anchor_survey, mse_mean_vs_target, pixel_frechet,
                             nmse_mean_vs_target, npixel_frechet
  histograms.png             per-galaxy MSE | per-galaxy pixel Fréchet distance
  histograms_normalised.png  per-galaxy normalised MSE | normalised pixel Fréchet

Usage:
    python experiments/euclid-cosmos/visualization/pixel_metrics.py \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5 \
        --gen-dir    /n03data/fontirro/cutouts/gen/v15 \
        --out-dir    /n03data/fontirro/euclid-cosmos/plots_model/.../pixel_metrics \
        --n-samples  1
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(__file__)
_experiment_dir = os.path.abspath(os.path.join(_here, ".."))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
sys.path.insert(0, _experiment_dir)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import NORM_DICT
from visualization.compute_fid import DIRECTIONS, find_generated, load_real, load_generated

# anchor survey -> NORM_DICT key of its 64 x 64 h5 images
NORM_KEY = {"euclid": "euclid_up", "cosmos": "cosmos_ds"}


def pixel_frechet(target, generated):
    """Per-image Fréchet distance between Gaussians fitted to the pixel values.

    target, generated: (B, C, H, W) -> (B,)
    """
    t, g = target.flatten(1), generated.flatten(1)
    mu_t, mu_g = t.mean(dim=1), g.mean(dim=1)
    sigma_t, sigma_g = t.std(dim=1, unbiased=False), g.std(dim=1, unbiased=False)
    return (mu_t - mu_g) ** 2 + (sigma_t - sigma_g) ** 2


def target_variance(target):
    """Per-image pixel variance, (B, C, H, W) -> (B,).

    Population variance (unbiased=False), so MSE / variance is exactly 1 - R^2.
    """
    return target.flatten(1).var(dim=1, unbiased=False)


def metric_hist(ax, values, xlabel, title):
    """Histogram of a non-negative per-galaxy metric.

    Log axis when the values span two orders of magnitude or more (a linear one
    would squash the bulk into the first bins); linear otherwise, where log
    tick labels would overlap.
    """
    values = values[np.isfinite(values)]
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


def plot_normalised_histograms(per_galaxy, n_samples, path):
    fig, (ax_mse, ax_fd) = plt.subplots(1, 2, figsize=(12, 4.5))
    n = len(per_galaxy)
    metric_hist(ax_mse, per_galaxy["nmse_mean_vs_target"],
                "MSE / variance of the target (1 - R²)",
                f"Per-galaxy normalised MSE (mean of {n_samples} generations, N={n})")
    metric_hist(ax_fd, per_galaxy["npixel_frechet"],
                "Pixel Fréchet distance / variance of the target",
                f"Per-galaxy normalised pixel Fréchet distance (N={n})")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",          required=True)
    p.add_argument("--gen-dir",     required=True,
                   help="Folder of FITS files from generate_test_set.py, e.g. /n03data/fontirro/cutouts/gen/v15")
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--n-samples",   type=int, default=1,
                   help="Draws per galaxy (_n_1 ... _n_<N>), averaged into the mean")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Generated files, one {prefix: {h5 idx: path}} per draw ---
    found = [find_generated(args.gen_dir, k) for k in range(1, args.n_samples + 1)]

    rows = []
    for prefix, (survey, key) in DIRECTIONS.items():
        # Only galaxies with every draw on disk, so each mean is over n_samples draws
        with_any = set().union(*(f[prefix] for f in found))
        idx = np.array(sorted(set(found[0][prefix]).intersection(*(f[prefix] for f in found[1:]))))
        if len(with_any) > len(idx):
            print(f"[WARN] {prefix}: {len(with_any) - len(idx)} galaxies lack some of draws "
                  f"1..{args.n_samples}, skipped")
        if len(idx) == 0:
            print(f"[WARN] {prefix}: no galaxies with all {args.n_samples} draws in {args.gen_dir}")
            continue
        print(f"{prefix}: {len(idx)} galaxies")

        # Both sets in the target survey's normalised units, as in training
        mean, std = NORM_DICT[NORM_KEY[survey]]
        anchor = (load_real(args.h5, key, idx) - mean) / std
        draws = torch.stack([load_generated([f[prefix][i] for i in idx]) for f in found])
        mean_generation = (draws.mean(dim=0) - mean) / std  # (N, 1, H, W)
        if mean_generation.shape != anchor.shape:
            raise ValueError(f"{prefix}: real {tuple(anchor.shape)} vs generated {tuple(mean_generation.shape)}")

        mse = ((mean_generation - anchor) ** 2).mean(dim=(1, 2, 3))
        fd  = pixel_frechet(anchor, mean_generation)
        var_t = target_variance(anchor)
        for i, galaxy in enumerate(idx):
            rows.append({"idx": int(galaxy), "anchor_survey": survey,
                         "mse_mean_vs_target": mse[i].item(),
                         "pixel_frechet": fd[i].item(),
                         "nmse_mean_vs_target": (mse[i] / var_t[i]).item(),
                         "npixel_frechet": (fd[i] / var_t[i]).item()})

    if not rows:
        raise SystemExit(f"No generated FITS files with all {args.n_samples} draws in {args.gen_dir}")

    per_galaxy = pd.DataFrame(rows)
    # %.6g (significant digits), not %.6f: pixel_frechet goes down to ~1e-5.
    per_galaxy.to_csv(os.path.join(args.out_dir, "per_galaxy.csv"), index=False, float_format="%.6g")

    print(f"\n=== Test Results (mean of {args.n_samples} generations, N = {len(per_galaxy)}) ===")
    for col, label in [("mse_mean_vs_target", "MSE"), ("pixel_frechet", "Pixel Fréchet"),
                       ("nmse_mean_vs_target", "Normalised MSE"),
                       ("npixel_frechet", "Normalised Fréchet")]:
        v = per_galaxy[col]
        print(f"{label:<18}: mean {v.mean():.6g}  median {v.median():.6g}  std {v.std():.6g}")

    plot_histograms(per_galaxy, args.n_samples, os.path.join(args.out_dir, "histograms.png"))
    plot_normalised_histograms(per_galaxy, args.n_samples,
                               os.path.join(args.out_dir, "histograms_normalised.png"))
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
