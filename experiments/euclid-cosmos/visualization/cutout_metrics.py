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
Both are in the normalised units of the target survey, which differ between
Euclid and COSMOS, so pooling both directions gives a bimodal histogram. So
also, each divided by the variance of the target image's pixels:
  - normalised MSE            MSE / variance = 1 - R^2: the fraction of the
                              target's pixel variance the mean generation
                              leaves unexplained.
  - normalised pixel Fréchet  pixel Fréchet / variance
                                = ((mu_target - mu_gen) / sigma_target)^2
                                  + (1 - sigma_gen / sigma_target)^2
Both are unitless and unchanged by each survey's (x - mean) / std
normalisation, so comparable across surveys.

And over the whole test set, per direction:
  - FID            Fréchet distance between Gaussians fitted to the Inception-v3
                   pool features (2048-d, standard FID weights from pytorch-fid)
                   of the real targets and of the generations. Uses the first
                   draw of each galaxy, not the mean: averaging blurs the image,
                   which FID would penalise, and it keeps FID independent of
                   --n-samples. Images go to Inception through one linear
                   stretch per direction, fixed from the real targets' 0.5 / 99.5
                   pixel percentiles and shared with the generations, so
                   flux-level differences are kept (a per-image stretch would
                   erase them).
  - FID floor      FID is biased upwards at small N, so its absolute value says
                   little on its own. The test galaxies are split in two random
                   halves A, B and
                       floor_half = FID(real A, real B)
                       fid_half   = FID(real A, generated B)
                   are computed on the same sizes: a model indistinguishable from
                   the real data has fid_half ~ floor_half.

The test set is loaded bidirectionally, as in training: even h5 rows have a
Euclid target (COSMOS input), odd rows a COSMOS target (Euclid input).

Writes to --out-dir:
  per_galaxy.csv             idx, anchor_survey, mse_mean_vs_target, pixel_frechet,
                             nmse_mean_vs_target, npixel_frechet
  fid.csv                    anchor_survey, n, fid, fid_half, floor_half
  histograms.png             per-galaxy MSE | per-galaxy pixel Fréchet distance
  histograms_normalised.png  per-galaxy normalised MSE | normalised pixel Fréchet

Needs pytorch-fid. Its Inception weights are downloaded on first use into
~/.cache/torch/hub/checkpoints; if compute nodes have no internet, fetch them
once on the login node:
    python -c "from pytorch_fid.inception import fid_inception_v3; fid_inception_v3()"

Usage:
    python experiments/euclid-cosmos/visualization/cutout_metrics.py \
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


def inception_features(images, inception, lo, hi, batch_size, device):
    """Inception-v3 pool features, (N, 1, H, W) in normalised units -> (N, 2048).

    Pixels are mapped to [0, 1] by the fixed stretch lo -> 0, hi -> 1 and the
    single band repeated to 3 channels; InceptionV3 itself resizes to 299 x 299
    and rescales to [-1, 1].
    """
    feats = []
    with torch.no_grad():
        for batch in images.split(batch_size):
            x = ((batch - lo) / (hi - lo)).clamp(0, 1).repeat(1, 3, 1, 1).to(device)
            feats.append(inception(x)[0].flatten(1).cpu())
    return torch.cat(feats).double().numpy()


def fid(feats_a, feats_b):
    """Fréchet distance between Gaussians fitted to two (N, D) feature sets."""
    return float(calculate_frechet_distance(feats_a.mean(axis=0), np.cov(feats_a, rowvar=False),
                                            feats_b.mean(axis=0), np.cov(feats_b, rowvar=False)))


def direction_fid(real, generated, inception, batch_size, device, rng):
    """FID of one direction, plus the real-vs-real floor on equal-size halves.

    real, generated: (N, 1, H, W), row i of both the same galaxy.
    """
    lo, hi = np.percentile(real.numpy(), [0.5, 99.5])
    f_real = inception_features(real, inception, lo, hi, batch_size, device)
    f_gen  = inception_features(generated, inception, lo, hi, batch_size, device)

    half = len(real) // 2
    perm = rng.permutation(len(real))
    a, b = perm[:half], perm[half:2 * half]
    return {"n": len(real),
            "fid": fid(f_real, f_gen),
            "fid_half": fid(f_real[a], f_gen[b]),
            "floor_half": fid(f_real[a], f_real[b])}


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
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--h5",          required=True)
    p.add_argument("--indices",     required=True, help="test_indices.npy saved by train.py")
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--n-samples",   type=int, default=3,
                   help="Generations per input, averaged into the mean (FID uses only the first)")
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
    # Kept whole for FID, which needs the full population of each direction
    real_targets, first_draws, surveys = [], [], []
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
            draws = torch.stack([
                model.sample(
                    cond_image_samegal=cond,
                    cond_image_sameins=sameins,
                    masks=masks,
                    num_steps=args.num_steps,
                )
                for _ in range(args.n_samples)
            ]).float()  # (n_samples, B, 1, H, W); metrics in fp32 whatever the precision
            mean_generation = draws.mean(dim=0)

            real_targets.append(anchor.cpu())
            first_draws.append(draws[0].cpu())
            surveys.extend(m["anchor_survey"] for m in metadata)

            mse = ((mean_generation - anchor) ** 2).mean(dim=(1, 2, 3)).cpu()
            fd  = pixel_frechet(anchor, mean_generation).cpu()
            var_t = target_variance(anchor).cpu()
            for i, m in enumerate(metadata):
                rows.append({"idx": m["idx"], "anchor_survey": m["anchor_survey"],
                             "mse_mean_vs_target": mse[i].item(),
                             "pixel_frechet": fd[i].item(),
                             "nmse_mean_vs_target": (mse[i] / var_t[i]).item(),
                             "npixel_frechet": (fd[i] / var_t[i]).item()})

    per_galaxy = pd.DataFrame(rows)
    # %.6g (significant digits), not %.6f: pixel_frechet goes down to ~1e-5.
    per_galaxy.to_csv(os.path.join(args.out_dir, "per_galaxy.csv"), index=False, float_format="%.6g")

    print(f"\n=== Test Results (mean of {args.n_samples} generations, N = {len(per_galaxy)}) ===")
    for col, label in [("mse_mean_vs_target", "MSE"), ("pixel_frechet", "Pixel Fréchet"),
                       ("nmse_mean_vs_target", "Normalised MSE"),
                       ("npixel_frechet", "Normalised Fréchet")]:
        v = per_galaxy[col]
        print(f"{label:<18}: mean {v.mean():.6g}  median {v.median():.6g}  std {v.std():.6g}")

    # --- FID, per direction ---
    real_targets = torch.cat(real_targets)
    first_draws  = torch.cat(first_draws)
    surveys      = np.array(surveys)
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    rng = np.random.default_rng(args.seed)

    fid_rows = []
    print("\n=== FID (first generation per galaxy) ===")
    for survey in ("euclid", "cosmos"):
        sel = torch.from_numpy(surveys == survey)
        n = int(sel.sum())
        if n < 4:
            print(f"[WARN] {n} galaxies with a {survey} target, FID skipped")
            continue
        if n < 2048:
            print(f"[WARN] {n} galaxies with a {survey} target, fewer than the 2048 "
                  f"feature dimensions: FID is strongly biased, compare it with floor_half")
        r = direction_fid(real_targets[sel], first_draws[sel], inception,
                          args.batch_size, device, rng)
        fid_rows.append({"anchor_survey": survey, **r})
        print(f"{survey} target (N={n}): FID {r['fid']:.4g}  |  "
              f"half-size: FID {r['fid_half']:.4g}  floor (real vs real) {r['floor_half']:.4g}")
    pd.DataFrame(fid_rows).to_csv(os.path.join(args.out_dir, "fid.csv"), index=False, float_format="%.6g")

    plot_histograms(per_galaxy, args.n_samples, os.path.join(args.out_dir, "histograms.png"))
    plot_normalised_histograms(per_galaxy, args.n_samples,
                               os.path.join(args.out_dir, "histograms_normalised.png"))
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
