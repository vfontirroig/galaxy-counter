"""
FID between real test-set cutouts (h5) and their generated counterparts (FITS
files written by generate_test_set.py), per direction, over every galaxy with a
generated file.

FID is the Fréchet distance between Gaussians fitted to the Inception-v3 pool
features (2048-d, standard FID weights from pytorch-fid) of the real targets and
of the generations. Images go to Inception through one linear stretch per
direction, fixed from the real targets' 0.5 / 99.5 pixel percentiles and shared
with the generations, so flux-level differences are kept (a per-image stretch
would erase them).

Generated files are matched to the h5 by name, <input>_<target>_idx_<idx>_n_<k>.fits:
  cos_euc -> real target euclid_images_upscaled[idx]
  euc_cos -> real target cosmos_images_downscaled[idx]
Both are in the h5's preprocessed units (generate_test_set.py undoes the
dataset normalisation before writing), so they are compared as they are.
Only draw --draw of each galaxy is used.

Writes to --out-dir:
  fid.csv   anchor_survey, n, fid

Needs pytorch-fid. Its Inception weights are downloaded on first use into
~/.cache/torch/hub/checkpoints; if compute nodes have no internet, fetch them
once on the login node:
    python -c "from pytorch_fid.inception import fid_inception_v3; fid_inception_v3()"

Usage:
    python experiments/euclid-cosmos/visualization/compute_fid.py \
        --h5          /n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5 \
        --gen-dir     /n03data/fontirro/cutouts/gen/v15 \
        --out-dir     /n03data/fontirro/euclid-cosmos/plots_model/.../fid
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import h5py
import torch
from astropy.io import fits
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

# file prefix <input>_<target> -> (anchor survey, h5 dataset of the real target)
DIRECTIONS = {
    "cos_euc": ("euclid", "euclid_images_upscaled"),
    "euc_cos": ("cosmos", "cosmos_images_downscaled"),
}
FILE_RE = re.compile(r"^(cos_euc|euc_cos)_idx_(\d+)_n_(\d+)\.fits$")


def find_generated(gen_dir, draw):
    """{prefix: {h5 idx: path}} for every FITS file of draw `draw` in gen_dir."""
    found = {prefix: {} for prefix in DIRECTIONS}
    for path in glob.glob(os.path.join(gen_dir, "*.fits")):
        m = FILE_RE.match(os.path.basename(path))
        if m and int(m.group(3)) == draw:
            found[m.group(1)][int(m.group(2))] = path
    return found


def load_real(h5_path, key, idx):
    """(N, 1, H, W) real targets for sorted h5 rows idx, read in one go."""
    with h5py.File(h5_path, "r") as f:
        images = f[key][idx]
    return torch.from_numpy(images.reshape(len(idx), 1, *images.shape[-2:]).astype(np.float32))


def load_generated(paths):
    """(N, 1, H, W) generated cutouts, in the order of paths."""
    # astype: FITS data is big-endian, which torch.from_numpy rejects
    images = np.stack([fits.getdata(p).astype(np.float32) for p in tqdm(paths, desc="Reading FITS")])
    return torch.from_numpy(images.reshape(len(paths), 1, *images.shape[-2:]))


def inception_features(images, inception, lo, hi, batch_size, device):
    """Inception-v3 pool features, (N, 1, H, W) -> (N, 2048).

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


def direction_fid(real, generated, inception, batch_size, device):
    """FID of one direction. real, generated: (N, 1, H, W)."""
    lo, hi = np.percentile(real.numpy(), [0.5, 99.5])
    f_real = inception_features(real, inception, lo, hi, batch_size, device)
    f_gen  = inception_features(generated, inception, lo, hi, batch_size, device)
    return fid(f_real, f_gen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5",         required=True)
    p.add_argument("--gen-dir",    required=True,
                   help="Folder of FITS files from generate_test_set.py, e.g. /n03data/fontirro/cutouts/gen/v15")
    p.add_argument("--out-dir",    required=True)
    p.add_argument("--draw",       type=int, default=1, help="Which draw (_n_<k>) to use per galaxy")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()

    found = find_generated(args.gen_dir, args.draw)
    fid_rows = []
    print(f"\n=== FID (draw {args.draw}, {args.gen_dir}) ===")
    for prefix, (survey, key) in DIRECTIONS.items():
        idx = np.array(sorted(found[prefix]))  # sorted: h5py fancy indexing needs increasing rows
        n = len(idx)
        if n < 4:
            print(f"[WARN] {n} {prefix} files for draw {args.draw}, FID skipped")
            continue
        if n < 2048:
            print(f"[WARN] {n} galaxies with a {survey} target, fewer than the 2048 "
                  f"feature dimensions: FID is strongly biased upwards")

        real = load_real(args.h5, key, idx)
        generated = load_generated([found[prefix][i] for i in idx])
        if real.shape != generated.shape:
            raise ValueError(f"{prefix}: real {tuple(real.shape)} vs generated {tuple(generated.shape)}")

        value = direction_fid(real, generated, inception, args.batch_size, device)
        fid_rows.append({"anchor_survey": survey, "n": n, "fid": value})
        print(f"{survey} target (N={n}): FID {value:.4g}")

    pd.DataFrame(fid_rows).to_csv(os.path.join(args.out_dir, "fid.csv"), index=False, float_format="%.6g")
    print(f"Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
