"""
Generate cutouts for every galaxy of the Euclid x COSMOS test set and save each
one as a FITS file, so metrics such as FID can be computed later without
sampling again.

The test set is loaded bidirectionally, as in training and cutout_metrics.py:
even h5 rows have a Euclid target (COSMOS input), odd rows a COSMOS target
(Euclid input). Each galaxy is conditioned as in training: the same galaxy in
the other survey + its precomputed same-instrument sky neighbour
(train.py's collate_fn). --n-samples independent draws per galaxy (default 1).

One file per draw in <out-dir>/<version>/ (created if missing), so versions
don't overwrite each other:
    <input>_<target>_idx_<idx>_n_<k>.fits
  e.g. euc_cos_idx_2838_n_1.fits: Euclid input, generated COSMOS, h5 row 2838,
  first draw. idx is the h5 row of the target galaxy.

Pixels are in the h5's preprocessed units (euclid_images_upscaled /
cosmos_images_downscaled), i.e. the model output with the dataset's
(x - mean) / std normalisation undone; NORMMEAN and NORMSTD in the header
give the values used.

Usage:
    python experiments/euclid-cosmos/visualization/generate_test_set.py \
        --checkpoint /n03data/fontirro/euclid-cosmos/checkpoints/.../best-epoch=75-step=92000.ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5 \
        --indices    /n03data/fontirro/euclid-cosmos/checkpoints/.../test_indices.npy \
        --out-dir    /n03data/fontirro/cutouts/gen/ \
        --version    v4
"""

import os
import sys
import argparse
from functools import partial
import numpy as np
import torch
import pytorch_lightning as pl
from astropy.io import fits
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_here = os.path.dirname(__file__)
_experiment_dir = os.path.abspath(os.path.join(_here, ".."))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
sys.path.insert(0, _experiment_dir)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn

# anchor (generated) survey -> (file prefix <input>_<target>, NORM_DICT key of its h5 images)
SURVEY_INFO = {
    "euclid": ("cos_euc", "euclid_up"),
    "cosmos": ("euc_cos", "cosmos_ds"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--h5",          required=True)
    p.add_argument("--indices",     required=True, help="test_indices.npy saved by train.py")
    p.add_argument("--out-dir",     default="/n03data/fontirro/cutouts/gen/")
    p.add_argument("--version",     required=True,
                   help="Model version, e.g. v4; files go to <out-dir>/<version>/")
    p.add_argument("--n-samples",   type=int, default=1, help="Generations per galaxy, one file each")
    p.add_argument("--num-steps",   type=int, default=100, help="ODE integration steps")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42,
                   help="Fixes the neighbour fallbacks and the sampling noise, "
                        "so reruns reproduce the same cutouts")
    p.add_argument("--precision",   default="bf16-mixed", choices=["bf16-mixed", "32"],
                   help="Sampling precision. bf16-mixed matches training (train.py's "
                        "Trainer precision, which also covers its validation sampling) and "
                        "is several times faster; 32 is full fp32.")
    args = p.parse_args()
    pl.seed_everything(args.seed, workers=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = os.path.join(args.out_dir, args.version)
    os.makedirs(out_dir, exist_ok=True)

    # On CPU the whole test set takes days and the job hits its time limit, so
    # stop now rather than after hours of sampling.
    if not torch.cuda.is_available():
        sys.exit(f"No GPU visible to torch {torch.__version__} (CUDA build: {torch.version.cuda}, "
                 f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}); refusing to run on CPU.")
    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

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
    print(f"Test set: {len(test_indices)} galaxies, {args.n_samples} generation(s) each")

    # --- Generate and save ---
    n_files = 0
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=args.precision == "bf16-mixed")
    with torch.no_grad(), autocast:
        for anchor, cond, sameins, masks, metadata in tqdm(loader, desc="Generating"):
            cond    = cond.to(device)
            sameins = sameins.to(device)
            masks   = masks.to(device)

            for k in range(1, args.n_samples + 1):
                generated = model.sample(
                    cond_image_samegal=cond,
                    cond_image_sameins=sameins,
                    masks=masks,
                    num_steps=args.num_steps,
                ).float().cpu()  # (B, 1, H, W), normalised units

                for i, m in enumerate(metadata):
                    prefix, norm_key = SURVEY_INFO[m["anchor_survey"]]
                    mean, std = dataset.norm_dict[norm_key]
                    image = generated[i, 0].numpy() * std + mean

                    header = fits.Header()
                    header["DIRECTN"]  = (prefix, "<input>_<target> survey")
                    header["GALIDX"]   = (m["idx"], "h5 row of the target galaxy")
                    header["GENNUM"]   = (k, "draw number for this galaxy")
                    header["NSAMPLES"] = (args.n_samples, "draws per galaxy")
                    header["SAMEIDX"]  = (m["sameins_idx"], "h5 row of the sameins neighbour, -1 random")
                    header["NSTEPS"]   = (args.num_steps, "ODE integration steps")
                    header["SEED"]     = (args.seed, "random seed")
                    header["PRECISN"]  = (args.precision, "sampling precision")
                    header["NORMMEAN"] = (mean, "pixels = model output * NORMSTD + NORMMEAN")
                    header["NORMSTD"]  = (std, "pixels = model output * NORMSTD + NORMMEAN")
                    header["VERSION"]  = (args.version, "model version")
                    header["CKPT"]     = os.path.basename(args.checkpoint)

                    path = os.path.join(out_dir, f"{prefix}_idx_{m['idx']}_n_{k}.fits")
                    fits.writeto(path, image.astype(np.float32), header, overwrite=True)
                    n_files += 1

    print(f"Saved {n_files} FITS files to: {out_dir}")


if __name__ == "__main__":
    main()
