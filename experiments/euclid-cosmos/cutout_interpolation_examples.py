"""
To check the different effects of cutout interpolation modes, 
"""


import sys
import os

# ---------------------------------------------------------------------------
# CONFIG 
# ---------------------------------------------------------------------------

CATALOG_PATH = "/n03data/fontirro/data_files/cat_crossmatch_mag27_mag25.csv"  # path to the paired catalog

EUCLID_COL = "file_euclid_vis"              # column name for the Euclid FITS file name
COSMOS_COL = "file_cosmos_f150w"            # column name for the COSMOS FITS file name

EUCLID_PATH_PREFIX = "/n03data/fontirro/cutouts/euclid/40_cutouts/40_cutouts-vis/"  # prefix path for Euclid cutouts
COSMOS_PATH_PREFIX = "/n03data/fontirro/cutouts/cosmos/120_cutouts/f150w/"  # prefix path for COSMOS cutouts

OUTPUT_DIR = "/n03data/fontirro/plots_examples/cutout_interpolation_examples"  # output directory for the cutout examples

MODES = ["nearest", "nearest-exact", "bilinear", "bicubic", "area"]  # interpolation modes to test

EUCLID_HDU = 1   # HDU index for Euclid data (usually 1 for science extension)
COSMOS_HDU = 0   # HDU index for COSMOS data (usually 0)

NUM_WORKERS = 16  # parallel threads for loading + preprocessing

H_SIZE = 64  # target spatial size for both Euclid and COSMOS (COSMOS will be downscaled to match Euclid)
W_SIZE = 64  # target spatial size for both Euclid and COSMOS (COSMOS will be downscaled to match Euclid)

# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from image_preprocessing import preprocess_image_v2


def load_fits(path: str, hdu: int) -> torch.Tensor:
    """Load a FITS file and return a (1, 1, H, W) float32 tensor."""
    with fits.open(path) as hdul:
        data = hdul[hdu].data.astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis]
    return torch.from_numpy(data).unsqueeze(0)  # (1, 1, H, W)


def get_spatial_size(path: str, hdu: int) -> tuple[int, int]:
    """Return (H, W) of the first valid file."""
    with fits.open(path) as hdul:
        data = hdul[hdu].data
    return data.shape[-2], data.shape[-1]

def euclid_zero_frac(path: str) -> float:
    """Return fraction of zero pixels in a Euclid FITS cutout (numpy only, no torch)."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul[EUCLID_HDU].data.astype(np.float32)
    return float(np.mean(data == 0))

def main():
    # Load the catalog
    df = pd.read_csv(CATALOG_PATH)  

    #randomly select one galaxy from the filtered catalog
    df = df.sample(n=1)
    cos_id = df.iloc[0]["id"]
    euc_id = df.iloc[0]["object_id"]
    vis_mag = df.iloc[0]["vis_AB_mag"]
    f150w_mag = df.iloc[0][ "mag_model_f150w"]
    euclid_path = EUCLID_PATH_PREFIX + df.iloc[0][EUCLID_COL]
    cosmos_path = COSMOS_PATH_PREFIX + df.iloc[0][COSMOS_COL]

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load Euclid and COSMOS cutouts
    euclid_cutout = load_fits(euclid_path, EUCLID_HDU)
    cosmos_cutout = load_fits(cosmos_path, COSMOS_HDU)

    # Preprocess images (normaliation and range compression)
    euclid_cutout = preprocess_image_v2(euclid_cutout)
    cosmos_cutout = preprocess_image_v2(cosmos_cutout)

    fig, axes = plt.subplots(2, len(MODES) + 1, figsize=(30, 10))

    # Display original Euclid and COSMOS cutouts
    axes[0, 0].imshow(euclid_cutout.squeeze().numpy(), cmap='gray')
    axes[0, 0].set_title(f"Euclid Cutout\nID: {euc_id}\nVis Mag: {vis_mag:.2f}")
    axes[0, 0].axis('off')

    axes[1, 0].imshow(cosmos_cutout.squeeze().numpy(), cmap='gray')
    axes[1, 0].set_title(f"COSMOS Cutout\nID: {cos_id}\nF150W Mag: {f150w_mag:.2f}")
    axes[1, 0].axis('off')  

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cutout_interpolation_examples_{cos_id}_{euc_id}.png"), dpi=300)

    print(f"Saved cutout interpolation examples for COSMOS ID {cos_id} and Euclid ID {euc_id} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()