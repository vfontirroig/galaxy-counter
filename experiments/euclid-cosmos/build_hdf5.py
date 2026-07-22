"""
Build an HDF5 file with paired Euclid (VIS) and COSMOS (F150W) cutouts.

The native spatial size of the first file in each instrument is used to
pre-allocate the datasets. All images of the same instrument are assumed
to share the same size. Datasets follow the (N, C, H, W) layout from
new_dataset_guide.md.

HDF5 layout:
    euclid_images         — (N, 1, H_euc, W_euc) float32
    cosmos_images         — (N, 1, H_cos, W_cos) float32
    catalog/euclid_paths  — string array (N,)
    catalog/cosmos_paths  — string array (N,)
    attrs: num_pairs, num_channels, euclid_shape, cosmos_shape

Preprocessing (via preprocess_image_v2):
    Euclid : ZP rescaling (ZP 26.2 → 23.9), no range compression
    COSMOS : no ZP rescaling (already at 23.9), range compression applied

Edit the CONFIG block below, then run:
    python experiments/euclid-cosmos/build_hdf5.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

CATALOG_PATH = "/n03data/fontirro/data_files/cat_crossmatch_mag27_mag25.csv"  # path to the paired catalog

EUCLID_COL = "40_file_euclid_vis"              # column name for the Euclid FITS file path
COSMOS_COL = "file_cosmos_f150w"            # column name for the COSMOS FITS file path

EUCLID_EXISTS_COL = "cutout_euc_40_vis"    # boolean column: True if Euclid cutout exists
COSMOS_EXISTS_COL = "cutout_cos_256_rot_f150w"  # boolean column: True if COSMOS cutout exists

EUCLID_DIR_PATH = "/n03data/fontirro/cutouts/euclid/40_cutouts/40_cutouts-vis"  # base directory for Euclid VIS cutouts.
COSMOS_DIR_PATH = "/n03data/fontirro/cutouts/cosmos/256_cutouts_rotated/f150w/"  # base directory for COSMOS F150W cutouts.

EUCLID_HDU = 1   # HDU index for Euclid data (usually 1 for science extension)
COSMOS_HDU = 0   # HDU index for COSMOS data (usually 0)

OUTPUT_H5 = "/n03data/fontirro/data_files/euclid_cosmos_pairs_vis_f150w_v2.h5"

NUM_WORKERS = 16  # parallel threads for loading + preprocessing

H_SIZE = 64  # target spatial size for both Euclid and COSMOS 
W_SIZE = 64  # target spatial size for both Euclid and COSMOS 

# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import torch
import torch.nn.functional as F
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


def process_pair(args: tuple) -> tuple:
    """Load, preprocess, downscale cosmos, and upscale euclid. Returns (i, euc, cos, cos_down, euc_up, error)."""
    i, ep, cp  = args
    try:
        euc_tensor = load_fits(ep, EUCLID_HDU)
        cos_tensor = load_fits(cp, COSMOS_HDU)
        euc = preprocess_image_v2(euc_tensor, bands=["VIS"]).squeeze(0).numpy()
        cos = preprocess_image_v2(cos_tensor, bands=["F150W"]).squeeze(0).numpy()
        cos_down = F.interpolate(
            torch.from_numpy(cos).unsqueeze(0), size=(H_SIZE, W_SIZE),
            mode="linear",
        ).squeeze(0).numpy() #(1, H_SIZE, W_SIZE)
        euc_up = F.interpolate(
            torch.from_numpy(euc).unsqueeze(0), size=(H_SIZE, W_SIZE),
            mode="linear",
        ).squeeze(0).numpy() #(1, H_SIZE, W_SIZE)
        return i, euc, cos, cos_down, euc_up, None
    except Exception as e:
        return i, None, None, None, None, str(e)


def main():
    catalog = pd.read_csv(CATALOG_PATH)
    print(f"Catalog loaded: {len(catalog)} pairs")
    print(f"Columns: {list(catalog.columns[-30:])}")

    mask = (catalog[EUCLID_EXISTS_COL] == 1) & (catalog[COSMOS_EXISTS_COL] == 1)
    catalog = catalog[mask].reset_index(drop=True)
    print(f"Pairs with both cutouts present: {len(catalog)}")

    #check if the files exist
    missing_euclid = []
    missing_cosmos = []
    for i, row in catalog.iterrows():
        euclid_path = os.path.join(EUCLID_DIR_PATH, row[EUCLID_COL])
        cosmos_path = os.path.join(COSMOS_DIR_PATH, row[COSMOS_COL])
        if not os.path.isfile(euclid_path):
            missing_euclid.append(euclid_path)
        if not os.path.isfile(cosmos_path):
            missing_cosmos.append(cosmos_path)

    print(f"Missing Euclid files: {len(missing_euclid)}")
    print(f"Missing COSMOS files: {len(missing_cosmos)}")

    # euclid_paths = [os.path.join(EUCLID_DIR_PATH, p) for p in catalog[EUCLID_COL]]
    # cosmos_paths = [os.path.join(COSMOS_DIR_PATH, p) for p in catalog[COSMOS_COL]]
    # N = len(euclid_paths)

    # H_euc, W_euc = get_spatial_size(euclid_paths[0], EUCLID_HDU)
    # H_cos, W_cos = get_spatial_size(cosmos_paths[0], COSMOS_HDU)
    # print(f"Euclid image size : {H_euc} x {W_euc}")
    # print(f"COSMOS image size : {H_cos} x {W_cos}")


    # # ------------------------------------------------------------------
    # # Pass 1: filter out Euclid cutouts with >= 10% zero pixels
    # # Uses threads (I/O bound, no torch) so no fork/spawn overhead.
    # # ------------------------------------------------------------------
    # print(f"\nScanning {N} Euclid files for zero-pixel fraction...")
    # with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
    #     zero_fracs = list(tqdm(
    #         pool.map(euclid_zero_frac, euclid_paths),
    #         total=N, desc="Scanning", mininterval=5,
    #     ))
    # valid = [zf < 0.10 for zf in zero_fracs]
    # euclid_paths = [p for p, v in zip(euclid_paths, valid) if v]
    # cosmos_paths = [p for p, v in zip(cosmos_paths, valid) if v]
    # catalog      = catalog[valid].reset_index(drop=True)
    # N_valid = len(euclid_paths)
    # print(f"Valid pairs after filtering: {N_valid}/{N}  ({N - N_valid} skipped, zero_frac >= 10%)")

    # if N_valid == 0:
    #     print("[ERROR] No valid pairs found — check EUCLID_HDU and file paths.")
    #     sys.exit(1)

    # # ------------------------------------------------------------------
    # # Quick sanity check on first valid pair
    # # ------------------------------------------------------------------
    # print("\nTesting first pair (sequential)...")
    # _, t_euc, t_cos, _, t_euc_up, t_err = process_pair(
    #     (0, euclid_paths[0], cosmos_paths[0])
    # )
    # if t_err:
    #     print(f"[ERROR] First pair failed:\n{t_err}")
    #     sys.exit(1)
    # print(f"  euc [{t_euc.min():.4f}, {t_euc.max():.4f}]  euc_up [{t_euc_up.min():.4f}, {t_euc_up.max():.4f}]  cos [{t_cos.min():.4f}, {t_cos.max():.4f}]")
    # print("OK.\n")

    # H_euc, W_euc = get_spatial_size(euclid_paths[0], EUCLID_HDU)
    # H_cos, W_cos = get_spatial_size(cosmos_paths[0], COSMOS_HDU)
    # print(f"Euclid image size : {H_euc} x {W_euc}")
    # print(f"COSMOS image size : {H_cos} x {W_cos}")

   
    # ------------------------------------------------------------------
    # Pass 2: process and write — pairs that fail processing are dropped
    # (not left as zero-filled rows); datasets are shrunk to the final
    # count at the end so the catalogue only contains real cutouts.
    # ------------------------------------------------------------------

    # args_list = [(i, ep, cp) for i, (ep, cp) in enumerate(zip(euclid_paths, cosmos_paths))]

    # with h5py.File(OUTPUT_H5, "w") as f:
    #     euc_ds = f.create_dataset("euclid_images", shape=(N_valid, 1, H_euc, W_euc),
    #                                maxshape=(N_valid, 1, H_euc, W_euc), chunks=True, dtype=np.float32)
    #     cos_ds = f.create_dataset("cosmos_images", shape=(N_valid, 1, H_cos, W_cos),
    #                                maxshape=(N_valid, 1, H_cos, W_cos), chunks=True, dtype=np.float32)
    #     cos_down_ds = f.create_dataset("cosmos_images_downscaled", shape=(N_valid, 1, H_SIZE, W_SIZE),
    #                                     maxshape=(N_valid, 1, H_SIZE, W_SIZE), chunks=True, dtype=np.float32)
    #     euc_up_ds = f.create_dataset("euclid_images_upscaled", shape=(N_valid, 1, H_SIZE, W_SIZE),
    #                                   maxshape=(N_valid, 1, H_SIZE, W_SIZE), chunks=True, dtype=np.float32)

    #     kept_euclid_paths = []
    #     kept_cosmos_paths = []
    #     kept_cat_indices  = []
    #     write_idx = 0
    #     skipped = 0
    #     with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    #         results = executor.map(process_pair, args_list, chunksize=100)
    #         for result in tqdm(results, total=N_valid, desc="Processing", mininterval=60, dynamic_ncols=False):
    #             i, euc, cos, cos_down, euc_up, err = result
    #             if err:
    #                 print(f"\n  [WARN] skipping pair {i}: {err}")
    #                 skipped += 1
    #                 continue
    #             euc_ds[write_idx] = euc
    #             cos_ds[write_idx] = cos
    #             cos_down_ds[write_idx] = cos_down
    #             euc_up_ds[write_idx] = euc_up
    #             kept_euclid_paths.append(euclid_paths[i])
    #             kept_cosmos_paths.append(cosmos_paths[i])
    #             kept_cat_indices.append(i)
    #             write_idx += 1

    #     N_final = write_idx
    #     if N_final < N_valid:
    #         euc_ds.resize((N_final, 1, H_euc, W_euc))
    #         cos_ds.resize((N_final, 1, H_cos, W_cos))
    #         cos_down_ds.resize((N_final, 1, H_SIZE, W_SIZE))
    #         euc_up_ds.resize((N_final, 1, H_SIZE, W_SIZE))

    #     cat_grp = f.create_group("catalog")
    #     dt = h5py.string_dtype()
    #     cat_grp.create_dataset("euclid_paths", data=np.array(kept_euclid_paths, dtype=object), dtype=dt)
    #     cat_grp.create_dataset("cosmos_paths", data=np.array(kept_cosmos_paths, dtype=object), dtype=dt)

    #     kept_catalog = catalog.iloc[kept_cat_indices].reset_index(drop=True)
    #     feat_grp = cat_grp.create_group("features")
    #     for col in kept_catalog.columns:
    #         vals = kept_catalog[col]
    #         try:
    #             feat_grp.create_dataset(col, data=vals.to_numpy(dtype=np.float32, na_value=np.nan))
    #         except (ValueError, TypeError):
    #             feat_grp.create_dataset(col, data=np.array(vals.astype(str).tolist(), dtype=object), dtype=dt)
    #     f.attrs["num_pairs"] = N_final
    #     f.attrs["num_channels"] = 1
    #     f.attrs["euclid_shape"] = [H_euc, W_euc]
    #     f.attrs["cosmos_shape"] = [H_cos, W_cos]
    #     f.attrs["euclid_upscaled_shape"] = [H_SIZE, W_SIZE]
    #     f.attrs["cosmos_downscaled_shape"] = [H_SIZE, W_SIZE]

    # print(f"\nDone. {N_final}/{N_valid} pairs written to {OUTPUT_H5}")
    # if skipped:
    #     print(f"  {skipped} pairs skipped due to processing errors and dropped from the catalogue.")


if __name__ == "__main__":
    main()
