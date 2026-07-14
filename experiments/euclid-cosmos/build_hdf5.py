"""
Build an HDF5 file with paired Euclid NIR-Y and NIR-J cutouts.

The native spatial size of the first file in each band is used to
pre-allocate the datasets. All images of the same band are assumed
to share the same size. Datasets follow the (N, C, H, W) layout from
new_dataset_guide.md.

HDF5 layout:
    euclid_nir_y_images     — (N, 1, H_euc_y, W_euc_y) float32, NIR-Y band
    euclid_nir_j_images     — (N, 1, H_euc_j, W_euc_j) float32, NIR-J band
    catalog/euclid_y_paths  — string array (N,)
    catalog/euclid_j_paths  — string array (N,)
    attrs: num_pairs, num_channels, euclid_y_shape, euclid_j_shape

Preprocessing: none. Both bands are Euclid-native, so there's no
cross-survey (COSMOS) zeropoint to reconcile — raw pixel values are
written as-is.

Edit the CONFIG block below, then run:
    python experiments/euclid-cosmos/build_hdf5.py
"""

import sys
import os

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

CATALOG_PATH = "/n03data/fontirro/data_files/cat_crossmatch_mag27_mag25.csv"  # path to the paired catalog

# EUCLID_COL = "file_euclid_vis"              # column name for the Euclid FITS file path
# COSMOS_COL = "file_cosmos_f115w"            # column name for the COSMOS FITS file path

EUCLID_Y  = "40_file_euclid_nir_y"  # column name for the Euclid NIR-Y FITS file path
EUCLID_J  = "40_file_euclid_nir_j"  # column name for the Euclid NIR-J FITS file path

# EUCLID_EXISTS_COL = "cutout_euc_40_vis"    # boolean column: True if Euclid cutout exists
# COSMOS_EXISTS_COL = "cutout_cos_120_115w"  # boolean column: True if COSMOS cutout exists

EUCLID_Y_EXISTS_COL = "cutout_euc_40_nir_y"    # boolean column: 1 if Euclid NIR-Y cutout exists
EUCLID_J_EXISTS_COL = "cutout_euc_40_nir_j"    # boolean column: 1 if Euclid NIR-J cutout exists

# EUCLID_DIR_PATH = "/n03data/fontirro/cutouts/euclid/40_cutouts/40_cutouts-vis/"  # base directory for Euclid VIS cutouts.
# COSMOS_DIR_PATH = "/n03data/fontirro/cutouts/cosmos/120_cutouts/f115w/"  # base directory for COSMOS F115W cutouts.

EUCLID_Y_DIR_PATH = "/n03data/fontirro/cutouts/euclid/40_cutouts/NIR-Y/"  # base directory for Euclid NIR-Y cutouts.
EUCLID_J_DIR_PATH = "/n03data/fontirro/cutouts/euclid/40_cutouts/NIR-J/"  # base directory for Euclid NIR-J cutouts.

EUCLID_HDU = 1   # HDU index for Euclid data (usually 1 for science extension); used for both NIR-Y and NIR-J

OUTPUT_H5 = "/n03data/fontirro/data_files/euclid_pairs_nir_y_nir_j.h5"

NUM_WORKERS = 16  # parallel threads for loading + preprocessing

# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import h5py
from astropy.io import fits
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import torch


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
    """Load a Euclid NIR-Y / NIR-J pair. Returns (i, euc_y, euc_j, error).
    No zeropoint rescaling: both bands are Euclid-native, so there's no
    cross-survey (COSMOS) zeropoint to reconcile here."""
    i, ep_y, ep_j = args
    try:
        euc_y = load_fits(ep_y, EUCLID_HDU).squeeze(0).numpy()
        euc_j = load_fits(ep_j, EUCLID_HDU).squeeze(0).numpy()
        return i, euc_y, euc_j, None
    except Exception as e:
        return i, None, None, str(e)


def main():
    catalog = pd.read_csv(CATALOG_PATH)
    print(f"Catalog loaded: {len(catalog)} pairs")
    #print(f"Columns: {list(catalog.columns)}")

    mask = (catalog[EUCLID_Y_EXISTS_COL] == 1) & (catalog[EUCLID_J_EXISTS_COL] == 1)
    catalog = catalog[mask].reset_index(drop=True)
    print(f"Pairs with both cutouts present: {len(catalog)} / {len(mask)}")

    # euclid_paths = [os.path.join(EUCLID_DIR_PATH, p) for p in catalog[EUCLID_COL]]
    # cosmos_paths = [os.path.join(COSMOS_DIR_PATH, p) for p in catalog[COSMOS_COL]]

    euclid_y_paths = [os.path.join(EUCLID_Y_DIR_PATH, p) for p in catalog[EUCLID_Y]]
    euclid_j_paths = [os.path.join(EUCLID_J_DIR_PATH, p) for p in catalog[EUCLID_J]]    

    N_y = len(euclid_y_paths)
    N_j = len(euclid_j_paths)

    # ------------------------------------------------------------------
    # Pass 1: filter out Euclid cutouts with >= 10% zero pixels
    # Uses threads (I/O bound, no torch) so no fork/spawn overhead.
    # ------------------------------------------------------------------
    print(f"\nScanning {N_y} Euclid Y files and {N_j} Euclid J files for zero-pixel fraction...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        zero_fracs_y = list(tqdm(
            pool.map(euclid_zero_frac, euclid_y_paths),
            total=N_y, desc="Scanning", mininterval=5,
        ))
        zero_fracs_j = list(tqdm(
            pool.map(euclid_zero_frac, euclid_j_paths),
            total=N_j, desc="Scanning", mininterval=5,
        ))

    valid = [zf_y < 0.10 and zf_j < 0.10 for zf_y, zf_j in zip(zero_fracs_y, zero_fracs_j)]
    euclid_y_paths = [p for p, v in zip(euclid_y_paths, valid) if v]
    euclid_j_paths = [p for p, v in zip(euclid_j_paths, valid) if v]
    catalog = catalog[valid].reset_index(drop=True)
    N_valid = len(catalog)

    print(f"Valid pairs after filtering: {N_valid}/{N_y}  ({N_y - N_valid} skipped, zero_frac >= 10%)")


    if N_valid == 0:
        print("[ERROR] No valid pairs found — check EUCLID_HDU and file paths.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Quick sanity check on first valid pair
    # ------------------------------------------------------------------
    print("\nTesting first pair (sequential)...")
    _, t_euc_y, t_euc_j, t_err = process_pair(
        (0, euclid_y_paths[0], euclid_j_paths[0])
    )
    if t_err:
        print(f"[ERROR] First pair failed:\n{t_err}")
        sys.exit(1)
    print(f"  euc_y [{t_euc_y.min():.4f}, {t_euc_y.max():.4f}]  euc_j [{t_euc_j.min():.4f}, {t_euc_j.max():.4f}]")
    print("OK.\n")

    H_euc_y, W_euc_y = get_spatial_size(euclid_y_paths[0], EUCLID_HDU)
    H_euc_j, W_euc_j = get_spatial_size(euclid_j_paths[0], EUCLID_HDU)
    print(f"EUCLID NIR Y image size : {H_euc_y} x {W_euc_y}")
    print(f"EUCLID NIR J image size : {H_euc_j} x {W_euc_j}")



   
    # ------------------------------------------------------------------
    # Pass 2: process and write — pairs that fail processing are dropped
    # (not left as zero-filled rows); datasets are shrunk to the final
    # count at the end so the catalogue only contains real cutouts.
    # ------------------------------------------------------------------

    args_list = [(i, ep_y, ep_j) for i, (ep_y, ep_j) in enumerate(zip(euclid_y_paths, euclid_j_paths))]

    with h5py.File(OUTPUT_H5, "w") as f:
        euc_y_ds = f.create_dataset("euclid_nir_y_images", shape=(N_valid, 1, H_euc_y, W_euc_y),
                                   maxshape=(N_valid, 1, H_euc_y, W_euc_y), chunks=True, dtype=np.float32)
        euc_j_ds = f.create_dataset("euclid_nir_j_images", shape=(N_valid, 1, H_euc_j, W_euc_j),
                                   maxshape=(N_valid, 1, H_euc_j, W_euc_j), chunks=True, dtype=np.float32)

        kept_euclid_y_paths = []
        kept_euclid_j_paths = []
        kept_cat_indices  = []
        write_idx = 0
        skipped = 0
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = executor.map(process_pair, args_list, chunksize=100)
            for result in tqdm(results, total=N_valid, desc="Processing", mininterval=60, dynamic_ncols=False):
                i, euc_y, euc_j, err = result
                if err:
                    print(f"\n  [WARN] skipping pair {i}: {err}")
                    skipped += 1
                    continue
                euc_y_ds[write_idx] = euc_y
                euc_j_ds[write_idx] = euc_j
                kept_euclid_y_paths.append(euclid_y_paths[i])
                kept_euclid_j_paths.append(euclid_j_paths[i])
                kept_cat_indices.append(i)
                write_idx += 1

        N_final = write_idx
        if N_final < N_valid:
            euc_y_ds.resize((N_final, 1, H_euc_y, W_euc_y))
            euc_j_ds.resize((N_final, 1, H_euc_j, W_euc_j))

        cat_grp = f.create_group("catalog")
        dt = h5py.string_dtype()
        cat_grp.create_dataset("euclid_y_paths", data=np.array(kept_euclid_y_paths, dtype=object), dtype=dt)
        cat_grp.create_dataset("euclid_j_paths", data=np.array(kept_euclid_j_paths, dtype=object), dtype=dt)

        kept_catalog = catalog.iloc[kept_cat_indices].reset_index(drop=True)
        feat_grp = cat_grp.create_group("features")
        for col in kept_catalog.columns:
            vals = kept_catalog[col]
            try:
                feat_grp.create_dataset(col, data=vals.to_numpy(dtype=np.float32, na_value=np.nan))
            except (ValueError, TypeError):
                feat_grp.create_dataset(col, data=np.array(vals.astype(str).tolist(), dtype=object), dtype=dt)
        f.attrs["num_pairs"] = N_final
        f.attrs["num_channels"] = 1
        f.attrs["euclid_y_shape"] = [H_euc_y, W_euc_y]
        f.attrs["euclid_j_shape"] = [H_euc_j, W_euc_j]

    print(f"\nDone. {N_final}/{N_valid} pairs written to {OUTPUT_H5}")
    if skipped:
        print(f"  {skipped} pairs skipped due to processing errors and dropped from the catalogue.")


if __name__ == "__main__":
    main()
