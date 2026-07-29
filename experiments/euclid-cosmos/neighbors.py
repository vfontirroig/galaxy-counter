"""
Add same-survey 1 nearest neighbor indices to an existing paired HDF5 file
(built by build_hdf5.py), based on pixel-level image similarity. This is the
"same-instrument neighbor" signal described in new_dataset_guide.md step 3
(f['neighbor_idx_a'] / f['neighbor_idx_b']): flatten each image and run
sklearn NearestNeighbors to find, for every image, the closest *other* image
in the same survey's array.

Run after build_hdf5.py has produced the paired file:
    python experiments/euclid-cosmos/neighbors.py

Writes, at the top level of the HDF5 file:
    neighbor_idx_euclid   — (N, 1) int64, index into euclid_images, -1 = none
    neighbor_dist_euclid  — (N,) float64, pixel-space distance to that neighbor
    neighbor_idx_cosmos   — (N, 1) int64, index into cosmos_images
    neighbor_dist_cosmos  — (N,) float64

At training time, which array to index into (neighbor_idx_euclid vs.
neighbor_idx_cosmos) depends on which survey the current anchor is drawn
from — see anchor_survey / anchor_is_hsc in src/galaxy_counter/neighbors.py
for the existing pattern.
"""

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

H5_PATH = "/n03data/fontirro/data_files/euclid_cosmos_pairs_vis_f150w_v3.h5"

# image dataset name -> output neighbor dataset suffix
SURVEY_IMAGE_KEYS = {
    "euclid_images": "euclid",
    "cosmos_images": "cosmos",
}

MAX_NEIGHBOR_PIXEL_DIST = None  # optional cap on pixel-space distance; farther matches become -1

RA_COL = "ra"    # column name for right ascension (degrees) in catalog/features, for the example below
DEC_COL = "dec"  # column name for declination (degrees) in catalog/features

# ---------------------------------------------------------------------------

import numpy as np
import h5py
from sklearn.neighbors import NearestNeighbors


def nearest_neighbor_pixel(images, max_distance=None, metric="euclidean"):
    """
    For each image in `images`, find the index of its closest *other* image
    in the same array by flattened pixel distance.

    Args:
    - images : (N, C, H, W) array-like.
    - max_distance : float or None.
        Matches farther than this (in flattened pixel distance) are
        discarded (-1 index, inf distance).
    - metric : str, passed to sklearn.neighbors.NearestNeighbors.

    Returns:
    - neighbor_idx : (N,) int64 array. Index into `images` of the nearest
        other image, or -1 if none found within max_distance.
    - distance : (N,) float64 array. Pixel-space distance to that neighbor
        (np.inf where neighbor_idx == -1).
    """
    images = np.asarray(images)
    n = images.shape[0]
    if n < 2:
        return np.full(n, -1, dtype=np.int64), np.full(n, np.inf)

    flat = images.reshape(n, -1).astype(np.float32)

    # n_neighbors=2: the closest match to itself is always itself at
    # distance 0 (index 0 after sorting), so we take the second column.
    nn = NearestNeighbors(n_neighbors=2, metric=metric)
    nn.fit(flat)
    dist, idx = nn.kneighbors(flat)

    neighbor_idx = idx[:, 1].astype(np.int64)
    distance = dist[:, 1].astype(np.float64)

    if max_distance is not None:
        too_far = distance > max_distance
        neighbor_idx[too_far] = -1
        distance[too_far] = np.inf

    return neighbor_idx, distance


def add_neighbors_to_h5(h5_path, survey_image_keys, max_distance=None, metric="euclidean"):
    """
    Open an existing HDF5 file and add same-survey 1-NN pixel neighbor
    indices for each image dataset listed in survey_image_keys.

    Args:
    - h5_path : path to the HDF5 file (opened in append mode).
    - survey_image_keys : dict mapping an existing image dataset name
        (e.g. "euclid_images") to an output suffix (e.g. "euclid"); writes
        f["neighbor_idx_<suffix>"] and f["neighbor_dist_<suffix>"].
    - max_distance, metric : passed to nearest_neighbor_pixel.
    """
    with h5py.File(h5_path, "a") as f:
        for image_key, suffix in survey_image_keys.items():
            print(f"Computing pixel-kNN for '{image_key}'...")
            images = f[image_key][:]
            neighbor_idx, distance = nearest_neighbor_pixel(images, max_distance=max_distance, metric=metric)
            n_with_neighbor = int((neighbor_idx != -1).sum())
            print(f"  {n_with_neighbor}/{len(neighbor_idx)} images have a same-survey neighbor")

            idx_key, dist_key = f"neighbor_idx_{suffix}", f"neighbor_dist_{suffix}"
            for key in (idx_key, dist_key):
                if key in f:
                    del f[key]
            f.create_dataset(idx_key, data=neighbor_idx[:, None])
            f.create_dataset(dist_key, data=distance)

def main():
    # Example: compute the neighbor for one object, per survey, without
    # writing anything back to the HDF5 file (read-only).
    with h5py.File(H5_PATH, "r") as f:
        ra = f[f"catalog/features/{RA_COL}"][:]
        dec = f[f"catalog/features/{DEC_COL}"][:]

        i = 0
        for image_key, suffix in SURVEY_IMAGE_KEYS.items():
            images = f[image_key][:]
            neighbor_idx, distance = nearest_neighbor_pixel(images, max_distance=MAX_NEIGHBOR_PIXEL_DIST)
            j = neighbor_idx[i]
            print(
                f"Object {i} ({suffix}): anchor ra={ra[i]:.6f} dec={dec[i]:.6f}; "
                f"nearest same-survey neighbor is index {j} (pixel distance {distance[i]:.4f}), "
                + (f"ra={ra[j]:.6f} dec={dec[j]:.6f}" if j != -1 else "no neighbor found")
            )
        print("end.")

    #Adding the neighbors to the HDF5 file
    add_neighbors_to_h5(H5_PATH, SURVEY_IMAGE_KEYS, max_distance=MAX_NEIGHBOR_PIXEL_DIST, metric="euclidean")

if __name__ == "__main__":
    main()