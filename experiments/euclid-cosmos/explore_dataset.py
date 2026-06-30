import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# Per-survey [mean, std] of preprocessed pixel values.
NORM_DICT = {
    "euclid": [0.019, 0.019],
    "euclid_up": [0.019, 0.019],
    "cosmos": [0.044, 0.161],
    "cosmos_ds": [0.044, 0.121],
}


class EuclidCosmosDataset(Dataset):
    """
    Lazy-loading dataset for paired Euclid/COSMOS cutouts.

    Each sample is one galaxy observed by both instruments.
    Returns (anchor, cond, metadata) where:
      - bidirectional=False (default): anchor=Euclid, cond=COSMOS always.
      - bidirectional=True: even indices → anchor=Euclid, cond=COSMOS;
                            odd  indices → anchor=COSMOS, cond=Euclid.
    metadata["anchor_survey"] tells which survey is the anchor.
    """

    def __init__(self, hdf5_path: str, norm_dict: dict = NORM_DICT,
                 bidirectional: bool = False):
        self.hdf5_path = hdf5_path
        self.norm_dict = norm_dict
        self.bidirectional = bidirectional
        self.file = None  # opened lazily, once per worker

        with h5py.File(hdf5_path, "r") as f:
            self.N = int(f.attrs["num_pairs"])

    def _open_file(self):
        if self.file is None:
            self.file = h5py.File(self.hdf5_path, "r", libver="latest", swmr=True)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        self._open_file()

        euc = torch.from_numpy(self.file["euclid_images_upscaled"][idx].copy())
        cos = torch.from_numpy(self.file["cosmos_images_downscaled"][idx].copy())

        euc_mean, euc_std = self.norm_dict["euclid_up"]
        cos_mean, cos_std = self.norm_dict["cosmos_ds"] 
        euc = (euc - euc_mean) / euc_std
        cos = (cos - cos_mean) / cos_std

        if self.bidirectional and idx % 2 == 1:
            anchor, cond, survey = cos, euc, "cosmos"
        else:
            anchor, cond, survey = euc, cos, "euclid"

        metadata = {"idx": idx, "anchor_survey": survey}
        return anchor, cond, metadata


def collate_pairs(batch):
    """Stack (euclid, cosmos, metadata) samples into a batch."""
    euclid = torch.stack([b[0] for b in batch])
    cosmos = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return euclid, cosmos, metadata