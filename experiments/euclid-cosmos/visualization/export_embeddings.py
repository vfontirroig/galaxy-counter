"""
Export the raw encoder embeddings (pre-UMAP) from the trained Euclid x COSMOS model.

umap_latent.py encodes the images and then keeps only the 2D UMAP coordinates.
Distances in a UMAP embedding are not quantitative, so any cosine-similarity or
bimodality analysis has to be done in the original dimension - this script writes
those full-dimensional vectors to a single .npz.

Every pair in the h5 is encoded, in dataset order: row i of each saved array is
h5 index i, so the arrays join straight to catalog/features and to the
dataset_idx column of umap_groups_encoder_*.csv.

Usage:
    python experiments/euclid-cosmos/visualization/export_embeddings.py \
        --checkpoint /n03data/fontirro/checkpoints/euclid-cosmos-phase1/best-epoch=21-step=98000.ckpt \
        --h5         /n03data/fontirro/data_files/euclid_cosmos_pairs.h5 \
        --out        /n03data/fontirro/results_model/embeddings.npz
"""

import os
import sys
import argparse
from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_here = os.path.dirname(__file__)
_experiment_dir = os.path.abspath(os.path.join(_here, ".."))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
sys.path.insert(0, _experiment_dir)
sys.path.insert(0, os.path.join(_repo_root, "src"))

from dataset import EuclidCosmosDataset
from train import EuclidCosmosModel, collate_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--h5",         required=True)
    p.add_argument("--out",        default="embeddings.npz")
    p.add_argument("--n-samples",  type=int, default=-1,
                   help="Encode only the first N pairs, for a quick test run. "
                        "Default -1 encodes every pair.")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print(f"Loading checkpoint: {args.checkpoint}")
    model = EuclidCosmosModel.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)
    torch.set_grad_enabled(False)

    # --- Dataset, in order, no sampling ---
    dataset = EuclidCosmosDataset(args.h5)
    n = len(dataset) if args.n_samples == -1 else min(args.n_samples, len(dataset))
    loader = DataLoader(
        Subset(dataset, range(n)) if n < len(dataset) else dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, dataset=dataset),
    )
    print(f"Encoding {n} pairs...")

    # --- Encode all images ---
    euclid_emb1_list, cosmos_emb1_list = [], []
    euclid_emb2_list, cosmos_emb2_list = [], []

    with torch.no_grad():
        for euclid, cosmos, _, _, _ in loader:
            euclid = euclid.to(device)
            cosmos = cosmos.to(device)
            euclid_emb1_list.append(model.encoder_1(euclid).flatten(1).cpu())
            cosmos_emb1_list.append(model.encoder_1(cosmos).flatten(1).cpu())
            euclid_emb2_list.append(model.encoder_2(euclid).flatten(1).cpu())
            cosmos_emb2_list.append(model.encoder_2(cosmos).flatten(1).cpu())

    euclid_emb1 = torch.cat(euclid_emb1_list).numpy()
    cosmos_emb1 = torch.cat(cosmos_emb1_list).numpy()
    euclid_emb2 = torch.cat(euclid_emb2_list).numpy()
    cosmos_emb2 = torch.cat(cosmos_emb2_list).numpy()
    print(f"Encoded {len(euclid_emb1)} pairs. Embedding dim: {euclid_emb1.shape[1]}")

    # --- Save ---
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        euclid_emb1=euclid_emb1,
        cosmos_emb1=cosmos_emb1,
        euclid_emb2=euclid_emb2,
        cosmos_emb2=cosmos_emb2,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
