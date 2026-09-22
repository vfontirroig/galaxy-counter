#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_umap
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_umap_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_umap_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1/v1"
PLOT_DIR="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/v1"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/umap_latent.py \
    --checkpoint "${CKPT_DIR}/best-epoch=24-step=30000.ckpt" \
    --h5         "/n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v1.h5" \
    --out        "${PLOT_DIR}/umap_val.png" \
    --out-cutouts "${PLOT_DIR}/umap_val_cutouts.png" \
    --indices    "${CKPT_DIR}/val_indices.npy"