#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_train
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/embd_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/embd_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # matches NUM_WORKERS in train.py
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=3-00:00:00
#SBATCH --partition=pscompl      # change to your GPU partition
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1/v14"
H5_FILE="/n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5"
OUT_DIR="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1/v14/embeddings"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/export_embeddings.py \
--checkpoint  "${CKPT_DIR}/best-epoch=52-step=64000.ckpt" \
--h5          "${H5_FILE}" \
--out         "${OUT_DIR}/embeddings.npz" \