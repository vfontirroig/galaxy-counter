#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_cosine
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_cosine_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_cosine_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

# Must be the same CKPT_DIR and h5 the embeddings were exported with
CKPT_DIR="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1/v13"
H5_FILE="/n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5"
OUT_DIR="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/v13/cosine_similarity"

mkdir -p "${OUT_DIR}"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/cosine_similarity.py \
    --embeddings "${CKPT_DIR}/embeddings/embeddings.npz" \
    --ckpt-dir   "${CKPT_DIR}" \
    --h5         "${H5_FILE}" \
    --hist-centering global \
    --out-dir    "${OUT_DIR}/cosine_similarity"
