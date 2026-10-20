#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_cos_sim_all
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_cos_sim_all_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_cos_sim_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

# One subfolder per version (v4 ... v14), each with embeddings/embeddings.npz
CKPT_ROOT="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1"
OUT_DIR="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/cos_sim_all"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/cos_sim_all.py \
    --ckpt-root "${CKPT_ROOT}" \
    --centering global \
    --out-dir   "${OUT_DIR}"
