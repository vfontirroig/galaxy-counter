#!/usr/bin/env bash
#SBATCH --job-name=check_encoders
#SBATCH --output=/n03data/fontirro/logs/check_encoders_%j.out
#SBATCH --error=/n03data/fontirro/logs/check_encoders_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=compl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1/v1"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/check_encoders.py \
    --checkpoint "${CKPT_DIR}/best-epoch=40-step=50000.ckpt" \
    --h5         "/n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v1.h5"
