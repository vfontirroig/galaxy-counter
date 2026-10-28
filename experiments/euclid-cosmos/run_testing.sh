#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_test
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_test_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=05:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1/v5"
PLOT_DIR="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/v5/test"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/testing.py \
    --checkpoint "${CKPT_DIR}/best-epoch=75-step=92000.ckpt" \
    --h5         "/n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5" \
    --indices    "${CKPT_DIR}/test_indices.npy" \
    --out        "${PLOT_DIR}/test_results_cos_euc.png" \
    --n-plot     5 \
    --num-steps  100 \
    --direction "cosmos-to-euclid" \
    --metrics "${PLOT_DIR}/metrics_cos_euc.csv"