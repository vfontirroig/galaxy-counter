#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_test
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_test_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=05:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

CKPT_DIR="/n03data/fontirro/checkpoints/euclid-cosmos-vis-f150w/test-4-phase1"
PLOT_DIR="/n03data/fontirro/plots_model/euclid-cosmos-vis-f150w/test-1-phase1"

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/testing.py \
    --checkpoint "${CKPT_DIR}/best-epoch=218-step=195000.ckpt" \
    --h5         "/n03data/fontirro/data_files/euclid_cosmos_pairs_vis_f150w.h5" \
    --indices    "${CKPT_DIR}/test_indices.npy" \
    --out        "${PLOT_DIR}/test_results.png" \
    --n-plot     8 \
    --num-steps  100 \
    --direction "euclid-to-cosmos"
