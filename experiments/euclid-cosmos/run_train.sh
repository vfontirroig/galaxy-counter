#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_train
##SBATCH --array=0-1             # one task per entry in SEEDS below
#SBATCH --array=0                # single seed run
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/train_%A_%a.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/train_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # matches NUM_WORKERS in train.py
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=3-00:00:00
#SBATCH --partition=pscompl      # change to your GPU partition
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

# Each array task trains with its own seed into CKPT_DIR/seed<SEED> (see train.py).
# SEEDS=(1337 2024)
SEEDS=(42)
export SEED=${SEEDS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "Training with SEED=${SEED}"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/train.py
