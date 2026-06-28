#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_train
#SBATCH --output=/home/fontirro/logs/euclid_cosmos_train_%j.out
#SBATCH --error=/home/fontirro/logs/euclid_cosmos_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # matches NUM_WORKERS in train.py
#SBATCH --mem=32G
#SBATCH --gres=gpu:RTX8000:1        # request 1 RTX8000 GPU specifically
#SBATCH --time=1-00:00:00
#SBATCH --partition=compl      # change to your GPU partition
#SBATCH --chdir=/n03data/fontirro/galaxy-counter
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

python3 /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/train.py
