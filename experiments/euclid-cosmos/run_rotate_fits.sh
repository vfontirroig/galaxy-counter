#!/usr/bin/env bash
#SBATCH --job-name=rotate_fits
#SBATCH --output=/n03data/fontirro/logs/rotate_fits_%j.out
#SBATCH --error=/n03data/fontirro/logs/rotate_fits_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # set NUM_WORKERS in rotate_fits.py to match this
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

python /n03data/fontirro/galaxy-counter/experiments/euclid-cosmos/rotate_fits.py
