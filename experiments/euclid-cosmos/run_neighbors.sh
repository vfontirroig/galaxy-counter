#!/usr/bin/env bash
#SBATCH --job-name=neighbors
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/neighbors_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/neighbors_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/neighbors.py
