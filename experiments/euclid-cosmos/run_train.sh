#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_train
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/train_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # matches NUM_WORKERS in train.py
#SBATCH --mem=32G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2-00:00:00
#SBATCH --partition=pscompl      # change to your GPU partition
set -euo pipefail

# Don't `source .../activate` — the venv's activate script bakes in the
# automounter-canonicalized `/automnt/n03data/...` prefix, which is
# unreachable from compute nodes and silently falls through PATH to the
# system platform-python instead. Call the venv's python3 directly via the
# working /n03data/... path so it resolves the venv's own site-packages.
python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/train.py
