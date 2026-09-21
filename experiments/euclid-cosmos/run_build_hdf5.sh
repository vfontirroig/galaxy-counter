#!/usr/bin/env bash
#SBATCH --job-name=build_hdf5
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/build_hdf5_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/build_hdf5_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24        # set NUM_WORKERS in build_hdf5.py to match this
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -euo pipefail

# Don't `source .../activate` — the venv's activate script bakes in the
# automounter-canonicalized `/automnt/n03data/...` prefix, which is
# unreachable from compute nodes and silently falls through PATH to the
# system platform-python instead. Call the venv's python3 directly via the
# working /n03data/... path so it resolves the venv's own site-packages.
/n03data/fontirro/.galaxy-counter-env/bin/python3 \
    /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/build_hdf5.py