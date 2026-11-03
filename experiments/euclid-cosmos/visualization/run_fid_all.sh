#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_fid_all
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_fid_all_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_fid_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

# One subfolder per version (v4 ... v15), each with fid/fid.csv from run_compute_fid.sh
ROOT="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1"
OUT_DIR="${ROOT}/fid_all"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/fid_all.py \
    --root    "${ROOT}" \
    --out-dir "${OUT_DIR}"
