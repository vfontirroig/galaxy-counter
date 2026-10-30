#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_fid
#SBATCH --array=15
##SBATCH --array=4-14             # one task per version: v4 ... v14
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_fid_%A_v%a.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_fid_%A_v%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=02:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

VERSION="v${SLURM_ARRAY_TASK_ID}"
RUN_NAME="fid"                    # output subfolder; change it for runs with other settings
GEN_DIR="/n03data/fontirro/cutouts/gen/${VERSION}"   # written by run_generate_test_set.sh
OUT_DIR="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/${VERSION}/${RUN_NAME}"

# Refuse to run on a missing or empty folder (e.g. generation not run yet for this version).
shopt -s nullglob
FILES=("${GEN_DIR}"/*.fits)
if (( ${#FILES[@]} == 0 )); then
    echo "No FITS files in ${GEN_DIR}; run run_generate_test_set.sh for ${VERSION} first" >&2
    exit 1
fi
echo "Version ${VERSION}: ${#FILES[@]} FITS files in ${GEN_DIR}"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/compute_fid.py \
    --h5          "/n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5" \
    --gen-dir     "${GEN_DIR}" \
    --out-dir     "${OUT_DIR}" \
    --draw        1 \
    --batch-size  64
