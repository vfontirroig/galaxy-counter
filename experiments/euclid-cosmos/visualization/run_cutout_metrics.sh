#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_fid
#SBATCH --array=4-14             # one task per version: v4 ... v14
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_fid_%A_v%a.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_fid_%A_v%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=08:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

VERSION="v${SLURM_ARRAY_TASK_ID}"
RUN_NAME="fid_v4"                # output subfolder; change it for runs with other settings
CKPT_DIR="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1/${VERSION}"
OUT_DIR="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/${VERSION}/${RUN_NAME}"

# The best checkpoint's name differs per version (best-epoch=XX-step=YYYY.ckpt),
# so find it; refuse to guess if there is not exactly one.
shopt -s nullglob
CKPTS=("${CKPT_DIR}"/best-*.ckpt)
if (( ${#CKPTS[@]} != 1 )); then
    echo "Expected one best-*.ckpt in ${CKPT_DIR}, found ${#CKPTS[@]}: ${CKPTS[*]:-none}" >&2
    exit 1
fi
echo "Version ${VERSION}: ${CKPTS[0]}"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/cutout_metrics.py \
    --checkpoint  "${CKPTS[0]}" \
    --h5          "/n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5" \
    --indices     "${CKPT_DIR}/test_indices.npy" \
    --out-dir     "${OUT_DIR}" \
    --n-samples   3 \
    --num-steps   100 \
    --precision   bf16-mixed \
    --seed        42
