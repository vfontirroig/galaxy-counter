#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_fid
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_fid_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_fid_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=08:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

# Compute nodes may have no internet, so the Inception weights must already be
# cached. Fetch them once on a login node with:
#   python -c "from pytorch_fid.inception import InceptionV3; InceptionV3()"
INCEPTION_WEIGHTS="${TORCH_HOME:-${HOME}/.cache/torch}/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth"
if [[ ! -f "${INCEPTION_WEIGHTS}" ]]; then
    echo "Missing Inception weights: ${INCEPTION_WEIGHTS}" >&2
    exit 1
fi

CKPT_DIR="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1/v13"
OUT_DIR="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/v13/fid"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/fid_plot.py \
    --checkpoint  "${CKPT_DIR}/best-epoch=75-step=92000.ckpt" \
    --h5          "/n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5" \
    --indices     "${CKPT_DIR}/test_indices.npy" \
    --out-dir     "${OUT_DIR}" \
    --n-samples   3 \
    --num-steps   100 \
    --n-bootstrap 100 \
    --seed        42
