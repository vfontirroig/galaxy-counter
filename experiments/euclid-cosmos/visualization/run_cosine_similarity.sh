#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_cosine
#SBATCH --array=15            
##SBATCH --array=4-14             # one task per version: v4 ... v14
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_cosine_%A_v%a.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_cosine_%A_v%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

VERSION="v${SLURM_ARRAY_TASK_ID}"
# Must be the same CKPT_DIR and h5 the embeddings were exported with
CKPT_DIR="/n03data/fontirro/euclid-cosmos/checkpoints/euclid-cosmos-nir-h-f150w/test-1-phase1/${VERSION}"
H5_FILE="/n03data/fontirro/data_files/euclid_cosmos_pairs_nir_h_f150w_v2.h5"
OUT_DIR="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/${VERSION}/cosine_similarity"

EMBEDDINGS="${CKPT_DIR}/embeddings/embeddings.npz"
if [[ ! -f "${EMBEDDINGS}" ]]; then
    echo "No embeddings for ${VERSION} at ${EMBEDDINGS}; run run_export_embeddings.sh first." >&2
    exit 1
fi
echo "Version ${VERSION}: ${EMBEDDINGS}"

mkdir -p "${OUT_DIR}"

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/cosine_similarity.py \
    --embeddings "${EMBEDDINGS}" \
    --ckpt-dir   "${CKPT_DIR}" \
    --h5         "${H5_FILE}" \
    --hist-centering global \
    --out-dir    "${OUT_DIR}"
