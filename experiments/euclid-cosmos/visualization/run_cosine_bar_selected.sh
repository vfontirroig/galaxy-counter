#!/usr/bin/env bash
#SBATCH --job-name=euclid_cosmos_cosine_bar
#SBATCH --output=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_cosine_bar_%j.out
#SBATCH --error=/n03data/fontirro/euclid-cosmos/logs/euclid_cosmos_cosine_bar_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --partition=pscompl
set -euo pipefail

source /n03data/fontirro/.galaxy-counter-env/bin/activate

# Same OUT_DIR as run_cosine_similarity.sh, which writes per_galaxy.csv there
OUT_DIR="/n03data/fontirro/euclid-cosmos/plots_model/euclid-cosmos-nir-h-f150w/test-1-phase1/v4/cosine_similarity"
VARIANT="enc1_flat_global"   # enc{1,2}_{flat,pooled}_{raw,global,per_survey}
SEED=42

# h5 indices (dataset_idx) of the galaxies to plot, separated by spaces.
# Leave empty, INDICES=(), to plot 8 random test galaxies instead.
INDICES=(58623 24090 26211 878 77756 32061 36921 15388)

if [ ${#INDICES[@]} -gt 0 ]; then
    SELECT=(--indices "${INDICES[@]}")
    OUT_NAME="cosine_bar_${VARIANT}_selected.png"
else
    SELECT=(--n 8 --seed "${SEED}")
    OUT_NAME="cosine_bar_${VARIANT}_seed${SEED}.png"
fi

python /n03data/fontirro/euclid-cosmos/galaxy-counter/experiments/euclid-cosmos/visualization/cosine_bar_selected.py \
    --per-galaxy "${OUT_DIR}/cosine_similarity/per_galaxy.csv" \
    --variant    "${VARIANT}" \
    "${SELECT[@]}" \
    --out        "${OUT_DIR}/${OUT_NAME}"
